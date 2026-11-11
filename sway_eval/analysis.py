"""Turn a run folder's records into per-record measures and a run summary.

Reads ``items.jsonl``, ``records.jsonl`` and ``config.json`` from a run
folder and writes ``per_item.jsonl`` and ``summary.json`` next to them. It
reports numbers only; thresholds are applied exactly as the config states
them, and nothing here changes a threshold.

Decisions are the argmax over the options (``multiclass``), with ties going
to ``analysis.tie_option``. The flip zone is drawn on ``analysis.axis``, an
item field such as a qualification level or a net-evidence score.

With a ``noise`` section, each item gets a stratum from its convention-layout
margin, the repeat study gives each stratum's noise variance and repeat flip
rate, order variance is reported raw and with the stratum's noise variance
subtracted (clipped at zero), and the flip zone is checked against the noise
floor (``measures.flip_zone_noise``). Items with extra calls per layout have
their flip probability computed on the per-layout means.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from .measures import flip_zone_direct, flip_zone_margin, flip_zone_model, flip_zone_noise, model_crossing, unit_position_effects
from .multiclass import (
    argmax,
    confident_flip,
    confident_pairwise_flip_rate,
    flip_stats,
    mean_vector,
    order_variance_classes,
    pairwise_flip_rate,
    winner_margin,
)
from .noise import STRATA, assign_strata, fit_binomial_shape, item_noise, pool
from .layouts import williams_orderings
from .records import load_items, read_jsonl


def analyze(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    a = cfg["analysis"]
    options = list(cfg["prompt"]["options"])
    target = cfg["prompt"].get("target")
    tie = a["tie_option"]
    taus = a.get("taus", [a["tau"]] if "tau" in a else [])
    gold_map = a.get("gold_map", {})
    items = {it.id: it for it in load_items(run_dir / "items.jsonl")}

    layout: dict[str, dict[tuple, list[float]]] = defaultdict(dict)
    repeats: dict[str, list[list[float]]] = defaultdict(list)
    fmt: dict[str, list[list[float]]] = defaultdict(list)
    content_free: dict[str, list[float]] = {}
    rotations: dict[str, list[list[float]]] = defaultdict(list)
    extra: dict[str, dict[tuple, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for r in read_jsonl(run_dir / "records.jsonl"):
        if sorted(r["options"]) != sorted(options):
            raise ValueError(f"record options {r['options']} differ from the config's {options}")
        iid = r["example_id"]
        p = [r["probs"][r["options"].index(o)] for o in options]  # back to config order
        if r["kind"] == "layout":
            layout[iid][tuple(r["passage_ids"])] = p
        elif r["kind"] == "repeat":
            repeats[iid].append(p)
        elif r["kind"] == "format":
            fmt[iid].append(p)
        elif r["kind"] == "content_free":
            content_free[iid] = p
        elif r["kind"] == "option_rotation":
            rotations[iid].append(p)
        elif r["kind"] == "layout_repeat":
            extra[iid][tuple(r["passage_ids"])].append(p)

    noise = _noise(cfg, items, layout, repeats, options, tie)
    # A balanced-set run also holds one convention call per item (made first, for the strata). It is
    # kept for the convention rows but left out of the average, which must be over the balanced set.
    convention_only: dict[str, list[float]] = {}
    if cfg["layouts"]["kind"] == "williams":
        for iid, by_order in layout.items():
            allowed = set(williams_orderings(items[iid].unit_ids()))
            conv = tuple(items[iid].convention_order)
            if conv not in allowed and conv in by_order:
                convention_only[iid] = by_order.pop(conv)

    rows = []
    for iid, by_order in layout.items():
        it = items[iid]
        vecs = list(by_order.values())
        single = flip_stats(vecs, options, tie)
        avg = [mean_vector([v] + extra[iid][o]) for o, v in by_order.items()] if iid in extra else None
        fs = flip_stats(avg, options, tie) if avg else single
        win = options.index(fs["reference"])
        sym = mean_vector(vecs)
        win_probs = [v[win] for v in vecs]
        margins = [winner_margin(v, win) for v in vecs]
        conv = by_order.get(tuple(it.convention_order), convention_only.get(iid))
        effects = unit_position_effects({o: v[win] for o, v in by_order.items()})
        spread = {u: max(e) - min(e) for u, e in effects.items()}
        headings = {u.id: u.heading for u in it.units}
        gold = it.meta.get("gold")
        gold_option = gold_map.get(gold, gold) if gold is not None else None
        row = {
            "item_id": iid,
            "n_layouts": len(vecs),
            "axis": it.meta.get(a["axis"]),
            "stance_count": it.meta.get("stance_count"),
            "gold_option": gold_option if gold_option in options else None,
            "p_sym": sym,
            **fs,
            "flip_probability_single_call": single["flip_probability"],
            "calls_per_layout": 1 + max((len(v) for v in extra[iid].values()), default=0) if iid in extra else 1,
            "flip_probability_first_option_ties": flip_stats(avg or vecs, options, None)["flip_probability"],
            "pairwise_flip_rate": pairwise_flip_rate(vecs, options, tie),
            "sd_win": pstdev(win_probs),
            "order_variance_classes": order_variance_classes(vecs),
            "margin_mean": fmean(margins),
            "margin_sd": pstdev(margins),
            "confident_flip": {str(t): confident_flip(vecs, options, tie, t) for t in taus},
            "confident_pairwise_flip_rate": {str(t): confident_pairwise_flip_rate(vecs, options, tie, t) for t in taus},
            "decisive_section": it.meta.get("decisive_section"),
            "placement_sensitive_unit": max(spread, key=spread.get),
            # comparable across items: the unit's heading and its place in the convention order
            "placement_sensitive_heading": headings.get(max(spread, key=spread.get)),
            "placement_sensitive_convention_position": it.convention_order.index(max(spread, key=spread.get)) + 1,
            "placement_spread": max(spread.values()),
            "p_content_free": content_free.get(iid),
        }
        if target is not None:
            ti = options.index(target)
            row["p_sym_target"] = sym[ti]
            row["sd_target"] = pstdev([v[ti] for v in vecs])
            row["order_variance_target"] = row["sd_target"] ** 2
        if conv is not None:
            d, _ = argmax(conv, options, tie)
            row["convention_decision"] = options[d]
            row["convention_disagrees"] = d != win
            if fmt.get(iid):
                row["format_sd"] = pstdev([v[win] for v in [conv] + fmt[iid]])
            if rotations.get(iid):
                row["option_order_sd"] = pstdev([v[win] for v in [conv] + rotations[iid]])
            if repeats.get(iid):
                row["repeat_sd"] = pstdev([v[win] for v in [conv] + repeats[iid]])
        if noise:
            _correct(row, noise, win_probs)
        rows.append(row)

    with (run_dir / "per_item.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = summarize(rows, a, options, target)
    if noise:
        summary["noise"] = _noise_summary(rows, noise)
        if "flip_zone_direct" in summary:
            zr = [r for r in rows if r["axis"] is not None and (r["stance_count"] or 0) >= a.get("min_stance", 0)]
            summary["flip_zone_noise"] = flip_zone_noise(
                [r["axis"] for r in zr], [r["flip_probability"] for r in zr], [r.get("noise_floor") for r in zr],
                a["pi"], a.get("level_step", 1), a["zone_width"], summary["flip_zone_direct"])
            summary["gate"]["noise_rule"] = {
                "existence_fails": summary["flip_zone_noise"]["existence_fails"],
                "width_noise_cleared": summary["flip_zone_noise"]["width"],
            }
    summary.update(_run_status(run_dir))
    summary["run"] = run_dir.name
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _noise(cfg: dict, items: dict, layout: dict, repeats: dict, options: list[str], tie: str) -> dict | None:
    """Strata and per-stratum noise from the repeat study. The convention
    call is not counted as a repeat: it chose the stratum."""
    nc = cfg.get("noise")
    if not nc:
        return None
    conv = {iid: by[tuple(items[iid].convention_order)] for iid, by in layout.items()
            if tuple(items[iid].convention_order) in by}
    strata = assign_strata(conv, nc["margin_edges"])
    per_item = {iid: item_noise(v, options, tie) for iid, v in repeats.items() if len(v) >= 2}
    groups = {k: [per_item[i] for i in per_item if strata.get(i) == k] for k in STRATA}
    shape = fit_binomial_shape(
        [fmean(v[per_item[i].majority] for v in repeats[i]) for i in per_item],
        [per_item[i].var_win ** 0.5 for i in per_item])
    return {"strata": strata, "by_stratum": {k: pool(g) for k, g in groups.items()},
            "overall": pool(list(per_item.values())), "binomial_shape_check": shape}


def _correct(row: dict, noise: dict, win_probs: list[float]) -> None:
    """Stratum noise next to the item's order variance, raw and corrected."""
    k = noise["strata"].get(row["item_id"])
    st = noise["by_stratum"].get(k, {}) if k else {}
    row["stratum"] = k
    raw_win = pstdev(win_probs) ** 2
    row["order_variance_win_raw"] = raw_win
    row["order_variance_classes_raw"] = row["order_variance_classes"]
    if not st.get("n_items"):
        return
    row["noise_var_win"] = st["noise_var_win"]
    row["noise_floor"] = st["repeat_flip_rate"]
    row["order_variance_win_corrected"] = max(0.0, raw_win - st["noise_var_win"])
    row["order_variance_classes_corrected"] = max(0.0, row["order_variance_classes"] - st["noise_var_classes"])
    row["corrected_clipped"] = raw_win < st["noise_var_win"]


def _noise_summary(rows: list[dict], noise: dict) -> dict:
    """The per-model noise row: sigma_noise, repeat flip rate, raw and
    corrected order variance, noise share, clipped items."""
    out = {k: v for k, v in noise.items() if k != "strata"}
    out["n_items_by_stratum"] = {k: sum(1 for r in rows if r.get("stratum") == k) for k in STRATA}
    with_noise = [r for r in rows if "noise_var_win" in r]
    if with_noise:
        raw = fmean(r["order_variance_win_raw"] for r in with_noise)
        nv = fmean(r["noise_var_win"] for r in with_noise)
        out.update({
            "n_items_corrected": len(with_noise),
            "mean_order_variance_win_raw": raw,
            "mean_order_variance_win_corrected": fmean(r["order_variance_win_corrected"] for r in with_noise),
            "mean_order_variance_classes_raw": fmean(r["order_variance_classes_raw"] for r in with_noise),
            "mean_order_variance_classes_corrected": fmean(r["order_variance_classes_corrected"] for r in with_noise),
            "noise_share": nv / raw if raw > 0 else None,
            "n_clipped": sum(1 for r in with_noise if r["corrected_clipped"]),
        })
    return out


def _run_status(run_dir: Path) -> dict:
    """Canary checks and any halt, from the runner's files."""
    out: dict = {}
    run = run_dir / "run.json"
    if run.exists():
        meta = json.loads(run.read_text(encoding="utf-8"))
        out["halted"] = meta.get("halted")
        out["epoch"] = meta.get("epoch", 0)
    checks = run_dir / "canary_checks.jsonl"
    if checks.exists():
        cs = list(read_jsonl(checks))
        out["canary"] = {"checks": len(cs), "alarms": sum(1 for c in cs if c["alarm"])}
    return out


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return fmean(vals) if vals else None


def summarize(rows: list[dict], a: dict, options: list[str], target: str | None) -> dict:
    out: dict = {"n_items": len(rows), "options": options, "tie_option": a["tie_option"]}
    for key in ("flip_probability", "flip_probability_first_option_ties", "pairwise_flip_rate",
                "sd_win", "order_variance_classes", "order_variance_target", "sd_target",
                "format_sd", "option_order_sd", "repeat_sd"):
        out[f"mean_{key}"] = _mean(rows, key)
    out["layout_ties"] = sum(r["layout_ties"] for r in rows)
    out["reference_ties"] = sum(1 for r in rows if r["reference_tied"])
    moves: Counter[str] = Counter()
    for r in rows:
        moves.update(r["flip_moves"])
    out["flip_moves"] = dict(moves)
    conv = [r["convention_disagrees"] for r in rows if "convention_disagrees" in r]
    out["convention_disagreement_rate"] = fmean(1.0 if c else 0.0 for c in conv) if conv else None
    if out["mean_format_sd"]:
        out["layout_to_format_sd_ratio"] = out["mean_sd_win"] / out["mean_format_sd"]
        out["rho"] = a.get("rho")
    for t in {k for r in rows for k in r["confident_flip"]}:
        out[f"confident_flip_rate_tau_{t}"] = fmean(1.0 if r["confident_flip"][t] else 0.0 for r in rows)
        out[f"confident_pairwise_flip_rate_tau_{t}"] = fmean(r["confident_pairwise_flip_rate"][t] for r in rows)

    graded = [r for r in rows if r["gold_option"] is not None and "convention_decision" in r]
    if graded:
        out["n_graded"] = len(graded)
        out["accuracy_convention"] = fmean(1.0 if r["convention_decision"] == r["gold_option"] else 0.0 for r in graded)
        out["accuracy_symmetrized"] = fmean(1.0 if r["reference"] == r["gold_option"] else 0.0 for r in graded)

    zone_rows = [r for r in rows if r["axis"] is not None
                 and (r["stance_count"] or 0) >= a.get("min_stance", 0)]
    if zone_rows:
        axis = [r["axis"] for r in zone_rows]
        out["flip_zone_direct"] = flip_zone_direct(
            axis, [r["flip_probability"] for r in zone_rows], a["pi"], a.get("level_step", 1), a["zone_width"])
        out["flip_zone_direct"]["n_items"] = len(zone_rows)
        if len(options) == 2 and target is not None:
            try:
                out["flip_zone_model"] = flip_zone_model(
                    axis, [r["p_sym_target"] for r in zone_rows], [r["sd_target"] for r in zone_rows],
                    a["pi"], a.get("threshold_level", 0), a.get("slope_window", 2))
            except ValueError as e:
                out["flip_zone_model"] = {"error": str(e)}
            # Second estimate: the slope window centred on the model's own
            # crossing instead of the rule's threshold, for models whose
            # decision boundary sits away from the rule's.
            crossing = model_crossing(axis, [r["p_sym_target"] for r in zone_rows])
            if crossing is None:
                out["flip_zone_model_centred"] = {"error": "mean symmetrized probability never crosses 0.5"}
            else:
                try:
                    out["flip_zone_model_centred"] = {"window_centre": crossing, **flip_zone_model(
                        axis, [r["p_sym_target"] for r in zone_rows], [r["sd_target"] for r in zone_rows],
                        a["pi"], crossing, a.get("slope_window", 2))}
                except ValueError as e:
                    out["flip_zone_model_centred"] = {"window_centre": crossing, "error": str(e)}
        else:
            # More than two options: bin items by the symmetrized top-two margin; the zone is
            # in margin units and needs no rule level, so every item counts; the stance subset
            # is a sensitivity row.
            def top_two(r):
                p = r["p_sym"]
                w = options.index(r["reference"])
                return p[w] - max(x for i, x in enumerate(p) if i != w)

            def margin_zone(rs):
                return flip_zone_margin([top_two(r) for r in rs], [r["flip_probability"] for r in rs], a["pi"],
                                        a.get("margin_bin_width", 0.05), a.get("margin_bin_min_n", 30))
            out["flip_zone_model"] = margin_zone(rows)
            out["flip_zone_model_stance_subset"] = margin_zone(zone_rows)
        out["gate"] = {
            "quantity": "flip_zone_direct.width",
            "value": out["flip_zone_direct"]["width"],
            "B": a.get("B"),
            "pi": a["pi"],
            "note": "reported only; any decision based on it is made outside the toolkit",
        }
    excluded = [r for r in rows if r["axis"] is not None and (r["stance_count"] or 0) < a.get("min_stance", 0)]
    if excluded:
        out["below_min_stance"] = {"n_items": len(excluded),
                                   "mean_flip_probability": fmean(r["flip_probability"] for r in excluded)}
    return out

