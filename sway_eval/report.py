"""The audit report: one page a deployer can read, written next to a run.

``build_report(run_dir)`` analyzes the run (``analysis.analyze``), adds the
defense comparison (``defenses.compare_defenses``) when every ordering of each
record was scored on a two-option task, and returns the report as a dict and as
Markdown. ``write_report`` saves both as ``report.json`` and ``report.md``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .analysis import analyze
from .defenses import compare_defenses
from .records import load_items

DEFENSE_LABELS = {
    "convention": "Convention layout",
    "held_out_best_fixed_order": "Held-out best fixed layout",
    "k_random_1": "1 random layout",
    "k_random_2": "2 random layouts",
    "k_random_3": "3 random layouts",
    "balanced_williams": "Balanced set (Williams)",
    "full_symmetrization": "Full average",
}


def _defenses(run_dir: Path, cfg: dict) -> dict | None:
    options = list(cfg["prompt"]["options"])
    target = cfg["prompt"].get("target")
    if len(options) != 2 or target is None:
        return {"skipped": "the defense comparison covers two-option tasks"}
    items = {it.id: it for it in load_items(run_dir / "items.jsonl")}
    lay, reps = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    with (run_dir / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] not in ("layout", "layout_repeat") or r.get("format_index", 0) or r.get("option_rotation", 0):
                continue
            p = r["probs"][r["options"].index(target)]
            reps[r["example_id"]][tuple(r["passage_ids"])].append(p)
            if r["kind"] == "layout":
                lay[r["example_id"]][tuple(r["passage_ids"])] = p
    meta = {i: items[i].meta for i in lay}
    gold = {i: (m.get("gold") == target) if m.get("gold") is not None else None for i, m in meta.items()}
    try:
        return compare_defenses(
            lay, {i: items[i].convention_order for i in lay}, repeats=reps, gold=gold,
            group={i: m.get("decisive_section") for i, m in meta.items()},
            strata={i: m.get("level") for i, m in meta.items()},
            tie_to_target=cfg["analysis"]["tie_option"] == target,
            seed=int(cfg.get("seed", 0)) + 20261104)
    except ValueError as e:
        return {"skipped": str(e)}


CALL_LABELS = {
    "layout_set": "layouts in the audited set",
    "layout_convention": "template-order calls (for the noise strata and the convention rows)",
    "format": "formatting variants at the template order",
    "option_rotation": "option rotations at the template order",
    "content_free": "content-free calls",
    "repeat": "noise study repeats",
    "layout_repeat": "extra layout repeats for noisy models",
    "canary": "canary checks",
}


def _calls(run_dir: Path, cfg: dict) -> dict[str, int]:
    """Calls made, by purpose (records.jsonl)."""
    from .runner import layouts_for
    items = {it.id: it for it in load_items(run_dir / "items.jsonl")}
    lay = cfg["layouts"]
    sets = {}
    out: dict[str, int] = {}
    with (run_dir / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r["kind"]
            if k == "layout":
                iid = r["example_id"]
                if iid not in sets:
                    sets[iid] = set(layouts_for(items[iid], lay["kind"], lay.get("n"), lay.get("seed")))
                k = "layout_set" if tuple(r["passage_ids"]) in sets[iid] else "layout_convention"
            out[k] = out.get(k, 0) + 1
    return out


def build_report(run_dir: str | Path) -> tuple[dict, str]:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    s = analyze(run_dir)
    d = _defenses(run_dir, cfg)
    n_units = len(load_items(run_dir / "items.jsonl")[0].convention_order)
    report = {"run": run_dir.name, "model": run.get("model_version") or cfg["model"].get("name"),
              "n_units": n_units, "calls": _calls(run_dir, cfg), "summary": s, "defenses": d}
    return report, _markdown(report, cfg, run)


def _f(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _markdown(rep: dict, cfg: dict, run: dict) -> str:
    s, d = rep["summary"], rep["defenses"]
    a = cfg["analysis"]
    z = s.get("flip_zone_direct") or {}
    lines = [f"# Audit report: {rep['run']}", "",
             f"Model: {rep['model']}. Records: {s['n_items']}. Options: {', '.join(cfg['prompt']['options'])}; "
             f"ties go to {a['tie_option']}.", ""]
    if s.get("halted") or run.get("halted"):
        lines += [f"**The run halted:** {s.get('halted') or run.get('halted')}. Results cover the calls made "
                  "before the halt and are not a complete audit.", ""]
    lines += ["## How often layout decides", "",
              f"- Mean flip probability (share of orderings whose decision differs from the averaged decision): "
              f"{_f(s['mean_flip_probability'])}; pairwise flip rate {_f(s['mean_pairwise_flip_rate'])}"]
    if z:
        lv = z.get("zone_levels") or []
        lines.append(f"- Flip zone at pi = {a['pi']}: " + (f"levels {lv[0]} to {lv[-1]}, width {_f(z.get('width'), 0)}"
                     + (f", holes {z['holes']}" if z.get("holes") else "") if lv else "no level reaches pi"))
        lines.append("- Mean flip probability by level: " + ", ".join(
            f"{k}: {v:.3f}" for k, v in z.get("mean_flip_by_level", {}).items()))
    zm = s.get("flip_zone_model") or {}
    if "share_in_zone" in zm:
        lines.append(f"- Share of records inside the margin zone: {_f(zm['share_in_zone'])}")
    lines += ["", "## What it costs, and whether it is order or brittleness", "",
              f"- Order SD (sigma_layout): {_f(s['mean_sd_win'], 4)}; formatting SD: {_f(s['mean_format_sd'], 4)}; "
              f"option-order SD: {_f(s.get('mean_option_order_sd'), 4)}",
              f"- rho = sigma_layout / sigma_format: {_f(s.get('layout_to_format_sd_ratio'), 2)}",
              f"- Order variance, summed over options (the Brier cost that averaging removes): {_f(s['mean_order_variance_classes'], 4)}",
              f"- Accuracy against the stated rule: convention layout {_f(s.get('accuracy_convention'))}, "
              f"averaged {_f(s.get('accuracy_symmetrized'))}"]
    n = s.get("noise") or {}
    lines += ["", "## Noise and canaries", ""]
    if n:
        lines.append(f"- Run-to-run noise share of order variance: {_f(n.get('noise_share'))}; "
                     f"corrected order variance {_f(n.get('mean_order_variance_classes_corrected'), 4)}")
    else:
        lines.append("- No repeat study in this config.")
    c = s.get("canary") or {}
    lines.append(f"- Canary checks: {c.get('checks', 0)}, alarms {c.get('alarms', 0)}"
                 + (" (each alarm re-checked at once; the run halts only when the re-check also alarms)" if c else ""))
    lines += ["", "## What each defense buys", ""]
    if d and "table" in d:
        lines += ["| Defense | Calls per record | Agreement with the full average | Section gap | Residual |",
                  "|---|---|---|---|---|"]
        rows = {**{k: d["table"][k] for k in DEFENSE_LABELS if k in d["table"]},
                **{k: v for k, v in d["table"].items() if k.startswith("adaptive_screen")}}
        for k, x in rows.items():
            label = DEFENSE_LABELS.get(k) or f"Adaptive screen ({k.rsplit('_', 1)[1].replace('sigma', ' sigma')})"
            ci = x.get("agreement_ci95") or [None, None]
            lines.append(f"| {label} | {x['calls']:.1f} | {x['agreement']:.3f} [{_f(ci[0], 2)}, {_f(ci[1], 2)}] | "
                         f"{_f(x.get('section_gap'), 2)} | {x['residual']:.5f} |")
        lines += ["", "Agreement: share of records whose decision matches the decision on the average over all "
                  "orderings, with a 95% bootstrap interval over records. Section gap: accuracy of records whose "
                  "decisive section is Experience minus that of records whose decisive section is Skills, where the "
                  "records carry that field. Residual: mean squared deviation from the full average."]
    elif cfg["layouts"]["kind"] == "williams":
        n_units = rep["n_units"]
        full = math.factorial(n_units)
        lines += [f"- This audit used the balanced set ({_balanced_calls(n_units)} orderings per record, plus one "
                  "call at the convention layout for the convention rows), the recommended default: the averaged "
                  "decisions above are the balanced-set decisions, in which every position effect and first-order "
                  "carryover is averaged out exactly.",
                  f"- Comparing every defense against the full average needs `layouts.kind: all` ({full:,} orderings "
                  f"per record, {full * s['n_items']:,} calls for these records); that option exists to validate the balanced set on a model, not for routine audits."]
    else:
        lines.append(f"- Not computed: {(d or {}).get('skipped', 'no layouts')}")
    c = rep["calls"]
    n = s["n_items"]
    lines += ["", f"## Calls made: {sum(c.values()):,} for {n} records", ""]
    for k, label in CALL_LABELS.items():
        if c.get(k):
            lines.append(f"- {c[k]:,} {label} ({c[k] / n:.1f} per record)")
    return "\n".join(lines) + "\n"


def _balanced_calls(n_units: int) -> int:
    """Orderings in a Williams set: n for even n, 2n for odd n."""
    return n_units if n_units % 2 == 0 else 2 * n_units


def write_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    rep, md = build_report(run_dir)
    (run_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    return run_dir / "report.md"
