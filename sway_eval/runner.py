"""Run an experiment: plan the calls, confirm the cost, call the model, save records.

Every call goes through the response cache, so an interrupted run resumes
where it stopped and re-running costs nothing. A run that costs money, or is
expected to take more than an hour, needs ``yes=True`` (``--yes`` on the
command line) after its plan has been printed.

With a ``noise`` section the run goes in stages (see ``noise``):

1. the convention layout of every item, which puts items in strata;
2. the repeat study: items drawn per stratum, called ``noise.repeats``
   times at the convention layout;
3. every other call, in batches, with a canary check at the start and end
   of each batch (with a ``canary`` section); and
4. if the noise share of order variance misses ``noise.target_share``,
   extra calls per layout for items in the near and at strata, averaged in
   the analysis.

A canary check that alarms is repeated at once; if it alarms again the run
halts, and ``run.json`` says why. Data from before and after a halt are
never merged: the next run uses a new ``epoch``.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev, stdev

from .adapters.base import DecisionRequest
from .cache import CachedAdapter, ResponseCache
from .config import formats, resolve, rotated_options
from .layouts import all_orderings, williams_orderings
from .multiclass import flip_stats
from .noise import (
    STRATA,
    CanarySet,
    assign_strata,
    by_stratum,
    calibrate,
    canary_alarm,
    draw_per_stratum,
    item_noise,
    layout_repeats,
    pool,
)
from .records import item_to_dict
from .units import Item, Unit, render, render_blank, render_unit

CONTENT_FREE = "N/A"
LONG_RUN_S = 3600
# Repeat indices are part of the cache key; these ranges keep the kinds apart.
LAYOUT_REPEAT_BASE = 1000
CANARY_BASE = 100_000
CHECKS_PER_BATCH = 2  # canary at the start and at the end of each batch


class Halted(Exception):
    """A canary check alarmed twice in a row."""


@dataclass(frozen=True)
class Call:
    """One planned model call."""

    item_id: str
    kind: str  # layout | content_free | repeat | format | option_rotation | canary | layout_repeat | subset | blank | reference
    order: tuple[str, ...]
    format_index: int = 0
    repeat: int = 0
    option_rotation: int = 0
    tag: str = ""  # which canary check a canary call belongs to


@dataclass(frozen=True)
class RunPlan:
    n_items: int
    calls_by_kind: dict
    total_calls: int
    est_input_tokens: int
    est_cost_usd: float
    est_wall_s: float
    paid: bool

    def needs_confirmation(self) -> bool:
        return self.paid or self.est_wall_s > LONG_RUN_S

    def text(self) -> str:
        kinds = ", ".join(f"{k}: {v:,}" for k, v in self.calls_by_kind.items())
        bound = " at most" if any("at most" in k for k in self.calls_by_kind) else ""
        return (
            f"Planned calls:{bound} {self.total_calls:,} over {self.n_items} items ({kinds})\n"
            f"Estimated input tokens: {self.est_input_tokens:,} (about 4 characters per token)\n"
            f"Estimated cost: ${self.est_cost_usd:,.2f}"
            f"{' (paid API)' if self.paid else ' (no API cost)'}\n"
            f"Estimated wall time:{bound} {self.est_wall_s / 60:,.1f} min, before cache hits"
        )


def layouts_for(item: Item, kind: str, n: int | None = None, seed: int | str | None = None) -> list[tuple[str, ...]]:
    """The layouts an item is scored in. ``sample``: the convention order plus
    ``n - 1`` other orderings drawn without replacement, seeded by ``seed`` and
    the item id, so every model sees the same layouts."""
    if kind == "all":
        return all_orderings(item.unit_ids())
    if kind == "williams":
        return williams_orderings(item.unit_ids())
    if kind == "sample":
        if not n or seed is None:
            raise ValueError("layouts.kind 'sample' needs layouts.n and layouts.seed")
        conv = tuple(item.convention_order)
        rest = [o for o in all_orderings(item.unit_ids()) if o != conv]
        if n - 1 > len(rest):
            raise ValueError(f"layouts.n {n} exceeds the {len(rest) + 1} orderings of item {item.id}")
        return [conv] + random.Random(f"{seed}|layouts|{item.id}").sample(rest, n - 1)
    raise ValueError(f"unknown layouts.kind {kind!r}")


def _layouts(item: Item, cfg: dict) -> list[tuple[str, ...]]:
    lay = cfg["layouts"]
    return layouts_for(item, lay["kind"], lay.get("n"), lay.get("seed"))


def reference_units(cfg: dict) -> list[dict]:
    """The configured reference units ({heading, text}); ``reference_unit`` may be one or a list."""
    ref = cfg.get("reference_unit") or []
    return [ref] if isinstance(ref, dict) else list(ref)


def plan_calls(items: list[Item], cfg: dict) -> list[Call]:
    """Every call whose count is fixed in advance, in a fixed order.
    Calls that depend on earlier answers (repeat study, canaries, extra
    layout repeats) are made by ``execute``."""
    calls: list[Call] = []
    n_formats = len(formats(cfg))
    for it in items:
        lay = _layouts(it, cfg)
        if tuple(it.convention_order) not in lay:  # made first anyway, for the strata (``execute``)
            calls.append(Call(it.id, "layout", tuple(it.convention_order)))
        calls += [Call(it.id, "layout", o) for o in lay]
        if cfg.get("content_free", True):
            calls.append(Call(it.id, "content_free", ()))
        calls += [Call(it.id, "format", it.convention_order, format_index=f) for f in range(1, n_formats)]
        if cfg.get("subsets", False):
            # every non-empty proper subset, units in convention relative order ("reversed": reversed order)
            base = tuple(it.convention_order)
            if cfg["subsets"] == "reversed":
                base = base[::-1]
            elif isinstance(cfg["subsets"], list):  # an explicit order, by unit heading
                by_heading = {it.unit(u).heading: u for u in base}
                if sorted(cfg["subsets"]) != sorted(by_heading):
                    raise ValueError(f"subsets order {cfg['subsets']} must name each heading of {it.id} once")
                base = tuple(by_heading[h] for h in cfg["subsets"])
            elif cfg["subsets"] not in (True, "convention"):
                raise ValueError("subsets is true, 'convention', 'reversed' or a list of headings")
            sizes = cfg.get("subset_sizes") or range(1, len(base))  # e.g. [1]: each unit alone
            if any(not 1 <= k < len(base) for k in sizes):
                raise ValueError(f"subset_sizes {list(sizes)} must lie in 1..{len(base) - 1}")
            for k in sizes:
                calls += [Call(it.id, "subset", sub) for sub in combinations(base, k)]
            if cfg.get("blank_baseline", False):  # the headings alone, in the same order: the empty record
                calls.append(Call(it.id, "blank", base))
        # one fixed unit as the whole record, a reference for unit weights; a list gives several
        calls += [Call(it.id, "reference", (f"reference:{k}",)) for k in range(len(reference_units(cfg)))]
        if cfg.get("option_rotations", False):
            n_opt = len(cfg["prompt"]["options"])
            calls += [Call(it.id, "option_rotation", it.convention_order, option_rotation=k) for k in range(1, n_opt)]
    return calls


def conditional_counts(items: list[Item], cfg: dict) -> dict[str, int]:
    """Upper bounds on the calls ``execute`` decides during the run."""
    nc, cc = cfg.get("noise"), cfg.get("canary")
    if not nc:
        return {}
    n_layouts = len(_layouts(items[0], cfg)) if items else 0
    out = {"repeat (at most)": min(len(items), 3 * nc["items_per_stratum"]) * nc["repeats"]}
    extra = (nc["max_layout_repeats"] - 1) * n_layouts * len(items)
    out["layout_repeat (at most)"] = extra
    if cc:
        n_canary = sum(cc["per_stratum"].values())
        main = len(plan_calls(items, cfg)) - len(items)
        batches = math.ceil(main / cc["batch_size"]) + math.ceil(extra / cc["batch_size"])
        out["canary (at most, before confirmations)"] = batches * CHECKS_PER_BATCH * n_canary
    return out


def request_for(call: Call, item: Item, cfg: dict) -> DecisionRequest:
    p = cfg["prompt"]
    fmt = formats(cfg)[call.format_index]
    if call.kind == "content_free":
        context = CONTENT_FREE
    elif call.kind == "blank":
        context = render_blank(item, call.order, fmt)
    elif call.kind == "reference":
        ref = reference_units(cfg)[int(call.order[0].split(":")[1]) if call.order else 0]
        context = render_unit(Unit(id="reference", heading=ref.get("heading", ""), lines=(ref["text"],)), fmt)
    else:
        context = render(item, call.order, fmt, subset=call.kind == "subset")
    instruction = p["instruction"].format(**item.meta.get("prompt_fields", {}))
    return DecisionRequest(instruction=instruction, context=context, options=rotated_options(cfg, call.option_rotation))


def make_plan(items: list[Item], cfg: dict, paid: bool) -> RunPlan:
    calls = plan_calls(items, cfg)
    by_id = {it.id: it for it in items}
    chars = 0
    by_kind: dict[str, int] = {}
    for c in calls:
        r = request_for(c, by_id[c.item_id], cfg)
        chars += len(r.instruction) + len(r.context) + sum(len(o) for o in r.options)
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    extra = conditional_counts(items, cfg)
    by_kind.update(extra)
    n = len(calls) + sum(extra.values())
    tokens = chars // 4 + (chars // 4 // len(calls) * sum(extra.values()) if calls else 0)
    m = cfg["model"]
    cost = tokens / 1e6 * float(m.get("price_per_m_input_tokens", 0.0))
    wall = n * float(m.get("est_latency_s", 0.0)) / max(1, int(m.get("concurrency", 1)))
    return RunPlan(len(items), by_kind, n, tokens, cost, wall, paid)


def run_dir_name(cfg: dict, model_version: str) -> str:
    """Date, experiment and model version; a repo@commit version is shortened
    to repo-commit12 (the full commit is in run.json)."""
    if "@" in model_version:
        repo, commit = model_version.split("@", 1)
        model_version = f"{repo.split('/')[-1]}-{commit[:12]}"
    safe = model_version.replace("/", "-").replace(" ", "_")
    return f"{cfg['date']}_{cfg['name']}_{safe}"


def execute(items: list[Item], cfg: dict, adapter, yes: bool = False, log=print) -> Path | None:
    """Run every planned call and write the run folder. Returns its path,
    or None if the run needs confirmation and ``yes`` is False."""
    plan = make_plan(items, cfg, paid=getattr(adapter, "paid", False))
    log(plan.text())
    if plan.needs_confirmation() and not yes:
        log("Not run: this run costs money or takes over an hour. Re-run with --yes to confirm.")
        return None

    launched = datetime.now(timezone.utc)
    if cfg["date"] == "launch":  # stamped by the harness, not typed: the UTC date the run starts
        cfg = {**cfg, "date": launched.date().isoformat()}
    cache = ResponseCache(resolve(cfg, cfg.get("cache", "cache.sqlite")))
    model = CachedAdapter(adapter, cache, epoch=int(cfg.get("epoch", 0)))
    by_id = {it.id: it for it in items}
    convention = [Call(it.id, "layout", it.convention_order) for it in items]

    # The first call tells us the exact model version, which names the run folder.
    first, _ = model.decide(request_for(convention[0], by_id[convention[0].item_id], cfg), 0)
    out = _new_run_dir(resolve(cfg, cfg["output_root"]), run_dir_name(cfg, first.model_version))
    _save_config(cfg, out)
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(item_to_dict(it), ensure_ascii=False) + "\n")

    started = time.time()
    meta: dict = {"plan": asdict(plan), "epoch": int(cfg.get("epoch", 0)),
                  "launched_utc": launched.isoformat(timespec="seconds")}
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        s = _Session(cfg, adapter, model, by_id, out, f, first.model_version, log)
        try:
            conv = {c.item_id: s.call(c) for c in convention}
            rest = [c for c in plan_calls(items, cfg)
                    if not (c.kind == "layout" and c.order == by_id[c.item_id].convention_order)]
            if not cfg.get("noise"):
                s.run_batches(rest, None, "main")
            else:
                setup = _repeat_study(s, conv)
                s.run_batches(rest, setup.get("canary_set"), "main")
                setup["layout_repeats"] = _layout_repeats(s, setup)
                _write_setup(out, setup)
        except Halted as e:
            meta["halted"] = str(e)
            log(f"HALTED: {e}")
    if hasattr(adapter, "info"):
        meta["adapter"] = adapter.info()
    meta.update({"cache_hits": model.hits, "cache_misses": model.misses,
                 "model_version": first.model_version, "seconds": round(time.time() - started, 1)})
    (out / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cache.close()
    log(f"Wrote {out}")
    return out


class _Session:
    """One run's open state: makes a call, writes its record, runs canaries."""

    def __init__(self, cfg, adapter, model, by_id, out, f, version, log) -> None:
        self.cfg, self.adapter, self.model, self.by_id = cfg, adapter, model, by_id
        self.out, self.f, self.version, self.log = out, f, version, log
        self.pinned = cfg["model"].get("version") or version
        self.options = list(cfg["prompt"]["options"])
        self.tie = cfg["analysis"]["tie_option"]
        self.n = 0
        self.canary_seq = 0
        self.layouts: dict[str, dict[tuple, list[float]]] = {}
        self.setup: dict = {}

    def call(self, c: Call) -> list[float]:
        """Make one call, write its record; return probabilities in config option order."""
        req = request_for(c, self.by_id[c.item_id], self.cfg)
        t0 = time.time()
        d, cached = self.model.decide(req, c.repeat)
        rec = {
            "run": self.out.name,
            "example_id": c.item_id,
            "kind": c.kind,
            "passage_ids": list(c.order),
            "format_index": c.format_index,
            "repeat": c.repeat,
            "model": self.adapter.model_id,
            "model_version": d.model_version,
            "options": list(req.options),
            "probs": list(d.probs),
            "logits": d.raw.get("logits") if isinstance(d.raw, dict) else None,
            "content_free": c.kind == "content_free",
            "option_rotation": c.option_rotation,
            "tag": c.tag,
            "epoch": self.model.epoch,
            "raw": d.raw,
            "cost_usd": _cost(d.raw, self.cfg),
            "cached": cached,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "latency_ms": None if cached else round((time.time() - t0) * 1000),
        }
        self.f.write(json.dumps(rec) + "\n")
        if d.model_version != self.version:
            raise RuntimeError(f"model version changed mid-run: {self.version} -> {d.model_version}")
        self.n += 1
        if self.n % 1000 == 0:
            self.log(f"{self.n:,} calls")
        p = [d.probs[req.options.index(o)] for o in self.options]
        if c.kind == "layout":
            self.layouts.setdefault(c.item_id, {})[c.order] = p
        return p

    def run_batches(self, calls: list[Call], canary: tuple | None, phase: str) -> None:
        size = self.cfg["canary"]["batch_size"] if canary else max(1, len(calls))
        for b, start in enumerate(range(0, len(calls), size)):
            if canary:
                self.check(canary, f"{phase}-{b}-start")
            for c in calls[start:start + size]:
                self.call(c)
            if canary:
                self.check(canary, f"{phase}-{b}-end")

    def check(self, canary: tuple, tag: str) -> None:
        """One canary check, repeated once if it alarms; halts on a second alarm."""
        cs, rule = canary
        for attempt in ("", "-confirm"):
            values = {}
            for iid in cs.ids:
                self.canary_seq += 1
                order = self.by_id[iid].convention_order
                p = self.call(Call(iid, "canary", order, repeat=CANARY_BASE + self.canary_seq, tag=tag + attempt))
                values[iid] = p[cs.option[iid]]
            reasons = canary_alarm(values, cs, rule)
            with (self.out / "canary_checks.jsonl").open("a", encoding="utf-8") as g:
                g.write(json.dumps({"tag": tag + attempt, "alarm": reasons, "values": values}) + "\n")
            if not reasons:
                return
            self.log(f"canary alarm at {tag + attempt}: {'; '.join(reasons)}")
        raise Halted(f"canary alarm confirmed at {tag}: {'; '.join(reasons)}")


def _repeat_study(s: _Session, conv: dict[str, list[float]]) -> dict:
    """Strata from the convention calls, the repeat study, and the canary set.
    The convention call used for stratifying is not one of the repeats, so
    the noise estimate is not selected on its own values."""
    cfg, nc, seed = s.cfg, s.cfg["noise"], s.cfg["seed"]
    strata = assign_strata(conv, nc["margin_edges"])
    chosen = draw_per_stratum(by_stratum(strata), {k: nc["items_per_stratum"] for k in STRATA}, seed, "repeat-study")
    reps: dict[str, list[list[float]]] = {}
    for k in STRATA:
        for iid in chosen[k]:
            order = s.by_id[iid].convention_order
            reps[iid] = [s.call(Call(iid, "repeat", order, repeat=r)) for r in range(1, nc["repeats"] + 1)]
    noises = {iid: item_noise(v, s.options, s.tie) for iid, v in reps.items()}
    setup: dict = {
        "strata": strata,
        "repeat_items": chosen,
        "shortfall": {k: nc["items_per_stratum"] - len(chosen[k]) for k in STRATA},
        "noise_by_stratum": {k: pool([noises[i] for i in chosen[k]]) for k in STRATA},
    }
    s.log("repeat study: " + ", ".join(f"{k} {len(chosen[k])} items" for k in STRATA))
    cc = cfg.get("canary")
    if cc:
        picks = draw_per_stratum(chosen, cc["per_stratum"], seed, "canary")
        ids = tuple(i for k in STRATA for i in picks[k])
        win = {i: [v[noises[i].majority] for v in reps[i]] for i in ids}
        cs = CanarySet(ids, {i: strata[i] for i in ids}, {i: noises[i].majority for i in ids},
                       {i: fmean(win[i]) for i in ids}, {i: stdev(win[i]) for i in ids})
        cal = calibrate(win, cs, cc)
        rule = cal["rule"]
        setup["canary"] = {"items": picks, **{k: v for k, v in cal.items() if k != "rule"},
                           "rule": asdict(rule), "baseline_mean": cs.mean, "baseline_sd": cs.sd}
        setup["canary_set"] = (cs, rule)
        s.log(f"canary: T = {rule.T:.2f}, M in [{rule.L:+.4f}, {rule.U:+.4f}], expected false alarms "
              f"{cal['expected_false_alarm_rate_per_check']:.3f} per check")
    _write_setup(s.out, setup)
    return setup


def _layout_repeats(s: _Session, setup: dict) -> dict:
    """Extra calls per layout for near and at items if the noise share of
    order variance misses the target (``noise.layout_repeats``)."""
    nc = s.cfg["noise"]
    noise_var = {k: v.get("noise_var_win") for k, v in setup["noise_by_stratum"].items()}
    pairs = []
    for iid, by_order in s.layouts.items():
        nv = noise_var.get(setup["strata"].get(iid))
        if nv is None:
            continue
        vecs = list(by_order.values())
        win = s.options.index(flip_stats(vecs, s.options, s.tie)["reference"])
        pairs.append((nv, pstdev([v[win] for v in vecs]) ** 2))
    if not pairs:
        return {"skipped": "no item has a stratum with repeat-study noise"}
    d = layout_repeats(fmean(p[0] for p in pairs), fmean(p[1] for p in pairs),
                       nc["target_share"], nc["max_layout_repeats"])
    s.log(f"noise share of order variance {d['noise_share']:.3f}; layout repeats R = {d['R_needed']}"
          + (" (too noisy: above the cap, none made)" if d["too_noisy"] else ""))
    if d["R"] and d["R"] > 1:
        extra = [Call(iid, "layout_repeat", order, repeat=LAYOUT_REPEAT_BASE + r)
                 for iid, by_order in s.layouts.items() if setup["strata"].get(iid) in ("near", "at")
                 for order in by_order for r in range(1, d["R"])]
        d["n_calls"] = len(extra)
        s.run_batches(extra, setup.get("canary_set"), "layout_repeats")
    return d


def _write_setup(out: Path, setup: dict) -> None:
    clean = {k: v for k, v in setup.items() if k != "canary_set"}
    (out / "noise_setup.json").write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")


def _cost(raw: dict, cfg: dict) -> float | None:
    """Cost from the tokens the API reports, at the config's price."""
    tokens = (raw.get("usage") or {}).get("input_tokens")
    price = cfg["model"].get("price_per_m_input_tokens")
    return None if tokens is None or price is None else tokens / 1e6 * float(price)


def _new_run_dir(root: Path, name: str) -> Path:
    """A fresh run folder. Existing runs are never overwritten: a repeat run
    on the same day gets a suffix (_r2, _r3, ...)."""
    out, k = root / name, 1
    while out.exists():
        k += 1
        out = root / f"{name}_r{k}"
    out.mkdir(parents=True)
    return out


def _save_config(cfg: dict, out: Path) -> None:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (out / "config.json").write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
    src = cfg.get("_source")
    if src:
        shutil.copy2(src, out / Path(src).name)
