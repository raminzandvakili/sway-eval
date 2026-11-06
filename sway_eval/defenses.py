"""Compare defenses against layout sensitivity by calls and agreement.

Each defense turns a record's per-layout probabilities into one estimate of the
target option's probability and one decision. The reference is full
symmetrization: the mean over every admissible layout, each layout averaged over
its repeats where repeats exist. Defenses use single calls, as a deployer would.

Defenses:

- the convention layout (1 call);
- the fixed layout that agrees most with the reference on one half of the
  records, evaluated on the other half, both directions pooled (1 call);
- k random layouts for k = 1..10 (expectation over seeded draws; k calls);
- the balanced Williams set, units in convention order as symbols (2n calls
  for odd n);
- full symmetrization (n! calls);
- the adaptive screen: one random layout, and the balanced set if that score
  lies within k_sigma times the mean layout SD of 0.5 (1 + |balanced| * q calls).

For each: agreement with the reference decision, accuracy against gold labels
where they exist, the gap in accuracy between two named groups (the section gap
when records carry a decisive-section field), and the residual (mean squared
deviation of the estimate from the reference), with bootstrap intervals over
records for agreement and the gap. Needs every ordering of each record.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import permutations
from statistics import fmean, pstdev

from .layouts import williams_orderings


def _decide(p: float, tie_to_target: bool) -> bool:
    """True when the target option wins; an exact tie goes as the tie rule says."""
    return p > 0.5 or (tie_to_target and p == 0.5)


def _per_item(draws, ref, gold, tie):
    out = {}
    for i in ref:
        ests = [d[i] for d in draws]
        out[i] = (fmean(_decide(e, tie) == _decide(ref[i], tie) for e in ests),
                  fmean(_decide(e, tie) == gold[i] for e in ests) if gold.get(i) is not None else None,
                  fmean((e - ref[i]) ** 2 for e in ests))
    return out


def _gap(rows, group, ids, groups):
    a = [rows[i][1] for i in ids if group.get(i) == groups[0] and rows[i][1] is not None]
    b = [rows[i][1] for i in ids if group.get(i) == groups[1] and rows[i][1] is not None]
    return fmean(a) - fmean(b) if a and b else None


def _summarize(rows, group, groups, rng, boot):
    ids = sorted(rows)
    boots_a, boots_g = [], []
    for _ in range(boot):
        b = rng.choices(ids, k=len(ids))
        boots_a.append(fmean(rows[i][0] for i in b))
        g = _gap(rows, group, b, groups)
        if g is not None:
            boots_g.append(g)
    boots_a.sort()
    boots_g.sort()

    def ci(xs):
        return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]] if xs else None

    acc = [rows[i][1] for i in ids if rows[i][1] is not None]
    return {"agreement": fmean(rows[i][0] for i in ids), "agreement_ci95": ci(boots_a),
            "accuracy": fmean(acc) if acc else None, "section_gap": _gap(rows, group, ids, groups),
            "section_gap_ci95": ci(boots_g), "residual": fmean(rows[i][2] for i in ids)}


def compare_defenses(
    layouts: Mapping[str, Mapping[tuple[str, ...], float]],
    convention: Mapping[str, Sequence[str]],
    repeats: Mapping[str, Mapping[tuple[str, ...], Sequence[float]]] | None = None,
    gold: Mapping[str, bool | None] | None = None,
    group: Mapping[str, str] | None = None,
    groups: tuple[str, str] = ("Experience", "Skills"),
    strata: Mapping[str, object] | None = None,
    tie_to_target: bool = False,
    k_sigmas: Sequence[float] = (2.0, 3.0),
    seed: int = 20261104,
    draws: int = 500,
    boot: int = 2000,
) -> dict:
    """Defense table for one model.

    ``layouts[item][order]`` is the target option's probability from one call
    at that ordering of the item's unit ids, for every ordering, in the order
    the calls were made; ``convention[item]`` is the item's convention order.
    ``repeats`` holds every call per ordering (including the first) where
    layouts were repeated. ``gold[item]`` is True when the target option is
    correct; ``group`` names the two groups compared by the section gap;
    ``strata`` stratifies the held-out split (e.g. qualification level).
    """
    ids = sorted(layouts)
    n = len(convention[ids[0]])
    gold = dict(gold or {})
    group = dict(group or {})
    strata = dict(strata or {})
    reps = repeats or {}
    for i in ids:
        if len(layouts[i]) != len(list(permutations(range(n)))):
            raise ValueError(f"defense comparison needs every ordering; {i} has {len(layouts[i])}")

    def at(i, symbols):
        return layouts[i][tuple(convention[i][s] for s in symbols)]

    ref = {i: fmean(fmean(reps[i][o]) if i in reps and o in reps[i] else p for o, p in layouts[i].items())
           for i in ids}
    rng, brng = random.Random(seed), random.Random(seed + 1)

    def row(calls, ds, sub=None, **extra):
        refs = {i: ref[i] for i in (sub or ids)}
        return {"calls": calls, **extra, **_summarize(_per_item(ds, refs, gold, tie_to_target), group, groups, brng, boot)}

    rows = {}
    conv = {i: at(i, range(n)) for i in ids}
    rows["convention"] = row(1, [conv])
    by_stratum = defaultdict(list)
    for i in ids:
        by_stratum[strata.get(i)].append(i)
    half_a, half_b = [], []
    for _, its in sorted(by_stratum.items(), key=lambda kv: (kv[0] is None, kv[0])):
        its = sorted(its)
        rng.shuffle(its)
        half_a += its[: len(its) // 2]
        half_b += its[len(its) // 2:]
    pooled, chosen = {}, []
    for fit, test in ((half_a, half_b), (half_b, half_a)):
        best = max(permutations(range(n)),
                   key=lambda o: sum(_decide(at(i, o), tie_to_target) == _decide(ref[i], tie_to_target) for i in fit))
        chosen.append(list(best))
        pooled.update({i: at(i, best) for i in test})
    rows["held_out_best_fixed_order"] = row(1, [pooled], chosen_orders=chosen)
    frontier = {}
    for k in range(1, 11):
        ds = [{i: fmean(rng.sample(list(layouts[i].values()), k)) for i in ids} for _ in range(draws)]
        frontier[k] = row(k, ds)
    for k in (1, 2, 3):
        rows[f"k_random_{k}"] = frontier[k]
    W = williams_orderings(list(range(n)))
    bal = {i: fmean(at(i, w) for w in W) for i in ids}
    rows["balanced_williams"] = row(len(W), [bal])
    rows["full_symmetrization"] = row(len(layouts[ids[0]]), [{i: fmean(layouts[i].values()) for i in ids}])
    sigma = fmean(pstdev(list(layouts[i].values())) for i in ids)
    superseded = {}
    for kk in k_sigmas:
        ds, qs = [], []
        for _ in range(draws):
            scr = {i: rng.choice(list(layouts[i].values())) for i in ids}
            hit = {i: abs(scr[i] - 0.5) < kk * sigma for i in ids}
            qs.append(fmean(hit.values()))
            ds.append({i: (bal[i] if hit[i] else scr[i]) for i in ids})
        q = fmean(qs)
        rows[f"adaptive_screen_random_{kk:g}sigma"] = row(1 + len(W) * q, ds, screened_share=q)
        hit = {i: abs(conv[i] - 0.5) < kk * sigma for i in ids}
        q = fmean(hit.values())
        superseded[f"adaptive_screen_convention_{kk:g}sigma"] = row(
            1 + len(W) * q, [{i: (bal[i] if hit[i] else conv[i]) for i in ids}], screened_share=q)
    n_rep = fmean(len(reps[i][o]) if i in reps and o in reps[i] else 1 for i in ids for o in layouts[i])
    return {"sigma_layout": sigma, "mean_calls_per_layout_in_reference": n_rep, "table": rows,
            "superseded_convention_screen": superseded, "frontier_k_random": frontier}
