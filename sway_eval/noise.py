"""Run-to-run noise: the repeat study, per-stratum noise, and canaries.

A layout measured with one call carries the model's run-to-run noise on top
of the layout effect, so measured order variance is layout variance plus
noise variance (assuming the two are independent). The repeat study calls a
sample of items many times at the convention layout to estimate the noise
part, which the analysis subtracts.

Noise depends on where an item sits, so items are put in strata by their
top-2 margin at the convention layout: the distance from flipping, in the
same units for any number of options (with two options the margin is
2 * |p - 0.5|). Repeat items are drawn per stratum with a fixed seed.

Canaries are a few repeat items called again at batch boundaries. A run
halts when a canary check alarms twice in a row (see ``canary_alarm``):
either the canaries' mean deviation M is outside a band, or their squared
deviations D, each scaled by the canary's own repeat-study noise, sum too
high. Both thresholds come from a leave-one-out reference built from the
repeat study itself (``calibrate``), because a fresh call scaled by a mean
and SD estimated from a few calls is wider than N(0, 1).
All thresholds come from the config; nothing here has a default for them.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean, stdev, variance

from .multiclass import argmax

Vector = Sequence[float]
STRATA = ("far", "near", "at")
DEVIATION_BINS = (0.0, 0.01, 0.02)  # 0, one grid step, two steps, more
_EPS = 1e-9  # probabilities on a 0.01 grid don't subtract exactly


def top2_margin(p: Vector) -> float:
    """Largest probability minus the second largest."""
    s = sorted(p, reverse=True)
    return s[0] - s[1]


def stratum(margin: float, edges: Mapping[str, float]) -> str:
    """``far`` at or above ``edges["far"]``, ``near`` at or above ``edges["near"]``, else ``at``."""
    if margin >= edges["far"] - _EPS:
        return "far"
    if margin >= edges["near"] - _EPS:
        return "near"
    return "at"


def assign_strata(convention: Mapping[str, Vector], edges: Mapping[str, float]) -> dict[str, str]:
    return {iid: stratum(top2_margin(p), edges) for iid, p in convention.items()}


def draw_per_stratum(ids_by_stratum: Mapping[str, Sequence[str]], counts: Mapping[str, int], seed: int,
                     salt: str) -> dict[str, list[str]]:
    """Draw ``counts[s]`` ids from each stratum with a fixed seed; all of them
    if the stratum is smaller (the shortfall is visible in the result)."""
    rng = random.Random(f"{seed}|{salt}")
    out = {}
    for s in STRATA:
        pool = sorted(ids_by_stratum.get(s, []))
        out[s] = sorted(rng.sample(pool, min(counts.get(s, 0), len(pool))))
    return out


def by_stratum(strata: Mapping[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {s: [] for s in STRATA}
    for iid, s in strata.items():
        out[s].append(iid)
    return out


@dataclass(frozen=True)
class ItemNoise:
    """Noise in one item's repeated calls."""

    n: int
    majority: int  # index of the majority decision
    var_win: float  # sample variance (n - 1) of the majority option's probability
    var_classes: float  # summed sample variance over options
    identical_share: float  # share of repeats equal to the modal response
    repeat_flips: int  # repeats whose decision differs from the majority
    deviations: dict = field(default_factory=dict)  # largest |p - modal| per repeat, binned


def item_noise(vectors: Sequence[Vector], options: Sequence[str], tie_option: str | None) -> ItemNoise:
    if len(vectors) < 2:
        raise ValueError("need at least two repeated calls")
    vecs = [tuple(v) for v in vectors]
    decisions = [argmax(v, options, tie_option)[0] for v in vecs]
    counts = Counter(decisions)
    top = max(counts.values())
    tied = [d for d in range(len(options)) if counts.get(d) == top]
    majority = options.index(tie_option) if tie_option in options and options.index(tie_option) in tied else tied[0]
    modal = Counter(vecs).most_common(1)[0][0]
    dev = Counter(_bin(max(abs(a - b) for a, b in zip(v, modal))) for v in vecs)
    return ItemNoise(
        n=len(vecs),
        majority=majority,
        var_win=variance([v[majority] for v in vecs]),
        var_classes=sum(variance([v[c] for v in vecs]) for c in range(len(options))),
        identical_share=sum(1 for v in vecs if v == modal) / len(vecs),
        repeat_flips=sum(1 for d in decisions if d != majority),
        deviations=dict(dev),
    )


def _bin(x: float) -> str:
    if x <= DEVIATION_BINS[0] + _EPS:
        return "0"
    for edge in DEVIATION_BINS[1:]:
        if x <= edge + _EPS:
            return f"<={edge}"
    return f">{DEVIATION_BINS[-1]}"


def pool(noises: Sequence[ItemNoise]) -> dict:
    """Noise of a group of items: mean within-item variance, pooled rates."""
    if not noises:
        return {"n_items": 0}
    n_calls = sum(x.n for x in noises)
    dev: Counter[str] = Counter()
    for x in noises:
        dev.update(x.deviations)
    var_win = fmean(x.var_win for x in noises)
    return {
        "n_items": len(noises),
        "n_calls": n_calls,
        "noise_var_win": var_win,
        "sigma_noise": math.sqrt(var_win),
        "noise_var_classes": fmean(x.var_classes for x in noises),
        "identical_share": sum(x.identical_share * x.n for x in noises) / n_calls,
        "repeat_flip_rate": sum(x.repeat_flips for x in noises) / n_calls,
        "deviations": dict(dev),
    }


def fit_binomial_shape(means: Sequence[float], sds: Sequence[float]) -> dict:
    """Least-squares fit of sigma_noise = c * p (1 - p) through the origin,
    where p is each item's mean winning probability (an appendix check)."""
    xs = [m * (1 - m) for m in means]
    sxx = sum(x * x for x in xs)
    if not sxx:
        return {"c": None, "n_items": len(xs)}
    c = sum(x * s for x, s in zip(xs, sds)) / sxx
    rss = sum((s - c * x) ** 2 for x, s in zip(xs, sds))
    return {"c": c, "rmse": math.sqrt(rss / len(xs)), "n_items": len(xs)}


def layout_repeats(noise_var: float, measured_var: float, target_share: float, max_r: int) -> dict:
    """How many calls per layout, averaged, bring the noise share of order
    variance down to ``target_share``.

    Averaging R calls divides the noise variance by R, so the share is at
    most ``target_share`` when R >= (1 - t) / t * noise / layout (9 * noise /
    layout at t = 0.10). A model needing more than ``max_r`` is too noisy for
    single-layout measurement.
    """
    layout_var = measured_var - noise_var
    share = noise_var / measured_var if measured_var > 0 else math.inf
    if layout_var <= 0:
        r = math.inf
    else:
        r = max(1, math.ceil((1 - target_share) / target_share * noise_var / layout_var - _EPS))
    return {"noise_share": share, "noise_var": noise_var, "measured_var": measured_var,
            "R": r if r <= max_r else None, "R_needed": r if math.isfinite(r) else None,
            "too_noisy": r > max_r}


@dataclass(frozen=True)
class CanarySet:
    """Canary items with their repeat-study baseline: the majority option's
    probability, its mean and sample SD (n - 1) over the item's repeats."""

    ids: tuple[str, ...]
    strata: dict
    option: dict  # item id -> option index followed
    mean: dict
    sd: dict


@dataclass(frozen=True)
class CanaryRule:
    """A check alarms if D > ``T`` or M is outside [``L``, ``U``]."""

    T: float
    L: float
    U: float
    sigma_floor: float


def canary_stats(values: Mapping[str, float], cs: CanarySet, sigma_floor: float) -> tuple[float, float]:
    """D = sum of ((x - mean) / max(sd, floor))^2 and M = mean of (x - mean)."""
    d = sum(((values[i] - cs.mean[i]) / max(cs.sd[i], sigma_floor)) ** 2 for i in cs.ids)
    m = fmean(values[i] - cs.mean[i] for i in cs.ids)
    return d, m


def canary_alarm(values: Mapping[str, float], cs: CanarySet, rule: CanaryRule) -> list[str]:
    """Reasons a canary check alarms (empty if it passes). ``values`` is each
    canary's probability for its followed option in this check."""
    d, m = canary_stats(values, cs, rule.sigma_floor)
    reasons = []
    if not rule.L - _EPS <= m <= rule.U + _EPS:
        reasons.append(f"mean deviation M = {m:+.4f} outside [{rule.L:+.4f}, {rule.U:+.4f}]")
    if d > rule.T + _EPS:
        reasons.append(f"dispersion D = {d:.2f} > T = {rule.T:.2f}")
    return reasons


def loo_reference(repeats: Mapping[str, Sequence[float]], ids: Sequence[str], sigma_floor: float,
                  n_checks: int, seed: int) -> tuple[list[float], list[float]]:
    """Reference values of D and M with no drift, by leave-one-out.

    For each canary and each of its calls, the call is held out and scaled
    by the mean and sample SD (floored) of the others. A reference check
    draws one held-out call per canary, independently per canary.
    """
    contrib = {}
    for i in ids:
        xs = list(repeats[i])
        if len(xs) < 3:
            raise ValueError("leave-one-out needs at least three repeats per canary")
        rows = []
        for j, x in enumerate(xs):
            rest = xs[:j] + xs[j + 1:]
            mu = fmean(rest)
            rows.append((((x - mu) / max(stdev(rest), sigma_floor)) ** 2, x - mu))
        contrib[i] = rows
    rng = random.Random(seed)
    ds, ms = [], []
    for _ in range(n_checks):
        picks = [contrib[i][rng.randrange(len(contrib[i]))] for i in ids]
        ds.append(sum(p[0] for p in picks))
        ms.append(fmean(p[1] for p in picks))
    return ds, ms


def quantile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank quantile: the smallest value with at least ``q`` of the data at or below it."""
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def calibrate(repeats: Mapping[str, Sequence[float]], cs: CanarySet, cc: Mapping) -> dict:
    """Thresholds from the leave-one-out reference, with floors.

    ``T`` is the ``dispersion_pct`` quantile of reference D and [L, U] the
    ``mean_band`` quantiles of reference M. Floors: T is at least the D of
    ``floor_changes`` one-grid-step changes on floored canaries, and [L, U]
    contains the M of that many same-direction steps, so a single step
    never alarms on a near-deterministic model.
    """
    floor, grid, k, n = cc["sigma_floor"], cc["grid"], cc["floor_changes"], len(cs.ids)
    ds, ms = loo_reference(repeats, cs.ids, floor, cc["reference_checks"], cc["reference_seed"])
    lo_q, hi_q = cc["mean_band"]
    raw = {"T": quantile(ds, cc["dispersion_pct"]), "L": quantile(ms, lo_q), "U": quantile(ms, hi_q)}
    t_floor, m_floor = k * (grid / floor) ** 2, k * grid / n
    rule = CanaryRule(T=max(raw["T"], t_floor), L=min(raw["L"], -m_floor), U=max(raw["U"], m_floor),
                      sigma_floor=floor)
    alarms = sum(1 for d, m in zip(ds, ms)
                 if d > rule.T + _EPS or not rule.L - _EPS <= m <= rule.U + _EPS)
    return {
        "rule": rule,
        "raw": raw,
        "floors": {"T": t_floor, "M": m_floor},
        "expected_false_alarm_rate_per_check": alarms / len(ds),
        "analytic_T_cross_check": analytic_dispersion_quantile(
            cc["dispersion_pct"], n, min(len(repeats[i]) for i in cs.ids), cc["reference_checks"], cc["reference_seed"]),
    }


def analytic_dispersion_quantile(pct: float, n_canaries: int, n_repeats: int, n_sim: int, seed: int) -> float:
    """The ``pct`` quantile of D if every canary's noise were normal and
    continuous: a sum of (1 + 1/n) * t^2 with n - 1 degrees of freedom
    (by simulation). A cross-check for noisy models only."""
    rng = random.Random(f"{seed}|analytic")
    df = n_repeats - 1
    vals = []
    for _ in range(n_sim):
        tot = 0.0
        for _ in range(n_canaries):
            z = rng.gauss(0.0, 1.0)
            chi = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(df))
            tot += (1 + 1 / n_repeats) * z * z / (chi / df)
        vals.append(tot)
    return quantile(vals, pct)
