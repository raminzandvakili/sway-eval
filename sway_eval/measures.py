"""Order-sensitivity measures for one record and across records.

Everything is defined against one reference, the symmetrized prediction:
the target option's probability averaged over a set of legitimate layouts.
The primary unit is that probability, not the log-odds margin, because the
Brier identity below is written in it and hosted APIs clip small
probabilities (which saturates log-odds).

All functions are for a binary decision on a fixed ``target`` option.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean

TIE_RULES = ("target", "other")


def decide(p: float, tie_rule: str) -> bool:
    """True if the target option wins at probability ``p``.

    A probability of exactly 0.5 is a tie (common on APIs with a 0.01 grid);
    ``tie_rule`` says who wins it and must be fixed before a run.
    """
    if tie_rule not in TIE_RULES:
        raise ValueError(f"tie_rule must be one of {TIE_RULES}")
    if p > 0.5:
        return True
    if p < 0.5:
        return False
    return tie_rule == "target"


def symmetrized(probs: Sequence[float]) -> float:
    """Mean target probability over layouts: the position-neutral prediction."""
    if not probs:
        raise ValueError("no probabilities")
    return fmean(probs)


def order_variance(probs: Sequence[float]) -> float:
    """Variance of the target probability across layouts (population form).

    By the ambiguity decomposition (Krogh & Vedelsby, 1995) this equals the
    mean Brier score over layouts minus the Brier score of the symmetrized
    prediction, for any label: the Brier cost that symmetrization removes.
    """
    m = symmetrized(probs)
    return fmean((p - m) ** 2 for p in probs)


def brier(p: float, y: int) -> float:
    """Binary Brier score of probability ``p`` for outcome ``y`` (0 or 1)."""
    return (p - y) ** 2


def flip_probability(probs: Sequence[float], tie_rule: str) -> float:
    """Share of layouts whose decision differs from the symmetrized decision."""
    ref = decide(symmetrized(probs), tie_rule)
    return fmean(1.0 if decide(p, tie_rule) != ref else 0.0 for p in probs)


def disagreement(p_fixed: float, probs: Sequence[float], tie_rule: str) -> bool:
    """Does the decision at one fixed layout differ from the symmetrized one?"""
    return decide(p_fixed, tie_rule) != decide(symmetrized(probs), tie_rule)


@dataclass(frozen=True)
class RecordSummary:
    """Per-record measures over a set of layouts."""

    item_id: str
    n_layouts: int
    p_sym: float
    decision_sym: bool
    order_variance: float
    order_sd: float
    flip_probability: float
    p_min: float
    p_max: float


def summarize_record(item_id: str, probs: Sequence[float], tie_rule: str) -> RecordSummary:
    v = order_variance(probs)
    return RecordSummary(
        item_id=item_id,
        n_layouts=len(probs),
        p_sym=symmetrized(probs),
        decision_sym=decide(symmetrized(probs), tie_rule),
        order_variance=v,
        order_sd=math.sqrt(v),
        flip_probability=flip_probability(probs, tie_rule),
        p_min=min(probs),
        p_max=max(probs),
    )


def unit_position_effects(probs_by_order: Mapping[tuple[str, ...], float]) -> dict[str, list[float]]:
    """For each unit, the mean target probability with that unit at each position,
    minus the record's overall mean.

    Used only to find which unit's placement moves the decision most (the
    decisive unit); it is a per-record diagnostic, not a model-level profile.
    """
    orders = list(probs_by_order)
    if not orders:
        raise ValueError("no orderings")
    n = len(orders[0])
    overall = fmean(probs_by_order.values())
    sums: dict[str, list[float]] = {u: [0.0] * n for u in orders[0]}
    counts: dict[str, list[int]] = {u: [0] * n for u in orders[0]}
    for order, p in probs_by_order.items():
        for pos, u in enumerate(order):
            sums[u][pos] += p
            counts[u][pos] += 1
    return {
        u: [sums[u][k] / counts[u][k] - overall if counts[u][k] else math.nan for k in range(n)]
        for u in sums
    }


def flip_zone_direct(
    levels: Sequence[float],
    flip_probs: Sequence[float],
    pi: float,
    level_step: float,
    width_convention: str,
) -> dict:
    """Flip zone estimated level by level.

    The zone is the set of qualification levels whose mean flip probability
    is at least ``pi``. ``width_convention`` fixes how a set of discrete
    levels becomes a width, and must be chosen before a run:

    - ``"span"``: inclusive span between the outer edges, in increments:
      (highest - lowest zone level) / ``level_step`` + 1, so one level alone
      has width 1 and two adjacent levels width 2. Interior levels that miss
      the floor are reported as ``holes``, so a fragmented zone can't hide
      inside a wide span.
    - ``"count"``: number of zone levels (kept for sensitivity checks)

    ``censored`` is True when an edge of the zone is the first or last level
    swept; the width is then a lower bound ("at least").
    """
    if len(levels) != len(flip_probs):
        raise ValueError("levels and flip_probs differ in length")
    by_level: dict[float, list[float]] = {}
    for lv, fp in zip(levels, flip_probs):
        by_level.setdefault(lv, []).append(fp)
    means = {lv: fmean(v) for lv, v in sorted(by_level.items())}
    levels_all = list(means)
    zone = [lv for lv, m in means.items() if m >= pi]
    holes = [lv for lv in levels_all if zone and min(zone) < lv < max(zone) and lv not in zone]
    if width_convention == "span":
        width = (max(zone) - min(zone)) / level_step + 1 if zone else 0
    elif width_convention == "count":
        width = len(zone)
    else:
        raise ValueError('width_convention must be "span" or "count"')
    censored = bool(zone) and (min(zone) == levels_all[0] or max(zone) == levels_all[-1])
    return {
        "mean_flip_by_level": means,
        "zone_levels": zone,
        "lower_edge": min(zone) if zone else None,
        "upper_edge": max(zone) if zone else None,
        "width": width,
        "width_is_lower_bound": censored,
        "holes": holes,
        "n_holes": len(holes),
    }


def flip_zone_model(
    levels: Sequence[float],
    p_sym: Sequence[float],
    order_sds: Sequence[float],
    pi: float,
    threshold_level: float,
    window: float,
) -> dict:
    """Flip zone from a normal approximation.

    Near the threshold, a record whose symmetrized probability is m flips
    under a random layout with probability about Phi(-|m - 0.5| / sigma), so
    the zone is where |m - 0.5| < z * sigma with Phi(-z) = pi. Converting with
    the local slope of m per level gives a full width of 2 z sigma / slope.
    ``window`` selects the levels within that distance of ``threshold_level``
    used for the slope fit and for sigma.
    """
    pts = [(lv, m, s) for lv, m, s in zip(levels, p_sym, order_sds) if abs(lv - threshold_level) <= window]
    if len(pts) < 3 or len({lv for lv, _, _ in pts}) < 2:
        raise ValueError("not enough levels near the threshold to fit a slope")
    xs = [lv for lv, _, _ in pts]
    ys = [m for _, m, _ in pts]
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    sigma = math.sqrt(fmean(s * s for _, _, s in pts))
    z = -NormalDist().inv_cdf(pi)
    width = 2 * z * sigma / abs(slope) if slope else math.inf
    centre = threshold_level + (0.5 - (my - slope * mx) - slope * threshold_level) / slope if slope else math.nan
    return {"slope": slope, "sigma": sigma, "z": z, "width": width, "centre": centre,
            "lower_edge": centre - width / 2, "upper_edge": centre + width / 2, "n_points": len(pts)}


def flip_zone_noise(
    levels: Sequence[float],
    flip_probs: Sequence[float],
    floors: Sequence[float | None],
    pi: float,
    level_step: float,
    width_convention: str,
    raw_zone: Mapping,
) -> dict:
    """The flip zone checked against run-to-run noise.

    ``floors`` is each item's noise floor: the repeat flip rate of its
    stratum (None if unknown). A level's floor is the mean over its items.
    Two rules, reported side by side with the raw zone:

    - existence: over the raw zone (all levels from its lower to its upper
      edge, inclusive), the average layout flip probability must exceed the
      average noise floor;
    - width: edges are the outermost levels that clear both ``pi`` and their
      own floor; interior levels failing either are holes. A level with no
      floor can't be an edge.
    """
    flips: dict[float, list[float]] = {}
    fl: dict[float, list[float]] = {}
    for lv, fp, f in zip(levels, flip_probs, floors):
        flips.setdefault(lv, []).append(fp)
        if f is not None:
            fl.setdefault(lv, []).append(f)
    mean_flip = {lv: fmean(v) for lv, v in sorted(flips.items())}
    floor = {lv: fmean(v) for lv, v in sorted(fl.items())}
    levels_all = list(mean_flip)
    out: dict = {"mean_floor_by_level": floor}
    lo, hi = raw_zone.get("lower_edge"), raw_zone.get("upper_edge")
    if lo is None:
        out.update({"zone_average_flip": None, "zone_average_floor": None, "existence_fails": None})
    else:
        span = [lv for lv in levels_all if lo <= lv <= hi]
        avg_floor = [floor[lv] for lv in span if lv in floor]
        out["zone_average_flip"] = fmean(mean_flip[lv] for lv in span)
        out["zone_average_floor"] = fmean(avg_floor) if avg_floor else None
        out["existence_fails"] = None if not avg_floor else not out["zone_average_flip"] > out["zone_average_floor"]
    cleared = [lv for lv in levels_all if mean_flip[lv] >= pi and lv in floor and mean_flip[lv] > floor[lv]]
    holes = [lv for lv in levels_all if cleared and min(cleared) < lv < max(cleared) and lv not in cleared]
    if width_convention == "span":
        width = (max(cleared) - min(cleared)) / level_step + 1 if cleared else 0
    elif width_convention == "count":
        width = len(cleared)
    else:
        raise ValueError('width_convention must be "span" or "count"')
    out.update({
        "cleared_levels": cleared,
        "lower_edge": min(cleared) if cleared else None,
        "upper_edge": max(cleared) if cleared else None,
        "width": width,
        "width_is_lower_bound": bool(cleared) and (min(cleared) == levels_all[0] or max(cleared) == levels_all[-1]),
        "holes": holes,
        "n_holes": len(holes),
    })
    return out


def model_crossing(levels: Sequence[float], p_sym: Sequence[float]) -> float | None:
    """Where the mean symmetrized target probability crosses 0.5, by linear
    interpolation between adjacent levels (the first crossing from below);
    None if it never crosses."""
    by: dict[float, list[float]] = {}
    for lv, p in zip(levels, p_sym):
        by.setdefault(lv, []).append(p)
    pts = sorted((lv, fmean(v)) for lv, v in by.items())
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 < 0.5 <= y1:
            return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
    return None


def flip_zone_margin(margins: Sequence[float], flip_probs: Sequence[float], pi: float, bin_width: float,
                     min_n: int = 30) -> dict:
    """Model-based flip zone for any number of options, in margin units.

    Items are binned by their symmetrized top-two margin (the symmetrized
    winner's probability minus the largest other) on a fixed grid of
    ``bin_width`` starting at 0, the same edges for every run. A bin inside
    or at the edge of the zone (including the first bin past its outer edge)
    with fewer than ``min_n`` items is merged with
    its inner neighbour (the innermost bin, having none, with its outer one),
    repeatedly, so merged bins still have grid edges. The zone is the span of
    bins whose mean flip probability is at least ``pi``; bins inside the span
    that miss ``pi`` are ``holes``. ``share_in_zone`` is the share of items
    whose margin falls inside the zone.
    """
    if len(margins) != len(flip_probs):
        raise ValueError("margins and flip_probs differ in length")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    cells: dict[int, list[float]] = {}
    for m, fp in zip(margins, flip_probs):
        cells.setdefault(int(m // bin_width), []).append(fp)
    bins = [[k, k + 1, v] for k, v in sorted(cells.items())]  # [lo cell, hi cell, flips]

    def zone_of(bs):
        return [i for i, (_, _, v) in enumerate(bs) if fmean(v) >= pi]

    merges = []
    while True:
        z = zone_of(bins)
        if not z:
            break
        # the zone's bins and the first bin past its outer edge, which decides where the edge falls
        thin = [i for i in range(0, min(max(z) + 2, len(bins))) if len(bins[i][2]) < min_n]
        if not thin or len(bins) == 1:
            break
        i = thin[-1]  # outermost thin bin first
        j = i - 1 if i > 0 else i + 1
        a, b = sorted((i, j))
        merges.append([round(bins[a][0] * bin_width, 6), round(bins[b][1] * bin_width, 6)])
        bins[a:b + 1] = [[bins[a][0], bins[b][1], bins[a][2] + bins[b][2]]]
    key = lambda b: f"{b[0] * bin_width:.3f}-{b[1] * bin_width:.3f}"
    out = {
        "bin_width": bin_width,
        "min_n": min_n,
        "mean_flip_by_bin": {key(b): round(fmean(b[2]), 6) for b in bins},
        "n_by_bin": {key(b): len(b[2]) for b in bins},
        "merged": merges,
        "n_items": len(margins),
    }
    z = zone_of(bins)
    if z:
        lo, hi = bins[min(z)][0] * bin_width, bins[max(z)][1] * bin_width
        out.update(zone_bins=[key(bins[i]) for i in z], lower_edge=round(lo, 6), upper_edge=round(hi, 6),
                   width=round(hi - lo, 6), holes=[key(bins[i]) for i in range(min(z), max(z) + 1) if i not in z],
                   share_in_zone=round(sum(1 for m in margins if lo <= m < hi) / len(margins), 6))
    else:
        out.update(zone_bins=[], lower_edge=None, upper_edge=None, width=0.0, holes=[], share_in_zone=0.0)
    return out


def slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Ordinary least-squares slope of ``ys`` on ``xs``; None if ``xs`` is constant."""
    mx, my = fmean(xs), fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None
