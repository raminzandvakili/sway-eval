"""Order-sensitivity measures for decisions over any number of options.

A layout's decision is the argmax of its probability vector; the reference
decision is the argmax of the mean vector over layouts (the symmetrized
prediction). Ties go to a fixed option named in advance (``tie_option``),
never toward the reference, so ties can't hide flips. With two options these
reduce to the binary measures in ``measures``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from statistics import fmean, pstdev

Vector = Sequence[float]


def argmax(p: Vector, options: Sequence[str], tie_option: str | None) -> tuple[int, bool]:
    """Index of the winning option, and whether the top probability was tied.

    A tie goes to ``tie_option`` when it is among the tied options, and
    otherwise to the first tied option in ``options`` order.
    """
    top = max(p)
    tied = [i for i, v in enumerate(p) if v == top]
    if len(tied) == 1:
        return tied[0], False
    if tie_option is not None and options.index(tie_option) in tied:
        return options.index(tie_option), True
    return tied[0], True


def mean_vector(vectors: Sequence[Vector]) -> list[float]:
    if not vectors:
        raise ValueError("no vectors")
    k = len(vectors[0])
    return [fmean(v[c] for v in vectors) for c in range(k)]


def order_variance_classes(vectors: Sequence[Vector]) -> float:
    """Sum over options of each option's probability variance across layouts.

    By the ambiguity decomposition applied option by option, this is the
    multiclass Brier cost that symmetrization removes. With two options it is
    twice the variance of either option's probability.
    """
    k = len(vectors[0])
    return sum(pstdev([v[c] for v in vectors]) ** 2 for c in range(k))


def winner_margin(p: Vector, win: int) -> float:
    """Probability of option ``win`` minus the largest other probability."""
    return p[win] - max(v for i, v in enumerate(p) if i != win)


def flip_stats(vectors: Sequence[Vector], options: Sequence[str], tie_option: str | None) -> dict:
    """Reference decision, flip probability, flip types and tie counts."""
    ref, ref_tied = argmax(mean_vector(vectors), options, tie_option)
    moves: Counter[str] = Counter()
    flips = ties = 0
    for v in vectors:
        d, tied = argmax(v, options, tie_option)
        ties += tied
        if d != ref:
            flips += 1
            moves[f"{options[ref]}->{options[d]}"] += 1
    return {
        "reference": options[ref],
        "reference_tied": ref_tied,
        "flip_probability": flips / len(vectors),
        "flip_moves": dict(moves),
        "layout_ties": ties,
    }


def confident_flip(vectors: Sequence[Vector], options: Sequence[str], tie_option: str | None, tau: float) -> bool:
    """Do two layouts reach different decisions, each with winning probability >= tau?"""
    confident = {}
    for v in vectors:
        d, _ = argmax(v, options, tie_option)
        if max(v) >= tau:
            confident[d] = True
    return len(confident) > 1


def confident_pairwise_flip_rate(vectors: Sequence[Vector], options: Sequence[str], tie_option: str | None,
                                 tau: float) -> float:
    """Probability that two layouts drawn without replacement reach different
    decisions, each with winning probability >= tau. Unlike ``confident_flip``
    it does not grow with the number of layouts scored."""
    n = len(vectors)
    if n < 2:
        return 0.0
    counts: dict[int, int] = {}
    for v in vectors:
        if max(v) >= tau:
            d, _ = argmax(v, options, tie_option)
            counts[d] = counts.get(d, 0) + 1
    c = list(counts.values())
    opposite = sum(c[i] * c[j] for i in range(len(c)) for j in range(i + 1, len(c)))
    return opposite / (n * (n - 1) / 2)


def pairwise_flip_rate(vectors: Sequence[Vector], options: Sequence[str], tie_option: str | None) -> float:
    """Share of layout pairs whose decisions differ (what neighbours report)."""
    ds = [argmax(v, options, tie_option)[0] for v in vectors]
    pairs = list(combinations(ds, 2))
    return sum(a != b for a, b in pairs) / len(pairs) if pairs else 0.0
