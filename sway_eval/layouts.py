"""Sets of legitimate layouts: all orderings, balanced (Williams) sets, samples.

Every function returns orderings of unit ids. Orderings are tuples, so they
can be used as dictionary keys and compared across runs.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import permutations


def all_orderings(unit_ids: Sequence[str]) -> list[tuple[str, ...]]:
    """Every ordering of the units, in a fixed (lexicographic) order."""
    return [tuple(p) for p in permutations(sorted(unit_ids))]


def williams_square(n: int) -> list[list[int]]:
    """A Williams design on positions 0..n-1 (Williams, 1949).

    Every unit appears in every position equally often, and every unit
    immediately follows every other unit equally often. For even n this needs
    n sequences; for odd n it needs 2n (each sequence plus its reverse).
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return [[0]]
    first = [0]
    lo, hi = 1, n - 1
    take_lo = True
    while len(first) < n:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    rows = [[(v + r) % n for v in first] for r in range(n)]
    if n % 2 == 1:
        rows += [list(reversed(row)) for row in rows]
    return rows


def williams_orderings(unit_ids: Sequence[str]) -> list[tuple[str, ...]]:
    """Balanced set of orderings for these units.

    Units are labelled by sorted id, not by the order they were submitted in,
    so two layouts of the same record get the same set of orderings.
    """
    labels = sorted(unit_ids)
    return [tuple(labels[i] for i in row) for row in williams_square(len(labels))]


def random_orderings(unit_ids: Sequence[str], k: int, seed: int) -> list[tuple[str, ...]]:
    """k orderings drawn uniformly at random (with replacement), reproducibly."""
    rng = random.Random(seed)
    labels = sorted(unit_ids)
    out = []
    for _ in range(k):
        perm = labels[:]
        rng.shuffle(perm)
        out.append(tuple(perm))
    return out
