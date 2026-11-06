"""Size of the search space of ordered passage subsets, and call budgets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import permutations
from math import perm
from typing import TypeVar

T = TypeVar("T")


def count_sequences(n: int, k: int) -> int:
    """Number of ordered sequences of exactly k distinct passages out of n.

    This is n! / (n - k)!. For example, count_sequences(30, 5) == 17_100_720.
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative")
    if k > n:
        return 0
    return perm(n, k)


def count_sequences_up_to(n: int, k: int) -> int:
    """Number of ordered sequences of 1 to k distinct passages out of n."""
    return sum(count_sequences(n, j) for j in range(1, min(k, n) + 1))


def ordered_subsets(items: Sequence[T], k: int) -> Iterator[tuple[T, ...]]:
    """Yield every ordered sequence of k distinct items."""
    return permutations(items, k)


def exhaustive_calls(n_examples: int, n_passages: int, k: int | None = None) -> int:
    """Model calls needed to score every ordering of k passages for each example.

    With k omitted, every passage is used, so each example contributes
    n_passages! orderings. Use this before launching a pilot: 50 examples
    with 8 passages each take 2,016,000 calls.
    """
    k = n_passages if k is None else k
    return n_examples * count_sequences(n_passages, k)
