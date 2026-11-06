from collections import Counter
from math import factorial

import pytest

from sway_eval.layouts import all_orderings, random_orderings, williams_orderings, williams_square


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_williams_balances_positions_and_carryover(n):
    rows = williams_square(n)
    assert len(rows) == (n if n % 2 == 0 else 2 * n)
    for row in rows:
        assert sorted(row) == list(range(n))
    reps = len(rows) // n
    for pos in range(n):
        assert Counter(row[pos] for row in rows) == {u: reps for u in range(n)}
    pairs = Counter((row[i], row[i + 1]) for row in rows for i in range(n - 1))
    assert set(pairs.values()) == {reps}
    assert len(pairs) == n * (n - 1)


def test_williams_set_does_not_depend_on_submitted_order():
    assert williams_orderings(["c", "a", "b"]) == williams_orderings(["b", "c", "a"])


def test_all_orderings_count_and_uniqueness():
    o = all_orderings(list("abcde"))
    assert len(o) == factorial(5) == len(set(o))


def test_random_orderings_reproducible():
    assert random_orderings(list("abcd"), 5, seed=3) == random_orderings(list("abcd"), 5, seed=3)
