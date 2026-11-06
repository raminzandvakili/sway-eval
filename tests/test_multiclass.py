import pytest

from sway_eval.measures import flip_probability, order_variance
from sway_eval.multiclass import (
    argmax,
    confident_flip,
    confident_pairwise_flip_rate,
    flip_stats,
    order_variance_classes,
    pairwise_flip_rate,
    winner_margin,
)

OPTS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def test_ties_go_to_named_option_not_reference():
    assert argmax([0.34, 0.33, 0.33], OPTS, "NOT ENOUGH INFO") == (0, False)
    assert argmax([0.33, 0.34, 0.34], OPTS, "NOT ENOUGH INFO") == (2, True)
    assert argmax([0.4, 0.4, 0.2], OPTS, "NOT ENOUGH INFO") == (0, True)  # NEI not tied: first in order


def test_binary_reduces_to_binary_measures():
    ps = [0.6, 0.7, 0.4, 0.8, 0.45]
    vecs = [[p, 1 - p] for p in ps]
    opts = ["ADVANCE", "DO NOT ADVANCE"]
    assert flip_stats(vecs, opts, "DO NOT ADVANCE")["flip_probability"] == pytest.approx(flip_probability(ps, "other"))
    assert order_variance_classes(vecs) == pytest.approx(2 * order_variance(ps))
    assert winner_margin([0.7, 0.3], 0) == pytest.approx(2 * 0.7 - 1)


def test_flip_moves_and_confident_flip():
    vecs = [[0.6, 0.2, 0.2], [0.85, 0.1, 0.05], [0.1, 0.85, 0.05], [0.2, 0.3, 0.5]]
    s = flip_stats(vecs, OPTS, "NOT ENOUGH INFO")
    assert s["reference"] == "SUPPORTS"
    assert s["flip_moves"] == {"SUPPORTS->REFUTES": 1, "SUPPORTS->NOT ENOUGH INFO": 1}
    assert confident_flip(vecs, OPTS, "NOT ENOUGH INFO", 0.8) is True
    assert confident_flip(vecs, OPTS, "NOT ENOUGH INFO", 0.9) is False
    assert pairwise_flip_rate(vecs, OPTS, "NOT ENOUGH INFO") == pytest.approx(5 / 6)
    # one confident pair with different decisions (rows 2 and 3) among 6 pairs
    assert confident_pairwise_flip_rate(vecs, OPTS, "NOT ENOUGH INFO", 0.8) == pytest.approx(1 / 6)
    assert confident_pairwise_flip_rate(vecs, OPTS, "NOT ENOUGH INFO", 0.9) == 0.0
    assert confident_pairwise_flip_rate(vecs * 3, OPTS, "NOT ENOUGH INFO", 0.8) == pytest.approx(9 / 66)
