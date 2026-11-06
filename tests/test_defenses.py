"""Defense comparison on the mock: known answers."""

import math
from itertools import permutations

import pytest

from sway_eval.adapters import MockAdapter, text_scorer
from sway_eval.defenses import compare_defenses

UNITS = ("a", "b", "c", "d", "e")


def table(weights, scores):
    m = MockAdapter(text_scorer(dict(zip(UNITS, scores))), weights, bias=-sum(scores) * sum(weights) / 5)
    layouts, conv = {}, {}
    for k in range(6):
        iid = f"r{k}"
        shift = (k - 2.5) * 0.4
        layouts[iid] = {o: 1 / (1 + math.exp(-(m.logit(" ".join(o)) + shift))) for o in permutations(UNITS)}
        conv[iid] = UNITS
    return compare_defenses(layouts, conv, draws=20, boot=50)


def test_flat_weights_make_every_defense_exact():
    t = table([1.0] * 5, [2, 1, 0.5, 1.5, 1])["table"]
    for name, row in t.items():
        assert row["agreement"] == pytest.approx(1.0), name
        assert row["residual"] == pytest.approx(0.0, abs=1e-12), name


def test_balanced_set_beats_one_random_layout_on_position_weights():
    res = table([1.6, 1.3, 1.0, 0.7, 0.4], [2, 1, 0.5, 1.5, 1])
    t = res["table"]
    assert t["balanced_williams"]["calls"] == 10 and t["full_symmetrization"]["calls"] == 120
    assert t["balanced_williams"]["residual"] < 0.2 * t["k_random_1"]["residual"]
    assert t["full_symmetrization"]["residual"] == pytest.approx(0.0, abs=1e-12)


def test_missing_orderings_are_refused():
    layouts = {"r0": {UNITS: 0.6}}
    with pytest.raises(ValueError):
        compare_defenses(layouts, {"r0": UNITS})
