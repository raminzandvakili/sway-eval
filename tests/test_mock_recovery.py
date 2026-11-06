"""The analysis must recover what the mock model was built with."""

from itertools import permutations
from statistics import fmean

import pytest

from sway_eval.adapters import DecisionRequest, MockAdapter, text_scorer
from sway_eval.measures import order_variance, unit_position_effects
from sway_eval.synth import RubricSpec, generate, rubric_scorer
from sway_eval.units import FormatSpec, render

SCORES = {"u0": 2.0, "u1": -1.0, "u2": 0.5, "u3": 1.5}
WEIGHTS = [1.6, 1.1, 0.8, 0.5]


def logits_by_order(weights):
    m = MockAdapter(text_scorer(SCORES), weights)
    out = {}
    for order in permutations(SCORES):
        out[order] = m.logit(" ".join(order))
    return out


def test_position_effects_match_closed_form():
    # For an additive model, E[z | u at p] - E[z] =
    #   c_u w_p + (C - c_u)(S - w_p)/(n - 1) - C S / n
    z = logits_by_order(WEIGHTS)
    eff = unit_position_effects(z)
    n, C, S = len(WEIGHTS), sum(SCORES.values()), sum(WEIGHTS)
    for u, c in SCORES.items():
        for p, w in enumerate(WEIGHTS):
            expect = c * w + (C - c) * (S - w) / (n - 1) - C * S / n
            assert eff[u][p] == pytest.approx(expect)


def test_flat_weights_give_no_order_variance():
    z = logits_by_order([1.0] * 4)
    assert order_variance(list(z.values())) == pytest.approx(0.0)
    assert order_variance(list(logits_by_order(WEIGHTS).values())) > 0


def test_rubric_scorer_reads_points_under_any_layout_and_format():
    spec = RubricSpec()
    items = generate(spec, levels=[-3, 0, 2], per_level=4, seed=7)
    scorer = rubric_scorer(spec)
    fmts = [FormatSpec(), FormatSpec(separator="\n", heading_style="## {h}", bullet="* "),
            FormatSpec(heading_style="{H}", bullet=""), FormatSpec(heading_style="[{h}]")]
    for it in items:
        total = sum(it.meta["points"].values())
        assert total == spec.threshold + it.meta["level"]
        assert it.meta["label"] == (it.meta["level"] >= 0)
        for order in list(permutations(it.convention_order))[:10]:
            for f in fmts:
                assert sum(scorer(render(it, order, f))) == total


def test_mock_on_rubric_threshold_at_zero_logit():
    spec = RubricSpec()
    it = generate(spec, levels=[0], per_level=1, seed=1)[0]
    m = MockAdapter(rubric_scorer(spec), [1.0] * 5, bias=-spec.threshold)
    d = m.decide(DecisionRequest("x", render(it, it.convention_order), spec.options))
    assert d.probs[0] == pytest.approx(0.5)


def test_levels_are_uniform():
    items = generate(RubricSpec(), levels=list(range(-5, 5)), per_level=5, seed=0)
    counts = {}
    for it in items:
        counts[it.meta["level"]] = counts.get(it.meta["level"], 0) + 1
    assert set(counts.values()) == {5} and len(counts) == 10
    assert fmean(1.0 if it.meta["label"] else 0.0 for it in items) == pytest.approx(0.5)


def test_one_year_is_singular_and_still_scored():
    from sway_eval.synth import RubricSpec, generate, rubric_scorer
    from sway_eval.units import FormatSpec, render
    items = generate(RubricSpec(), list(range(-5, 5)), 5, seed=0)
    texts = [render(it, it.convention_order, FormatSpec()) for it in items]
    assert not any("1 years" in t for t in texts)
    one = [(it, t) for it, t in zip(items, texts) if "1 year of relevant" in t]
    it, t = one[0]
    assert sum(rubric_scorer(RubricSpec())(t)) == sum(it.meta["points"].values())


def test_carryover_adds_the_adjacent_product_term():
    plain = MockAdapter(text_scorer(SCORES), WEIGHTS)
    carry = MockAdapter(text_scorer(SCORES), WEIGHTS, carryover=0.3)
    order = ("u0", "u1", "u2", "u3")
    adjacent = 2.0 * -1.0 + -1.0 * 0.5 + 0.5 * 1.5
    assert carry.logit(" ".join(order)) - plain.logit(" ".join(order)) == pytest.approx(0.3 * adjacent)


def test_format_effect_applies_without_dash_bullets():
    m = MockAdapter(text_scorer(SCORES), WEIGHTS, format_effect=-0.4)
    bulleted = "- u0\n- u1\n- u2\n- u3"
    plain = "u0\nu1\nu2\nu3"
    assert m.logit(plain) - m.logit(bulleted) == pytest.approx(-0.4)


def test_drift_changes_weights_after_the_given_call():
    m = MockAdapter(text_scorer(SCORES), WEIGHTS, drift_after=2, drift_weights=[1.0] * 4)
    req = DecisionRequest(instruction="q", context="u0 u1 u2 u3", options=("A", "B"))
    before = [m.decide(req).probs[0] for _ in range(2)]
    after = m.decide(req).probs[0]
    assert before[0] == before[1] != after
    with pytest.raises(ValueError):
        MockAdapter(text_scorer(SCORES), WEIGHTS, drift_after=2)
