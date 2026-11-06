import math

import pytest

from sway_eval import (
    calibrate,
    log_softmax,
    margin_from_probs,
    margin_vs_rest,
    target_margin,
    top2_margin,
)


def softmax(z):
    return [math.exp(v) for v in log_softmax(z)]


Z = [2.0, 0.5, -1.0]


def test_top2_margin_basic():
    assert top2_margin(Z) == pytest.approx(1.5)


def test_binary_margin_is_absolute_log_odds():
    assert top2_margin([0.3, 1.7]) == pytest.approx(1.4)
    assert top2_margin([1.0, 1.0]) == 0.0


@pytest.mark.parametrize("shift", [-100.0, -3.5, 0.0, 7.0, 250.0])
def test_margins_are_shift_invariant(shift):
    shifted = [v + shift for v in Z]
    assert top2_margin(shifted) == pytest.approx(top2_margin(Z))
    assert margin_vs_rest(shifted) == pytest.approx(margin_vs_rest(Z))
    assert target_margin(shifted, 1) == pytest.approx(target_margin(Z, 1))


def test_margin_from_probs_matches_logits():
    assert margin_from_probs(softmax(Z)) == pytest.approx(top2_margin(Z))


def test_margin_from_probs_ignores_normalization():
    p = softmax(Z)
    assert margin_from_probs([3 * v for v in p]) == pytest.approx(margin_from_probs(p))


def test_margin_from_probs_rounded_runner_up_is_inf():
    assert margin_from_probs([1.0, 0.0, 0.0]) == math.inf


def test_margin_vs_rest_is_log_odds_of_winner():
    p1 = max(softmax(Z))
    assert margin_vs_rest(Z) == pytest.approx(math.log(p1 / (1 - p1)))


def test_margin_vs_rest_equals_top2_for_binary():
    assert margin_vs_rest([0.2, 1.9]) == pytest.approx(top2_margin([0.2, 1.9]))


def test_target_margin_sign():
    assert target_margin(Z, 0) == pytest.approx(1.5)
    assert target_margin(Z, 2) == pytest.approx(-3.0)


def test_uniform_content_free_leaves_margins_unchanged():
    assert top2_margin(calibrate(Z, [0.4, 0.4, 0.4])) == pytest.approx(top2_margin(Z))


def test_calibration_removes_label_prior():
    # The model favors option 0 even with no content; calibration must shrink option 0's margin.
    before = target_margin(Z, 0)
    after = target_margin(calibrate(Z, [1.0, 0.0, 0.0]), 0)
    assert after == pytest.approx(before - 1.0)


def test_calibration_matches_zhao_probability_form():
    # Zhao et al. (2021): q is proportional to p / p_cf. Check against the log-space version.
    cf = [0.9, -0.2, 0.1]
    p, p_cf = softmax(Z), softmax(cf)
    q = [a / b for a, b in zip(p, p_cf)]
    q = [v / sum(q) for v in q]
    assert margin_from_probs(q) == pytest.approx(top2_margin(calibrate(Z, cf)))


@pytest.mark.parametrize("bad", [[], [1.0], [1.0, float("nan")]])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        top2_margin(bad)


def test_rejects_negative_probs():
    with pytest.raises(ValueError):
        margin_from_probs([0.7, -0.1, 0.4])


def test_target_out_of_range():
    with pytest.raises(IndexError):
        target_margin(Z, 3)


def test_calibrate_length_mismatch():
    with pytest.raises(ValueError):
        calibrate(Z, [0.0, 0.0])
