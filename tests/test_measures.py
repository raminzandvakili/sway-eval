import random
from statistics import fmean

import pytest

from sway_eval.measures import (
    brier,
    decide,
    flip_probability,
    flip_zone_direct,
    flip_zone_model,
    order_variance,
    symmetrized,
)


@pytest.mark.parametrize("y", [0, 1])
def test_brier_identity_holds_for_any_label(y):
    rng = random.Random(1)
    probs = [rng.random() for _ in range(120)]
    mean_brier = fmean(brier(p, y) for p in probs)
    assert mean_brier - brier(symmetrized(probs), y) == pytest.approx(order_variance(probs))


def test_tie_rule():
    assert decide(0.5, "target") is True and decide(0.5, "other") is False
    with pytest.raises(ValueError):
        decide(0.5, "coin")


def test_flip_probability():
    assert flip_probability([0.6, 0.7, 0.4, 0.8], "other") == pytest.approx(0.25)
    assert flip_probability([0.9, 0.95], "other") == 0.0


def test_flip_zone_direct_conventions():
    levels = [-2, -1, 0, 1, 2]
    fp = [0.0, 0.2, 0.4, 0.15, 0.0]
    span = flip_zone_direct(levels, fp, 0.10, 1, "span")
    assert span["zone_levels"] == [-1, 0, 1] and span["width"] == 3
    assert span["n_holes"] == 0 and span["width_is_lower_bound"] is False
    one = flip_zone_direct(levels, [0, 0, 0.5, 0, 0], 0.10, 1, "span")
    assert one["width"] == 1
    holed = flip_zone_direct(levels, [0, 0.2, 0.05, 0.3, 0], 0.10, 1, "span")
    assert holed["width"] == 3 and holed["holes"] == [0]
    assert flip_zone_direct(levels, [0.2, 0.3, 0, 0, 0], 0.10, 1, "span")["width_is_lower_bound"] is True
    with pytest.raises(ValueError):
        flip_zone_direct(levels, fp, 0.10, 1, "other")


def test_flip_zone_model_recovers_linear_case():
    levels = [-2, -1, 0, 1, 2]
    p_sym = [0.5 + 0.1 * lv for lv in levels]
    z = flip_zone_model(levels, p_sym, [0.05] * 5, pi=0.10, threshold_level=0, window=2)
    assert z["slope"] == pytest.approx(0.1)
    assert z["centre"] == pytest.approx(0.0)
    assert z["width"] == pytest.approx(2 * z["z"] * 0.05 / 0.1)


def test_model_crossing_interpolates():
    from sway_eval.measures import model_crossing
    assert model_crossing([0, 0, 1, 1, 2, 2], [0.1, 0.1, 0.3, 0.3, 0.7, 0.7]) == pytest.approx(1.5)
    assert model_crossing([0, 1], [0.1, 0.2]) is None


def test_run_dir_name_shortens_commit():
    from sway_eval.runner import run_dir_name
    cfg = {"date": "2026-10-28", "name": "a_x"}
    assert run_dir_name(cfg, "Cloudflare/clef-flash@" + "ab" * 20) == "2026-10-28_a_x_clef-flash-abababababab"
    assert run_dir_name(cfg, "jev-1.14.0") == "2026-10-28_a_x_jev-1.14.0"


def test_flip_zone_margin_bins_by_margin():
    from sway_eval.measures import flip_zone_margin, slope
    margins = [0.01, 0.03, 0.07, 0.12, 0.14, 0.31, 0.6]
    flips = [0.4, 0.2, 0.15, 0.05, 0.06, 0.12, 0.0]
    z = flip_zone_margin(margins, flips, 0.10, 0.05, min_n=1)
    assert z["n_by_bin"]["0.000-0.050"] == 2 and z["zone_bins"] == ["0.000-0.050", "0.050-0.100", "0.300-0.350"]
    assert z["lower_edge"] == 0.0 and z["upper_edge"] == 0.35 and z["width"] == 0.35
    assert z["holes"] == ["0.100-0.150"]  # bins with no items are not holes
    assert z["share_in_zone"] == pytest.approx(6 / 7, abs=1e-6) and z["merged"] == []
    assert flip_zone_margin([0.5], [0.0], 0.10, 0.05)["width"] == 0.0
    assert slope([0, 1, 2], [1, 3, 5]) == 2 and slope([1, 1], [0, 1]) is None


def test_flip_zone_margin_merges_thin_bins_inward():
    from sway_eval.measures import flip_zone_margin
    # bin 0: 3 items, bin 1: 1 item (thin, in the zone), bin 2: 3 items below pi
    margins = [0.01] * 3 + [0.06] + [0.11] * 3
    flips = [0.3] * 3 + [0.0] + [0.0] * 3
    z = flip_zone_margin(margins, flips, 0.10, 0.05, min_n=2)
    assert z["merged"] == [[0.0, 0.1]] and z["n_by_bin"]["0.000-0.100"] == 4
    assert z["zone_bins"] == ["0.000-0.100"] and z["width"] == 0.1 and z["share_in_zone"] == pytest.approx(4 / 7, abs=1e-6)
    # a thin innermost bin merges outward
    z = flip_zone_margin([0.01, 0.06, 0.07], [0.5, 0.5, 0.5], 0.10, 0.05, min_n=2)
    assert z["merged"] == [[0.0, 0.1]] and z["width"] == 0.1
