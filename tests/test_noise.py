import random
from statistics import fmean, stdev

import pytest

from sway_eval.measures import flip_zone_noise
from sway_eval.noise import (
    CanaryRule,
    CanarySet,
    assign_strata,
    by_stratum,
    calibrate,
    canary_alarm,
    canary_stats,
    draw_per_stratum,
    item_noise,
    layout_repeats,
    pool,
    quantile,
    stratum,
    top2_margin,
)

OPTS = ["A", "B"]
EDGES = {"far": 0.6, "near": 0.2}


def test_strata_use_margin_units():
    assert top2_margin([0.8, 0.2]) == pytest.approx(0.6)
    assert stratum(top2_margin([0.8, 0.2]), EDGES) == "far"  # 0.6 exactly, despite float error
    assert stratum(top2_margin([0.6, 0.4]), EDGES) == "near"
    assert stratum(top2_margin([0.55, 0.45]), EDGES) == "at"
    assert stratum(top2_margin([0.5, 0.3, 0.2]), EDGES) == "near"


def test_draw_is_seeded_and_takes_all_when_short():
    strata = assign_strata({f"i{k}": [0.5 + k / 100, 0.5 - k / 100] for k in range(30)}, EDGES)  # margins 0 to 0.58
    a = draw_per_stratum(by_stratum(strata), {"far": 10, "near": 10, "at": 10}, 0, "x")
    assert a == draw_per_stratum(by_stratum(strata), {"far": 10, "near": 10, "at": 10}, 0, "x")
    assert len(a["at"]) == 10 and len(a["near"]) == 10 and a["far"] == []


def test_item_noise_counts_flicker_and_flips():
    reps = [[0.50, 0.50], [0.51, 0.49], [0.51, 0.49], [0.49, 0.51], [0.51, 0.49]]
    x = item_noise(reps, OPTS, tie_option="B")
    assert OPTS[x.majority] == "A"
    assert x.repeat_flips == 2  # the tie (to B) and the 0.49
    assert x.identical_share == pytest.approx(0.6)
    assert x.deviations == {"0": 3, "<=0.01": 1, "<=0.02": 1}
    assert x.var_classes == pytest.approx(2 * x.var_win)


def test_deterministic_item_has_zero_noise():
    x = item_noise([[0.7, 0.3]] * 10, OPTS, None)
    p = pool([x, x])
    assert p["sigma_noise"] == 0 and p["repeat_flip_rate"] == 0 and p["identical_share"] == 1


def test_layout_repeats_formula():
    ok = layout_repeats(noise_var=0.001, measured_var=0.02, target_share=0.1, max_r=5)
    assert ok["R"] == 1 and ok["noise_share"] == pytest.approx(0.05)
    # noise 0.004, layout 0.006: R >= 9 * 0.004 / 0.006 = 6 -> above the cap
    bad = layout_repeats(0.004, 0.010, 0.1, 5)
    assert bad["R_needed"] == 6 and bad["R"] is None and bad["too_noisy"]
    mid = layout_repeats(0.002, 0.010, 0.1, 5)  # 9 * 0.002 / 0.008 = 2.25 -> 3
    assert mid["R"] == 3
    allnoise = layout_repeats(0.01, 0.01, 0.1, 5)
    assert allnoise["too_noisy"] and allnoise["R_needed"] is None


def canaries(sd=0.01, n=20):
    ids = tuple(f"c{k}" for k in range(n))
    strata = {i: ("at" if k < 10 else "near" if k < 15 else "far") for k, i in enumerate(ids)}
    return CanarySet(ids, strata, {i: 0 for i in ids}, {i: 0.5 for i in ids}, {i: sd for i in ids})


CC = {"dispersion_pct": 0.99, "mean_band": [0.005, 0.995], "sigma_floor": 0.005, "grid": 0.01,
      "floor_changes": 2, "reference_checks": 4000, "reference_seed": 20261120}


def test_quantile_nearest_rank():
    xs = list(range(1, 101))
    assert quantile(xs, 0.99) == 99 and quantile(xs, 0.005) == 1 and quantile(xs, 1.0) == 100


def test_canary_alarm_reasons():
    cs = canaries()
    rule = CanaryRule(T=40, L=-0.005, U=0.005, sigma_floor=0.005)
    assert canary_alarm({i: 0.5 for i in cs.ids}, cs, rule) == []
    assert canary_alarm({i: 0.51 for i in cs.ids}, cs, rule)[0].startswith("mean deviation")
    # each canary moved by 1.5 sigma in alternating directions: M = 0, D = 45
    spread = {i: 0.5 + (0.015 if k % 2 else -0.015) for k, i in enumerate(cs.ids)}
    assert canary_stats(spread, cs, 0.005)[0] == pytest.approx(45)
    assert [r.split(" =")[0] for r in canary_alarm(spread, cs, rule)] == ["dispersion D"]


def test_deterministic_canaries_get_the_floors():
    cs = canaries(sd=0.0)
    reps = {i: [0.5] * 10 for i in cs.ids}
    cal = calibrate(reps, cs, CC)
    r = cal["rule"]
    assert r.T == pytest.approx(8) and r.U == pytest.approx(0.001) and r.L == pytest.approx(-0.001)
    assert cal["expected_false_alarm_rate_per_check"] == 0
    one = {i: 0.51 if k == 0 else 0.5 for k, i in enumerate(cs.ids)}
    two = {i: 0.51 if k < 2 else 0.5 for k, i in enumerate(cs.ids)}
    three = {i: 0.51 if k < 3 else 0.5 for k, i in enumerate(cs.ids)}
    assert canary_alarm(one, cs, r) == []
    assert canary_alarm(two, cs, r) == []  # exactly two steps: no alarm despite float rounding
    assert len(canary_alarm(three, cs, r)) == 2


def test_loo_threshold_is_calibrated_for_noisy_canaries():
    """Fresh calls from the same noise should alarm near the nominal 2% per
    check, where a chi-square threshold would alarm far more often."""
    rng = random.Random(7)
    draw = lambda: min(0.99, max(0.01, round((0.5 + rng.gauss(0, 0.03)) / 0.01) * 0.01))
    alarms = checks = 0
    for _ in range(40):
        reps = {f"c{k}": [draw() for _ in range(10)] for k in range(20)}
        cs = CanarySet(tuple(reps), {i: "near" for i in reps}, {i: 0 for i in reps},
                       {i: fmean(v) for i, v in reps.items()}, {i: stdev(v) for i, v in reps.items()})
        cal = calibrate(reps, cs, {**CC, "reference_checks": 2000})
        assert 40 < cal["rule"].T < 90 and 50 < cal["analytic_T_cross_check"] < 80
        for _ in range(50):
            checks += 1
            alarms += bool(canary_alarm({i: draw() for i in reps}, cs, cal["rule"]))
    assert alarms / checks < 0.06


def test_flip_zone_noise_rules():
    levels = [0, 1, 2, 3, 4]
    flips = [0.05, 0.30, 0.12, 0.40, 0.05]
    raw = {"lower_edge": 1, "upper_edge": 3}
    z = flip_zone_noise(levels, flips, [0.01, 0.10, 0.20, 0.10, 0.01], 0.10, 1, "span", raw)
    assert z["cleared_levels"] == [1, 3] and z["holes"] == [2] and z["width"] == 3
    assert z["existence_fails"] is False
    noisy = flip_zone_noise(levels, flips, [0.5] * 5, 0.10, 1, "span", raw)
    assert noisy["existence_fails"] is True and noisy["width"] == 0
    edge = flip_zone_noise(levels, flips, [0.01, 0.35, 0.01, 0.01, 0.01], 0.10, 1, "span", raw)
    assert edge["lower_edge"] == 2 and edge["width"] == 2  # noise can't make level 1 an edge
