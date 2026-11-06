import json

import pytest

from sway_eval.analysis import analyze
from sway_eval.config import build_adapter, validate
from sway_eval.records import save_items
from sway_eval.runner import execute, make_plan
from sway_eval.synth import RubricSpec, generate


def config(tmp_path, **model):
    spec = RubricSpec()
    return {
        "name": "a_test_mock", "paper": "A", "date": "2026-10-25", "seed": 0,
        "model": {"provider": "mock", "name": "mock-1", "position_weights": [1.4, 1.2, 1.0, 0.8, 0.6],
                  "scale": 0.8, "noise": 0.2, "grid": 0.01, **model},
        "dataset": {"kind": "synthetic_rubric", "items": str(tmp_path / "items.jsonl"),
                    "levels": [-2, 2], "per_level": 2},
        "prompt": {"instruction": spec.instruction(), "options": list(spec.options), "target": "ADVANCE"},
        "layouts": {"kind": "all"},
        "content_free": True,
        "noise": {"margin_edges": {"far": 0.6, "near": 0.2}, "items_per_stratum": 2, "repeats": 10,
                  "target_share": 0.1, "max_layout_repeats": 3},
        "canary": {"per_stratum": {"at": 1, "near": 1, "far": 1}, "batch_size": 400, "dispersion_pct": 0.99,
                   "mean_band": [0.005, 0.995], "sigma_floor": 0.005, "grid": 0.01, "floor_changes": 2,
                   "reference_checks": 2000, "reference_seed": 1},
        "format_variants": [{"separator": "\n"}, {"heading_style": "## {h}", "bullet": "* "}],
        "analysis": {"tie_option": "DO NOT ADVANCE", "axis": "level", "taus": [0.8, 0.9], "zone_width": "span", "pi": 0.10, "B": 2, "level_step": 1,
                     "threshold_level": 0, "slope_window": 2},
        "output_root": str(tmp_path / "results"),
        "cache": str(tmp_path / "cache.sqlite"),
        "_base": str(tmp_path),
    }


def items_for(cfg):
    return generate(RubricSpec(), list(range(-2, 3)), 2, seed=0)


def test_end_to_end_with_mock(tmp_path):
    cfg = config(tmp_path)
    validate(cfg)
    items = items_for(cfg)
    save_items(items, cfg["dataset"]["items"])
    plan = make_plan(items, cfg, paid=False)
    fixed = {k: v for k, v in plan.calls_by_kind.items() if "at most" not in k}
    assert fixed == {"layout": 10 * 120, "content_free": 10, "format": 20}
    assert plan.calls_by_kind["repeat (at most)"] == 6 * 10
    assert plan.calls_by_kind["layout_repeat (at most)"] == 2 * 120 * 10
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    assert out is not None and out.name == "2026-10-25_a_test_mock_mock-1"
    kinds = [json.loads(l)["kind"] for l in (out / "records.jsonl").open()]
    assert kinds.count("layout") == 1200 and kinds[:10] == ["layout"] * 10  # convention calls first
    assert kinds.count("repeat") % 10 == 0 and kinds.count("canary") > 0
    assert len(kinds) <= plan.total_calls + kinds.count("canary")
    s = analyze(out)
    assert s["n_items"] == 10
    assert s["mean_order_variance_target"] > 0
    assert s["mean_order_variance_classes"] == pytest.approx(2 * s["mean_order_variance_target"])
    assert s["n_graded"] == 10 and "confident_flip_rate_tau_0.8" in s and "confident_pairwise_flip_rate_tau_0.8" in s
    assert "flip_zone_direct" in s and "gate" in s
    assert "noise" in s and "noise_rule" in s["gate"] and s["canary"]["checks"] > 0 and s["halted"] is None
    row = json.loads((out / "per_item.jsonl").open().readline())
    item = items[0] if items[0].id == row["item_id"] else next(i for i in items if i.id == row["item_id"])
    pos = row["placement_sensitive_convention_position"]
    assert item.convention_order[pos - 1] == row["placement_sensitive_unit"]
    assert row["placement_sensitive_heading"] == {u.id: u.heading for u in item.units}[row["placement_sensitive_unit"]]
    setup = json.loads((out / "noise_setup.json").read_text())
    assert set(setup["repeat_items"]) == {"far", "near", "at"} and "layout_repeats" in setup
    assert json.loads((out / "config.json").read_text())["name"] == "a_test_mock"
    again = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    assert again.name == out.name + "_r2" and (out / "records.jsonl").exists()


def test_runs_without_noise_section(tmp_path):
    cfg = config(tmp_path)
    del cfg["noise"], cfg["canary"]
    items = items_for(cfg)[:2]
    plan = make_plan(items, cfg, paid=False)
    assert plan.calls_by_kind == {"layout": 240, "content_free": 2, "format": 4}
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    assert sum(1 for _ in (out / "records.jsonl").open()) == plan.total_calls
    assert "noise" not in analyze(out)


def test_old_noise_floor_key_is_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg["noise_floor"] = {"items": 2, "repeats": 3}
    with pytest.raises(ValueError):
        validate(cfg)
    cfg = config(tmp_path)
    del cfg["noise"]
    with pytest.raises(ValueError):  # canary without noise
        validate(cfg)


class Drifting:
    """Wraps an adapter and shifts the target probability after ``after`` calls."""

    def __init__(self, inner, after, shift):
        self.inner, self.after, self.shift, self.calls = inner, after, shift, 0
        self.model_id, self.paid = inner.model_id + "-drift", False

    def decide(self, request, repeat=0):
        from sway_eval.adapters.base import Decision
        d = self.inner.decide(request, repeat)
        self.calls += 1
        if self.calls <= self.after:
            return d
        t = request.options.index("ADVANCE")
        p = min(0.99, d.probs[t] + self.shift)
        probs = [0.0, 0.0]
        probs[t], probs[1 - t] = p, 1 - p
        return Decision(tuple(probs), d.model_version, d.raw)


def test_canary_halts_on_confirmed_drift(tmp_path):
    cfg = config(tmp_path)
    items = items_for(cfg)
    adapter = Drifting(build_adapter(cfg), after=600, shift=0.3)
    msgs = []
    out = execute(items, cfg, adapter, log=msgs.append)
    meta = json.loads((out / "run.json").read_text())
    assert meta["halted"] and "canary alarm confirmed" in meta["halted"]
    checks = [json.loads(l) for l in (out / "canary_checks.jsonl").open()]
    assert checks[-1]["tag"].endswith("-confirm") and checks[-1]["alarm"] and checks[-2]["alarm"]
    assert analyze(out)["halted"] == meta["halted"]


def test_new_epoch_does_not_replay_cache(tmp_path):
    cfg = config(tmp_path)
    items = items_for(cfg)[:2]
    del cfg["noise"], cfg["canary"]
    adapter = build_adapter(cfg)
    execute(items, cfg, adapter, log=lambda *_: None)
    n = adapter.calls
    execute(items, cfg, adapter, log=lambda *_: None)
    assert adapter.calls == n  # same epoch: all cached
    cfg["epoch"] = 1
    execute(items, cfg, adapter, log=lambda *_: None)
    assert adapter.calls > n


def test_paid_run_needs_yes(tmp_path):
    cfg = config(tmp_path)
    items = items_for(cfg)
    adapter = build_adapter(cfg)
    adapter.paid = True
    msgs = []
    assert execute(items, cfg, adapter, yes=False, log=msgs.append) is None
    assert any("Planned calls" in m for m in msgs) and adapter.calls == 0


def test_config_requires_pre_run_choices(tmp_path):
    cfg = config(tmp_path)
    cfg["analysis"]["tie_option"] = "MAYBE"
    with pytest.raises(ValueError):
        validate(cfg)
    cfg = config(tmp_path)
    cfg["model"] = {"provider": "typesafe", "name": "jev", "version": ""}
    with pytest.raises(ValueError):
        validate(cfg)


def test_option_rotations_are_a_separate_baseline(tmp_path):
    cfg = config(tmp_path)
    cfg["option_rotations"] = True
    items = items_for(cfg)[:2]
    plan = make_plan(items, cfg, paid=False)
    assert plan.calls_by_kind["option_rotation"] == 2
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    recs = [json.loads(l) for l in (out / "records.jsonl").open()]
    layout = [r for r in recs if r["kind"] == "layout"]
    rot = [r for r in recs if r["kind"] == "option_rotation"]
    assert all(r["option_rotation"] == 0 and r["options"] == ["ADVANCE", "DO NOT ADVANCE"] for r in layout)
    assert all(r["option_rotation"] == 1 and r["options"] == ["DO NOT ADVANCE", "ADVANCE"] for r in rot)
    s = analyze(out)
    assert s["mean_option_order_sd"] is not None


def test_rotation_baseline_sees_option_position_bias(tmp_path):
    cfg = config(tmp_path, noise=0.0, grid=None)
    cfg["option_rotations"] = True
    items = items_for(cfg)[:4]
    flat = analyze(execute(items, cfg, build_adapter(cfg), log=lambda *_: None))["mean_option_order_sd"]
    cfg2 = config(tmp_path, noise=0.0, grid=None, first_option_bias=1.0, name="mock-bias")
    cfg2["option_rotations"] = True
    biased = analyze(execute(items, cfg2, build_adapter(cfg2), log=lambda *_: None))["mean_option_order_sd"]
    assert flat == pytest.approx(0.0) and biased > 0.05

def test_json_config_round_trip(tmp_path):
    from sway_eval.config import load_config
    cfg = config(tmp_path)
    cfg.pop("_base")
    path = tmp_path / "a_test_mock.json"
    path.write_text(json.dumps(cfg))
    loaded = load_config(path)
    assert loaded["date"] == "2026-10-25" and loaded["_base"] == str(tmp_path)


def test_sampled_layouts_are_seeded_and_keep_the_convention_order(tmp_path):
    from sway_eval.runner import layouts_for
    cfg = config(tmp_path)
    it = items_for(cfg)[0]
    a = layouts_for(it, "sample", 36, 7)
    assert a == layouts_for(it, "sample", 36, 7) and len(set(a)) == 36
    assert a[0] == tuple(it.convention_order) and a != layouts_for(it, "sample", 36, 8)
    with pytest.raises(ValueError):
        layouts_for(it, "sample", 121, 7)
    with pytest.raises(ValueError):
        layouts_for(it, "sample", 36, None)


def test_sample_items_is_seeded_and_stratified(tmp_path):
    from sway_eval.records import Item, Unit, sample_items
    items = [Item(id=f"c{i}", units=(Unit(id=f"c{i}:a", heading=None, lines=("x",)),), convention_order=(f"c{i}:a",),
                  meta={"gold": "S" if i < 60 else ("R" if i < 90 else "N")}) for i in range(100)]
    a = sample_items(items, 20, 7, "gold")
    assert a == sample_items(items, 20, 7, "gold") and len(a) == 20
    counts = {g: sum(1 for it in a if it.meta["gold"] == g) for g in "SRN"}
    assert counts == {"S": 12, "R": 6, "N": 2}
    assert [it.id for it in a] == sorted((it.id for it in a), key=lambda x: int(x[1:]))  # original order
    assert a != sample_items(items, 20, 8, "gold") and len(sample_items(items, 5, 1)) == 5
    with pytest.raises(ValueError):
        sample_items(items, 101, 1)


def test_launch_date_is_stamped_in_utc(tmp_path):
    from datetime import datetime, timezone
    cfg = config(tmp_path)
    cfg["date"] = "launch"
    items = items_for(cfg)
    save_items(items, cfg["dataset"]["items"])
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    today = datetime.now(timezone.utc).date().isoformat()
    assert out.name.startswith(today + "_")
    meta = json.loads((out / "run.json").read_text())
    assert meta["launched_utc"].startswith(today)
    assert json.loads((out / "config.json").read_text())["date"] == today


def test_subsets_plan_every_proper_subset_in_convention_order(tmp_path):
    from sway_eval.runner import plan_calls
    cfg = config(tmp_path)
    cfg["subsets"] = True
    items = items_for(cfg)
    subs = [c for c in plan_calls(items, cfg) if c.kind == "subset" and c.item_id == items[0].id]
    n = len(items[0].convention_order)
    assert len(subs) == 2 ** n - 2
    pos = {u: i for i, u in enumerate(items[0].convention_order)}
    assert all(list(c.order) == sorted(c.order, key=pos.get) for c in subs)
    save_items(items, cfg["dataset"]["items"])
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    kinds = [json.loads(l)["kind"] for l in (out / "records.jsonl").open()]
    assert kinds.count("subset") == len(items) * (2 ** n - 2)
    assert analyze(out)["n_items"] == len(items)  # analysis ignores the subset calls


def test_subset_sizes_limit_the_subsets(tmp_path):
    from sway_eval.runner import plan_calls
    cfg = config(tmp_path)
    cfg["subsets"], cfg["subset_sizes"] = True, [1]
    items = items_for(cfg)
    it = items[0]
    subs = [c for c in plan_calls(items, cfg) if c.kind == "subset" and c.item_id == it.id]
    assert [c.order for c in subs] == [(u,) for u in it.convention_order]
    cfg["subset_sizes"] = [len(it.convention_order)]
    with pytest.raises(ValueError):
        plan_calls(items, cfg)


def test_reference_unit_is_one_call_per_item(tmp_path):
    from sway_eval.runner import plan_calls, request_for
    cfg = config(tmp_path)
    cfg["reference_unit"] = {"heading": "Violin", "text": "The violin is a string instrument."}
    items = items_for(cfg)
    refs = [c for c in plan_calls(items, cfg) if c.kind == "reference"]
    assert [c.item_id for c in refs] == [it.id for it in items]
    text = request_for(refs[0], items[0], cfg).context
    assert "Violin" in text and "The violin is a string instrument." in text
    assert not any(line in text for u in items[0].units for line in u.lines)
    cfg["reference_unit"] = [cfg["reference_unit"], {"heading": "Chess", "text": "Chess is a board game."}]
    refs = [c for c in plan_calls(items, cfg) if c.kind == "reference" and c.item_id == items[0].id]
    texts = [request_for(c, items[0], cfg).context for c in refs]
    assert len(refs) == 2 and "Violin" in texts[0] and "Chess is a board game." in texts[1]


def test_reversed_subsets_and_blank_baseline(tmp_path):
    from sway_eval.runner import plan_calls, request_for
    cfg = config(tmp_path)
    cfg["subsets"], cfg["blank_baseline"] = "reversed", True
    items = items_for(cfg)
    it = items[0]
    calls = [c for c in plan_calls(items, cfg) if c.item_id == it.id]
    subs = [c for c in calls if c.kind == "subset"]
    rev = {u: i for i, u in enumerate(reversed(it.convention_order))}
    assert all(list(c.order) == sorted(c.order, key=rev.get) for c in subs)
    blank = [c for c in calls if c.kind == "blank"]
    assert len(blank) == 1 and blank[0].order == tuple(reversed(it.convention_order))
    text = request_for(blank[0], it, cfg).context
    assert all(u.heading in text for u in it.units if u.heading)
    assert not any(line in text for u in it.units for line in u.lines)
    cfg["subsets"] = "sideways"
    with pytest.raises(ValueError):
        plan_calls(items, cfg)


def test_subsets_in_an_explicit_heading_order(tmp_path):
    from sway_eval.runner import plan_calls
    cfg = config(tmp_path)
    items = items_for(cfg)
    it = items[0]
    heads = [it.unit(u).heading for u in it.convention_order]
    order = heads[:1] + heads[2:4] + heads[1:2] + heads[4:]
    cfg["subsets"], cfg["blank_baseline"] = order, True
    calls = [c for c in plan_calls(items, cfg) if c.item_id == it.id]
    blank = [c for c in calls if c.kind == "blank"][0]
    assert [it.unit(u).heading for u in blank.order] == order
    pos = {u: i for i, u in enumerate(blank.order)}
    assert all(list(c.order) == sorted(c.order, key=pos.get) for c in calls if c.kind == "subset")
    cfg["subsets"] = heads[:4]
    with pytest.raises(ValueError):
        plan_calls(items, cfg)


def test_example_config_loads_without_a_paper_letter():
    from pathlib import Path

    from sway_eval.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "examples" / "mock_audit.yaml")
    assert cfg["model"]["provider"] == "mock" and "paper" not in cfg


def test_report_after_a_mock_run(tmp_path):
    from sway_eval.report import write_report
    cfg = config(tmp_path)
    items = items_for(cfg)
    save_items(items, cfg["dataset"]["items"])
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    md = write_report(out).read_text(encoding="utf-8")
    for heading in ("How often layout decides", "What it costs", "Noise and canaries", "What each defense buys"):
        assert heading in md
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    t = rep["defenses"]["table"]
    assert t["balanced_williams"]["calls"] == 10 and t["full_symmetrization"]["agreement"] == 1.0


def test_balanced_set_average_leaves_out_the_extra_convention_call(tmp_path):
    cfg = config(tmp_path)
    cfg["layouts"] = {"kind": "williams"}
    items = items_for(cfg)
    save_items(items, cfg["dataset"]["items"])
    assert make_plan(items, cfg, paid=False).calls_by_kind["layout"] == len(items) * 11  # 10 balanced + template order
    out = execute(items, cfg, build_adapter(cfg), log=lambda *_: None)
    analyze(out)
    rows = [json.loads(l) for l in (out / "per_item.jsonl").open()]
    assert {r["n_layouts"] for r in rows} == {10}
    assert all("convention_decision" in r for r in rows)
    from sway_eval.report import write_report
    md = write_report(out).read_text(encoding="utf-8")
    assert f"- {len(items) * 10:,} layouts in the audited set" in md and "template-order calls" in md
