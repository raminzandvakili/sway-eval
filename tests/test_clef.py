import pytest

from sway_eval.adapters.base import DecisionRequest
from sway_eval.adapters.clef import ClefAdapter

CRIT = {"ADVANCE": "meets the rule", "DO NOT ADVANCE": "does not"}
REV = "f" * 40


def test_record_has_system_one_shape_and_keeps_request_order():
    a = ClefAdapter("Cloudflare/clef-flash", REV, CRIT)
    r = a.record(DecisionRequest("Should this candidate advance?", "Skills: SQL", ("DO NOT ADVANCE", "ADVANCE")))
    q = r["questions"]["decision"]
    assert r["state"] == "Skills: SQL" and q["type"] == "choice"
    assert list(q["criteria"]) == ["DO NOT ADVANCE", "ADVANCE"]


def test_pins_a_full_commit_and_needs_criteria():
    with pytest.raises(ValueError):
        ClefAdapter("Cloudflare/clef-flash", "main", CRIT)
    a = ClefAdapter("Cloudflare/clef-flash", REV, {"ADVANCE": "x"})
    with pytest.raises(ValueError):
        a.record(DecisionRequest("q", "s", ("ADVANCE", "DO NOT ADVANCE")))


def test_nothing_is_loaded_until_a_call():
    a = ClefAdapter("Cloudflare/clef-flash", REV, CRIT)
    assert a.info() == {} and a._model is None and a.model_id == f"Cloudflare/clef-flash@{REV}"


def test_decision2_questions_keep_request_order():
    from sway_eval.adapters.decision2 import Decision2Adapter
    a = Decision2Adapter("vllm-sr/Decision-2.0-Nox-4B", REV, CRIT)
    q = a.questions(DecisionRequest("Advance?", "s", ("DO NOT ADVANCE", "ADVANCE")))["decision"]
    assert list(q["criteria"]) == ["DO NOT ADVANCE", "ADVANCE"] and q["type"] == "choice"
    assert a.info() == {} and a._model is None
    with pytest.raises(ValueError):
        Decision2Adapter("vllm-sr/Decision-2.0-Nox-4B", "main", CRIT)


def test_evicting_runs_on_plain_files(tmp_path):
    from sway_eval.adapters._memory import evict, evicting
    (tmp_path / "a.bin").write_bytes(b"x" * 4096)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y")
    (tmp_path / "link").symlink_to(tmp_path / "a.bin")
    assert evict(tmp_path) == 2  # links skipped
    with evicting(tmp_path, every_s=0.01):
        pass
    with evicting(None):
        pass


def test_cloning_before_copy_is_exact_and_restores_the_loader():
    pytest.importorskip("transformers")
    import torch
    from transformers import core_model_loading as cml

    from sway_eval.adapters._memory import cloning_before_copy
    from transformers import modeling_utils as mu
    original, opener = cml._materialize_copy, mu.safe_open
    t = torch.arange(6, dtype=torch.bfloat16)
    with cloning_before_copy():
        out = cml._materialize_copy(t, device="cpu")
        assert torch.equal(out, t) and out.data_ptr() != t.data_ptr()
    assert cml._materialize_copy is original and mu.safe_open is opener
    with cloning_before_copy(False):
        assert cml._materialize_copy is original


def test_low_peak_dtypes_follow_the_package_rule():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from torch import nn

    from sway_eval.adapters._d2_lowpeak import to_package_dtypes

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.ModuleDict({"emb": nn.Embedding(4, 3), "lin": nn.Linear(3, 3), "norm": nn.LayerNorm(3)})
            self.head = nn.Linear(3, 1)
            self.backbone.lin.register_buffer("scale", torch.ones(3))

    m = Model().to(torch.bfloat16)
    before = m.backbone.lin.weight.detach().clone()
    to_package_dtypes(m, torch)
    assert m.backbone.lin.weight.dtype == torch.bfloat16 and torch.equal(m.backbone.lin.weight, before)
    assert m.backbone.lin.bias.dtype == torch.bfloat16
    for p in (m.backbone.emb.weight, m.backbone.norm.weight, m.head.weight, m.backbone.lin.scale):
        assert p.dtype == torch.float32
