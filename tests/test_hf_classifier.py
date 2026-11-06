"""The generic classifier adapter, with a fake classifier (no download, no torch)."""

import math

import pytest

from sway_eval.adapters import DecisionRequest
from sway_eval.adapters.hf_classifier import HFClassifierAdapter

REQ = DecisionRequest(instruction="Should this candidate advance?", context="Experience:\n- 3 years",
                      options=("ADVANCE", "DO NOT ADVANCE"))


def fake(logits, n=12):
    seen = []

    def fn(instruction, record):
        seen.append((instruction, record))
        return logits, n
    return fn, seen


def test_softmax_over_the_mapped_labels_in_request_order():
    fn, seen = fake([0.5, 2.0, -1.0])  # three labels; options use labels 1 and 2
    a = HFClassifierAdapter(fn, {"ADVANCE": 1, "DO NOT ADVANCE": 2}, max_length=512, model_id="toy@abc")
    d = a.decide(REQ)
    p = math.exp(2.0) / (math.exp(2.0) + math.exp(-1.0))
    assert d.probs == pytest.approx((p, 1 - p))
    assert d.model_version == "toy@abc" and seen == [(REQ.instruction, REQ.context)]
    rotated = DecisionRequest(REQ.instruction, REQ.context, ("DO NOT ADVANCE", "ADVANCE"))
    assert a.decide(rotated).probs == pytest.approx((1 - p, p))


def test_refuses_long_records_and_unmapped_options():
    fn, _ = fake([0.0, 1.0], n=600)
    a = HFClassifierAdapter(fn, {"ADVANCE": 0, "DO NOT ADVANCE": 1}, max_length=512, model_id="toy@abc")
    with pytest.raises(ValueError, match="over the 512 limit"):
        a.decide(REQ)
    b = HFClassifierAdapter(fake([0.0, 1.0])[0], {"ADVANCE": 0}, max_length=512, model_id="toy@abc")
    with pytest.raises(ValueError, match="no classifier label"):
        b.decide(REQ)


def test_from_pretrained_needs_a_full_commit_hash():
    with pytest.raises(ValueError, match="40-character"):
        HFClassifierAdapter.from_pretrained("some/model", "main", {"ADVANCE": 0})
