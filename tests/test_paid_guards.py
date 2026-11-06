import json

import pytest

from sway_eval.adapters.base import Decision, DecisionRequest
from sway_eval.adapters.microsoft import MicrosoftDecisionAdapter, endpoint
from sway_eval.adapters.spend import SpendCapped, SpendCapReached

CRIT = {"ADVANCE": "meets the rule", "DO NOT ADVANCE": "does not"}
REQ = DecisionRequest("Advance?", "Skills: SQL", ("ADVANCE", "DO NOT ADVANCE"))


class Fake:
    model_id = "fake/1"
    paid = True

    def __init__(self, tokens=None, fail=False):
        self.tokens, self.fail, self.calls = tokens, fail, 0

    def decide(self, request, repeat=0):
        self.calls += 1
        if self.fail:
            raise RuntimeError("parse failed after the call")
        raw = {"usage": {"input_tokens": self.tokens}} if self.tokens else {}
        return Decision((0.5, 0.5), self.model_id, raw)


def test_spend_cap_counts_reported_tokens_and_refuses_past_the_cap(tmp_path):
    ledger = tmp_path / "spend.json"
    a = SpendCapped(Fake(tokens=20), cap_usd=1.0, price_per_m_input_tokens=0.042, ledger=ledger)
    est = a.estimate(REQ)
    a.cap_usd = 2.5 * est  # room for two calls, not three, on the conservative estimate
    a.decide(REQ)
    a.decide(REQ)
    assert a.spent() == pytest.approx(2 * 20 / 1e6 * 0.042)  # reported tokens, not the estimate
    a3 = SpendCapped(Fake(), cap_usd=2.5 * est, price_per_m_input_tokens=0.042, ledger=tmp_path / "s2.json")
    a3.decide(REQ)
    a3.decide(REQ)
    with pytest.raises(SpendCapReached):
        a3.decide(REQ)
    again = SpendCapped(Fake(), cap_usd=2.5 * est, price_per_m_input_tokens=0.042, ledger=tmp_path / "s2.json")
    assert again.spent() == pytest.approx(2 * est)  # kept across runs


def test_spend_cap_refuses_before_calling(tmp_path):
    inner = Fake(tokens=10)
    a = SpendCapped(inner, cap_usd=1e-12, price_per_m_input_tokens=0.042, ledger=tmp_path / "s.json")
    with pytest.raises(SpendCapReached):
        a.decide(REQ)
    assert inner.calls == 0


def test_spend_cap_keeps_the_estimate_when_the_call_fails(tmp_path):
    a = SpendCapped(Fake(fail=True), cap_usd=1.0, price_per_m_input_tokens=0.042, ledger=tmp_path / "s.json")
    with pytest.raises(RuntimeError):
        a.decide(REQ)
    assert a.spent() == pytest.approx(a.estimate(REQ)) and a.spent() > 0


def test_endpoint_accepts_resource_or_full_url():
    assert endpoint("https://x.services.ai.azure.com/") == "https://x.services.ai.azure.com/providers/microsoft/v1/systemone"
    full = "https://x.services.ai.azure.com/providers/microsoft/v1/systemone"
    assert endpoint(full) == full
    assert endpoint("https://x.services.ai.azure.com/models") == "https://x.services.ai.azure.com/providers/microsoft/v1/systemone"


def _env(monkeypatch):
    monkeypatch.setenv("MICROSOFT_DECISION_1_TARGET_URL", "https://x.services.ai.azure.com")
    monkeypatch.setenv("MICROSOFT_DECISION_1_KEY", "k")
    monkeypatch.setenv("MICROSOFT_DECISION_1_DEPLOYMENT", "dep")


def test_microsoft_waits_a_minute_on_429_and_sends_the_documented_body(monkeypatch, tmp_path):
    _env(monkeypatch)
    replies = [(429, {}, b"busy"), (429, {"retry-after": "90"}, b"busy"), (200, {}, b'{"answers": {}}')]
    sent, slept = [], []

    def transport(url, body, headers):
        sent.append((url, json.loads(body), dict(headers)))
        return replies.pop(0)

    a = MicrosoftDecisionAdapter(CRIT, "1", transport=transport, sleep=slept.append)
    assert a.fetch_sample(REQ, tmp_path / "sample.json") == {"answers": {}}
    assert slept == [60.0, 90.0] and a.rate_limit_waits == 2
    url, body, headers = sent[-1]
    assert url.endswith("/providers/microsoft/v1/systemone") and headers["api-key"] == "k"
    assert body["model"] == "dep" and list(body["questions"]["decision"]["criteria"]) == ["ADVANCE", "DO NOT ADVANCE"]
    assert json.loads((tmp_path / "sample.json").read_text()) == {"answers": {}}


def test_microsoft_parses_the_live_response_shape(monkeypatch):
    _env(monkeypatch)
    resp = {"model": "microsoft-decision-1",
            "answers": {"decision": {"choice": "ADVANCE", "confidence": 0.5, "type": "choice",
                                     "probabilities": {"DO NOT ADVANCE": 0.25, "ADVANCE": 0.75}}},
            "usage": {"input_tokens": 100, "output_tokens": 1}}
    a = MicrosoftDecisionAdapter(CRIT, "1", transport=lambda u, b, h: (200, {}, json.dumps(resp).encode()),
                                 sleep=lambda s: None)
    d = a.decide(REQ)
    assert d.probs == (0.75, 0.25) and d.raw["usage"]["input_tokens"] == 100
    bad = dict(resp, model="something-else")
    with pytest.raises(RuntimeError):
        a.parse(bad, REQ)
    wrong = {**resp, "answers": {"decision": {**resp["answers"]["decision"], "probabilities": {"ADVANCE": 1.0}}}}
    with pytest.raises(RuntimeError):
        a.parse(wrong, REQ)


def test_microsoft_needs_its_environment(monkeypatch):
    for k in ("MICROSOFT_DECISION_1_TARGET_URL", "MICROSOFT_DECISION_1_KEY", "MICROSOFT_DECISION_1_DEPLOYMENT"):
        monkeypatch.delenv(k, raising=False)
    a = MicrosoftDecisionAdapter(CRIT, "1", transport=lambda u, b, h: (200, {}, b"{}"))
    with pytest.raises(RuntimeError):
        a.call(REQ)
    with pytest.raises(ValueError):
        MicrosoftDecisionAdapter(CRIT, "")


def test_charge_estimate_records_and_refuses(tmp_path):
    a = SpendCapped(Fake(), cap_usd=1.0, price_per_m_input_tokens=0.042, ledger=tmp_path / "s.json")
    a.charge_estimate(REQ)
    assert a.spent() == pytest.approx(a.estimate(REQ))
    a.cap_usd = a.spent()
    with pytest.raises(SpendCapReached):
        a.charge_estimate(REQ)


def test_openai_rule_placements_and_parse(monkeypatch):
    from sway_eval.adapters.openai_decisions import OpenAIDecisionsAdapter, Refusal
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    req = DecisionRequest("Score points. Should this candidate advance?", "Skills: SQL", ("ADVANCE", "DO NOT ADVANCE"))
    b = OpenAIDecisionsAdapter(CRIT, "input").body(req)
    assert b["input"] == "Screening rule: Score points.\n\nCandidate record:\nSkills: SQL"
    assert b["questions"][0]["instructions"] == "Should this candidate advance under the screening rule?"
    assert [c["value"] for c in b["questions"][0]["choices"]] == ["ADVANCE", "DO NOT ADVANCE"]
    a = OpenAIDecisionsAdapter(CRIT, "instructions").body(req)
    assert a["input"] == "Skills: SQL" and a["questions"][0]["instructions"] == req.instruction
    assert a["questions"][0]["choices"] == b["questions"][0]["choices"]
    resp = {"model": "gpt-6-luna", "answers": [{"type": "choice", "name": "decision", "choice": "ADVANCE",
            "probabilities": [{"value": "DO NOT ADVANCE", "probability": 0.3}, {"value": "ADVANCE", "probability": 0.7}],
            "confidence": 0.4}], "usage": {"input_tokens": 10}}
    replies = [(429, {"retry-after": "1"}, b""), (200, {}, json.dumps(resp).encode())]
    slept = []
    ad = OpenAIDecisionsAdapter(CRIT, transport=lambda body, h: replies.pop(0), sleep=slept.append)
    d = ad.decide(req)
    assert d.probs == (0.7, 0.3) and slept == [1.0] and d.model_version.endswith("rule-in-input")
    refusal = {"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "decision"}]}
    with pytest.raises(Refusal):
        ad.parse(refusal, req)
