import json

import pytest

from sway_eval.adapters import DecisionRequest, TypeSafeAdapter

CRIT = {"ADVANCE": "Meets the rule", "DO NOT ADVANCE": "Does not meet the rule"}
REQ = DecisionRequest("Apply the rule.", "Experience:\n- 5 years", ("DO NOT ADVANCE", "ADVANCE"))


def ok(model="jev-1.14.0", probs=None):
    probs = probs or {"ADVANCE": 0.73, "DO NOT ADVANCE": 0.27}
    return json.dumps({"model": model, "usage": {"input_tokens": 40, "output_tokens": 5},
                       "answers": {"decision": {"type": "choice", "choice": "ADVANCE",
                                                "confidence": 0.46, "probabilities": probs}}}).encode()


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, body, headers):
        self.sent.append((json.loads(body), dict(headers)))
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def test_body_keeps_option_order_and_pin():
    a = TypeSafeAdapter("jev-1.14.0", CRIT, transport=Fake([]))
    body = a.body(REQ)
    assert body["model"] == "jev-1.14.0" and body["state"] == REQ.context
    assert list(body["questions"]["decision"]["criteria"]) == ["DO NOT ADVANCE", "ADVANCE"]


def test_parse_maps_probabilities_to_request_order_and_keeps_raw():
    fake = Fake([(200, {}, ok())])
    d = TypeSafeAdapter("jev-1.14.0", CRIT, transport=fake).decide(REQ)
    assert d.probs == (0.27, 0.73) and d.model_version == "jev-1.14.0"
    assert d.raw["usage"]["input_tokens"] == 40
    assert fake.sent[0][1]["Authorization"] == "Bearer test-key"


def test_version_mismatch_is_an_error():
    with pytest.raises(RuntimeError):
        TypeSafeAdapter("jev-1.14.0", CRIT, transport=Fake([(200, {}, ok(model="jev-1.13.0"))])).decide(REQ)


def test_retries_honor_retry_after():
    waits = []
    fake = Fake([(429, {"Retry-After": "2"}, b"slow down"), (529, {}, b"busy"), (200, {}, ok())])
    a = TypeSafeAdapter("jev-1.14.0", CRIT, transport=fake, sleep=waits.append, backoff_s=0.5)
    assert a.decide(REQ).probs == (0.27, 0.73)
    assert waits == [2.0, 1.0]


def test_client_errors_are_not_retried():
    fake = Fake([(422, {}, b'{"detail": []}')])
    with pytest.raises(RuntimeError):
        TypeSafeAdapter("jev-1.14.0", CRIT, transport=fake).decide(REQ)
    assert len(fake.sent) == 1


def test_guards(monkeypatch):
    with pytest.raises(ValueError):
        TypeSafeAdapter("jev-latest", CRIT)
    with pytest.raises(ValueError):
        TypeSafeAdapter("jev-1.14.0", {"ADVANCE": "x"}).body(REQ)
    with pytest.raises(ValueError):
        TypeSafeAdapter("jev-1.14.0", CRIT).check_length(DecisionRequest("i", "x" * 200_000, REQ.options))
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(RuntimeError):
        TypeSafeAdapter("jev-1.14.0", CRIT, transport=Fake([])).decide(REQ)
