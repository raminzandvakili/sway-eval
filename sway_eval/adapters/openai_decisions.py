"""Adapter for the OpenAI Decisions API (gpt-6-luna).

Request and response shapes from OpenAI's Decisions guide
(https://developers.openai.com/api/docs/guides/decisions, read 2026-11-03) and a
live response saved on 2026-11-03:

    POST https://api.openai.com/v1/decisions
    {"model": "gpt-6-luna", "input": "<shared evidence>",
     "questions": [{"type": "choice", "name": "decision", "instructions": "...",
                    "choices": [{"value": "<option>", "description": "..."}, ...]}]}

    -> {"model": "gpt-6-luna",
        "answers": [{"type": "choice", "name": "decision", "choice": "...",
                     "probabilities": [{"value": ..., "probability": ...}, ...],
                     "confidence": ...}],
        "usage": {"input_tokens": ..., ...}}

``placement`` sets where the request's instruction (the rule) goes:
``"input"`` puts it above the record in ``input`` and asks ``question`` in the
question's instructions (the guide: evidence all questions share goes in
``input``); ``"instructions"`` puts it in the question's instructions with the
record alone in ``input`` (the System One mapping). Choices keep the request's
option order. A refusal answer raises, so a run stops rather than recording a
decision the model did not make. Rate limits (429) wait for ``retry-after`` or
back off exponentially.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from .base import Adapter, Decision, DecisionRequest

ENDPOINT = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
QUESTION_ID = "decision"
RETRY = {429, 500, 502, 503, 504}

Transport = Callable[[bytes, Mapping[str, str]], tuple[int, Mapping[str, str], bytes]]


def _urllib_transport(timeout: float) -> Transport:
    def send(body: bytes, headers: Mapping[str, str]) -> tuple[int, Mapping[str, str], bytes]:
        req = urllib.request.Request(ENDPOINT, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), e.read()

    return send


class Refusal(RuntimeError):
    """The model returned a refusal instead of a choice."""


class OpenAIDecisionsAdapter(Adapter):
    """gpt-6-luna through /v1/decisions, one ``choice`` question per call."""

    paid = True

    def __init__(
        self,
        criteria: Mapping[str, str],
        placement: str = "input",
        question: str = "Should this candidate advance under the screening rule?",
        rule_label: str = "Screening rule: ",
        record_label: str = "Candidate record:\n",
        drop_from_rule: str = " Should this candidate advance?",
        api_key_env: str = "OPENAI_API_KEY",
        transport: Transport | None = None,
        max_retries: int = 8,
        backoff_s: float = 2.0,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if placement not in ("input", "instructions"):
            raise ValueError("placement is 'input' or 'instructions'")
        self.criteria = dict(criteria)
        self.placement = placement
        self.question, self.rule_label, self.record_label = question, rule_label, record_label
        self.drop_from_rule = drop_from_rule
        self.api_key_env = api_key_env
        self.model_id = f"openai/{MODEL}@rule-in-{placement}"
        self.transport = transport or _urllib_transport(timeout_s)
        self.max_retries, self.backoff_s, self.sleep = max_retries, backoff_s, sleep

    def body(self, request: DecisionRequest) -> dict:
        """The exact JSON body sent for a request (choices in request order)."""
        missing = [o for o in request.options if o not in self.criteria]
        if missing:
            raise ValueError(f"no criteria description for options {missing}")
        if self.placement == "input":
            rule = request.instruction.replace(self.drop_from_rule, "")
            text, instructions = f"{self.rule_label}{rule}\n\n{self.record_label}{request.context}", self.question
        else:
            text, instructions = request.context, request.instruction
        return {"model": MODEL, "input": text,
                "questions": [{"type": "choice", "name": QUESTION_ID, "instructions": instructions,
                               "choices": [{"value": o, "description": self.criteria[o]} for o in request.options]}]}

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"set {self.api_key_env} in code/.env to call the OpenAI Decisions API")
        data = json.dumps(self.body(request)).encode("utf-8")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            status, resp_headers, payload = self.transport(data, headers)
            if status == 200:
                return self.parse(json.loads(payload), request)
            if status in RETRY and attempt < self.max_retries:
                lower = {k.lower(): v for k, v in resp_headers.items()}
                try:
                    wait = float(lower["retry-after"])
                except (KeyError, ValueError):
                    wait = self.backoff_s * 2**attempt
                self.sleep(max(0.0, wait))
                continue
            raise RuntimeError(f"OpenAI Decisions returned {status}: {payload[:500].decode('utf-8', 'replace')}")
        raise AssertionError("unreachable")

    def parse(self, resp: dict, request: DecisionRequest) -> Decision:
        """Probabilities in request option order, with the raw response kept."""
        if resp.get("model") != MODEL:
            raise RuntimeError(f"expected model {MODEL!r}, the response says {resp.get('model')!r}")
        answers = [a for a in resp.get("answers", []) if a.get("name") == QUESTION_ID]
        if len(answers) != 1:
            raise RuntimeError(f"expected one answer named {QUESTION_ID!r}: {str(resp.get('answers'))[:200]}")
        ans = answers[0]
        if ans.get("type") == "refusal":
            raise Refusal(f"refusal: {str(ans)[:300]}")
        if ans.get("type") != "choice":
            raise RuntimeError(f"unexpected answer type {ans.get('type')!r}")
        probs = {p["value"]: float(p["probability"]) for p in ans.get("probabilities", [])}
        if set(probs) != set(request.options):
            raise RuntimeError(f"probabilities {probs} do not match options {list(request.options)}")
        return Decision(probs=tuple(probs[o] for o in request.options), model_version=self.model_id, raw=resp)

    def info(self) -> dict:
        return {"placement": self.placement, "question": self.question if self.placement == "input" else None}
