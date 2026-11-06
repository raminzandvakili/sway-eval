"""Adapter for TypeSafe's Jev decision model (System One API).

Request and response shapes follow TypeSafe's API documentation
(https://docs.typesafe.ai/api, accessed 2026-10-26):

    POST https://api.typesafe.ai/v1/systemone
    {"model": "jev-1.14.0", "state": "<record text>",
     "questions": {"decision": {"type": "choice", "instructions": "...",
                                "criteria": {"<option>": "<description>", ...}}}}

    -> {"model": "jev-1.14.0",
        "answers": {"decision": {"type": "choice", "choice": "...",
                                 "confidence": ..., "probabilities": {...}}},
        "usage": {"input_tokens": ..., "output_tokens": ...}}

Option order is JSON key order, so the request keeps the options in the order
given. The response's ``model`` must equal the pinned version, or the call
fails. The full raw response is kept with every decision.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from .base import Adapter, Decision, DecisionRequest

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
QUESTION_ID = "decision"
PER_QUESTION_TOKEN_LIMIT = 32_000
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}

# (body bytes, headers) -> (status, response headers, response body bytes)
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


class TypeSafeAdapter(Adapter):
    """Jev through the System One endpoint, one ``choice`` question per call.

    ``criteria`` maps each option to the description sent with it; every
    option in a request must have one. The API key is read from the
    environment variable ``api_key_env`` and never logged.
    """

    paid = True

    def __init__(
        self,
        model: str,
        criteria: Mapping[str, str],
        api_key_env: str = "TYPESAFE_API_KEY",
        transport: Transport | None = None,
        max_retries: int = 5,
        backoff_s: float = 1.0,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model or model.endswith("latest") or model.endswith("preview"):
            raise ValueError("pin an exact Jev model ID such as jev-1.14.0, not an alias")
        self.model = model
        self.model_id = f"typesafe/{model}"
        self.criteria = dict(criteria)
        self.api_key_env = api_key_env
        self.transport = transport or _urllib_transport(timeout_s)
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.sleep = sleep

    def body(self, request: DecisionRequest) -> dict:
        """The exact JSON body sent for a request (options in request order)."""
        missing = [o for o in request.options if o not in self.criteria]
        if missing:
            raise ValueError(f"no criteria description for options {missing}")
        return {
            "model": self.model,
            "state": request.context,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": request.instruction,
                    "criteria": {o: self.criteria[o] for o in request.options},
                }
            },
        }

    def check_length(self, request: DecisionRequest) -> None:
        """Refuse requests that may exceed the per-question limit, so nothing
        is truncated. Uses a conservative estimate of 3 characters per token."""
        chars = len(request.context) + len(request.instruction) + sum(
            len(o) + len(self.criteria.get(o, "")) for o in request.options)
        if chars / 3 > PER_QUESTION_TOKEN_LIMIT:
            raise ValueError(f"request of {chars:,} characters may exceed {PER_QUESTION_TOKEN_LIMIT:,} tokens")

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        self.check_length(request)
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"set {self.api_key_env} (in code/.env) to call the TypeSafe API")
        body = json.dumps(self.body(request)).encode("utf-8")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            status, resp_headers, data = self.transport(body, headers)
            if status == 200:
                return self.parse(json.loads(data), request)
            if status in RETRY_STATUSES and attempt < self.max_retries:
                self.sleep(self._wait(resp_headers, attempt))
                continue
            raise RuntimeError(f"TypeSafe API returned {status}: {data[:500].decode('utf-8', 'replace')}")
        raise AssertionError("unreachable")

    def _wait(self, headers: Mapping[str, str], attempt: int) -> float:
        lower = {k.lower(): v for k, v in headers.items()}
        try:
            return max(0.0, float(lower["retry-after"]))
        except (KeyError, ValueError):
            return self.backoff_s * 2**attempt

    def parse(self, resp: dict, request: DecisionRequest) -> Decision:
        """Probabilities in request option order, with the raw response kept."""
        version = resp.get("model")
        if version != self.model:
            raise RuntimeError(f"requested {self.model} but the response says {version}")
        answer = resp["answers"][QUESTION_ID]
        probs = answer["probabilities"]
        if set(probs) != set(request.options):
            raise RuntimeError(f"response options {sorted(probs)} differ from {sorted(request.options)}")
        return Decision(
            probs=tuple(float(probs[o]) for o in request.options),
            model_version=version,
            raw=resp,
        )
