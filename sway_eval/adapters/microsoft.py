"""Adapter for Microsoft-Decision-1 on Microsoft Foundry (System One shape).

Request shape from Microsoft's documentation
(https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision,
updated 2026-10-09, accessed 2026-10-26):

    POST <resource endpoint>/providers/microsoft/v1/systemone
    api-key: <key>
    {"model": "<deployment name>", "state": "<record text>",
     "questions": {"decision": {"type": "choice", "instructions": "...",
                                "criteria": {"<option>": "<description>", ...}}}}

The documentation gives no complete response example; ``parse`` is written
against a live response saved with ``fetch_sample`` on 2026-11-03. Rate limiting (HTTP 429) waits 60 seconds, or
longer if the response's ``retry-after`` asks for it, and retries.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from .base import Adapter, Decision, DecisionRequest

PATH = "/providers/microsoft/v1/systemone"
QUESTION_ID = "decision"
EXPECTED_MODEL = "microsoft-decision-1"
CONTEXT_TOKENS = 32_768
RATE_LIMIT_WAIT_S = 60.0
SERVER_RETRY = {500, 502, 503, 504}

# (url, body bytes, headers) -> (status, response headers, response body bytes)
Transport = Callable[[str, bytes, Mapping[str, str]], tuple[int, Mapping[str, str], bytes]]


def _urllib_transport(timeout: float) -> Transport:
    def send(url: str, body: bytes, headers: Mapping[str, str]) -> tuple[int, Mapping[str, str], bytes]:
        req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), e.read()

    return send


def endpoint(target: str) -> str:
    """The System One URL from the resource endpoint or a full target URL. The
    deployment page's Target URI ends in ``/models`` (the model-inference path);
    System One sits under the bare resource endpoint, so that suffix is dropped."""
    target = target.rstrip("/")
    if target.endswith("/systemone"):
        return target
    if target.endswith("/models"):
        target = target[: -len("/models")]
    return target + PATH


class MicrosoftDecisionAdapter(Adapter):
    """Microsoft-Decision-1 through a pinned Foundry deployment, one ``choice`` question per call.

    The target URL, key and deployment name come from the environment
    variables named here (code/.env); the key is never logged.
    """

    paid = True

    def __init__(
        self,
        criteria: Mapping[str, str],
        model_version: str,
        url_env: str = "MICROSOFT_DECISION_1_TARGET_URL",
        key_env: str = "MICROSOFT_DECISION_1_KEY",
        deployment_env: str = "MICROSOFT_DECISION_1_DEPLOYMENT",
        transport: Transport | None = None,
        max_rate_limit_waits: int = 60,
        max_server_retries: int = 5,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model_version:
            raise ValueError("record the deployment's pinned model version (e.g. '1')")
        self.criteria = dict(criteria)
        self.model_version = str(model_version)
        self.url_env, self.key_env, self.deployment_env = url_env, key_env, deployment_env
        self.model_id = f"microsoft/microsoft-decision-1@{self.model_version}"
        self.transport = transport or _urllib_transport(timeout_s)
        self.max_rate_limit_waits = max_rate_limit_waits
        self.max_server_retries = max_server_retries
        self.sleep = sleep
        self.rate_limit_waits = 0

    def _env(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"set {name} in code/.env to call Microsoft-Decision-1")
        return value

    def body(self, request: DecisionRequest) -> dict:
        """The exact JSON body sent for a request (options in request order)."""
        missing = [o for o in request.options if o not in self.criteria]
        if missing:
            raise ValueError(f"no criteria description for options {missing}")
        return {
            "model": os.environ.get(self.deployment_env, "<deployment>"),
            "state": request.context,
            "questions": {QUESTION_ID: {"type": "choice", "instructions": request.instruction,
                                        "criteria": {o: self.criteria[o] for o in request.options}}},
        }

    def check_length(self, request: DecisionRequest) -> None:
        """Refuse requests that may exceed the context, so nothing is truncated (3 characters per token)."""
        if len(json.dumps(self.body(request))) / 3 > CONTEXT_TOKENS:
            raise ValueError(f"request may exceed the {CONTEXT_TOKENS:,}-token context")

    def call(self, request: DecisionRequest) -> dict:
        """POST one request and return the parsed JSON response."""
        self.check_length(request)
        url = endpoint(self._env(self.url_env))
        headers = {"api-key": self._env(self.key_env), "Content-Type": "application/json"}
        body = self.body(request)
        body["model"] = self._env(self.deployment_env)
        data = json.dumps(body).encode("utf-8")
        server_tries = rate_waits = 0
        while True:
            status, resp_headers, payload = self.transport(url, data, headers)
            if status == 200:
                return json.loads(payload)
            if status == 429 and rate_waits < self.max_rate_limit_waits:
                rate_waits += 1
                self.rate_limit_waits += 1
                self.sleep(self._wait(resp_headers))
                continue
            if status in SERVER_RETRY and server_tries < self.max_server_retries:
                self.sleep(2.0 ** server_tries)
                server_tries += 1
                continue
            raise RuntimeError(f"Microsoft-Decision-1 returned {status}: {payload[:500].decode('utf-8', 'replace')}")

    def _wait(self, headers: Mapping[str, str]) -> float:
        lower = {k.lower(): v for k, v in headers.items()}
        try:
            return max(RATE_LIMIT_WAIT_S, float(lower.get("retry-after", 0)))
        except ValueError:
            return RATE_LIMIT_WAIT_S

    def fetch_sample(self, request: DecisionRequest, path: str | Path) -> dict:
        """Make one live call and save the raw response, to write ``parse`` against."""
        resp = self.call(request)
        Path(path).write_text(json.dumps(resp, indent=2), encoding="utf-8")
        return resp

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        return self.parse(self.call(request), request)

    def parse(self, resp: dict, request: DecisionRequest) -> Decision:
        """Probabilities in request option order, with the raw response kept.

        Written against a live response (2026-11-03): ``model`` is the underlying
        model name without a version; ``answers.decision`` has ``type``,
        ``choice``, ``confidence`` (top minus second probability on the samples
        seen; stored, not used) and ``probabilities`` keyed by option; ``usage``
        has ``input_tokens``. Anything else fails loudly.
        """
        if resp.get("model") != EXPECTED_MODEL:
            raise RuntimeError(f"expected model {EXPECTED_MODEL!r}, the response says {resp.get('model')!r}")
        answer = (resp.get("answers") or {}).get(QUESTION_ID)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise RuntimeError(f"no choice answer for {QUESTION_ID!r}: {str(answer)[:200]}")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(request.options):
            raise RuntimeError(f"probabilities {probs!r} do not match options {list(request.options)}")
        return Decision(probs=tuple(float(probs[o]) for o in request.options),
                        model_version=self.model_id, raw=resp)

    def info(self) -> dict:
        return {"model_version": self.model_version, "rate_limit_waits": self.rate_limit_waits}
