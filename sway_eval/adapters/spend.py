"""A hard spending cap for a paid adapter, kept across runs in a ledger file.

Before each call the cap is checked against the spend so far plus a
conservative estimate of this call (3 characters per token, at the price per
million input tokens), and the estimate is reserved in the ledger before the
call is sent; a call that could cross the cap is refused with
``SpendCapReached``, which stops the run (finished calls stay in the response
cache, so a later run resumes). After the call the ledger adds the cost from
the tokens the API reports, or the estimate if it reports none. Only real API
calls reach this wrapper: it sits inside the response cache.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from .base import Adapter, Decision, DecisionRequest


class SpendCapReached(RuntimeError):
    """The next call could take the ledger past its cap."""


class SpendCapped(Adapter):
    """Wraps ``adapter`` with a cap of ``cap_usd`` recorded in ``ledger``."""

    def __init__(self, adapter: Adapter, cap_usd: float, price_per_m_input_tokens: float, ledger: str | Path) -> None:
        if cap_usd <= 0 or price_per_m_input_tokens <= 0:
            raise ValueError("a spending cap needs a positive cap and price")
        self.adapter = adapter
        self.cap_usd = float(cap_usd)
        self.price = float(price_per_m_input_tokens)
        self.ledger = Path(ledger)
        self.model_id = adapter.model_id
        self.paid = True
        self._lock = threading.Lock()

    def spent(self) -> float:
        """Dollars recorded in the ledger so far."""
        if not self.ledger.exists():
            return 0.0
        return float(json.loads(self.ledger.read_text(encoding="utf-8"))["spent_usd"])

    def estimate(self, request: DecisionRequest) -> float:
        """A conservative cost for one call: 3 characters per token over the whole body."""
        body = getattr(self.adapter, "body", None)
        text = json.dumps(body(request)) if callable(body) else request.instruction + request.context + "".join(request.options)
        return len(text) / 3 / 1e6 * self.price

    def charge_estimate(self, request: DecisionRequest) -> None:
        """Record a call made outside ``decide`` (such as a saved sample) at its
        estimated cost, refusing it if it could cross the cap."""
        with self._lock:
            spent, est = self.spent(), self.estimate(request)
            if spent + est > self.cap_usd:
                raise SpendCapReached(f"{self.model_id}: ${spent:.4f} spent of a ${self.cap_usd:.2f} cap")
            self._write(spent + est)

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        with self._lock:
            spent, est = self.spent(), self.estimate(request)
            if spent + est > self.cap_usd:
                raise SpendCapReached(
                    f"{self.model_id}: ${spent:.4f} spent of a ${self.cap_usd:.2f} cap; the next call could cross it")
            self._write(spent + est)  # reserved first: a call that fails after being billed stays counted
            decision = self.adapter.decide(request, repeat=repeat)
            tokens = (decision.raw.get("usage") or {}).get("input_tokens")
            cost = tokens / 1e6 * self.price if isinstance(tokens, (int, float)) else est
            self._write(spent + cost)
        return decision

    def _write(self, spent: float) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        data = {"model_id": self.model_id, "cap_usd": self.cap_usd, "price_per_m_input_tokens": self.price,
                "spent_usd": spent, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        fd, tmp = tempfile.mkstemp(dir=self.ledger.parent, prefix=self.ledger.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.ledger)

    def info(self) -> dict:
        inner = getattr(self.adapter, "info", lambda: {})()
        return {**inner, "spend_cap_usd": self.cap_usd, "spend_ledger": str(self.ledger)}
