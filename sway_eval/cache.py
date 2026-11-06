"""A sqlite response cache that every model call goes through.

The cache makes runs resumable and keeps repeated analyses free. A response
is keyed by the adapter's ``model_id``, the full request, and a repeat index,
so deliberate repeated calls (for run-to-run noise) are never collapsed into
one cached answer. A nonzero serving ``epoch`` is part of the key too, so
after a confirmed drift a new epoch calls the model afresh instead of
replaying answers from before the change.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from .adapters.base import Adapter, Decision, DecisionRequest


def cache_key(model_id: str, request: DecisionRequest, repeat: int = 0, epoch: int = 0) -> str:
    d = {"model": model_id, "request": request.as_dict(), "repeat": repeat}
    if epoch:
        d["epoch"] = epoch  # epoch 0 keeps the keys of runs made before epochs existed
    payload = json.dumps(d, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Key-value store of decisions on disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, model_id TEXT, response TEXT)"
        )
        self._db.commit()

    def get(self, key: str) -> Decision | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        d = json.loads(row[0])
        return Decision(probs=tuple(d["probs"]), model_version=d["model_version"], raw=d.get("raw", {}))

    def put(self, key: str, model_id: str, decision: Decision) -> None:
        payload = json.dumps(
            {"probs": list(decision.probs), "model_version": decision.model_version, "raw": decision.raw}
        )
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, model_id, response) VALUES (?, ?, ?)",
                (key, model_id, payload),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._db.close()


class CachedAdapter:
    """Wraps an adapter so every call is looked up in, and saved to, the cache."""

    def __init__(self, adapter: Adapter, cache: ResponseCache, epoch: int = 0) -> None:
        self.adapter = adapter
        self.cache = cache
        self.epoch = epoch
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self.adapter.model_id

    def decide(self, request: DecisionRequest, repeat: int = 0) -> tuple[Decision, bool]:
        """Return (decision, cached?)."""
        key = cache_key(self.adapter.model_id, request, repeat, self.epoch)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit, True
        decision = self.adapter.decide(request, repeat=repeat)
        self.cache.put(key, self.adapter.model_id, decision)
        self.misses += 1
        return decision, False
