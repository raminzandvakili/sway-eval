"""Adapter for the open-weight Decision-2.0 models (vLLM Semantic Router), run locally.

Uses the runtime shipped in the model snapshot, ``decision2.Decision2``
(the package's ``runtime.entry``; https://huggingface.co/vllm-sr, accessed
2026-10-28). ``from_pretrained`` checks every file against the package's
``MODEL_MANIFEST.json`` before loading. Requests have the System One shape:

    model.system_one(state="<record text>",
                     questions={"decision": {"type": "choice", "instructions": "...",
                                             "criteria": {"<option>": "<description>", ...}}})

    -> {"model": ..., "answers": {"decision": {"type": "choice", "choice": ...,
        "confidence": ..., "probabilities": {...}}}, "usage": {"input_tokens": ...}}

The runtime keeps options in the order given, refuses over-long input
instead of cutting it, and returns unrounded probabilities. It does not
return logits; the log probabilities stored as ``logits`` equal the head's
logits divided by the package's calibration temperature (1 when the package
has no calibration file), up to a constant.
``share_context`` stays off (the exact path), and the ROCm-only graph and
kernel paths are switched off. ``low_peak`` loads through ``_d2_lowpeak``
instead, for models whose official load peak does not fit in memory.

Needs the ``local-d2`` extra. Nothing is imported or loaded until the first call.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping

from ._memory import evicting
from .base import Adapter, Decision, DecisionRequest

QUESTION_ID = "decision"


def _materialized(snapshot: str, revision: str) -> str:
    """A copy of the snapshot with real files instead of links to the Hub
    cache's blobs, made once next to it: the runtime's manifest check
    refuses packages that contain links."""
    import shutil
    from pathlib import Path

    src = Path(snapshot)
    dst = src.parents[1] / "materialized" / revision
    if not dst.exists():
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp, symlinks=False, ignore=shutil.ignore_patterns(".gitattributes"))
        tmp.rename(dst)
    return str(dst)


class Decision2Adapter(Adapter):
    """A Decision-2.0 package pinned to one Hugging Face commit."""

    paid = False

    def __init__(
        self,
        repo: str,
        revision: str,
        criteria: Mapping[str, str],
        device: str = "cuda:0",
        deterministic: bool = True,
        evict_page_cache: bool = False,
        low_peak: bool = False,
    ) -> None:
        if len(revision) != 40:
            raise ValueError("pin a full 40-character commit hash, not a branch or tag")
        self.repo, self.revision = repo, revision
        self.model_id = f"{repo}@{revision}"
        self.criteria = dict(criteria)
        self.device = device
        self.deterministic = deterministic
        self.evict_page_cache = evict_page_cache
        self.low_peak = low_peak
        self._model = None
        self._info: dict = {}

    def questions(self, request: DecisionRequest) -> dict:
        """The ``questions`` argument sent for a request (options in request order)."""
        missing = [o for o in request.options if o not in self.criteria]
        if missing:
            raise ValueError(f"no criteria description for options {missing}")
        return {QUESTION_ID: {"type": "choice", "instructions": request.instruction,
                              "criteria": {o: self.criteria[o] for o in request.options}}}

    def _load(self) -> None:
        if self._model is not None:
            return
        import os

        import torch
        import transformers
        from huggingface_hub import snapshot_download

        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        path = _materialized(snapshot_download(self.repo, revision=self.revision), self.revision)
        sys.path.insert(0, path)
        from decision2 import Decision2  # the snapshot's own runtime

        with evicting(_hub_cache() if self.evict_page_cache else None, every_s=0.5 if self.low_peak else 2.0):
            if self.low_peak:
                from ._d2_lowpeak import load_low_peak

                self._model = load_low_peak(path, self.device)
            else:
                self._model = Decision2.from_pretrained(path, device=self.device, graphs=False, kernels=False,
                                                        share_context=False)
        self._info = {
            "repo": self.repo,
            "revision": self.revision,
            "package_model_name": self._model.model_name,
            "calibration": self._model.manifest.get("calibration"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if self.device.startswith("cuda") else self.device,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "evict_page_cache": self.evict_page_cache,
            "low_peak": self.low_peak,
            "max_input_tokens": self._model.max_input_tokens,
        }

    def info(self) -> dict:
        """Software and hardware the model ran on (empty before the first call)."""
        return dict(self._info)

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        self._load()
        t0 = time.time()
        res = self._model.system_one(state=request.context, questions=self.questions(request))
        ms = round((time.time() - t0) * 1000, 2)
        ans = res["answers"][QUESTION_ID]
        if "error" in ans:
            raise RuntimeError(f"Decision-2.0 refused the request: {ans['error']}")
        probs = ans["probabilities"]
        if set(probs) != set(request.options):
            raise RuntimeError(f"answer options {sorted(probs)} differ from {sorted(request.options)}")
        return Decision(
            probs=tuple(float(probs[o]) for o in request.options),
            model_version=self.model_id,
            raw={"logits": [math.log(max(float(probs[o]), 1e-300)) for o in request.options],
                 "logits_are": "log probabilities (logits / calibration temperature, up to a constant)", "answer": ans,
                 "input_tokens": res["usage"]["input_tokens"], "ms": ms},
        )


def _hub_cache() -> str:
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE
