"""Adapter for Cloudflare's open-weight Clef decision models, run locally.

Uses the official loader shipped in the model snapshot
(``joint_schema_model.py``: ``load_release_model``, ``encode_record``,
``collate_records``), as the model cards describe
(https://huggingface.co/Cloudflare/clef-flash, accessed 2026-10-28). The
record has the System One shape:

    {"state": "<record text>",
     "questions": {"decision": {"type": "choice", "instructions": "...",
                                "criteria": {"<option>": "<description>", ...}}}}

The joint schema head returns one logit per option; probabilities are their
softmax. Raw logits are kept with every decision.

Two properties of the official encoder matter here. It sorts the options of
a ``choice`` question by name, so the order of options in a request does not
reach the model. And it cuts the state silently at ``max_length`` tokens, so
this adapter encodes without a limit first and refuses longer requests.

Needs the ``local`` extra (torch, transformers, huggingface_hub). Nothing is
imported or loaded until the first call.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping, Sequence

from ._memory import cloning_before_copy, evicting
from .base import Adapter, Decision, DecisionRequest

QUESTION_ID = "decision"
MAX_LENGTH = 16_384


class ClefAdapter(Adapter):
    """A Clef model pinned to one Hugging Face commit.

    ``criteria`` maps each option to the description sent with it. With
    ``deterministic`` set, torch is asked for deterministic kernels; whether
    every kernel complied is recorded in ``info()``. ``evict_page_cache``
    keeps model files out of the page cache while loading, and
    ``clone_before_copy`` keeps the host copy of the weights from doubling
    the load peak on a GPU that shares memory with the CPU (both in ``_memory``).
    """

    paid = False

    def __init__(
        self,
        repo: str,
        revision: str,
        criteria: Mapping[str, str],
        dtype: str = "bfloat16",
        device: str = "cuda",
        deterministic: bool = True,
        evict_page_cache: bool = False,
        clone_before_copy: bool = False,
        max_length: int = MAX_LENGTH,
    ) -> None:
        if len(revision) != 40:
            raise ValueError("pin a full 40-character commit hash, not a branch or tag")
        self.repo, self.revision = repo, revision
        self.model_id = f"{repo}@{revision}"
        self.criteria = dict(criteria)
        self.dtype, self.device = dtype, device
        self.deterministic = deterministic
        self.evict_page_cache = evict_page_cache
        self.clone_before_copy = clone_before_copy
        self.max_length = max_length
        self._model = self._processor = self._lib = None
        self._info: dict = {}

    def record(self, request: DecisionRequest) -> dict:
        """The System One record sent for a request."""
        missing = [o for o in request.options if o not in self.criteria]
        if missing:
            raise ValueError(f"no criteria description for options {missing}")
        return {
            "state": request.context,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": request.instruction,
                    "criteria": {o: self.criteria[o] for o in request.options},
                }
            },
        }

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
        path = snapshot_download(self.repo, revision=self.revision)
        sys.path.insert(0, path)
        import joint_schema_model as lib  # the snapshot's own code

        self._lib = lib
        with evicting(_hub_cache() if self.evict_page_cache else None, every_s=0.5 if self.clone_before_copy else 2.0), \
                cloning_before_copy(self.clone_before_copy):
            self._model, self._processor = lib.load_release_model(path, device=self.device, dtype=getattr(torch, self.dtype))
        self._info = {
            "repo": self.repo,
            "revision": self.revision,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if self.device.startswith("cuda") else self.device,
            "dtype": self.dtype,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "evict_page_cache": self.evict_page_cache,
            "clone_before_copy": self.clone_before_copy,
            "max_length": self.max_length,
        }

    def info(self) -> dict:
        """Software and hardware the model ran on (empty before the first call)."""
        return dict(self._info)

    def encode(self, request: DecisionRequest):
        """Encode with the official encoder, refusing anything it would cut."""
        self._load()
        enc = self._lib.encode_record(self._processor.tokenizer, self.record(request), max_length=10**9)
        if len(enc.input_ids) > self.max_length:
            raise ValueError(f"request is {len(enc.input_ids):,} tokens, over the {self.max_length:,} limit")
        return enc

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        return self.decide_batch([request])[0]

    def decide_batch(self, requests: Sequence[DecisionRequest]) -> list[Decision]:
        """Score several requests in one forward pass (padded batch)."""
        import torch

        self._load()
        encoded = [self.encode(r) for r in requests]
        t0 = time.time()
        batch = self._lib.collate_records(encoded, self._processor.tokenizer.pad_token_id, torch.device(self.device))
        with torch.inference_mode():
            out = self._model(batch)
        ms = round((time.time() - t0) * 1000 / len(requests), 2)
        decisions = []
        for req, enc, per_record in zip(requests, encoded, out):
            q = enc.questions[0]
            logits = per_record[0].float()
            by_option = dict(zip(q.option_ids, logits.tolist()))
            probs = dict(zip(q.option_ids, logits.softmax(-1).tolist()))
            decisions.append(Decision(
                probs=tuple(probs[o] for o in req.options),
                model_version=self.model_id,
                raw={"logits": [by_option[o] for o in req.options], "encoder_option_order": list(q.option_ids),
                     "input_tokens": len(enc.input_ids), "batch_size": len(requests), "ms_per_record": ms},
            ))
        return decisions


def _hub_cache() -> str:
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE
