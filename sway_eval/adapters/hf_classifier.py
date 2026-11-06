"""Any Hugging Face sequence classifier with fixed labels, as a decision model.

The instruction and the record go to the classifier as a text pair (or joined
by a blank line), and each of the request's options maps to one of the
classifier's labels. The adapter returns a softmax over the mapped labels'
logits, renormalized over the options, in the request's order. Records longer
than the model's limit are refused rather than truncated, so the model never
silently loses the end of a record.

Loading needs the ``local`` extra (``transformers`` and ``torch``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

from .base import Adapter, Decision, DecisionRequest

LogitsFn = Callable[[str, str], tuple[Sequence[float], int]]  # (instruction, record) -> (logits, n_tokens)


class HFClassifierAdapter(Adapter):
    """A fixed-label classifier behind the decision interface.

    ``labels`` maps each option to a label name or index of the classifier.
    ``logits_fn`` returns all label logits and the token count for one input;
    ``from_pretrained`` builds it from a Hub model pinned to a commit.
    """

    paid = False

    def __init__(self, logits_fn: LogitsFn, labels: Mapping[str, int], max_length: int, model_id: str) -> None:
        self.logits_fn = logits_fn
        self.labels = dict(labels)
        self.max_length = max_length
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, repo: str, revision: str, labels: Mapping[str, str | int], device: str = "cpu",
                        input_mode: str = "pair", max_length: int | None = None) -> HFClassifierAdapter:
        if len(revision) != 40:
            raise ValueError("pin a full 40-character commit hash, not a branch or tag")
        if input_mode not in ("pair", "concat"):
            raise ValueError('input_mode is "pair" or "concat"')
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision).to(device).eval()
        names = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        index = {}
        for option, label in labels.items():
            if isinstance(label, int):
                index[option] = label
            elif label.lower() in names:
                index[option] = names[label.lower()]
            else:
                raise ValueError(f"label {label!r} for option {option!r} is not one of {sorted(names)}")
        limit = max_length or tok.model_max_length

        @torch.no_grad()
        def logits_fn(instruction: str, record: str):
            args = (instruction, record) if input_mode == "pair" else (instruction + "\n\n" + record,)
            enc = tok(*args, return_tensors="pt", truncation=False)
            n = int(enc["input_ids"].shape[1])
            if n > limit:
                return [], n
            return model(**{k: v.to(device) for k, v in enc.items()}).logits[0].float().tolist(), n

        return cls(logits_fn, index, limit, f"{repo}@{revision}")

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        missing = [o for o in request.options if o not in self.labels]
        if missing:
            raise ValueError(f"no classifier label for options {missing}")
        logits, n = self.logits_fn(request.instruction, request.context)
        if n > self.max_length:
            raise ValueError(f"request is {n:,} tokens, over the {self.max_length:,} limit")
        z = [logits[self.labels[o]] for o in request.options]
        top = max(z)
        e = [math.exp(x - top) for x in z]
        return Decision(probs=tuple(x / sum(e) for x in e), model_version=self.model_id,
                        raw={"logits": list(logits), "n_tokens": n})
