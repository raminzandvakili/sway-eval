"""The interface every model adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionRequest:
    """One decision: an instruction, the record text, and fixed options.

    ``context`` is the rendered record. Adapters decide how to lay out the
    three parts for their API; they must not change the text itself.
    """

    instruction: str
    context: str
    options: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"instruction": self.instruction, "context": self.context, "options": list(self.options)}


@dataclass(frozen=True)
class Decision:
    """The model's answer: one probability per option, in request order."""

    probs: tuple[float, ...]
    model_version: str
    raw: dict = field(default_factory=dict)


class Adapter(ABC):
    """A decision model behind one method.

    ``model_id`` identifies the model and the pinned version requested; it is
    part of every cache key. ``paid`` says whether calls cost money, so the
    runner can require confirmation.
    """

    model_id: str
    paid: bool = False

    @abstractmethod
    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        """Return the model's distribution over ``request.options``.

        ``repeat`` distinguishes deliberate repeated calls (for run-to-run
        noise); adapters that can't vary by it simply ignore it.
        """
