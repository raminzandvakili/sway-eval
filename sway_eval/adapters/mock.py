"""A deterministic mock decision model with known position weights.

The mock asks a scorer for the content score of each unit in the order the
units appear in the rendered record, then scores the target option with an
additive logit:

    logit = bias + scale * sum over positions k of score[k] * position_weight[k]
                 + carryover * sum over adjacent positions k of score[k-1] * score[k]
                 + format_effect * [the record has no line bulleted with "- "]

Because the content scores, position weights, carryover and formatting effect
are known, tests and evaluations can check that the analysis recovers what was
put in. ``drift_after`` and ``drift_weights`` change the position weights after
a given number of calls, a planted serving change for the canaries to catch.
It never touches the network.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence

from .base import Adapter, Decision, DecisionRequest

Scorer = Callable[[str], list[float]]


def text_scorer(unit_text: Mapping[str, float]) -> Scorer:
    """Scorer that finds each known text in the context and returns its score,
    in order of appearance. Each text should identify one unit."""

    def score(context: str) -> list[float]:
        found = sorted((context.find(t), s) for t, s in unit_text.items() if t in context)
        return [s for _, s in found]

    return score


class MockAdapter(Adapter):
    """Binary mock: option ``target`` gets probability sigmoid(logit).

    ``noise`` adds seeded Gaussian noise to the logit per (request, repeat),
    so repeated calls differ the way a nondeterministic API would. ``grid``
    rounds probabilities to a grid with a floor, as some hosted APIs do
    (e.g. 0.01).

    ``target`` is an option name or an index into the request's options, so
    the mock follows its option when option order is rotated.
    ``first_option_bias`` is added to the target's logit when it is listed
    first: a known option-position bias for testing the rotation baseline.
    ``carryover`` adds a first-order term on adjacent units' scores, and
    ``format_effect`` shifts the logit when the record is not bulleted with
    "- " (the default format is), a known formatting sensitivity. After
    ``drift_after`` calls, ``drift_weights`` replace the position weights.

    The response cache keys on ``model_id``, so give the mock a new id
    whenever its behaviour changes, or old answers will be replayed.
    """

    paid = False

    def __init__(
        self,
        scorer: Scorer,
        position_weights: Sequence[float],
        bias: float = 0.0,
        scale: float = 1.0,
        target: int | str = 0,
        first_option_bias: float = 0.0,
        noise: float = 0.0,
        grid: float | None = None,
        seed: int = 0,
        model_id: str = "mock-1",
        carryover: float = 0.0,
        format_effect: float = 0.0,
        drift_after: int | None = None,
        drift_weights: Sequence[float] | None = None,
    ) -> None:
        self.scorer = scorer
        self.position_weights = [float(w) for w in position_weights]
        self.bias = float(bias)
        self.scale = float(scale)
        self.target = target
        self.first_option_bias = float(first_option_bias)
        self.noise = float(noise)
        self.grid = grid
        self.seed = seed
        self.model_id = model_id
        self.carryover = float(carryover)
        self.format_effect = float(format_effect)
        self.drift_after = drift_after
        self.drift_weights = [float(w) for w in drift_weights] if drift_weights is not None else None
        if (drift_after is None) != (drift_weights is None):
            raise ValueError("drift_after and drift_weights go together")
        self.calls = 0

    def weights(self) -> list[float]:
        """Position weights in force now (the drift weights after ``drift_after`` calls)."""
        if self.drift_after is not None and self.calls > self.drift_after:
            return self.drift_weights
        return self.position_weights

    def logit(self, context: str) -> float:
        """The noise-free logit for a rendered record."""
        scores = self.scorer(context)
        w = self.weights()
        if len(scores) > len(w):
            raise ValueError("more units in context than position weights")
        z = self.bias + self.scale * sum(s * x for s, x in zip(scores, w))
        z += self.carryover * sum(a * b for a, b in zip(scores, scores[1:]))
        if self.format_effect and not any(line.startswith("- ") for line in context.splitlines()):
            z += self.format_effect
        return z

    def decide(self, request: DecisionRequest, repeat: int = 0) -> Decision:
        if len(request.options) != 2:
            raise ValueError("the mock supports binary decisions only")
        self.calls += 1
        t = request.options.index(self.target) if isinstance(self.target, str) else self.target
        z = self.logit(request.context) + (self.first_option_bias if t == 0 else 0.0)
        if self.noise:
            rng = random.Random(f"{self.seed}|{request.instruction}|{request.context}|{repeat}")
            z += rng.gauss(0.0, self.noise)
        p = 1.0 / (1.0 + math.exp(-z))
        if self.grid:
            p = min(1.0 - self.grid, max(self.grid, round(p / self.grid) * self.grid))
        probs = [0.0, 0.0]
        probs[t] = p
        probs[1 - t] = 1.0 - p
        return Decision(probs=tuple(probs), model_version=self.model_id, raw={"logit": z})
