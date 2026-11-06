"""Logit-margin utilities for decision models.

Adding the same constant to every logit leaves the softmax unchanged, so only
differences between logits carry meaning. Every function here works with
differences. That means the inputs can be raw logits or log-probabilities,
and the result is the same.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _check(values: Sequence[float], name: str = "logits") -> list[float]:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError(f"{name} must have at least two options, got {len(vals)}")
    if any(math.isnan(v) for v in vals):
        raise ValueError(f"{name} contain NaN")
    return vals


def _logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def log_softmax(logits: Sequence[float]) -> list[float]:
    """Log-probabilities from logits."""
    z = _check(logits)
    lse = _logsumexp(z)
    return [v - lse for v in z]


def top2_margin(logits: Sequence[float]) -> float:
    """Gap between the winning option and the runner-up: z(1) - z(2).

    Always >= 0. For two options this equals |z_1 - z_2|, the absolute log-odds.
    """
    z = sorted(_check(logits), reverse=True)
    return z[0] - z[1]


def margin_from_probs(probs: Sequence[float]) -> float:
    """Top-2 margin computed from probabilities: log p(1) - log p(2).

    This equals z(1) - z(2), so hosted APIs that return only probabilities
    are enough. The probabilities don't need to sum to 1, because rescaling
    them doesn't change the margin.

    Returns ``math.inf`` when the runner-up probability is 0. Some APIs round
    small probabilities down to 0, so check for this before averaging margins.
    """
    p = sorted(_check(probs, "probs"), reverse=True)
    if p[-1] < 0:
        raise ValueError("probs must be non-negative")
    if p[0] == 0:
        raise ValueError("all probabilities are zero")
    if p[1] == 0:
        return math.inf
    return math.log(p[0]) - math.log(p[1])


def margin_vs_rest(logits: Sequence[float]) -> float:
    """Winner against all other options: z(1) - logsumexp(other logits).

    This equals log(p1 / (1 - p1)), the log-odds of the winning option.
    Unlike the top-2 gap, it is smooth, and it shrinks when probability is
    spread over many runners-up.
    """
    z = sorted(_check(logits), reverse=True)
    return z[0] - _logsumexp(z[1:])


def target_margin(logits: Sequence[float], target: int) -> float:
    """Margin for a fixed option: z[target] - max of the other logits.

    Positive when ``target`` wins and negative when it loses. Fix ``target`` to
    the model's prediction on the full document, so the search can't switch
    to whichever label is easiest to make confident.
    """
    z = _check(logits)
    if not 0 <= target < len(z):
        raise IndexError(f"target {target} out of range for {len(z)} options")
    return z[target] - max(v for i, v in enumerate(z) if i != target)


def calibrate(logits: Sequence[float], content_free_logits: Sequence[float]) -> list[float]:
    """Contextual calibration (Zhao et al., 2021) in log space.

    Subtracts the model's scores on a content-free input, such as "N/A", so
    that label priors the model holds regardless of content are not rewarded.
    Up to a constant, this matches Zhao et al.'s diagonal correction with
    zero bias. Pass the result to any margin function.
    """
    z = _check(logits)
    c = _check(content_free_logits, "content_free_logits")
    if len(z) != len(c):
        raise ValueError(f"length mismatch: {len(z)} logits vs {len(c)} content-free logits")
    return [a - b for a, b in zip(z, c)]
