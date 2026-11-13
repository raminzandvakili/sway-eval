"""sway-eval: audit how much the order of a record's sections decides an AI decision model's verdict."""

from .margin import (
    calibrate,
    log_softmax,
    margin_from_probs,
    margin_vs_rest,
    target_margin,
    top2_margin,
)
from .space import (
    count_sequences,
    count_sequences_up_to,
    exhaustive_calls,
    ordered_subsets,
)

__version__ = "0.1.0"

__all__ = [
    "calibrate",
    "count_sequences",
    "count_sequences_up_to",
    "exhaustive_calls",
    "log_softmax",
    "margin_from_probs",
    "margin_vs_rest",
    "ordered_subsets",
    "target_margin",
    "top2_margin",
]
