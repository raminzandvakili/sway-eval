"""Model adapters. Every call goes through ``sway_eval.cache.CachedAdapter``."""

from .base import Adapter, Decision, DecisionRequest
from .clef import ClefAdapter
from .decision2 import Decision2Adapter
from .mock import MockAdapter, text_scorer
from .typesafe import TypeSafeAdapter

__all__ = ["Adapter", "ClefAdapter", "Decision2Adapter", "Decision", "DecisionRequest", "MockAdapter", "TypeSafeAdapter", "text_scorer"]
