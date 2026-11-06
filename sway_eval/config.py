"""Experiment configs: loading, validation, and building the pieces they name.

Configs are YAML (or JSON) files with the fields in
``experiments/_template.yaml``. Relative paths in a config are resolved
against the config file's own folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .adapters.base import Adapter
from .adapters.clef import ClefAdapter
from .adapters.decision2 import Decision2Adapter
from .adapters.mock import MockAdapter
from .adapters.microsoft import MicrosoftDecisionAdapter
from .adapters.openai_decisions import OpenAIDecisionsAdapter
from .adapters.spend import SpendCapped
from .adapters.typesafe import TypeSafeAdapter
from .synth import RubricSpec, rubric_scorer
from .units import FormatSpec

REQUIRED = ("name", "date", "seed", "model", "dataset", "prompt", "layouts", "analysis", "output_root")


def load_config(path: str | Path) -> dict:
    """Read a config file and attach its folder as ``_base``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("reading YAML configs needs PyYAML; use a .json config or install pyyaml") from e
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    cfg["date"] = str(cfg.get("date", ""))  # YAML reads 2026-11-01 as a date object
    cfg["_base"] = str(path.resolve().parent)
    cfg["_source"] = str(path.resolve())
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config is missing {missing}")
    if cfg.get("paper") and not str(cfg["name"]).lower().startswith(str(cfg["paper"]).lower() + "_"):
        raise ValueError("with a paper letter, the config name must start with it, e.g. a_pilot_jev")
    a = cfg["analysis"]
    options = cfg["prompt"].get("options", [])
    if a.get("tie_option") not in options:
        raise ValueError(f"analysis.tie_option must name one of the options {options} before a run")
    if not a.get("axis"):
        raise ValueError("analysis.axis must name the item field the flip zone is drawn on")
    if a.get("zone_width") not in ("count", "span"):
        raise ValueError('analysis.zone_width must be set to "count" or "span" before a run')
    _validate_noise(cfg)
    m = cfg["model"]
    if m.get("provider") != "mock" and not m.get("version"):
        raise ValueError("model.version must pin an exact model version")


NOISE_KEYS = ("margin_edges", "items_per_stratum", "repeats", "target_share", "max_layout_repeats")
CANARY_KEYS = ("per_stratum", "batch_size", "dispersion_pct", "mean_band", "sigma_floor", "grid",
               "floor_changes", "reference_checks", "reference_seed")


def _validate_noise(cfg: dict) -> None:
    """The noise and canary sections have no defaults: every value is a
    pre-run choice and must be in the config."""
    if "noise_floor" in cfg:
        raise ValueError("noise_floor is replaced by the noise section (see experiments/_template.yaml)")
    nc, cc = cfg.get("noise"), cfg.get("canary")
    if nc is not None:
        missing = [k for k in NOISE_KEYS if k not in nc]
        if missing:
            raise ValueError(f"noise is missing {missing}")
        if not 0 <= nc["margin_edges"]["near"] < nc["margin_edges"]["far"]:
            raise ValueError("noise.margin_edges needs 0 <= near < far")
        if nc["repeats"] < 2:
            raise ValueError("noise.repeats must be at least 2")
    if cc is not None:
        if nc is None:
            raise ValueError("canary needs a noise section: its baseline is the repeat study")
        missing = [k for k in CANARY_KEYS if k not in cc]
        if missing:
            raise ValueError(f"canary is missing {missing}")


def load_env(path: str | Path | None = None) -> None:
    """Read KEY=VALUE lines from code/.env into the environment, without
    overriding variables that are already set. Values are never printed."""
    path = Path(path) if path else Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def rotated_options(cfg: dict, k: int) -> tuple[str, ...]:
    """The config's canonical option order rotated cyclically by k positions.
    Layout runs always use k = 0; other rotations are a separate baseline."""
    opts = list(cfg["prompt"]["options"])
    k %= len(opts)
    return tuple(opts[k:] + opts[:k])


def resolve(cfg: dict, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(cfg["_base"]) / q


def rubric_spec(cfg: dict) -> RubricSpec:
    d = cfg["dataset"]
    kw = {k: d[k] for k in ("threshold", "max_years") if k in d}
    if "convention_order" in d:
        kw["convention_order"] = tuple(d["convention_order"])
    return RubricSpec(**kw)


def formats(cfg: dict) -> list[FormatSpec]:
    """The default format first, then the formatting variants."""
    p = cfg["prompt"]
    base = FormatSpec(**p.get("format", {}))
    return [base] + [FormatSpec(**v) for v in cfg.get("format_variants", [])]


def build_adapter(cfg: dict) -> Adapter:
    """The config's adapter, behind a spending cap if ``model.max_spend_usd`` is set."""
    adapter = _adapter(cfg)
    m = cfg["model"]
    if m.get("max_spend_usd"):
        if not m.get("spend_ledger"):
            raise ValueError("model.max_spend_usd needs model.spend_ledger (a file outside code/)")
        adapter = SpendCapped(adapter, m["max_spend_usd"], m["price_per_m_input_tokens"], resolve(cfg, m["spend_ledger"]))
    return adapter


def _adapter(cfg: dict) -> Adapter:
    m = cfg["model"]
    if m["provider"] == "mock":
        spec = rubric_spec(cfg)
        weights = m["position_weights"]
        scale = m.get("scale", 1.0)
        # Default bias puts the threshold at logit 0 when weights average 1.
        bias = m.get("bias", -scale * spec.threshold * sum(weights) / len(weights))
        return MockAdapter(
            scorer=rubric_scorer(spec),
            position_weights=weights,
            bias=bias,
            scale=scale,
            noise=m.get("noise", 0.0),
            target=cfg["prompt"]["target"],
            first_option_bias=m.get("first_option_bias", 0.0),
            grid=m.get("grid"),
            seed=cfg["seed"],
            model_id=m.get("name", "mock-1"),
            carryover=m.get("carryover", 0.0),
            format_effect=m.get("format_effect", 0.0),
            drift_after=m.get("drift_after"),
            drift_weights=m.get("drift_weights"),
        )
    if m["provider"] == "hf_classifier":
        from .adapters.hf_classifier import HFClassifierAdapter
        return HFClassifierAdapter.from_pretrained(
            m["repo"], m["version"], m["labels"], device=m.get("device", "cpu"),
            input_mode=m.get("input_mode", "pair"), max_length=m.get("max_length"))
    if m["provider"] == "typesafe":
        load_env()
        return TypeSafeAdapter(model=m["version"], criteria=cfg["prompt"]["criteria"])
    if m["provider"] == "clef":
        return ClefAdapter(repo=m["repo"], revision=m["version"], criteria=cfg["prompt"]["criteria"],
                           dtype=m.get("dtype", "bfloat16"), device=m.get("device", "cuda"),
                           deterministic=m.get("deterministic", True),
                           evict_page_cache=m.get("evict_page_cache", False),
                           clone_before_copy=m.get("clone_before_copy", False))
    if m["provider"] == "decision2":
        return Decision2Adapter(repo=m["repo"], revision=m["version"], criteria=cfg["prompt"]["criteria"],
                                device=m.get("device", "cuda:0"), deterministic=m.get("deterministic", True),
                                evict_page_cache=m.get("evict_page_cache", False), low_peak=m.get("low_peak", False))
    if m["provider"] == "openai":
        load_env()
        return OpenAIDecisionsAdapter(criteria=cfg["prompt"]["criteria"], placement=m.get("placement", "input"))
    if m["provider"] == "microsoft":
        load_env()
        return MicrosoftDecisionAdapter(criteria=cfg["prompt"]["criteria"], model_version=m["version"])
    raise NotImplementedError(f"no adapter for provider {m['provider']!r}")
