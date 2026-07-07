"""Strict Pattern-BN / Predicate-BN presets for GARplusMiner."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

STRICT_PATTERN_BN: Dict[str, Any] = {
    "enable_pattern_bn": True,
    "tau_p": 0.0,
    "pattern_bn_relative_tau": 0.5,
    "pattern_bn_top_k_per_spawn_node": 4,
    "pattern_bn_min_keep_per_spawn_node": 1,
    "augment_structural_features": True,
    "retrain_pattern_bn": True,
}

STRICT_PREDICATE_BN: Dict[str, Any] = {
    "enable_predicate_bn": True,
    "tau_x": 0.05,
    "predicate_bn_top_k_features": 24,
    "predicate_bn_min_keep_features": 6,
    "predicate_bn_max_parent_features": 16,
    "predicate_bn_feature_score": "bic",
    "retrain_predicate_bn": True,
}


def apply_strict_bn_config(cfg: Any) -> Any:
    """Return a copy of *cfg* with strict BN fields applied."""

    return replace(cfg, **STRICT_PATTERN_BN, **STRICT_PREDICATE_BN)