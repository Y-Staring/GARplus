"""BN feature engineering and presets for GARplusMiner."""

from .bn_config import STRICT_PATTERN_BN, STRICT_PREDICATE_BN, apply_strict_bn_config
from .pattern_bn_features import (
    BNEdgeLabelFn,
    BNNodeLabelFn,
    aggregate_spawn_node_label,
    augment_graph_structural_features,
    build_default_bn_label_fns,
    default_bn_edge_label,
    default_bn_node_label,
)

__all__ = [
    "BNEdgeLabelFn",
    "BNNodeLabelFn",
    "STRICT_PATTERN_BN",
    "STRICT_PREDICATE_BN",
    "aggregate_spawn_node_label",
    "apply_strict_bn_config",
    "augment_graph_structural_features",
    "build_default_bn_label_fns",
    "default_bn_edge_label",
    "default_bn_node_label",
]