from __future__ import annotations

"""Feature helpers for Pattern-BN: decouple matching labels from BN variables."""

import sys
from collections import Counter
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx

_MINER_ROOT = Path(__file__).resolve().parents[1] / "GARplusMiner"
if str(_MINER_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINER_ROOT))

from graph_types import DataGraph, Edge, FrequentPattern, GraphPattern, Vertex  # noqa: E402


BNEdgeLabelFn = Callable[[Edge, DataGraph], str]
BNNodeLabelFn = Callable[[object, DataGraph], str]


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"", "-", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _three_bin(value: float, q1: float, q2: float, low: str = "low", mid: str = "mid", high: str = "high") -> str:
    if q1 == q2:
        return mid
    if value <= q1:
        return low
    if value >= q2:
        return high
    return mid


def _quantile_cut(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = sorted(values)
    n = len(arr)
    if n == 1 or arr[0] == arr[-1]:
        return float(arr[0]), float(arr[-1])
    q1_idx = max(0, n // 3 - 1)
    q2_idx = max(0, (2 * n) // 3 - 1)
    return float(arr[q1_idx]), float(arr[q2_idx])


def augment_graph_structural_features(graph: DataGraph) -> None:
    """Add role / clustering_bin / core_bin on vertices for BN and predicate mining."""

    undirected = nx.Graph()
    for edge in graph.all_edges():
        undirected.add_edge(edge.src, edge.dst)

    degree_map = dict(undirected.degree())
    clustering_map = nx.clustering(undirected) if undirected.number_of_edges() > 0 else {n: 0.0 for n in undirected.nodes()}
    core_map = nx.core_number(undirected) if undirected.number_of_edges() > 0 else {n: 0 for n in undirected.nodes()}

    deg_values = [float(degree_map.get(n, 0)) for n in graph.vertices]
    clu_values = [float(clustering_map.get(n, 0.0)) for n in graph.vertices]
    core_values = [float(core_map.get(n, 0)) for n in graph.vertices]

    deg_q1, deg_q2 = _quantile_cut(deg_values)
    clu_q1, clu_q2 = _quantile_cut(clu_values)
    core_q1, core_q2 = _quantile_cut(core_values)

    score_values: List[float] = []
    for edge in graph.all_edges():
        val = _to_float(edge.attrs.get("score"))
        if val is not None:
            score_values.append(val)
    score_q1, score_q2 = _quantile_cut(score_values)

    for node_id, vertex in graph.vertices.items():
        degree = float(degree_map.get(node_id, 0))
        clustering = float(clustering_map.get(node_id, 0.0))
        core = float(core_map.get(node_id, 0))

        vertex.attrs.setdefault("degree", degree)
        vertex.attrs["role"] = _three_bin(degree, deg_q1, deg_q2, low="leaf", mid="mid", high="hub")
        vertex.attrs["clustering_bin"] = _three_bin(clustering, clu_q1, clu_q2)
        vertex.attrs["core_bin"] = _three_bin(float(core), core_q1, core_q2)

        if "degree_bucket" not in vertex.attrs:
            bucket = vertex.attrs["role"]
            if bucket == "leaf":
                vertex.attrs["degree_bucket"] = "low"
            elif bucket == "hub":
                vertex.attrs["degree_bucket"] = "high"
            else:
                vertex.attrs["degree_bucket"] = "medium"

    for edge in graph.all_edges():
        score = _to_float(edge.attrs.get("score"))
        if score is not None:
            edge.attrs["score_bin"] = _three_bin(score, score_q1, score_q2, low="low", mid="medium", high="high")
        exp_sys = edge.attrs.get("experimental_system") or edge.attrs.get("experimental system")
        if exp_sys is not None and str(exp_sys).strip():
            edge.attrs.setdefault("bn_edge_semantic", str(exp_sys).strip().replace(" ", "_")[:60])


def default_bn_edge_label(edge: Edge, graph: DataGraph) -> str:
    """BN edge variable: prefer experimental_system, then score_bin; never interaction_label."""

    for key in ("direct_evidence_category", "bn_edge_semantic", "experimental_system", "experimental system"):
        raw = edge.attrs.get(key)
        if raw is not None and str(raw).strip():
            text = str(raw).strip().replace(" ", "_")[:60]
            prefix = "direct_evidence" if key == "direct_evidence_category" else "exp"
            return f"{prefix}:{text}"

    score_bin = edge.attrs.get("score_bin")
    if score_bin is not None and str(score_bin).strip():
        return f"score_bin:{score_bin}"

    score = _to_float(edge.attrs.get("score"))
    if score is not None:
        return f"score:{score:.4g}"

    return f"struct:{edge.label}"


def default_bn_node_label(node_id: object, graph: DataGraph) -> str:
    """BN node variable: structural role from augmented graph attributes."""

    vertex = graph.vertices.get(node_id)
    if vertex is None:
        return "unknown"
    for key in ("role", "degree_bucket", "clustering_bin", "core_bin"):
        val = vertex.attrs.get(key)
        if val is not None and str(val).strip():
            return f"{key}:{val}"
    return str(vertex.label)


def aggregate_spawn_node_label(
    pattern: GraphPattern,
    spawn_node: int,
    graph: DataGraph,
    frequent_pattern: Optional[FrequentPattern],
    node_label_fn: BNNodeLabelFn,
) -> str:
    """Majority role label over instances bound to spawn_node."""

    if frequent_pattern is None or not frequent_pattern.instances:
        return default_bn_node_label_for_pattern_node(pattern, spawn_node, graph, node_label_fn)

    labels: List[str] = []
    for instance in frequent_pattern.instances:
        data_node = instance.node_map.get(spawn_node)
        if data_node is not None:
            labels.append(node_label_fn(data_node, graph))
    if not labels:
        return default_bn_node_label_for_pattern_node(pattern, spawn_node, graph, node_label_fn)
    return Counter(labels).most_common(1)[0][0]


def default_bn_node_label_for_pattern_node(
    pattern: GraphPattern,
    pattern_node: int,
    graph: DataGraph,
    node_label_fn: BNNodeLabelFn,
) -> str:
    return f"plabel:{pattern.node_labels[pattern_node]}"


def build_default_bn_label_fns(graph: DataGraph) -> tuple[BNEdgeLabelFn, BNNodeLabelFn]:
    augment_graph_structural_features(graph)
    return default_bn_edge_label, default_bn_node_label
