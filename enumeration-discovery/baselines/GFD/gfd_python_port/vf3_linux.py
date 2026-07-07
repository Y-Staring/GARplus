from __future__ import annotations

"""vf3py-backed global matcher with GAR edge-id materialization."""

import os
from collections import Counter
from typing import Dict, List, Optional

import networkx as nx
import vf3py

from exact_subgraph_matcher import find_matches_with_limit as exact_find_matches_with_limit
from graph_types import DataGraph, GraphInstance, GraphPattern


def _drop_data_self_loops_for_vf3() -> bool:
    value = os.environ.get("GAR_VF3_DROP_DATA_SELF_LOOPS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _data_self_loop_count(graph: DataGraph) -> int:
    return sum(1 for edge in graph.all_edges() if edge.src == edge.dst)


def _to_networkx(pattern: GraphPattern, graph: DataGraph):
    pattern_graph = nx.DiGraph()
    pattern_has_self_loop = any(edge.src == edge.dst for edge in pattern.edges)
    drop_data_self_loops = _drop_data_self_loops_for_vf3()
    for node_id, label in enumerate(pattern.node_labels):
        pattern_graph.add_node(node_id, label=label)
    for edge in pattern.edges:
        pattern_graph.add_edge(edge.src, edge.dst, label=edge.label)
    data_graph = nx.DiGraph()
    for node_id, vertex in graph.vertices.items():
        data_graph.add_node(node_id, label=vertex.label)
    skipped_self_loops = 0
    for edge in graph.all_edges():
        if edge.src == edge.dst and not pattern_has_self_loop and drop_data_self_loops:
            skipped_self_loops += 1
            continue
        data_graph.add_edge(edge.src, edge.dst, label=edge.label)
    if skipped_self_loops and not getattr(find_matches_with_limit, "_reported_skipped_self_loops", False):
        print(
            f"[VF3Preflight] skipped_data_self_loops={skipped_self_loops} "
            "because current pattern has no self-loop edges and GAR_VF3_DROP_DATA_SELF_LOOPS=1"
        )
        setattr(find_matches_with_limit, "_reported_skipped_self_loops", True)
    return pattern_graph, data_graph


def _print_vf3_preflight(
    pattern: GraphPattern,
    graph: DataGraph,
    limit: Optional[int],
    pivot_candidates: Optional[List[int]],
    target_edge_index: Optional[int],
    max_instances_per_target_edge: Optional[int],
    target_edge_undirected: bool,
) -> None:
    if getattr(find_matches_with_limit, "_reported_preflight", False):
        return
    setattr(find_matches_with_limit, "_reported_preflight", True)

    edge_signature_counts = Counter()
    edge_pair_counts = Counter()
    label_counts = Counter()
    self_loops = 0
    edge_total = 0
    for edge in graph.all_edges():
        edge_total += 1
        edge_signature_counts[(edge.src, edge.dst, edge.label)] += 1
        edge_pair_counts[(edge.src, edge.dst)] += 1
        label_counts[str(edge.label)] += 1
        if edge.src == edge.dst:
            self_loops += 1

    duplicate_signature_keys = sum(1 for count in edge_signature_counts.values() if count > 1)
    duplicate_signature_edges = sum(count - 1 for count in edge_signature_counts.values() if count > 1)
    max_parallel_signature = max(edge_signature_counts.values(), default=0)
    multi_edge_pairs = sum(1 for count in edge_pair_counts.values() if count > 1)
    networkx_compressed_edges = edge_total - len(edge_pair_counts)
    pattern_edges = [(edge.src, edge.dst, edge.label) for edge in pattern.edges]
    print(
        "[VF3Preflight] "
        "backend_candidate=vf3py "
        f"limit={limit} pivot_candidates={len(pivot_candidates) if pivot_candidates is not None else None} "
        f"target_edge_index={target_edge_index} "
        f"max_instances_per_target_edge={max_instances_per_target_edge} "
        f"target_edge_undirected={target_edge_undirected} "
        f"pattern_nodes={pattern.node_count()} pattern_edges={pattern.edge_count()} "
        f"pattern_edge_list={pattern_edges} "
        f"data_vertices={len(graph.vertices)} data_edges={edge_total} "
        f"unique_src_dst_label_edges={len(edge_signature_counts)} "
        f"duplicate_src_dst_label_keys={duplicate_signature_keys} "
        f"duplicate_src_dst_label_edges={duplicate_signature_edges} "
        f"max_parallel_same_label={max_parallel_signature} "
        f"unique_src_dst_pairs={len(edge_pair_counts)} "
        f"multi_edge_pairs={multi_edge_pairs} "
        f"networkx_compressed_edges={networkx_compressed_edges} "
        f"self_loops={self_loops} "
        f"top_labels={label_counts.most_common(5)} "
        f"vf3py_threads={max(1, int(os.environ.get('GARPLUS_VF3PY_THREADS', '1')))}"
    )


def _normalize_mapping(raw_mapping: Dict, pattern: GraphPattern) -> Optional[Dict[int, int]]:
    pattern_nodes = set(range(pattern.node_count()))
    mapping = {int(key): int(value) for key, value in raw_mapping.items()}
    if pattern_nodes.issubset(mapping):
        return {node_id: mapping[node_id] for node_id in pattern_nodes}
    if pattern_nodes.issubset(set(mapping.values())):
        inverse = {value: key for key, value in mapping.items()}
        return {node_id: inverse[node_id] for node_id in pattern_nodes}
    return None


def _append_edge_bindings(pattern, graph, node_map, results, limit) -> bool:
    choices = []
    for pattern_edge_index, edge in enumerate(pattern.edges):
        candidates = graph.find_edges(node_map[edge.src], node_map[edge.dst], edge.label)
        if not candidates:
            return False
        choices.append((pattern_edge_index, candidates))
    choices.sort(key=lambda item: len(item[1]))
    bindings = {}
    used_edge_ids = set()

    def bind(index: int) -> bool:
        if limit is not None and len(results) >= limit:
            return True
        if index == len(choices):
            edge_ids = tuple(sorted((graph.edges_by_id[eid].src, graph.edges_by_id[eid].dst, graph.edges_by_id[eid].label) for eid in bindings.values()))
            results.append(GraphInstance(node_map=dict(node_map), edge_ids=edge_ids, pivot=node_map.get(0), edge_bindings=dict(bindings)))
            return False
        pattern_edge_index, candidates = choices[index]
        for edge in candidates:
            if edge.edge_id in used_edge_ids:
                continue
            bindings[pattern_edge_index] = edge.edge_id
            used_edge_ids.add(edge.edge_id)
            should_stop = bind(index + 1)
            used_edge_ids.remove(edge.edge_id)
            bindings.pop(pattern_edge_index)
            if should_stop:
                return True
        return False

    return bind(0)


def find_matches_with_limit(
    pattern: GraphPattern,
    graph: DataGraph,
    limit: Optional[int] = None,
    pivot_candidates: Optional[List[int]] = None,
    target_edge_index: Optional[int] = None,
    max_instances_per_target_edge: Optional[int] = None,
    target_edge_undirected: bool = False,
) -> List[GraphInstance]:
    """Use vf3py's documented monomorphism API, then bind concrete GAR edges."""

    if pivot_candidates is not None or max_instances_per_target_edge is not None:
        if max_instances_per_target_edge is not None and not getattr(find_matches_with_limit, "_reported_target_cap_backend", False):
            print(
                f"[VF3Linux] max_instances_per_target_edge={max_instances_per_target_edge}; "
                "using exact matcher for target-edge capped rematch"
            )
            setattr(find_matches_with_limit, "_reported_target_cap_backend", True)
        return exact_find_matches_with_limit(
            pattern,
            graph,
            limit,
            pivot_candidates,
            target_edge_index=target_edge_index,
            max_instances_per_target_edge=max_instances_per_target_edge,
            target_edge_undirected=target_edge_undirected,
        )
    _print_vf3_preflight(
        pattern,
        graph,
        limit,
        pivot_candidates,
        target_edge_index,
        max_instances_per_target_edge,
        target_edge_undirected,
    )
    if any(edge.src == edge.dst for edge in pattern.edges):
        print("[VF3Linux] pattern_has_self_loop=True; using exact matcher because vf3py cannot add self-loop edges")
        return exact_find_matches_with_limit(
            pattern,
            graph,
            limit,
            pivot_candidates,
            target_edge_index=target_edge_index,
            max_instances_per_target_edge=max_instances_per_target_edge,
            target_edge_undirected=target_edge_undirected,
        )
    if not _drop_data_self_loops_for_vf3():
        data_self_loops = _data_self_loop_count(graph)
        if data_self_loops:
            print(
                f"[VF3Linux] data_self_loops={data_self_loops} "
                "GAR_VF3_DROP_DATA_SELF_LOOPS=0; using exact matcher instead of vf3py"
            )
            return exact_find_matches_with_limit(
                pattern,
                graph,
                limit,
                pivot_candidates,
                target_edge_index=target_edge_index,
                max_instances_per_target_edge=max_instances_per_target_edge,
                target_edge_undirected=target_edge_undirected,
            )
    pattern_graph, data_graph = _to_networkx(pattern, graph)
    threads = max(1, int(os.environ.get("GARPLUS_VF3PY_THREADS", "1")))
    try:
        raw_mappings = vf3py.get_subgraph_monomorphisms(
            pattern_graph,
            data_graph,
            node_match=lambda pattern_attrs, data_attrs: pattern_attrs.get("label") == data_attrs.get("label"),
            edge_match=lambda pattern_attrs, data_attrs: pattern_attrs.get("label") == data_attrs.get("label"),
            variant="L",
            num_threads=threads,
        )
    except Exception as exc:
        print(f"[VF3Linux] vf3py_failed={type(exc).__name__}; using exact matcher")
        return exact_find_matches_with_limit(pattern, graph, limit, pivot_candidates)
    if not getattr(find_matches_with_limit, "_reported_backend", False):
        print(f"[VF3Linux] backend=vf3py version={getattr(vf3py, '__version__', 'unknown')} variant=L threads={threads}")
        setattr(find_matches_with_limit, "_reported_backend", True)
    results = []
    for raw_mapping in raw_mappings:
        node_map = _normalize_mapping(raw_mapping, pattern)
        if node_map is not None and _append_edge_bindings(pattern, graph, node_map, results, limit):
            break
    return results


def find_matches(pattern: GraphPattern, graph: DataGraph) -> List[GraphInstance]:
    return find_matches_with_limit(pattern, graph, limit=None)
