from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Callable, Optional

from gfd_mining import GFDDependency, GFDDependencyMiner
from graph_types import DataGraph, FrequentPattern, PatternOptions
from pattern_extension import GraphSpawn


GraphLoader = Callable[..., DataGraph]
SeedBuilder = Callable[[DataGraph], FrequentPattern]


@dataclass
class GfdRunConfig:
    dataset_name: str
    interaction_csv_path: Optional[str]
    seed_builder: SeedBuilder
    csv_graph_loader: GraphLoader
    node_csv_path: Optional[str] = None
    auto_discover_if_missing: bool = False
    fallback_interaction_name: str = "protein_protein_signed.csv"
    fallback_node_name: str = "protein.csv"
    mode: str = "gfd"
    y_key: Optional[str] = "e0.interaction_label"
    max_rows: Optional[int] = None
    undirected: bool = True
    protein_index_column: str = "index"
    full_solution: bool = False
    pattern_support: int = 5
    min_support_count: int = 5
    min_confidence: float = 1.0
    min_value_support_count: int = 5
    max_lhs_size: int = 2
    max_conflicts: int = 3
    max_constant_literals_per_key: int = 5
    max_candidate_literals: int = 200
    discover_constant_rhs: bool = True
    discover_equality_rhs: bool = True
    discover_negative: bool = False
    max_radius: int = 2
    max_add_edge: int = 2
    node_max_add_edge: int = 4
    min_pattern_nodes: Optional[int] = None
    max_pattern_nodes: Optional[int] = None
    max_multi_support: int = 10000
    global_vspawn_instances: bool = False
    topology_only_pattern_dedup: bool = True
    topology_dedupe_respect_direction: bool = False
    pattern_extension_debug: bool = False
    pattern_extension_debug_limit: int = 500
    vf3_drop_data_self_loops: bool = True
    filter_degree_predicates: bool = False
    ignored_predicate_key_tokens: tuple[str, ...] = (
        "interaction_label",
        "edge_existing",
        "augmented_negative",
        "sampled_",
        "direction_role",
        "edgelabel",
        "ml_",
    )
    print_dependency_limit: int = 10


def resolve_csv_path(raw_path: Optional[str], fallback_name: str, auto_discover: bool) -> str:
    if raw_path:
        return raw_path
    if not auto_discover:
        raise FileNotFoundError(f"{fallback_name} is empty and auto discovery is disabled")
    search_root = Path(__file__).resolve().parents[3]
    matches = list(search_root.rglob(fallback_name))
    if not matches:
        raise FileNotFoundError(f"Could not auto-discover {fallback_name}")
    matches.sort(key=lambda candidate: len(str(candidate)))
    return str(matches[0])


def dependency_stats(dependencies: list[GFDDependency]) -> dict[str, object]:
    positive = [dep for dep in dependencies if dep.kind == "positive"]
    negative = [dep for dep in dependencies if dep.kind == "negative"]
    avg_conf = sum(dep.confidence for dep in dependencies) / len(dependencies) if dependencies else 0.0
    avg_pattern_size = (
        sum(dep.pattern_node_count for dep in dependencies) / len(dependencies)
        if dependencies
        else 0.0
    )
    return {
        "positive": len(positive),
        "negative": len(negative),
        "negative_ratio": len(negative) / len(dependencies) if dependencies else 0.0,
        "avg_confidence": avg_conf,
        "avg_pattern_size": avg_pattern_size,
        "total": len(dependencies),
    }


def predicate_key_has_ignored_token(key: str, ignored_tokens: tuple[str, ...]) -> bool:
    lowered = key.lower()
    return any(token.lower() in lowered for token in ignored_tokens)


def filtered_gfd_candidate_keys(
    miner: GFDDependencyMiner,
    graph: DataGraph,
    target_pattern: FrequentPattern,
    cfg: GfdRunConfig,
) -> Optional[list[str]]:
    if not cfg.filter_degree_predicates:
        return None

    rows = miner.prune_rows_by_value_support(miner.build_instance_rows(graph, target_pattern))
    all_keys = sorted({key for row in rows for key in row})
    candidate_keys = [
        key
        for key in all_keys
        if key != cfg.y_key and not predicate_key_has_ignored_token(key, cfg.ignored_predicate_key_tokens)
    ]
    dropped_keys = [
        key
        for key in all_keys
        if key != cfg.y_key and predicate_key_has_ignored_token(key, cfg.ignored_predicate_key_tokens)
    ]
    print(
        f"[PredicateFilter] pattern_id={target_pattern.pattern.pattern_id} y_key={cfg.y_key} "
        f"tokens={cfg.ignored_predicate_key_tokens} candidate_keys={len(candidate_keys)} "
        f"dropped_keys={len(dropped_keys)} sample={dropped_keys[:20]}"
    )
    return candidate_keys


def run_demo(cfg: GfdRunConfig) -> None:
    print(f"=== GFD {cfg.dataset_name} Demo ===")
    os.environ["GAR_VF3_DROP_DATA_SELF_LOOPS"] = "1" if cfg.vf3_drop_data_self_loops else "0"
    csv_path = resolve_csv_path(cfg.interaction_csv_path, cfg.fallback_interaction_name, cfg.auto_discover_if_missing)
    node_csv_path = (
        resolve_csv_path(cfg.node_csv_path, cfg.fallback_node_name, cfg.auto_discover_if_missing)
        if cfg.node_csv_path or cfg.auto_discover_if_missing
        else None
    )
    print(f"[Input] interaction_csv={csv_path}")
    print(f"[Input] node_csv={node_csv_path}")
    print(
        f"[Config] dataset={cfg.dataset_name} mode={cfg.mode} max_rows={cfg.max_rows} "
        f"y_key={cfg.y_key} min_support_count={cfg.min_support_count} "
        f"min_confidence={cfg.min_confidence} max_lhs_size={cfg.max_lhs_size}"
    )
    print(
        f"[GFDConfig] constant_rhs={cfg.discover_constant_rhs} equality_rhs={cfg.discover_equality_rhs} "
        f"negative={cfg.discover_negative} max_candidate_literals={cfg.max_candidate_literals}"
    )
    print(
        f"[VSpawnConfig] max_radius={cfg.max_radius} max_add_edge={cfg.max_add_edge} "
        f"node_max_add_edge={cfg.node_max_add_edge} min_pattern_nodes={cfg.min_pattern_nodes} "
        f"max_pattern_nodes={cfg.max_pattern_nodes} topology_only_dedup={cfg.topology_only_pattern_dedup} "
        f"global_vspawn_instances={cfg.global_vspawn_instances} "
        f"vf3_drop_data_self_loops={cfg.vf3_drop_data_self_loops}"
    )

    graph = cfg.csv_graph_loader(
        csv_path,
        max_rows=cfg.max_rows,
        undirected=cfg.undirected,
        protein_path=node_csv_path,
        protein_index_column=cfg.protein_index_column,
    )
    isolated_vertices = sum(1 for node_id in graph.vertices if not graph.out_edges.get(node_id) and not graph.in_edges.get(node_id))
    print(
        f"[Graph] vertices={len(graph.vertices)} out_edge_lists={sum(len(v) for v in graph.out_edges.values())} "
        f"isolated_vertices={isolated_vertices}"
    )

    seed = cfg.seed_builder(graph)
    spawn = GraphSpawn(
        graph,
        [seed],
        options=PatternOptions(
            pattern_support_threshold=cfg.pattern_support,
            max_radius=cfg.max_radius,
            max_add_edge=cfg.max_add_edge,
            node_max_add_edge=cfg.node_max_add_edge,
            max_pattern_nodes=cfg.max_pattern_nodes,
            full_solution=cfg.full_solution,
            max_multi_support=cfg.max_multi_support,
            topology_only_dedup=cfg.topology_only_pattern_dedup,
            topology_dedupe_respect_direction=cfg.topology_dedupe_respect_direction,
            global_vspawn_instances=cfg.global_vspawn_instances,
            extension_debug=cfg.pattern_extension_debug,
            extension_debug_limit=cfg.pattern_extension_debug_limit,
        ),
    )
    generated = []
    round_index = 0
    while spawn.unstoppable():
        round_generated = spawn.vspawn()
        round_index += 1
        generated.extend(round_generated)
        print(f"[VSpawn] round={round_index} generated={len(round_generated)} total={len(generated)}")
        stats = spawn.stats
        print(
            f"[VSpawnStats] round={round_index} candidates_seen={stats.candidates_seen} "
            f"bn_pruned={stats.bn_pruned} duplicate_pruned={stats.duplicate_pruned} "
            f"constraint_pruned={stats.constraint_pruned} no_match_pruned={stats.no_match_pruned} "
            f"support_pruned={stats.support_pruned}"
        )
    if not generated:
        raise RuntimeError("No pattern generated. Try lowering pattern_support or increasing max_radius/max_add_edge.")

    size_filtered_patterns = [
        item
        for item in generated
        if (cfg.min_pattern_nodes is None or item.pattern.node_count() >= cfg.min_pattern_nodes)
        and (cfg.max_pattern_nodes is None or item.pattern.node_count() <= cfg.max_pattern_nodes)
    ]
    patterns_to_mine = sorted(
        size_filtered_patterns,
        key=lambda item: (item.pattern.edge_count(), item.single_support(), item.multi_support()),
        reverse=True,
    )
    print(
        f"[Patterns] generated_total={len(generated)} mining_total={len(patterns_to_mine)} "
        f"size_filtered={len(generated) - len(size_filtered_patterns)} "
        f"min_pattern_nodes={cfg.min_pattern_nodes} max_pattern_nodes={cfg.max_pattern_nodes}"
    )

    if cfg.mode == "pattern-only":
        for pattern_index, pattern in enumerate(patterns_to_mine, start=1):
            edges = [(edge.src, edge.dst, edge.label) for edge in pattern.pattern.edges]
            print(
                f"[Pattern {pattern_index}/{len(patterns_to_mine)}] id={pattern.pattern.pattern_id} "
                f"labels={pattern.pattern.node_labels} edges={edges} "
                f"single_support={pattern.single_support()} multi_support={pattern.multi_support()}"
            )
        return
    if cfg.mode != "gfd":
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    miner = GFDDependencyMiner(
        min_support_count=cfg.min_support_count,
        min_confidence=cfg.min_confidence,
        min_value_support_count=cfg.min_value_support_count,
        max_lhs_size=cfg.max_lhs_size,
        max_conflicts=cfg.max_conflicts,
        max_constant_literals_per_key=cfg.max_constant_literals_per_key,
        max_candidate_literals=cfg.max_candidate_literals,
        discover_constant_rhs=cfg.discover_constant_rhs,
        discover_equality_rhs=cfg.discover_equality_rhs,
        discover_negative=cfg.discover_negative,
    )

    all_dependencies: list[GFDDependency] = []
    rule_mining_started = time.perf_counter()
    for pattern_index, target_pattern in enumerate(patterns_to_mine, start=1):
        edges = [(edge.src, edge.dst, edge.label) for edge in target_pattern.pattern.edges]
        print(
            f"[Pattern {pattern_index}/{len(patterns_to_mine)}] id={target_pattern.pattern.pattern_id} "
            f"labels={target_pattern.pattern.node_labels} edges={edges} "
            f"single_support={target_pattern.single_support()} multi_support={target_pattern.multi_support()}"
        )
        pattern_rule_mining_started = time.perf_counter()
        candidate_keys = filtered_gfd_candidate_keys(miner, graph, target_pattern, cfg)
        dependencies = miner.discover_dependencies(
            graph,
            target_pattern,
            y_key=cfg.y_key,
            candidate_keys=candidate_keys,
        )
        pattern_rule_mining_seconds = time.perf_counter() - pattern_rule_mining_started
        all_dependencies.extend(dependencies)
        stats = dependency_stats(dependencies)
        print(
            f"[GFDMining] pattern_id={target_pattern.pattern.pattern_id} dependencies={len(dependencies)} "
            f"positive={stats['positive']} negative={stats['negative']} y_key={cfg.y_key}"
        )
        print(
            f"[Timing] stage=rule_mining algorithm=GFD dataset={cfg.dataset_name} "
            f"pattern_id={target_pattern.pattern.pattern_id} dependencies={len(dependencies)} "
            f"seconds={pattern_rule_mining_seconds:.6f}"
        )

        for dependency in dependencies[: cfg.print_dependency_limit]:
            print("  " + dependency.format())
            if dependency.conflicts:
                print("    first_conflict:")
                pprint(dependency.conflicts[0].to_dict(dependency.determinant, dependency.dependent), width=120)

        if not dependencies:
            print("  [GFDMining] no dependency satisfied the thresholds for this pattern")

    rule_mining_seconds = time.perf_counter() - rule_mining_started
    summary = dependency_stats(all_dependencies)
    print(
        f"[Timing] stage=rule_mining_total algorithm=GFD dataset={cfg.dataset_name} "
        f"patterns={len(patterns_to_mine)} dependencies={len(all_dependencies)} "
        f"seconds={rule_mining_seconds:.6f}"
    )
    print(
        "[Summary] "
        f"patterns_mined={len(patterns_to_mine)} total_gfds={summary['total']} "
        f"positive={summary['positive']} negative={summary['negative']} "
        f"negative_ratio={summary['negative_ratio']:.4f} "
        f"avg_confidence={summary['avg_confidence']:.4f} avg_pattern_size={summary['avg_pattern_size']:.2f}"
    )
    if not all_dependencies:
        print("[GFDMining] no dependency satisfied the thresholds")
        print("  hint: lower min_confidence for approximate GFDs, lower min_support_count/min_value_support_count, or set y_key=None")
