from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MINER = ROOT / "enumeration-discovery" / "GARplusMiner"
sys.path.insert(0, str(MINER))

from garplus_demo_runner import GarplusRunConfig, run_demo  # noqa: E402
from garplus_ml_predicates import MLPredicateConfig  # noqa: E402
from graph_types import DataGraph, FrequentPattern, GraphInstance, GraphPattern, PatternEdge, Vertex  # noqa: E402

FLIP_ORIGINAL = {0: None, 1: 0, 2: 1, 3: 0, 4: 2, 5: 1, 6: 2}
CONFIG = {"dda": ("drug", "disease", "drug_disease"), "ti": ("gene", "disease", "gene_disease")}


def norm(value: object) -> str:
    value = str(value).strip()
    return value[:-2] if value.endswith(".0") else value


def recover_target_edges(dataset: str, train_c: Path, node_map_path: Path, big_nodes: Path) -> pd.DataFrame:
    source_type, target_type, relation = CONFIG[dataset]
    train = pd.read_csv(train_c, sep=r"\s+", header=None,
                        names=["src", "noisy_label", "dst", "flipped_flag", "flip_type"])
    local = pd.read_csv(node_map_path)
    local_map = dict(zip(local.node_id.astype(int), local.old_index.map(norm)))
    typed = pd.read_csv(node_map_path.with_name("edges_labeled_with_reason.csv"), usecols=["src"])
    source_block = int(typed.src.max()) + 1
    nodes = pd.read_csv(big_nodes, dtype=str)
    lookup = {(row.node_type.lower(), norm(row.node_id)): int(row.node_index) for row in nodes.itertuples()}

    def to_big(local_id: int) -> int:
        node_type = source_type if int(local_id) < source_block else target_type
        return lookup[(node_type, local_map[int(local_id)])]

    train["src_big"] = train.src.map(to_big)
    train["dst_big"] = train.dst.map(to_big)
    train["original_label"] = train.noisy_label
    changed = train.flip_type != 0
    train.loc[changed, "original_label"] = train.loc[changed, "flip_type"].map(FLIP_ORIGINAL)
    train["relation"] = relation
    return train


def enrich_node_attributes(nodes: pd.DataFrame, attribute_data_dir: Path | None) -> pd.DataFrame:
    if attribute_data_dir is None:
        return nodes
    parts = []
    for node_type, group in nodes.groupby("node_type"):
        path = attribute_data_dir / f"{node_type}.csv"
        if not path.exists():
            parts.append(group)
            continue
        wanted = set(group.node_id.map(norm))
        matched = []
        for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
            index_column = "index" if "index" in chunk.columns else "node_id"
            mask = chunk[index_column].map(norm).isin(wanted)
            if mask.any():
                matched.append(chunk.loc[mask].copy())
        attrs = pd.concat(matched, ignore_index=True) if matched else pd.DataFrame()
        if attrs.empty:
            parts.append(group)
            continue
        index_column = "index" if "index" in attrs.columns else "node_id"
        attrs["_join_id"] = attrs[index_column].map(norm)
        base = group.copy()
        base["_join_id"] = base.node_id.map(norm)
        merged = base.merge(attrs.drop(columns=[index_column], errors="ignore"), on="_join_id", how="left")
        parts.append(merged.drop(columns="_join_id"))
    return pd.concat(parts, ignore_index=True, sort=False)


def extract_subgraph(targets: pd.DataFrame, big_edges: Path, big_nodes: Path, hops: int, output: Path,
                     attribute_data_dir: Path | None = None, overfit_noise: bool = False) -> None:
    edges = pd.read_csv(big_edges, usecols=["relation", "x_index", "y_index"])
    edges = edges.rename(columns={"x_index": "src", "y_index": "dst"})
    edges[["src", "dst"]] = edges[["src", "dst"]].apply(pd.to_numeric, errors="coerce")
    edges = edges.dropna().astype({"src": int, "dst": int})
    incident: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(edges.itertuples(index=False)):
        incident[row.src].append(index)
        incident[row.dst].append(index)
    seeds = set(targets.loc[targets.flipped_flag == 1, "src_big"]) | set(targets.loc[targets.flipped_flag == 1, "dst_big"])
    visited, chosen = set(seeds), set()
    queue = deque((node, 0) for node in seeds)
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for edge_index in incident.get(node, []):
            chosen.add(edge_index)
            row = edges.iloc[edge_index]
            for nxt in (int(row.src), int(row.dst)):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
    structural = edges.iloc[sorted(chosen)].copy()
    structural["interaction_label"] = ""
    mining_targets = targets[targets.flipped_flag == 1] if overfit_noise else targets
    target_rows = mining_targets[["src_big", "dst_big", "relation", "original_label"]].rename(
        columns={"src_big": "src", "dst_big": "dst", "original_label": "interaction_label"}
    )
    combined = pd.concat([target_rows, structural], ignore_index=True).drop_duplicates(
        ["src", "dst", "relation"], keep="first"
    )
    used = set(combined.src) | set(combined.dst)
    nodes = pd.read_csv(big_nodes)
    nodes = nodes[nodes.node_index.isin(used)]
    nodes = enrich_node_attributes(nodes, attribute_data_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "subgraph_edges.csv", index=False)
    nodes.to_csv(output / "subgraph_nodes.csv", index=False)
    targets.to_csv(output / "target_edges_mapped.csv", index=False)
    (output / "subgraph_summary.json").write_text(json.dumps({
        "seed_nodes": len(seeds), "nodes": len(nodes), "structural_edges": len(structural),
        "target_edges": len(target_rows), "combined_edges": len(combined), "hops": hops,
        "overfit_noise": overfit_noise,
    }, indent=2), encoding="utf-8")


def load_subgraph(path: str, max_rows=None, undirected=False, **_kwargs) -> DataGraph:
    directory = Path(path).parent
    nodes = pd.read_csv(directory / "subgraph_nodes.csv")
    vertices = {}
    identity = {"node_index", "node_id", "node_type"}
    for row in nodes.to_dict("records"):
        attrs = {str(key).strip().lower().replace(" ", "_"): value for key, value in row.items()
                 if key not in identity and pd.notna(value) and str(value).strip()}
        vertices[int(row["node_index"])] = Vertex(int(row["node_index"]), str(row["node_type"]).title(), attrs)
    graph = DataGraph(vertices=vertices)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if max_rows is not None and index >= max_rows:
                break
            src, dst = int(row["src"]), int(row["dst"])
            # NetworkX core_number (used by Pattern BN) rejects self loops.
            # They are irrelevant to the anchored DDA/TI target patterns.
            if src == dst:
                continue
            attrs = {}
            attrs["relation_type"] = row["relation"]
            if row.get("interaction_label", "").strip():
                label = norm(row["interaction_label"])
                attrs["interaction_label"] = label
                if label == "0":
                    attrs["is_missing_edge"] = "true"
                    attrs["edge_semantics"] = "no_edge"
                else:
                    attrs["is_missing_edge"] = "false"
                    attrs["edge_semantics"] = "observed_edge"
            graph.add_edge(src, dst, row["relation"], attrs)
    return graph


def target_seed(graph: DataGraph, source_label: str, target_label: str, relation: str) -> FrequentPattern:
    pattern = GraphPattern([source_label.title(), target_label.title()], [PatternEdge(0, 1, relation)])
    instances = []
    for edge in graph.all_edges():
        if edge.label == relation and "interaction_label" in edge.attrs:
            instances.append(GraphInstance({0: edge.src, 1: edge.dst}, ((edge.src, edge.dst, relation),),
                                           edge.src, {0: edge.edge_id}))
    return FrequentPattern(pattern, instances)


def mine(dataset: str, output: Path, enable_bn: bool = False, overfit_noise: bool = False) -> None:
    source, target, relation = CONFIG[dataset]
    edge_path = output / "subgraph_edges.csv"
    cfg = GarplusRunConfig(
        dataset_name=f"{dataset.upper()}_FLIP", interaction_csv_path=str(edge_path),
        sampled_pt_path=None, sampled_graph_loader=None, use_sampled_pt_graph=False,
        csv_graph_loader=load_subgraph, verification_graph_loader=load_subgraph,
        seed_builder=partial(target_seed, source_label=source, target_label=target, relation=relation),
        force_edge_label=None, mode="fp-growth", fp_growth_max_itemset_size=4,
        y_key="e0.interaction_label", include_ml_predicate_targets=False,
        augment_negative_edges=False, balance_edge_labels=False, max_rows=None,
        pattern_support=1 if overfit_noise else 5,
        min_support=1 if overfit_noise else 10,
        min_confidence=0.5 if overfit_noise else 0.6,
        min_lift=1.0,
        min_value_support_count=1 if overfit_noise else 5,
        max_radius=2, max_add_edge=2, node_max_add_edge=2,
        undirected=False, undirected_pattern=False, topology_only_pattern_dedup=True,
        global_rematch_patterns=not overfit_noise,
        include_seed_pattern=overfit_noise,
        global_match_scope="sampled", rule_coverage_scope="sampled",
        # In overfit mode Predicate BN is intentionally disabled: its sparse
        # feature guards remove rare/high-cardinality entity attributes, which
        # are exactly what is needed to memorize injected-noise endpoints.
        enable_pattern_bn=enable_bn, tau_p=0.0,
        enable_predicate_bn=enable_bn and not overfit_noise, tau_x=0.05,
        predicate_bn_focus_targets=None if overfit_noise else ("0", "1", "2"),
        pattern_bn_cache_path=str(output / "pattern_bn.pkl"),
        predicate_bn_cache_path=str(output / "predicate_bn.pkl"),
        deduped_rules_output_path=str(output / "deduped_rules.txt"),
        pattern_instances_output_path=str(output / "pattern_instances.jsonl"),
        enable_rule_payload_generation=False, enable_target_recall=True,
        drop_unknown_target_rows=True, ignored_target_values=("", "unknown"),
        filter_degree_predicates=not overfit_noise,
        drop_identifier_predicates=not overfit_noise,
        # The controlled overfit run mines the injected rows themselves. Keep
        # endpoint attributes and labels on non-target edges so rules can
        # actually rematch those rows. The selector already excludes the exact
        # target key e0.interaction_label.
        drop_target_entity_features=not overfit_noise,
        ignored_predicate_key_tokens=(
            ("edge_existing", "edge_semantics", "is_missing_edge", "interaction_label_bin")
            if overfit_noise
            else ("degree", "node_name", "interaction_label")
        ),
        predicate_bn_max_feature_cardinality=100000 if overfit_noise else 50,
        print_deduped_rule_limit=100000 if overfit_noise else 50,
        ml_predicates=MLPredicateConfig(enabled=False), debug_match_expansion=False,
        debug_transaction_cost=False, print_instances=False,
    )
    run_demo(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=CONFIG)
    parser.add_argument("--train-c", required=True, type=Path)
    parser.add_argument("--node-map", required=True, type=Path)
    parser.add_argument("--big-nodes", required=True, type=Path)
    parser.add_argument("--big-edges", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attribute-data-dir", type=Path)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--enable-bn", action="store_true")
    parser.add_argument("--overfit-noise", action="store_true",
                        help="Mine only flipped rows and retain identifier/name predicates for maximum training repair.")
    args = parser.parse_args()
    targets = recover_target_edges(args.dataset, args.train_c, args.node_map, args.big_nodes)
    extract_subgraph(targets, args.big_edges, args.big_nodes, args.hops, args.output_dir,
                     args.attribute_data_dir, args.overfit_noise)
    if not args.prepare_only:
        mine(args.dataset, args.output_dir, args.enable_bn, args.overfit_noise)


if __name__ == "__main__":
    main()
