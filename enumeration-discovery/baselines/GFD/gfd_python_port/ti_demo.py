from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

from gfd_demo_runner import GfdRunConfig, run_demo
from relation_loader import RelationGraphConfig, build_source_seed_pattern, load_relation_csv_graph


DEFAULT_DATA_DIR = "D:/CodeWork/python/GAR+/\u6570\u636e/\u53bb\u75c5\u56fe\u6570\u636e/\u53bb\u75c5\u56fe\u6570\u636e"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", DEFAULT_DATA_DIR))


RELATION = RelationGraphConfig(
    relation_name="TI",
    source_label="Gene",
    target_label="Disease",
    source_index_column="gene_index",
    target_index_column="disease_index",
    default_edge_label="gene_disease",
    edge_csv_path=str(DATA_DIR / "gene_disease.csv"),
    source_node_csv_path=str(DATA_DIR / "gene.csv"),
    target_node_csv_path=str(DATA_DIR / "disease.csv"),
    source_node_index_column="index",
    target_node_index_column="index",
    load_node_attributes=True,
    source_edge_attr_columns=("geneid", "genesymbol"),
    target_edge_attr_columns=("diseaseid", "diseasename"),
    excluded_edge_attr_columns=("gene_index", "disease_index", "node_1", "node_2"),
)


CONFIG = GfdRunConfig(
    dataset_name="TI",
    mode="gfd",
    interaction_csv_path=RELATION.edge_csv_path,
    node_csv_path=None,
    csv_graph_loader=partial(load_relation_csv_graph, RELATION),
    seed_builder=partial(build_source_seed_pattern, source_label="Gene"),
    fallback_interaction_name="gene_disease.csv",
    fallback_node_name="gene.csv",
    y_key="v0.high_degree",
    max_rows=None,
    undirected=False,
    full_solution=False,
    pattern_support=5,
    min_support_count=5,
    min_confidence=1.0,
    min_value_support_count=5,
    max_lhs_size=2,
    max_conflicts=3,
    max_constant_literals_per_key=5,
    max_candidate_literals=200,
    discover_constant_rhs=True,
    discover_equality_rhs=True,
    discover_negative=False,
    max_radius=2,
    max_add_edge=2,
    node_max_add_edge=4,
    min_pattern_nodes=None,
    max_pattern_nodes=None,
    max_multi_support=10000,
    topology_only_pattern_dedup=True,
    topology_dedupe_respect_direction=True,
    vf3_drop_data_self_loops=True,
    filter_degree_predicates=True,
    ignored_predicate_key_tokens=(
        "interaction_label",
        "edge_existing",
        "augmented_negative",
        "sampled_",
        "direction_role",
        "edgelabel",
        "ml_",
    ),
)


def main() -> None:
    run_demo(CONFIG)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running cost:", time.time() - start_time)
