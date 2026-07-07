from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

from gar_demo_runner import GarRunConfig, run_demo
from relation_loader import RelationGraphConfig, build_source_seed_pattern, load_relation_csv_graph


DEFAULT_DATA_DIR = "/home/yangsiyi10504/baselines/去病图数据"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", DEFAULT_DATA_DIR))


RELATION = RelationGraphConfig(
    relation_name="DDA",
    source_label="Drug",
    target_label="Disease",
    source_index_column="chemical_index",
    target_index_column="disease_index",
    default_edge_label="drug_disease",
    edge_csv_path=str(DATA_DIR / "drug_disease.csv"),
    source_node_csv_path=str(DATA_DIR / "drug.csv"),
    target_node_csv_path=str(DATA_DIR / "disease.csv"),
    source_node_index_column="index",
    target_node_index_column="index",
    load_node_attributes=True,
    excluded_node_attr_columns=(
        "original_index",
        "source_node_id",
        "chemicalid",
        "chemicalname",
        "casrn",
        "synonyms",
        "description",
        "drug_interactions",
        "external_identifiers",
        "external_links",
        "general_references",
        "references",
        "patents",
    ),
    target_edge_attr_columns=("diseasename", "diseaseid"),
    excluded_edge_attr_columns=("chemical_index", "disease_index", "node_1", "node_2"),
)


CONFIG = GarRunConfig(
    dataset_name="DDA",
    mode="decision-tree",
    interaction_csv_path=RELATION.edge_csv_path,
    sampled_pt_path=None,
    sampled_graph_loader=None,
    node_csv_path=None,
    csv_graph_loader=partial(load_relation_csv_graph, RELATION),
    seed_builder=partial(build_source_seed_pattern, source_label="Drug"),
    fallback_interaction_name="drug_disease.csv",
    fallback_node_name="drug.csv",
    y_key="v0.high_degree",
    max_rows=None,
    undirected=False,
    full_solution=False,
    pattern_support=5,
    min_support=50,
    min_confidence=0.6,
    min_value_support_count=5,
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
