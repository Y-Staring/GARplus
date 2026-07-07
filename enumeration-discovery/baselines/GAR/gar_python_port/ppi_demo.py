from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

from gar_demo_runner import GarRunConfig, run_demo
from ppi_loader import build_ppi_seed_pattern, load_ppi_csv


DEFAULT_DATA_DIR = "/home/yangsiyi10504/baselines/去病图数据"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", DEFAULT_DATA_DIR))


CONFIG = GarRunConfig(
    dataset_name="PPI",
    mode="decision-tree",
    interaction_csv_path=str(DATA_DIR / "protein_protein_signed.csv"),
    node_csv_path=str(DATA_DIR / "protein.csv"),
    node_csv_label="protein_csv",
    sampled_pt_path=None,
    sampled_graph_loader=None,
    csv_graph_loader=partial(load_ppi_csv, edge_label_column="Experimental System", force_edge_label="candidate_interaction"),
    verification_graph_loader=partial(load_ppi_csv, edge_label_column="Experimental System", force_edge_label="candidate_interaction"),
    seed_builder=build_ppi_seed_pattern,
    fallback_interaction_name="protein_protein_signed.csv",
    fallback_node_name="protein.csv",
    force_edge_label="candidate_interaction",
    edge_label_column="Experimental System",
    augment_negative_edges=False,
    include_ml_predicate_targets=False,
    include_edge_existing_target=False,
    enable_pattern_bn=False,
    enable_predicate_bn=False,
    enable_target_recall=False,
    enable_rule_payload_generation=False,
    y_key="e0.interaction_label",
    min_support=50,
    min_confidence=0.6,
    min_value_support_count=5,
    pattern_support=5,
    max_radius=2,
    max_add_edge=2,
    node_max_add_edge=4,
    min_pattern_nodes=None,
    max_pattern_nodes=None,
    max_multi_support=10000,
    topology_only_pattern_dedup=True,
    topology_dedupe_respect_direction=False,
    global_rematch_patterns=True,
    global_rematch_max_instances=None,
    global_rematch_max_pattern_edges=3,
    global_rematch_target_edge_index=0,
    vf3_drop_data_self_loops=True,
    use_sampled_pt_graph=False,
    inject_sampled_frequent_patterns=False,
    filter_degree_predicates=False,
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
