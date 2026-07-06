from __future__ import annotations

import time
from functools import partial
from typing import Optional

from gfd_demo_runner import GfdRunConfig, run_demo
from ppi_loader import build_ppi_seed_pattern, load_ppi_csv


DATA_DIR = "D:/CodeWork/python/GAR+/\u6570\u636e/\u53bb\u75c5\u56fe\u6570\u636e/\u53bb\u75c5\u56fe\u6570\u636e"
CSV_PATH: Optional[str] = f"{DATA_DIR}/protein_protein_signed.csv"
PROTEIN_CSV_PATH: Optional[str] = f"{DATA_DIR}/protein.csv"
AUTO_DISCOVER_IF_MISSING = False


CONFIG = GfdRunConfig(
    dataset_name="PPI",
    mode="gfd",
    interaction_csv_path=CSV_PATH,
    node_csv_path=PROTEIN_CSV_PATH,
    auto_discover_if_missing=AUTO_DISCOVER_IF_MISSING,
    csv_graph_loader=partial(load_ppi_csv, edge_label_column="Experimental System", force_edge_label="candidate_interaction"),
    seed_builder=build_ppi_seed_pattern,
    fallback_interaction_name="protein_protein_signed.csv",
    fallback_node_name="protein.csv",
    y_key="e0.interaction_label",
    max_rows=None,
    undirected=True,
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
    global_vspawn_instances=False,
    topology_only_pattern_dedup=True,
    topology_dedupe_respect_direction=False,
    pattern_extension_debug=False,
    pattern_extension_debug_limit=500,
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
    print_dependency_limit=10,
)


def main() -> None:
    run_demo(CONFIG)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running cost:", time.time() - start_time)
