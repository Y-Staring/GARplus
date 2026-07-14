"""Batch-build GNN edge CSVs from GAR+ ablation negative-edge exports.

This script reuses each dataset's existing ``build_three_gnn_edge_csv.py``
logic, but drives it over many ablation variants. The expected negative-edge
input is the GARplusMiner export layout, for example:

    GARplusMiner/ablation_negative_edges_only/ti/full/rule_negative_pairs_*.csv

Outputs are written per dataset/variant and contain:

    baseline_edges.csv
    llm_augmented_edges.csv
    gar_augmented_edges.csv
    unified_node.csv
    dataset_stats.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType


DEFAULT_REPO_ROOT = Path("/home/yyyy/codework/GARplus")
DEFAULT_DATASETS = ("TI",)
DEFAULT_VARIANTS = (
    "full",
    "wo_order_embedding",
    "wo_bayesian_pruning",
    "wo_logicgar",
    "wo_neuralgar",
    "logicgar_only",
    "neuralgar_only",
)


DATASET_SETTINGS = {
    "TI": {
        "test_dir": "TI_test",
        "llm_src_type": "Gene",
        "llm_dst_type": "Disease",
        "gar_src_type": "Gene",
        "gar_dst_type": "Disease",
    },
    "DDA": {
        "test_dir": "DDA_test",
        "llm_src_type": "Drug",
        "llm_dst_type": "Disease",
        "gar_src_type": "Drug",
        "gar_dst_type": "Disease",
    },
    "PPI": {
        "test_dir": "PPI_test",
        "llm_src_type": "Protein",
        "llm_dst_type": "Protein",
        "gar_src_type": "Protein",
        "gar_dst_type": "Protein",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ablation GNN edge files from exported negative-edge CSVs."
    )
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=sorted(DATASET_SETTINGS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--negative-root",
        default=None,
        help=(
            "Root of GARplusMiner negative-edge exports. Defaults to "
            "<repo-root>/GARplusMiner/ablation_negative_edges_only."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root. Defaults to "
            "<repo-root>/experiments/ablation_gnn_batch/built_edges."
        ),
    )
    parser.add_argument(
        "--negative-file-name",
        default=None,
        help="Exact negative-edge CSV filename under each variant directory. If omitted, auto-detect.",
    )
    parser.add_argument(
        "--negative-glob",
        default="rule_negative_pairs*.csv",
        help="Glob used when --negative-file-name is omitted.",
    )
    parser.add_argument(
        "--negative-sampling-strategy",
        default="all",
        choices=("all", "random", "source_stratified"),
        help="Passed through to build_three_gnn_edge_csv.py.",
    )
    parser.add_argument(
        "--gar-only",
        action="store_true",
        help="Only write gar_augmented_edges.csv plus node/stats files; skip baseline and LLM edge files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the detected jobs and write no output edge files.",
    )
    return parser.parse_args()


def load_builder(builder_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"gnn_edge_builder_{builder_path.parent.name}", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load builder module: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_negative_csv(variant_dir: Path, exact_name: str | None, pattern: str) -> Path | None:
    if exact_name:
        path = variant_dir / exact_name
        return path if path.is_file() else None
    candidates = sorted(
        path for path in variant_dir.glob(pattern)
        if path.is_file() and path.name != "batch_negative_expansion_summary.csv"
    )
    if not candidates:
        return None
    preferred = [
        path for path in candidates
        if "anchored_existing_edge_labeling" in path.name or "existing_edge_labeling" in path.name
    ]
    return preferred[0] if preferred else candidates[0]


def configure_builder(
    module: ModuleType,
    dataset: str,
    repo_root: Path,
    negative_csv: Path,
    output_dir: Path,
    strategy: str,
    seed: int,
) -> None:
    settings = DATASET_SETTINGS[dataset]
    test_root = repo_root / "experiments" / "exp1_accuracy" / settings["test_dir"]

    module.LLM_EDGE_PATH = str(test_root / "data_signed" / "edges_labeled_with_reason.csv")
    module.NODE_CSV_PATH = str(test_root / "data_signed" / "node_labeled.csv")
    module.GAR_NEG_EDGE_PATH = str(negative_csv)
    module.OUTPUT_DIR = str(output_dir)
    module.RANDOM_SEED = seed
    module.NEGATIVE_SAMPLING_STRATEGY = strategy
    module.BUILD_TYPED_UNIFIED_NODE_CSV = True
    module.UNIFIED_NODE_CSV_NAME = "unified_node.csv"
    module.LLM_NODE_ID_COL = "node_id"
    module.LLM_CANONICAL_ID_COL = "old_index"
    module.LLM_SRC_NODE_TYPE = settings["llm_src_type"]
    module.LLM_DST_NODE_TYPE = settings["llm_dst_type"]
    module.GAR_SRC_NODE_TYPE = settings["gar_src_type"]
    module.GAR_DST_NODE_TYPE = settings["gar_dst_type"]
    module.GAR_SRC_COL = None
    module.GAR_DST_COL = None


def build_gar_only(module: ModuleType) -> None:
    llm_path = module.require_input_file(module.LLM_EDGE_PATH, "LLM_EDGE_PATH")
    gar_path = module.require_input_file(module.GAR_NEG_EDGE_PATH, "GAR_NEG_EDGE_PATH")
    node_path = module.require_input_file(module.NODE_CSV_PATH, "NODE_CSV_PATH")
    output_dir = Path(module.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_positive_raw, llm_negative_raw = module.read_llm_edges(llm_path)
    gar_negative_raw = module.read_gar_negative_edges(gar_path)
    llm_positive_raw_count = len(llm_positive_raw)
    llm_negative_raw_count = len(llm_negative_raw)
    gar_negative_raw_count = len(gar_negative_raw)

    llm_node_id_column = module.LLM_NODE_ID_COL or module.NODE_ID_COL
    gar_node_id_column = module.GAR_NODE_ID_COL or module.NODE_ID_COL
    unified_node_path = None
    gar_missing_src = gar_missing_dst = 0

    if module.BUILD_TYPED_UNIFIED_NODE_CSV:
        (
            llm_positive_raw,
            llm_negative_raw,
            gar_negative_raw,
            unified_node_rows,
            missing_counts,
        ) = module.build_typed_unified_edges(node_path, llm_positive_raw, llm_negative_raw, gar_negative_raw)
        num_nodes = len(unified_node_rows)
        unified_node_path = output_dir / module.UNIFIED_NODE_CSV_NAME
        module.write_unified_node_csv(unified_node_path, unified_node_rows)
        llm_positive_missing_node = missing_counts["llm_positive"]
        llm_negative_missing_node = missing_counts["llm_negative"]
        gar_negative_missing_node = 0
    else:
        llm_id_map, num_nodes = module.load_node_id_map(node_path, llm_node_id_column)
        gar_id_map, gar_num_nodes = module.load_node_id_map(node_path, gar_node_id_column)
        if gar_num_nodes != num_nodes:
            raise AssertionError("Node mapping views disagree on node_labeled row count.")
        llm_positive_raw, llm_positive_missing_node, _src, _dst = module.map_edges_to_dgl_ids(
            llm_positive_raw, llm_id_map
        )
        llm_negative_raw, llm_negative_missing_node, _src, _dst = module.map_edges_to_dgl_ids(
            llm_negative_raw, llm_id_map
        )
        gar_negative_raw, gar_negative_missing_node, gar_missing_src, gar_missing_dst = module.map_edges_to_dgl_ids(
            gar_negative_raw,
            gar_id_map,
            src_offset=module.GAR_SRC_NODE_ID_OFFSET,
            dst_offset=module.GAR_DST_NODE_ID_OFFSET,
        )

    positive_by_key = module.clean_and_dedupe(llm_positive_raw)
    llm_negative_by_key = module.clean_and_dedupe(llm_negative_raw)
    gar_negative_by_key = module.clean_and_dedupe(gar_negative_raw)
    llm_positive_after_dedup = len(positive_by_key)
    llm_negative_after_dedup = len(llm_negative_by_key)
    gar_negative_after_dedup = len(gar_negative_by_key)

    positive_keys = set(positive_by_key)
    gar_conflicts = set(gar_negative_by_key) & positive_keys
    for key in gar_conflicts:
        del gar_negative_by_key[key]

    if not positive_by_key or not gar_negative_by_key:
        empty = []
        if not positive_by_key:
            empty.append("LLM positive")
        if not gar_negative_by_key:
            empty.append("GAR negative")
        raise ValueError(f"Cannot build GAR-only dataset because these cleaned edge sets are empty: {', '.join(empty)}")

    rng = module.random.Random(module.RANDOM_SEED)
    shared_positive = list(positive_by_key.values())
    use_all_negative_edges = module.NEGATIVE_SAMPLING_STRATEGY == "all"
    if use_all_negative_edges:
        sampled_gar_negative = list(gar_negative_by_key.values())
        matched_negative_count = min(len(llm_negative_by_key), len(gar_negative_by_key))
    else:
        matched_negative_count = min(len(llm_negative_by_key), len(gar_negative_by_key))
        sampled_gar_negative = module.sample_negative_edges(
            gar_negative_by_key.values(), matched_negative_count, rng, "GAR negative"
        )

    gar_rows = module.labeled_rows(shared_positive, sampled_gar_negative)
    _gar_positive, gar_negative_count = module.validate_output_rows(gar_rows, "gar_augmented_edges.csv")
    module.assert_valid_node_ids(gar_rows, num_nodes, "gar_augmented_edges.csv")
    module.write_edge_csv(output_dir / "gar_augmented_edges.csv", gar_rows)

    stats = {
        "gar_only": True,
        "num_llm_positive_raw": llm_positive_raw_count,
        "num_llm_negative_raw": llm_negative_raw_count,
        "num_gar_negative_raw": gar_negative_raw_count,
        "num_llm_positive_filtered_missing_node": llm_positive_missing_node,
        "num_llm_negative_filtered_missing_node": llm_negative_missing_node,
        "num_gar_negative_filtered_missing_node": gar_negative_missing_node,
        "num_llm_positive_after_dedup": llm_positive_after_dedup,
        "num_llm_negative_after_dedup": llm_negative_after_dedup,
        "num_gar_negative_after_dedup": gar_negative_after_dedup,
        "num_gar_negative_conflict_with_positive_removed": len(gar_conflicts),
        "n_shared": len(shared_positive),
        "n_negative_per_setting": matched_negative_count if not use_all_negative_edges else None,
        "matched_negative_count": matched_negative_count,
        "gar_augmented": {"positive": len(shared_positive), "negative": gar_negative_count},
        "directed": module.DIRECTED,
        "random_seed": module.RANDOM_SEED,
        "negative_sampling_strategy": module.NEGATIVE_SAMPLING_STRATEGY,
        "sampled_gar_negative_src_top": module.source_distribution_summary(sampled_gar_negative),
        "num_nodes": num_nodes,
        "llm_node_id_column": llm_node_id_column,
        "gar_node_id_column": gar_node_id_column,
        "gar_src_node_id_offset": module.GAR_SRC_NODE_ID_OFFSET,
        "gar_dst_node_id_offset": module.GAR_DST_NODE_ID_OFFSET,
        "num_gar_negative_src_filtered_missing_node": gar_missing_src,
        "num_gar_negative_dst_filtered_missing_node": gar_missing_dst,
        "typed_unified_node_mapping": module.BUILD_TYPED_UNIFIED_NODE_CSV,
        "unified_node_csv": str(unified_node_path) if unified_node_path else None,
    }
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"[Output] gar_augmented_edges.csv: label1={len(shared_positive)}, label2={gar_negative_count}",
        flush=True,
    )
    print(f"[Done] GAR-only files saved to {output_dir}", flush=True)


def run_one(
    dataset: str,
    variant: str,
    repo_root: Path,
    negative_csv: Path | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "variant": variant,
        "status": "ok",
        "negative_csv": str(negative_csv) if negative_csv else "",
        "output_dir": str(output_dir),
        "error": "",
    }
    if negative_csv is None:
        row.update(status="missing_negative_csv", error="no matching negative-edge CSV")
        return row

    expected_output = output_dir / "gar_augmented_edges.csv"
    if expected_output.exists() and not args.overwrite:
        row.update(status="skipped_existing_output")
        return row

    if args.dry_run:
        row.update(status="dry_run")
        return row

    builder_path = (
        repo_root
        / "experiments"
        / "exp1_accuracy"
        / DATASET_SETTINGS[dataset]["test_dir"]
        / "build_three_gnn_edge_csv.py"
    )
    if not builder_path.is_file():
        row.update(status="missing_builder", error=f"missing {builder_path}")
        return row

    print(
        f"[BuildAblationGNN] dataset={dataset} variant={variant} "
        f"negative_csv={negative_csv} output_dir={output_dir}",
        flush=True,
    )
    try:
        module = load_builder(builder_path)
        configure_builder(
            module=module,
            dataset=dataset,
            repo_root=repo_root,
            negative_csv=negative_csv,
            output_dir=output_dir,
            strategy=args.negative_sampling_strategy,
            seed=args.seed,
        )
        if args.gar_only:
            build_gar_only(module)
        else:
            module.main()
        stats_path = output_dir / "dataset_stats.json"
        if stats_path.is_file():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            row.update(
                num_nodes=stats.get("num_nodes"),
                positives=stats.get("baseline", {}).get("positive"),
                gar_negatives=stats.get("gar_augmented", {}).get("negative"),
                llm_negatives=stats.get("llm_augmented", {}).get("negative"),
                baseline_negatives=stats.get("baseline", {}).get("negative"),
            )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        print(row["traceback"], flush=True)
    return row


def write_summary(rows: list[dict[str, object]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "build_summary.csv"
    preferred = [
        "dataset",
        "variant",
        "status",
        "num_nodes",
        "positives",
        "baseline_negatives",
        "llm_negatives",
        "gar_negatives",
        "negative_csv",
        "output_dir",
        "error",
    ]
    fieldnames = [key for key in preferred if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fieldnames and key != "traceback":
                fieldnames.append(key)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    return summary_path


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    negative_root = (
        Path(args.negative_root).expanduser().resolve()
        if args.negative_root
        else repo_root / "GARplusMiner" / "ablation_negative_edges_only"
    )
    if args.negative_root is None and not negative_root.exists():
        fallback_negative_root = (
            repo_root / "enumeration-discovery" / "GARplusMiner" / "ablation_negative_edges_only"
        )
        if fallback_negative_root.exists():
            negative_root = fallback_negative_root
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else repo_root / "experiments" / "ablation_gnn_batch" / "built_edges"
    )

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        dataset_key = dataset.lower()
        for variant in args.variants:
            variant_dir = negative_root / dataset_key / variant
            negative_csv = find_negative_csv(variant_dir, args.negative_file_name, args.negative_glob)
            output_dir = output_root / dataset_key / variant
            rows.append(run_one(dataset, variant, repo_root, negative_csv, output_dir, args))

    summary_path = write_summary(rows, output_root)
    print(f"[Done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
