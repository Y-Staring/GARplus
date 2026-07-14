from __future__ import annotations

import argparse
import csv
import time
import traceback
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
REFINEMENT_DIR = REPO_ROOT / "experiments" / "exp1_accuracy" / "deductive_refinement"
if str(REFINEMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REFINEMENT_DIR))

from negative_edge_expander import DATASET_CONFIGS, expand_negative_edges


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ABLATION_ROOT = ROOT_DIR.parent / "miner" / "ablation_results"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "ablation_negative_edges_only"
NEGATIVE_ONLY = True

DEFAULT_DATASETS = ("PPI", "DDA", "TI")
DEFAULT_VARIANTS = (
    "full",
    "wo_order_embedding",
    "wo_bayesian_pruning",
    "wo_logicgar",
    "wo_neuralgar",
    "logicgar_only",
    "neuralgar_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export negative edges for rules produced by ablation runs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=sorted(DATASET_CONFIGS),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Ablation variants to process.",
    )
    parser.add_argument(
        "--ablation-root",
        default=str(DEFAULT_ABLATION_ROOT),
        help="Root directory containing ablation_results/{dataset}/{variant}.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for exported negative edge CSV files.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory containing interaction/node CSVs. If set, overrides the "
            "paths inherited from negative_edge_expander.py. Example: "
            "/home/yangsiyi10504/baselines/去病图数据"
        ),
    )
    parser.add_argument(
        "--scope-input-csv",
        default=None,
        help=(
            "Interaction CSV used as the expansion/search scope. This overrides "
            "only input_csv, leaving node CSV paths from --data-dir or the default "
            "dataset config unchanged. Useful for running anchored labeling on an "
            "unlabeled large graph such as protein_protein.csv."
        ),
    )
    parser.add_argument(
        "--mode",
        default="anchored_existing_edge_labeling",
        choices=(
            "anchored_existing_edge_labeling",
            "existing_edge_labeling",
            "matched_existing",
            "candidate_non_edges",
            "body_rematch_non_edges",
        ),
        help="negative_edge_expander expansion mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if the target CSV already exists.",
    )
    parser.add_argument(
        "--debug-progress",
        action="store_true",
        help="Enable verbose progress logs inside negative_edge_expander.",
    )
    parser.add_argument(
        "--allow-positive-relabel",
        action="store_true",
        help="Allow labeling existing positive rows as negative.",
    )
    parser.add_argument(
        "--allow-existing-negative-relabel",
        action="store_true",
        help="Allow exporting rows that are already negative.",
    )
    parser.add_argument(
        "--only-labels",
        nargs="*",
        default=None,
        help=(
            "Labels eligible for relabeling. Omit to use the expander default; "
            "pass an empty list by setting --only-labels with no values."
        ),
    )
    return parser.parse_args()


def dataset_data_paths(dataset: str, data_dir: Path) -> dict[str, Path]:
    if dataset == "PPI":
        return {
            "input_csv": data_dir / "protein_protein_signed.csv",
            "source_node_csv": data_dir / "protein.csv",
            "target_node_csv": data_dir / "protein.csv",
        }
    if dataset == "DDA":
        return {
            "input_csv": data_dir / "drug_disease_signed.csv",
            "source_node_csv": data_dir / "drug.csv",
            "target_node_csv": data_dir / "disease.csv",
        }
    if dataset == "TI":
        return {
            "input_csv": data_dir / "gene_disease_signed.csv",
            "source_node_csv": data_dir / "gene.csv",
            "target_node_csv": data_dir / "disease.csv",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def variant_paths(
    ablation_root: Path,
    output_root: Path,
    dataset: str,
    variant: str,
    mode: str,
) -> tuple[Path, Path, Path]:
    variant_dir = ablation_root / dataset.lower() / variant
    rules_file = variant_dir / "deduped_rules.txt"
    pattern_instances_file = variant_dir / "pattern_instances.jsonl"
    output_csv = output_root / dataset.lower() / variant / f"negative_edges_only_{mode}.csv"
    return rules_file, pattern_instances_file, output_csv


def run_one(
    dataset: str,
    variant: str,
    rules_file: Path,
    pattern_instances_file: Path,
    output_csv: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()
    row: dict[str, object] = {
        "dataset": dataset,
        "variant": variant,
        "mode": args.mode,
        "negative_only": int(NEGATIVE_ONLY),
        "rules_file": str(rules_file),
        "pattern_instances_file": str(pattern_instances_file),
        "output_csv": str(output_csv),
        "status": "ok",
        "error": "",
    }

    if not rules_file.exists():
        row.update(status="missing_rules", error=f"missing {rules_file}")
        return row
    if args.mode in {"anchored_existing_edge_labeling", "matched_existing", "candidate_non_edges", "body_rematch_non_edges"}:
        if not pattern_instances_file.exists():
            row.update(status="missing_pattern_instances", error=f"missing {pattern_instances_file}")
            return row
    if output_csv.exists() and not args.overwrite:
        row.update(status="skipped_existing_output")
        return row

    base_config = DATASET_CONFIGS[dataset]
    path_overrides = {}
    if args.data_dir:
        path_overrides = dataset_data_paths(dataset, Path(args.data_dir))
        missing_paths = [path for path in path_overrides.values() if path is not None and not path.exists()]
        if missing_paths:
            row.update(
                status="missing_data",
                error="; ".join(f"missing {path}" for path in missing_paths),
            )
            return row
    if args.scope_input_csv:
        path_overrides["input_csv"] = Path(args.scope_input_csv)
        if not path_overrides["input_csv"].exists():
            row.update(
                status="missing_scope_input",
                error=f"missing {path_overrides['input_csv']}",
            )
            return row
    only_labels = base_config.only_labels if args.only_labels is None else set(args.only_labels)
    config = replace(
        base_config,
        **path_overrides,
        rules_file=rules_file,
        pattern_instances_file=pattern_instances_file,
        output_csv=output_csv,
        negative_value="negative",
        expansion_mode=args.mode,
        debug_progress=args.debug_progress,
        allow_positive_relabel=args.allow_positive_relabel,
        allow_existing_negative_relabel=args.allow_existing_negative_relabel,
        only_labels=only_labels,
    )

    print(
        "[BatchNegativeExpansion] "
        f"dataset={dataset} variant={variant} mode={args.mode} "
        f"rules={rules_file} output={output_csv}",
        flush=True,
    )
    try:
        summary = expand_negative_edges(config)
        row.update(summary)
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        print(row["traceback"], flush=True)
    finally:
        row["seconds"] = f"{time.perf_counter() - started:.6f}"
    return row


def write_summary(rows: list[dict[str, object]], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_negative_expansion_summary.csv"
    fieldnames: list[str] = []
    seen = set()
    preferred = [
        "dataset",
        "variant",
        "mode",
        "negative_only",
        "status",
        "seconds",
        "rows",
        "rules",
        "usable_rules",
        "checked_rows",
        "matched_rows",
        "exported_rows",
        "exported_pairs",
        "inferred_new_negative_pairs",
        "skipped_positive",
        "skipped_existing_negative",
        "skipped_label_not_allowed",
        "skipped_structural_rule",
        "skipped_no_schema",
        "skipped_not_expandable",
        "rules_file",
        "pattern_instances_file",
        "output_csv",
        "error",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen and key != "traceback":
                fieldnames.append(key)
                seen.add(key)

    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def main() -> None:
    args = parse_args()
    ablation_root = Path(args.ablation_root)
    output_root = Path(args.output_root)

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for variant in args.variants:
            rules_file, pattern_instances_file, output_csv = variant_paths(
                ablation_root,
                output_root,
                dataset,
                variant,
                args.mode,
            )
            rows.append(
                run_one(
                    dataset=dataset,
                    variant=variant,
                    rules_file=rules_file,
                    pattern_instances_file=pattern_instances_file,
                    output_csv=output_csv,
                    args=args,
                )
            )

    summary_path = write_summary(rows, output_root)
    ok = sum(1 for row in rows if row.get("status") == "ok")
    skipped = sum(1 for row in rows if str(row.get("status", "")).startswith("skipped"))
    missing = sum(1 for row in rows if str(row.get("status", "")).startswith("missing"))
    errors = sum(1 for row in rows if row.get("status") == "error")
    print(
        "[BatchNegativeExpansionSummary] "
        f"total={len(rows)} ok={ok} skipped={skipped} missing={missing} "
        f"errors={errors} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
