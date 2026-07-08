from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable

from garplus_demo_runner import GarplusRunConfig, run_demo
from garplus_ml_predicates import MLPredicateConfig

from ppi_demo import CONFIG as PPI_CONFIG
from ppi_loader import build_ppi_seed_pattern
from dda_demo import CONFIG as DDA_CONFIG
from ti_demo import CONFIG as TI_CONFIG


DATASETS: dict[str, GarplusRunConfig] = {
    "PPI": PPI_CONFIG,
    "DDA": DDA_CONFIG,
    "TI": TI_CONFIG,
}

ABLATIONS = (
    "full",
    "wo_order_embedding",
    "wo_bayesian_pruning",
    "wo_logicgar",
    "wo_neuralgar",
    "logicgar_only",
    "neuralgar_only",
)

RESULT_DIR = Path(__file__).resolve().parent / "ablation_results"

# =========================
# Active run parameters
# =========================
# 每次只跑一个组合，方便开多个终端并行跑。
# 你只需要改下面两行；可选值见上面的 DATASETS / ABLATIONS。
#
# Examples:
# ACTIVE_DATASET = "PPI"
# ACTIVE_ABLATION = "full"
#
# ACTIVE_DATASET = "DDA"
# ACTIVE_ABLATION = "wo_bayesian_pruning"
#
# ACTIVE_DATASET = "TI"
# ACTIVE_ABLATION = "wo_neuralgar"
ACTIVE_DATASET = "PPI"
ACTIVE_ABLATION = "full"

# Keep the runner comparable and bounded. Set to None if you want each dataset's
# original pattern-size limit.
MAX_PATTERN_NODES = 4

# NeuralGAR-only is a strict allowlist: keep the target label plus ML-derived
# predicates.  Everything else, including v*/e1*/e0 original attributes, is
# removed before rule mining.
ML_PREDICATE_KEEP_TOKENS = (
    "ml_",
    "similarity",
    "equivalence",
)

# LogicGAR-only keeps structural and attribute predicates but removes all
# ML-derived predicates, including precomputed attributes left in input files.
ML_PREDICATE_DROP_TOKENS = (
    "ml_",
    "similarity",
    "equivalence",
)

SUMMARY_RE = re.compile(
    r"\[Summary\]\s+dataset=(?P<dataset>\S+)\s+patterns_mined=(?P<patterns>\d+)\s+"
    r"raw_rules=(?P<raw>\d+)\s+deduped_rules=(?P<deduped>\d+)\s+"
    r"positive_rules=(?P<positive>\d+)\s+negative_rules=(?P<negative>\d+)"
)
RULE_TIMING_RE = re.compile(r"\[Timing\]\s+stage=rule_mining_total.*?seconds=(?P<seconds>[0-9.]+)")
VSPAWN_STATS_RE = re.compile(
    r"\[VSpawnStats\].*?candidates_seen=(?P<candidates>\d+).*?"
    r"bn_pruned=(?P<bn>\d+).*?duplicate_pruned=(?P<duplicate>\d+).*?"
    r"constraint_pruned=(?P<constraint>\d+).*?no_match_pruned=(?P<nomatch>\d+).*?"
    r"support_pruned=(?P<support>\d+)"
)
PATTERN_BN_RE = re.compile(r"\[PatternBN\].*?seen=(?P<seen>\d+)\s+kept=(?P<kept>\d+)\s+pruned=(?P<pruned>\d+)")
PREDICATE_BN_RE = re.compile(r"\[PredicateBN\].*?seen=(?P<seen>\d+)\s+kept=(?P<kept>\d+)\s+pruned=(?P<pruned>\d+)")
PATTERN_INSTANCES_RE = re.compile(r"\[PatternInstancesSummary\].*?total_instances=(?P<instances>\d+)\s+patterns=(?P<patterns>\d+)")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _merge_tokens(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen = set()
    for group in groups:
        for token in group:
            if token and token not in seen:
                merged.append(token)
                seen.add(token)
    return tuple(merged)


def _disable_ml(config: GarplusRunConfig) -> MLPredicateConfig:
    return replace(
        config.ml_predicates,
        enabled=False,
        equivalence_enabled=False,
        similarity_enabled=False,
        offline_enabled=False,
    )


def _base_config(dataset: str, config: GarplusRunConfig, variant: str) -> GarplusRunConfig:
    out_dir = RESULT_DIR / dataset.lower() / variant
    common = {
        "deduped_rules_output_path": str(out_dir / "deduped_rules.txt"),
        "pattern_instances_output_path": str(out_dir / "pattern_instances.jsonl"),
        "pattern_extension_debug": False,
        "debug_literal_keys": False,
        "debug_match_expansion": True,
        "debug_transaction_cost": True,
        "debug_sample_matches": 0,
        "print_rule_limit": 0,
        "print_deduped_rule_limit": 20,
    }
    if MAX_PATTERN_NODES is not None:
        common["max_pattern_nodes"] = MAX_PATTERN_NODES
    return replace(config, **common)


def full_config(dataset: str, config: GarplusRunConfig) -> GarplusRunConfig:
    return _base_config(dataset, config, "full")


def wo_order_embedding_config(dataset: str, config: GarplusRunConfig) -> GarplusRunConfig:
    cfg = _base_config(dataset, config, "wo_order_embedding")
    updates = {
        "use_sampled_pt_graph": False,
        # Once sampled/order-embedding input is disabled, the CSV loader must
        # see the real graph.  Keeping demo max_rows=50 leaves too few edges
        # after neutral/unknown filtering and prevents any pattern expansion.
        "max_rows": None,
        # DDA/TI configs often provide only verification_graph_loader because
        # the full model loads sampled .pt graphs first.  Once order-embedding
        # sampling is disabled, load_graph needs csv_graph_loader, so reuse the
        # verification loader when needed.
        "csv_graph_loader": cfg.csv_graph_loader or cfg.verification_graph_loader,
        "inject_sampled_frequent_patterns": False,
        "enable_sampled_frequent_patterns": False,
        "global_match_scope": "sampled",
    }
    if dataset == "PPI":
        updates["seed_builder"] = build_ppi_seed_pattern
    return replace(cfg, **updates)


def wo_bayesian_pruning_config(dataset: str, config: GarplusRunConfig) -> GarplusRunConfig:
    return replace(
        _base_config(dataset, config, "wo_bayesian_pruning"),
        enable_pattern_bn=False,
        enable_predicate_bn=False,
    )


def neuralgar_only_config(dataset: str, config: GarplusRunConfig, variant: str = "neuralgar_only") -> GarplusRunConfig:
    cfg = _base_config(dataset, config, variant)
    return replace(
        cfg,
        # NeuralGAR: only ML predicates M(x,y) are allowed as antecedents.
        # The target label y_key is always preserved by the selector.
        ml_predicates=replace(cfg.ml_predicates, enabled=True),
        filter_degree_predicates=True,
        ignored_predicate_key_tokens=(),
        kept_predicate_key_tokens=ML_PREDICATE_KEEP_TOKENS,
    )


def logicgar_only_config(dataset: str, config: GarplusRunConfig, variant: str = "logicgar_only") -> GarplusRunConfig:
    cfg = _base_config(dataset, config, variant)
    return replace(
        cfg,
        # LogicGAR: keep structural and symbolic attribute predicates, but
        # remove all ML/similarity/equivalence predicates.
        ml_predicates=_disable_ml(cfg),
        ignored_predicate_key_tokens=_merge_tokens(cfg.ignored_predicate_key_tokens, ML_PREDICATE_DROP_TOKENS),
        kept_predicate_key_tokens=(),
        filter_degree_predicates=True,
    )


VARIANT_BUILDERS: dict[str, Callable[[str, GarplusRunConfig], GarplusRunConfig]] = {
    "full": full_config,
    "wo_order_embedding": wo_order_embedding_config,
    "wo_bayesian_pruning": wo_bayesian_pruning_config,
    # Backward-compatible names used by the paper table:
    # w/o LogicGAR leaves NeuralGAR-only predicates.
    "wo_logicgar": lambda dataset, config: neuralgar_only_config(dataset, config, "wo_logicgar"),
    # w/o NeuralGAR leaves LogicGAR-only predicates.
    "wo_neuralgar": lambda dataset, config: logicgar_only_config(dataset, config, "wo_neuralgar"),
    # Clear aliases for standalone runs.
    "logicgar_only": logicgar_only_config,
    "neuralgar_only": neuralgar_only_config,
}


def parse_log(log_text: str) -> dict[str, str]:
    row: dict[str, str] = {}
    if match := SUMMARY_RE.search(log_text):
        row.update(
            {
                "patterns_mined": match.group("patterns"),
                "raw_rules": match.group("raw"),
                "deduped_rules": match.group("deduped"),
                "positive_rules": match.group("positive"),
                "negative_rules": match.group("negative"),
            }
        )
    if match := RULE_TIMING_RE.search(log_text):
        row["rule_mining_seconds"] = match.group("seconds")
    if matches := list(VSPAWN_STATS_RE.finditer(log_text)):
        last = matches[-1]
        row.update(
            {
                "vspawn_candidates_seen": last.group("candidates"),
                "vspawn_bn_pruned": last.group("bn"),
                "vspawn_duplicate_pruned": last.group("duplicate"),
                "vspawn_constraint_pruned": last.group("constraint"),
                "vspawn_no_match_pruned": last.group("nomatch"),
                "vspawn_support_pruned": last.group("support"),
            }
        )
    pattern_seen = pattern_kept = pattern_pruned = 0
    for match in PATTERN_BN_RE.finditer(log_text):
        pattern_seen += int(match.group("seen"))
        pattern_kept += int(match.group("kept"))
        pattern_pruned += int(match.group("pruned"))
    row.update(
        {
            "pattern_bn_seen": str(pattern_seen),
            "pattern_bn_kept": str(pattern_kept),
            "pattern_bn_pruned": str(pattern_pruned),
        }
    )
    predicate_seen = predicate_kept = predicate_pruned = 0
    for match in PREDICATE_BN_RE.finditer(log_text):
        predicate_seen += int(match.group("seen"))
        predicate_kept += int(match.group("kept"))
        predicate_pruned += int(match.group("pruned"))
    row.update(
        {
            "predicate_bn_seen": str(predicate_seen),
            "predicate_bn_kept": str(predicate_kept),
            "predicate_bn_pruned": str(predicate_pruned),
        }
    )
    if match := PATTERN_INSTANCES_RE.search(log_text):
        row["pattern_instances"] = match.group("instances")
    return row


def run_one(dataset: str, variant: str, base: GarplusRunConfig) -> dict[str, str]:
    cfg = VARIANT_BUILDERS[variant](dataset, base)
    out_dir = RESULT_DIR / dataset.lower() / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    print(f"\n=== AblationRun dataset={dataset} variant={variant} ===")
    started = time.perf_counter()
    status = "ok"
    error = ""
    buffer = io.StringIO()
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file, buffer)
        try:
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                run_demo(cfg)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                traceback.print_exc()

    wall_seconds = time.perf_counter() - started
    parsed = parse_log(buffer.getvalue())
    row = {
        "dataset": dataset,
        "variant": variant,
        "status": status,
        "error": error,
        "wall_seconds": f"{wall_seconds:.6f}",
        "accuracy": "",
        "log_path": str(log_path),
        "deduped_rules_path": str(out_dir / "deduped_rules.txt"),
        "pattern_instances_path": str(out_dir / "pattern_instances.jsonl"),
    }
    row.update(parsed)
    print(
        f"[AblationTiming] dataset={dataset} variant={variant} wall_seconds={wall_seconds:.6f} "
        f"rules={row.get('deduped_rules', 'NA')} patterns={row.get('patterns_mined', 'NA')} "
        f"status={status} log={log_path}"
    )
    return row


def main() -> None:
    global MAX_PATTERN_NODES

    parser = argparse.ArgumentParser(description="Run one GARplus ablation setting.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("GARPLUS_ABLATION_DATASET", ACTIVE_DATASET),
        choices=sorted(DATASETS),
        help="Dataset to run. Can also be set by GARPLUS_ABLATION_DATASET.",
    )
    parser.add_argument(
        "--ablation",
        default=os.environ.get("GARPLUS_ABLATION_VARIANT", ACTIVE_ABLATION),
        choices=ABLATIONS,
        help="Ablation variant to run. Can also be set by GARPLUS_ABLATION_VARIANT.",
    )
    parser.add_argument(
        "--max-pattern-nodes",
        default=os.environ.get(
            "GARPLUS_ABLATION_MAX_PATTERN_NODES",
            "" if MAX_PATTERN_NODES is None else str(MAX_PATTERN_NODES),
        ),
        help="Override max_pattern_nodes. Use empty string or 'none' for no override.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    ablation = args.ablation
    max_pattern_nodes_text = str(args.max_pattern_nodes).strip().lower()
    if max_pattern_nodes_text in {"", "none", "null"}:
        MAX_PATTERN_NODES = None
    else:
        MAX_PATTERN_NODES = int(max_pattern_nodes_text)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "[AblationConfig] "
        f"dataset={dataset} ablation={ablation} "
        f"max_pattern_nodes={MAX_PATTERN_NODES}"
    )
    rows = [run_one(dataset, ablation, DATASETS[dataset])]

    csv_path = RESULT_DIR / dataset.lower() / ablation / "ablation_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "variant",
        "accuracy",
        "wall_seconds",
        "rule_mining_seconds",
        "patterns_mined",
        "pattern_instances",
        "raw_rules",
        "deduped_rules",
        "positive_rules",
        "negative_rules",
        "vspawn_candidates_seen",
        "vspawn_bn_pruned",
        "vspawn_duplicate_pruned",
        "vspawn_constraint_pruned",
        "vspawn_no_match_pruned",
        "vspawn_support_pruned",
        "pattern_bn_seen",
        "pattern_bn_kept",
        "pattern_bn_pruned",
        "predicate_bn_seen",
        "predicate_bn_kept",
        "predicate_bn_pruned",
        "status",
        "error",
        "log_path",
        "deduped_rules_path",
        "pattern_instances_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"\n[AblationSummary] wrote={csv_path}")


if __name__ == "__main__":
    main()
