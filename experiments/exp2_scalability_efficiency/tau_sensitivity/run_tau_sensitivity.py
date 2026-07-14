from __future__ import annotations

import contextlib
import csv
import io
import argparse
import os
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MINER_DIR = REPO_ROOT / "enumeration-discovery" / "GARplusMiner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

from BNlearning.bn_config import STRICT_PATTERN_BN, STRICT_PREDICATE_BN
from garplus_demo_runner import GarplusRunConfig, run_demo
from ppi_demo import CONFIG as PPI_CONFIG
from dda_demo import CONFIG as DDA_CONFIG
from ti_demo import CONFIG as TI_CONFIG


DATASETS: dict[str, GarplusRunConfig] = {
    "PPI": PPI_CONFIG,
    "DDA": DDA_CONFIG,
    "TI": TI_CONFIG,
}

# tau_P is swept as the absolute Pattern-BN threshold. To make that effect
# visible, relative_tau is disabled for this sweep; otherwise the strict
# adaptive cutoff max(tau_p, relative_tau * max_score) may dominate.
TAU_P_VALUES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
TAU_X_VALUES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30)
TAU_P_SWEEP_RELATIVE_TAU = None

RESULT_DIR = Path(__file__).resolve().parent / "tau_sensitivity_results"

# Default single-run parameters. Override these from Slurm/CLI.
ACTIVE_DATASET = "DDA"
ACTIVE_SWEEP = "tau_p"
ACTIVE_VALUE = 0.10

# Keep runs bounded and comparable. Set to None to use each demo's original
# max_pattern_nodes.
MAX_PATTERN_NODES = 4
RULE_COVERAGE_SCOPE = "sampled"  # none, sampled, original
RULE_COVERAGE_MAX_INSTANCES = None

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
PATTERN_BN_RE = re.compile(
    r"\[PatternBN\].*?seen=(?P<seen>\d+)\s+kept=(?P<kept>\d+)\s+pruned=(?P<pruned>\d+).*?"
    r"threshold_pruned=(?P<threshold>\d+)\s+topk_pruned=(?P<topk>\d+)\s+min_keep_rescued=(?P<rescued>\d+)"
)
PREDICATE_BN_RE = re.compile(
    r"\[PredicateBN\].*?seen=(?P<seen>\d+)\s+kept=(?P<kept>\d+)\s+pruned=(?P<pruned>\d+)"
)
PREDICATE_DETAIL_RE = re.compile(
    r"tau_x=(?P<tau>[0-9.]+).*?tau_pruned=(?P<tau_pruned>\d+)\s+"
    r"feature_limit_pruned=(?P<limit>\d+)\s+topk_pruned=(?P<topk>\d+)\s+"
    r"min_keep_rescued=(?P<rescued>\d+)"
)
PATTERN_INSTANCES_RE = re.compile(
    r"\[PatternInstancesSummary\].*?total_instances=(?P<instances>\d+)\s+patterns=(?P<patterns>\d+)"
)
RULE_COVERAGE_RE = re.compile(
    r"\[RuleCoverage\]\s+scope=(?P<scope>\S+)\s+"
    r"positive_rules=(?P<positive_rules>\d+)\s+"
    r"positive_covered_edges=(?P<positive_covered>\d+)/(?P<positive_total>\d+)\s+"
    r"positive_edge_recall=(?P<positive_recall>[0-9.]+)\s+"
    r"negative_rules=(?P<negative_rules>\d+)\s+"
    r"negative_covered_edges=(?P<negative_covered>\d+)/(?P<negative_total>\d+)\s+"
    r"negative_edge_recall=(?P<negative_recall>[0-9.]+)\s+"
    r"overall_covered_edges=(?P<overall_covered>\d+)/(?P<overall_total>\d+)\s+"
    r"overall_edge_recall=(?P<overall_recall>[0-9.]+)"
)


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


def base_config(dataset: str, config: GarplusRunConfig, sweep: str, value: float) -> GarplusRunConfig:
    out_dir = RESULT_DIR / dataset.lower() / sweep / f"{value:g}"
    updates = {
        **STRICT_PATTERN_BN,
        **STRICT_PREDICATE_BN,
        "deduped_rules_output_path": str(out_dir / "deduped_rules.txt"),
        "pattern_instances_output_path": str(out_dir / "pattern_instances.jsonl"),
        "pattern_extension_debug": False,
        "debug_literal_keys": False,
        "debug_match_expansion": True,
        "debug_transaction_cost": True,
        "debug_sample_matches": 0,
        "print_rule_limit": 0,
        "print_deduped_rule_limit": 20,
        "rule_coverage_scope": RULE_COVERAGE_SCOPE,
        "rule_coverage_max_instances": RULE_COVERAGE_MAX_INSTANCES,
    }
    if MAX_PATTERN_NODES is not None:
        updates["max_pattern_nodes"] = MAX_PATTERN_NODES
    return replace(config, **updates)


def config_for(dataset: str, config: GarplusRunConfig, sweep: str, value: float) -> GarplusRunConfig:
    cfg = base_config(dataset, config, sweep, value)
    if sweep == "tau_p":
        return replace(
            cfg,
            tau_p=value,
            pattern_bn_relative_tau=TAU_P_SWEEP_RELATIVE_TAU,
            enable_pattern_bn=True,
            enable_predicate_bn=True,
        )
    if sweep == "tau_x":
        return replace(
            cfg,
            tau_x=value,
            enable_pattern_bn=True,
            enable_predicate_bn=True,
        )
    raise ValueError(f"Unsupported sweep: {sweep}")


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
    pattern_threshold = pattern_topk = pattern_rescued = 0
    for match in PATTERN_BN_RE.finditer(log_text):
        pattern_seen += int(match.group("seen"))
        pattern_kept += int(match.group("kept"))
        pattern_pruned += int(match.group("pruned"))
        pattern_threshold += int(match.group("threshold"))
        pattern_topk += int(match.group("topk"))
        pattern_rescued += int(match.group("rescued"))
    row.update(
        {
            "pattern_bn_seen": str(pattern_seen),
            "pattern_bn_kept": str(pattern_kept),
            "pattern_bn_pruned": str(pattern_pruned),
            "pattern_bn_threshold_pruned": str(pattern_threshold),
            "pattern_bn_topk_pruned": str(pattern_topk),
            "pattern_bn_min_keep_rescued": str(pattern_rescued),
        }
    )
    predicate_seen = predicate_kept = predicate_pruned = 0
    for match in PREDICATE_BN_RE.finditer(log_text):
        predicate_seen += int(match.group("seen"))
        predicate_kept += int(match.group("kept"))
        predicate_pruned += int(match.group("pruned"))
    predicate_tau = predicate_limit = predicate_topk = predicate_rescued = 0
    for match in PREDICATE_DETAIL_RE.finditer(log_text):
        predicate_tau += int(match.group("tau_pruned"))
        predicate_limit += int(match.group("limit"))
        predicate_topk += int(match.group("topk"))
        predicate_rescued += int(match.group("rescued"))
    row.update(
        {
            "predicate_bn_seen": str(predicate_seen),
            "predicate_bn_kept": str(predicate_kept),
            "predicate_bn_pruned": str(predicate_pruned),
            "predicate_bn_tau_pruned": str(predicate_tau),
            "predicate_bn_feature_limit_pruned": str(predicate_limit),
            "predicate_bn_topk_pruned": str(predicate_topk),
            "predicate_bn_min_keep_rescued": str(predicate_rescued),
        }
    )
    if match := PATTERN_INSTANCES_RE.search(log_text):
        row["pattern_instances"] = match.group("instances")
    if match := RULE_COVERAGE_RE.search(log_text):
        row.update(
            {
                "rule_coverage_scope": match.group("scope"),
                "coverage_positive_rules": match.group("positive_rules"),
                "coverage_positive_covered_edges": match.group("positive_covered"),
                "coverage_positive_total_edges": match.group("positive_total"),
                "coverage_positive_edge_recall": match.group("positive_recall"),
                "coverage_negative_rules": match.group("negative_rules"),
                "coverage_negative_covered_edges": match.group("negative_covered"),
                "coverage_negative_total_edges": match.group("negative_total"),
                "coverage_negative_edge_recall": match.group("negative_recall"),
                "coverage_overall_covered_edges": match.group("overall_covered"),
                "coverage_overall_total_edges": match.group("overall_total"),
                "coverage_overall_edge_recall": match.group("overall_recall"),
            }
        )
    return row


def ratio(numerator: str, denominator: str) -> str:
    try:
        den = float(denominator)
        if den <= 0:
            return ""
        return f"{float(numerator) / den:.6f}"
    except (TypeError, ValueError):
        return ""


def run_one(dataset: str, base: GarplusRunConfig, sweep: str, value: float) -> dict[str, str]:
    cfg = config_for(dataset, base, sweep, value)
    out_dir = RESULT_DIR / dataset.lower() / sweep / f"{value:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    print(f"\n=== TauSensitivity dataset={dataset} sweep={sweep} value={value:g} ===")
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
        "sweep": sweep,
        "value": f"{value:g}",
        "tau_p": f"{cfg.tau_p:g}",
        "pattern_bn_relative_tau": "" if cfg.pattern_bn_relative_tau is None else f"{cfg.pattern_bn_relative_tau:g}",
        "tau_x": f"{cfg.tau_x:g}",
        "wall_seconds": f"{wall_seconds:.6f}",
        "status": status,
        "error": error,
        "log_path": str(log_path),
        "deduped_rules_path": str(out_dir / "deduped_rules.txt"),
        "pattern_instances_path": str(out_dir / "pattern_instances.jsonl"),
    }
    row.update(parsed)
    row["pattern_bn_prune_rate"] = ratio(row.get("pattern_bn_pruned", ""), row.get("pattern_bn_seen", ""))
    row["predicate_bn_prune_rate"] = ratio(row.get("predicate_bn_pruned", ""), row.get("predicate_bn_seen", ""))
    print(
        f"[TauTiming] dataset={dataset} sweep={sweep} value={value:g} wall_seconds={wall_seconds:.6f} "
        f"rules={row.get('deduped_rules', 'NA')} pattern_prune_rate={row.get('pattern_bn_prune_rate') or 'NA'} "
        f"predicate_prune_rate={row.get('predicate_bn_prune_rate') or 'NA'} status={status}"
    )
    return row


def add_relative_coverage(rows: list[dict[str, str]]) -> None:
    best: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["dataset"], row["sweep"])
        try:
            rules = int(row.get("deduped_rules", "") or 0)
        except ValueError:
            rules = 0
        best[key] = max(best.get(key, 0), rules)
    for row in rows:
        key = (row["dataset"], row["sweep"])
        row["relative_rule_coverage"] = ratio(row.get("deduped_rules", ""), str(best.get(key, 0)))


def main() -> None:
    global MAX_PATTERN_NODES, RULE_COVERAGE_SCOPE, RULE_COVERAGE_MAX_INSTANCES

    parser = argparse.ArgumentParser(description="Run one tau_P/tau_X sensitivity setting.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("GARPLUS_TAU_DATASET", ACTIVE_DATASET),
        choices=sorted(DATASETS),
        help="Dataset to run. Can also be set by GARPLUS_TAU_DATASET.",
    )
    parser.add_argument(
        "--sweep",
        default=os.environ.get("GARPLUS_TAU_SWEEP", ACTIVE_SWEEP),
        choices=("tau_p", "tau_x"),
        help="Which threshold to vary. Can also be set by GARPLUS_TAU_SWEEP.",
    )
    parser.add_argument(
        "--value",
        default=os.environ.get("GARPLUS_TAU_VALUE", str(ACTIVE_VALUE)),
        help="Threshold value for the selected sweep. Can also be set by GARPLUS_TAU_VALUE.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the old full grid over all datasets and all tau values.",
    )
    parser.add_argument(
        "--rule-coverage-scope",
        default=os.environ.get("GARPLUS_TAU_RULE_COVERAGE_SCOPE", RULE_COVERAGE_SCOPE),
        choices=("none", "sampled", "original"),
        help="Where to evaluate mined-rule label coverage recall.",
    )
    parser.add_argument(
        "--rule-coverage-max-instances",
        default=os.environ.get("GARPLUS_TAU_RULE_COVERAGE_MAX_INSTANCES", ""),
        help="Optional cap for coverage rematch instances per pattern. Empty/none means no cap.",
    )
    parser.add_argument(
        "--max-pattern-nodes",
        default=os.environ.get("GARPLUS_TAU_MAX_PATTERN_NODES", "" if MAX_PATTERN_NODES is None else str(MAX_PATTERN_NODES)),
        help="Override max_pattern_nodes. Empty/none keeps dataset default.",
    )
    args = parser.parse_args()

    RULE_COVERAGE_SCOPE = args.rule_coverage_scope
    max_instances_text = str(args.rule_coverage_max_instances).strip().lower()
    RULE_COVERAGE_MAX_INSTANCES = None if max_instances_text in {"", "none", "null"} else int(max_instances_text)
    max_pattern_nodes_text = str(args.max_pattern_nodes).strip().lower()
    MAX_PATTERN_NODES = None if max_pattern_nodes_text in {"", "none", "null"} else int(max_pattern_nodes_text)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset
    sweep = args.sweep
    value = float(args.value)
    print(
        "[TauSensitivityConfig] "
        f"dataset={dataset} sweep={sweep} value={value:g} all={args.all} "
        f"rule_coverage_scope={RULE_COVERAGE_SCOPE} "
        f"rule_coverage_max_instances={RULE_COVERAGE_MAX_INSTANCES} "
        f"max_pattern_nodes={MAX_PATTERN_NODES}"
    )
    rows: list[dict[str, str]] = []
    if args.all:
        for dataset_name, config in DATASETS.items():
            for tau_value in TAU_P_VALUES:
                rows.append(run_one(dataset_name, config, "tau_p", tau_value))
            for tau_value in TAU_X_VALUES:
                rows.append(run_one(dataset_name, config, "tau_x", tau_value))
    else:
        rows.append(run_one(dataset, DATASETS[dataset], sweep, value))

    add_relative_coverage(rows)
    if args.all:
        csv_path = RESULT_DIR / "tau_sensitivity_summary.csv"
    else:
        csv_path = RESULT_DIR / dataset.lower() / sweep / f"{value:g}" / "tau_sensitivity_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "sweep",
        "value",
        "tau_p",
        "pattern_bn_relative_tau",
        "tau_x",
        "wall_seconds",
        "rule_mining_seconds",
        "patterns_mined",
        "pattern_instances",
        "raw_rules",
        "deduped_rules",
        "positive_rules",
        "negative_rules",
        "relative_rule_coverage",
        "rule_coverage_scope",
        "coverage_positive_rules",
        "coverage_positive_covered_edges",
        "coverage_positive_total_edges",
        "coverage_positive_edge_recall",
        "coverage_negative_rules",
        "coverage_negative_covered_edges",
        "coverage_negative_total_edges",
        "coverage_negative_edge_recall",
        "coverage_overall_covered_edges",
        "coverage_overall_total_edges",
        "coverage_overall_edge_recall",
        "vspawn_candidates_seen",
        "vspawn_bn_pruned",
        "vspawn_duplicate_pruned",
        "vspawn_constraint_pruned",
        "vspawn_no_match_pruned",
        "vspawn_support_pruned",
        "pattern_bn_seen",
        "pattern_bn_kept",
        "pattern_bn_pruned",
        "pattern_bn_prune_rate",
        "pattern_bn_threshold_pruned",
        "pattern_bn_topk_pruned",
        "pattern_bn_min_keep_rescued",
        "predicate_bn_seen",
        "predicate_bn_kept",
        "predicate_bn_pruned",
        "predicate_bn_prune_rate",
        "predicate_bn_tau_pruned",
        "predicate_bn_feature_limit_pruned",
        "predicate_bn_topk_pruned",
        "predicate_bn_min_keep_rescued",
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
    print(f"\n[TauSensitivitySummary] wrote={csv_path}")


if __name__ == "__main__":
    main()
