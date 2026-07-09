from __future__ import annotations

import contextlib
import csv
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from ti_demo import CONFIG
from garplus_demo_runner import run_demo


ALGORITHM = "GARplus"
DATASET = "TI"
PATTERN_SIZES = (8, 6, 4, 2)
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
PATTERN_EXTENSION_DEBUG = True
PATTERN_EXTENSION_DEBUG_LIMIT = 200
RULE_COVERAGE_SCOPE = "sampled"  # none, sampled, original
RULE_COVERAGE_MAX_INSTANCES = None
RESULT_DIR = Path("/home/yyyy/codework/GARplus/enumeration-discovery/online_result/pattern_size_recall/pattern_size_ti")
TIMING_RE = re.compile(r"\[Timing\] stage=rule_mining_total .*?seconds=([0-9.]+)")
PATTERNS_RE = re.compile(r"\[Patterns\].*?mining_total=(\d+)")
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


def config_for_pattern_size(pattern_size: int):
    return replace(
        CONFIG,
        min_pattern_nodes=pattern_size,
        max_pattern_nodes=pattern_size,
        max_radius=MAX_RADIUS,
        max_add_edge=MAX_ADD_EDGE,
        node_max_add_edge=NODE_MAX_ADD_EDGE,
        pattern_support=PATTERN_SUPPORT,
        pattern_extension_debug=PATTERN_EXTENSION_DEBUG,
        pattern_extension_debug_limit=PATTERN_EXTENSION_DEBUG_LIMIT,
        rule_coverage_scope=RULE_COVERAGE_SCOPE,
        rule_coverage_max_instances=RULE_COVERAGE_MAX_INSTANCES,
        deduped_rules_output_path=str(RESULT_DIR / f"deduped_rules_n{pattern_size}.txt"),
        pattern_instances_output_path=str(RESULT_DIR / f"pattern_instances_n{pattern_size}.jsonl"),
    )


def parse_result(log_text: str) -> tuple[str, str]:
    timing_match = TIMING_RE.search(log_text)
    patterns_match = PATTERNS_RE.search(log_text)
    return (
        timing_match.group(1) if timing_match else "",
        patterns_match.group(1) if patterns_match else "",
    )


def parse_rule_coverage(log_text: str) -> dict[str, str]:
    match = RULE_COVERAGE_RE.search(log_text)
    if not match:
        return {
            "rule_coverage_scope": RULE_COVERAGE_SCOPE,
            "coverage_positive_edge_recall": "",
            "coverage_negative_edge_recall": "",
            "coverage_overall_edge_recall": "",
        }
    return {
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


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pattern_size in PATTERN_SIZES:
        log_path = RESULT_DIR / f"{ALGORITHM.lower()}_{DATASET.lower()}_n{pattern_size}.log"
        print(f"\n=== BatchRun algorithm={ALGORITHM} dataset={DATASET} pattern_size={pattern_size} ===")
        started = time.perf_counter()
        status = "ok"
        error = ""
        with log_path.open("w", encoding="utf-8") as log_file:
            tee = Tee(sys.stdout, log_file)
            try:
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    run_demo(config_for_pattern_size(pattern_size))
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    traceback.print_exc()
        wall_seconds = time.perf_counter() - started
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        rule_mining_seconds, patterns_mined = parse_result(log_text)
        coverage = parse_rule_coverage(log_text)
        print(
            f"[BatchTiming] algorithm={ALGORITHM} dataset={DATASET} pattern_size={pattern_size} "
            f"wall_seconds={wall_seconds:.6f} rule_mining_seconds={rule_mining_seconds or 'NA'} "
            f"patterns_mined={patterns_mined or 'NA'} "
            f"overall_edge_recall={coverage.get('coverage_overall_edge_recall') or 'NA'} "
            f"status={status} log={log_path}"
        )
        row = {
            "algorithm": ALGORITHM,
            "dataset": DATASET,
            "pattern_size": pattern_size,
            "min_pattern_nodes": pattern_size,
            "max_pattern_nodes": pattern_size,
            "max_radius": MAX_RADIUS,
            "max_add_edge": MAX_ADD_EDGE,
            "node_max_add_edge": NODE_MAX_ADD_EDGE,
            "pattern_support": PATTERN_SUPPORT,
            "patterns_mined": patterns_mined,
            "rule_mining_seconds": rule_mining_seconds,
            "wall_seconds": f"{wall_seconds:.6f}",
            "status": status,
            "error": error,
            "log_path": str(log_path),
        }
        row.update(coverage)
        rows.append(row)
    csv_path = RESULT_DIR / f"{ALGORITHM.lower()}_{DATASET.lower()}_pattern_size_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[BatchSummary] wrote={csv_path}")


if __name__ == "__main__":
    main()
