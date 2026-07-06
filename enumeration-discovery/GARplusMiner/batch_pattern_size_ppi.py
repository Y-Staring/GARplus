from __future__ import annotations

import contextlib
import csv
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from ppi_demo import CONFIG
from garplus_demo_runner import run_demo


ALGORITHM = "GARplus"
DATASET = "PPI"
PATTERN_SIZES = (2, 4, 6, 8)
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
PATTERN_EXTENSION_DEBUG = True
PATTERN_EXTENSION_DEBUG_LIMIT = 200
RESULT_DIR = Path(__file__).resolve().parent / "batch_results" / "pattern_size_ppi"
TIMING_RE = re.compile(r"\[Timing\] stage=rule_mining_total .*?seconds=([0-9.]+)")
PATTERNS_RE = re.compile(r"\[Patterns\].*?mining_total=(\d+)")


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
        print(
            f"[BatchTiming] algorithm={ALGORITHM} dataset={DATASET} pattern_size={pattern_size} "
            f"wall_seconds={wall_seconds:.6f} rule_mining_seconds={rule_mining_seconds or 'NA'} "
            f"patterns_mined={patterns_mined or 'NA'} status={status} log={log_path}"
        )
        rows.append(
            {
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
        )
    csv_path = RESULT_DIR / f"{ALGORITHM.lower()}_{DATASET.lower()}_pattern_size_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[BatchSummary] wrote={csv_path}")


if __name__ == "__main__":
    main()
