from __future__ import annotations

import contextlib
import csv
import re
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

from dda_demo import CONFIG
from gfd_demo_runner import run_demo


ALGORITHM = "GFD"
DATASET = "DDA"
PATTERN_SIZE = 4
DEFAULT_SIGMA = 50
DEFAULT_CONFIDENCE = 0.7
SIGMAS = (50, 100, 150, 200)
CONFIDENCES = (0.3, 0.5, 0.7, 0.9)
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
PATTERN_EXTENSION_DEBUG = True
PATTERN_EXTENSION_DEBUG_LIMIT = 200
RESULT_DIR = Path(__file__).resolve().parent / "batch_results" / "sigma_confidence_dda"
TIMING_RE = re.compile(r"\[Timing\] stage=rule_mining_total .*?seconds=([0-9.]+)")
PATTERNS_RE = re.compile(r"\[Patterns\].*?mining_total=(\d+)")
SUMMARY_RE = re.compile(r"\[Summary\].*?total_gfds=(\d+)")


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


def confidence_tag(confidence: float) -> str:
    return str(confidence).replace(".", "p")


def config_for_run(sigma: int, confidence: float):
    return replace(
        CONFIG,
        min_pattern_nodes=PATTERN_SIZE,
        max_pattern_nodes=PATTERN_SIZE,
        max_radius=MAX_RADIUS,
        max_add_edge=MAX_ADD_EDGE,
        node_max_add_edge=NODE_MAX_ADD_EDGE,
        pattern_support=PATTERN_SUPPORT,
        min_support_count=sigma,
        min_confidence=confidence,
        pattern_extension_debug=PATTERN_EXTENSION_DEBUG,
        pattern_extension_debug_limit=PATTERN_EXTENSION_DEBUG_LIMIT,
    )


def parse_result(log_text: str) -> tuple[str, str, str]:
    timing_match = TIMING_RE.search(log_text)
    patterns_match = PATTERNS_RE.search(log_text)
    summary_match = SUMMARY_RE.search(log_text)
    return (
        timing_match.group(1) if timing_match else "",
        patterns_match.group(1) if patterns_match else "",
        summary_match.group(1) if summary_match else "",
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    runs = (
        [("sigma", sigma, DEFAULT_CONFIDENCE) for sigma in SIGMAS]
        + [("confidence", DEFAULT_SIGMA, confidence) for confidence in CONFIDENCES]
    )
    for varying_param, sigma, confidence in runs:
            tag = f"{varying_param}_s{sigma}_c{confidence_tag(confidence)}_n{PATTERN_SIZE}"
            log_path = RESULT_DIR / f"{ALGORITHM.lower()}_{DATASET.lower()}_{tag}.log"
            print(
                f"\n=== BatchRun algorithm={ALGORITHM} dataset={DATASET} "
                f"pattern_size={PATTERN_SIZE} varying={varying_param} "
                f"sigma={sigma} confidence={confidence} ==="
            )
            started = time.perf_counter()
            status = "ok"
            error = ""
            with log_path.open("w", encoding="utf-8") as log_file:
                tee = Tee(sys.stdout, log_file)
                try:
                    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                        run_demo(config_for_run(sigma, confidence))
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                        traceback.print_exc()
            wall_seconds = time.perf_counter() - started
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            rule_mining_seconds, patterns_mined, total_gfds = parse_result(log_text)
            print(
                f"[BatchTiming] algorithm={ALGORITHM} dataset={DATASET} pattern_size={PATTERN_SIZE} "
                f"varying={varying_param} sigma={sigma} confidence={confidence} wall_seconds={wall_seconds:.6f} "
                f"rule_mining_seconds={rule_mining_seconds or 'NA'} patterns_mined={patterns_mined or 'NA'} "
                f"total_gfds={total_gfds or 'NA'} status={status} log={log_path}"
            )
            rows.append(
                {
                    "algorithm": ALGORITHM,
                    "dataset": DATASET,
                    "pattern_size": PATTERN_SIZE,
                    "varying_param": varying_param,
                    "sigma": sigma,
                    "confidence": confidence,
                    "min_pattern_nodes": PATTERN_SIZE,
                    "max_pattern_nodes": PATTERN_SIZE,
                    "max_radius": MAX_RADIUS,
                    "max_add_edge": MAX_ADD_EDGE,
                    "node_max_add_edge": NODE_MAX_ADD_EDGE,
                    "pattern_support": PATTERN_SUPPORT,
                    "min_support_count": sigma,
                    "min_confidence": confidence,
                    "patterns_mined": patterns_mined,
                    "total_gfds": total_gfds,
                    "rule_mining_seconds": rule_mining_seconds,
                    "wall_seconds": f"{wall_seconds:.6f}",
                    "status": status,
                    "error": error,
                    "log_path": str(log_path),
                }
            )
    csv_path = RESULT_DIR / f"{ALGORITHM.lower()}_{DATASET.lower()}_sigma_confidence_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[BatchSummary] wrote={csv_path}")


if __name__ == "__main__":
    main()
