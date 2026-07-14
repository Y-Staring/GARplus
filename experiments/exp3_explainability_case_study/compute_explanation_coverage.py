#!/usr/bin/env python3
"""Compute prediction-level GAR explanation coverage.

This script expects prediction rows that have already been matched against GARs,
for example the ``*_sampled_refined.csv`` files produced by exp1_accuracy. It
does not mine rules or re-run structural matching. The output deliberately
separates rule support, rule rejection, and negative-prediction explanation so
that a paper does not report the tautological statement that every prediction
rejected by a rule has a rule explanation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Matched prediction CSV files or directories (searched recursively).",
    )
    parser.add_argument("--output", required=True, help="Output summary CSV.")
    parser.add_argument("--dataset", default="unknown", help="Dataset label.")
    parser.add_argument("--pred-col", default="pred_label")
    parser.add_argument("--positive-label", default="positive")
    parser.add_argument("--negative-label", default="negative")
    parser.add_argument("--positive-hit-col", default="gar_positive_hit")
    parser.add_argument("--negative-hit-col", default="gar_negative_hit")
    parser.add_argument("--changed-col", default="changed_by_gar")
    parser.add_argument("--group-col", default="prediction_group")
    return parser.parse_args()


def discover_csvs(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(path.rglob("*.csv"))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input does not exist: {value}")
    return sorted(set(paths))


def as_hit(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    textual = series.astype(str).str.strip().str.lower()
    return numeric.ne(0) | textual.isin({"true", "yes", "y"})


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def summarize_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_file: str,
    group: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    required = {args.pred_col, args.positive_hit_col, args.negative_hit_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_file}: missing columns {missing}")

    pred = frame[args.pred_col].astype(str).str.strip().str.lower()
    pos_label = str(args.positive_label).strip().lower()
    neg_label = str(args.negative_label).strip().lower()
    positive_pred = pred.eq(pos_label)
    negative_pred = pred.eq(neg_label)
    positive_hit = as_hit(frame[args.positive_hit_col])
    negative_hit = as_hit(frame[args.negative_hit_col])
    if args.changed_col in frame:
        changed = as_hit(frame[args.changed_col])
    else:
        changed = positive_pred & negative_hit

    positive_supported = positive_pred & positive_hit
    negative_explained = negative_pred & negative_hit
    positive_rejected = positive_pred & negative_hit
    any_explained = positive_hit | negative_hit
    rejected_with_rule = changed & negative_hit

    counts = {
        "rows": len(frame),
        "positive_predictions": int(positive_pred.sum()),
        "positive_supported": int(positive_supported.sum()),
        "negative_predictions": int(negative_pred.sum()),
        "negative_explained": int(negative_explained.sum()),
        "positive_rejected_by_negative_rule": int(positive_rejected.sum()),
        "gar_rejected_predictions": int(changed.sum()),
        "gar_rejected_with_rule": int(rejected_with_rule.sum()),
        "predictions_with_any_rule": int(any_explained.sum()),
    }
    return {
        "dataset": dataset,
        "source_file": source_file,
        "prediction_group": group,
        **counts,
        "positive_explanation_coverage_pct": percentage(
            counts["positive_supported"], counts["positive_predictions"]
        ),
        "negative_prediction_explanation_coverage_pct": percentage(
            counts["negative_explained"], counts["negative_predictions"]
        ),
        "positive_rejection_rate_pct": percentage(
            counts["positive_rejected_by_negative_rule"],
            counts["positive_predictions"],
        ),
        "gar_rejection_explanation_pct": percentage(
            counts["gar_rejected_with_rule"], counts["gar_rejected_predictions"]
        ),
        "any_rule_coverage_pct": percentage(
            counts["predictions_with_any_rule"], counts["rows"]
        ),
    }


def main() -> None:
    args = parse_args()
    files = discover_csvs(args.inputs)
    if not files:
        raise SystemExit("No CSV files found.")

    rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        required = {args.pred_col, args.positive_hit_col, args.negative_hit_col}
        if not required.issubset(frame.columns):
            print(f"warning: skipping {path}; it is not a matched prediction CSV")
            continue
        all_frames.append(frame)
        if args.group_col in frame.columns:
            groups = frame.groupby(args.group_col, dropna=False, sort=True)
        else:
            groups = [("all", frame)]
        for group, group_frame in groups:
            rows.append(
                summarize_frame(
                    group_frame,
                    dataset=args.dataset,
                    source_file=str(path),
                    group=str(group),
                    args=args,
                )
            )

    if not all_frames:
        raise SystemExit("No input CSV contained the required GAR hit columns.")
    combined = pd.concat(all_frames, ignore_index=True)
    rows.append(
        summarize_frame(
            combined,
            dataset=args.dataset,
            source_file="__ALL__",
            group="__ALL__",
            args=args,
        )
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False, float_format="%.6f")
    print(result.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
