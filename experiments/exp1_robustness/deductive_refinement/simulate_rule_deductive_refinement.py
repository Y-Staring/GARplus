#!/usr/bin/env python
"""Simulated/oracle rule-hit sensitivity analysis for rule deductive refinement.

This script does not report real rule mining performance.  It asks a controlled
counterfactual question: if negative rules could hit model positive false
positives with a chosen coverage and precision, how much could rule deductive
refinement (RDR) improve the final predictions?

The simulation is useful when real mined rules currently have weak test-set hit
rates.  If oracle-like hits at realistic precision/coverage still bring little
gain, the refinement mechanism or task setup may be the bottleneck.  If gains
become large once coverage increases, the current issue is more likely rule
coverage/precision rather than the RDR update itself.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one float")
    try:
        parsed = [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {value}") from exc
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulated/oracle negative-rule hit sensitivity analysis for "
            "rule deductive refinement."
        )
    )
    parser.add_argument("--input", required=True, help="Input prediction CSV.")
    parser.add_argument(
        "--output-dir",
        default="outputs/simulated_rdr",
        help="Directory for summary CSVs and optional simulated predictions.",
    )
    parser.add_argument("--true-col", default="true", help="Ground-truth label column.")
    parser.add_argument("--pred-col", default="pred", help="Predicted label column.")
    parser.add_argument(
        "--score-col",
        default="score",
        help="Optional positive-class score/probability/logit column.",
    )
    parser.add_argument("--pos-label", default="1", help="Positive label value.")
    parser.add_argument("--neg-label", default="2", help="Negative label value.")
    parser.add_argument(
        "--coverage-list",
        type=parse_float_list,
        default=parse_float_list("0.05,0.1,0.2,0.3,0.5"),
        help="Comma-separated error coverage values.",
    )
    parser.add_argument(
        "--precision-list",
        type=parse_float_list,
        default=parse_float_list("0.7,0.8,0.9,1.0"),
        help="Comma-separated rule precision values.",
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--score-after-hit",
        type=float,
        default=1e-6,
        help="Positive-class score assigned to simulated rule-hit rows.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save one prediction CSV for every coverage/precision/seed run.",
    )
    return parser.parse_args()


def label_series(series: pd.Series) -> pd.Series:
    """Normalize labels for robust comparison across int/string CSV columns."""
    return series.astype("string").fillna("<NA>")


def label_value(value: object) -> str:
    return str(value)


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required_cols = [args.true_col, args.pred_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if args.score_col and args.score_col not in df.columns:
        warnings.warn(
            f"score column '{args.score_col}' not found; score-based metrics/refinement disabled.",
            RuntimeWarning,
        )
    return df


def safe_metric(name: str, func, default: float = math.nan) -> float:
    try:
        value = func()
    except Exception as exc:  # noqa: BLE001 - metrics fail in many valid edge cases.
        warnings.warn(f"{name} could not be computed: {exc}", RuntimeWarning)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_metrics(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    pos_label: object,
    score_col: str | None = None,
) -> dict[str, float]:
    y_true = label_series(df[true_col])
    y_pred = label_series(df[pred_col])
    pos = label_value(pos_label)

    metrics: dict[str, float] = {}
    metrics["acc"] = safe_metric("accuracy", lambda: accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = safe_metric(
        "macro F1", lambda: f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    def pos_prf() -> tuple[float, float, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[pos],
            average=None,
            zero_division=0,
        )
        return float(precision[0]), float(recall[0]), float(f1[0])

    pos_precision, pos_recall, pos_f1 = safe_metric(
        "positive precision/recall/F1", pos_prf, default=(math.nan, math.nan, math.nan)
    )
    metrics["pos_precision"] = pos_precision
    metrics["pos_recall"] = pos_recall
    metrics["pos_f1"] = pos_f1

    metrics["roc_auc"] = math.nan
    metrics["pr_auc"] = math.nan
    if score_col and score_col in df.columns:
        y_binary = (y_true == pos).astype(int).to_numpy()
        scores = pd.to_numeric(df[score_col], errors="coerce")
        valid = scores.notna().to_numpy()
        if valid.sum() == 0:
            warnings.warn(f"score column '{score_col}' has no numeric values.", RuntimeWarning)
        elif len(np.unique(y_binary[valid])) < 2:
            warnings.warn(
                f"{score_col}: ROC-AUC/PR-AUC require both positive and negative true labels.",
                RuntimeWarning,
            )
        else:
            score_values = scores.to_numpy(dtype=float)
            metrics["roc_auc"] = safe_metric(
                "ROC-AUC", lambda: roc_auc_score(y_binary[valid], score_values[valid])
            )
            metrics["pr_auc"] = safe_metric(
                "PR-AUC",
                lambda: average_precision_score(y_binary[valid], score_values[valid]),
            )

    return metrics


def sample_indices(pool: pd.Index, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or len(pool) == 0:
        return np.array([], dtype=object)
    n = min(n, len(pool))
    return rng.choice(pool.to_numpy(), size=n, replace=False)


def add_prefixed_metrics(
    row: dict[str, float],
    before: dict[str, float],
    after: dict[str, float],
) -> None:
    metric_names = ["acc", "macro_f1", "pos_precision", "pos_recall", "pos_f1", "roc_auc", "pr_auc"]
    for name in metric_names:
        row[f"{name}_before"] = before.get(name, math.nan)
        row[f"{name}_after"] = after.get(name, math.nan)
        if name in {"acc", "macro_f1"}:
            row[f"{name}_gain"] = row[f"{name}_after"] - row[f"{name}_before"]


def simulate_one_run(
    df: pd.DataFrame,
    coverage: float,
    precision: float,
    seed: int,
    true_col: str,
    pred_col: str,
    pos_label: object,
    neg_label: object,
    score_col: str | None = None,
    score_after_hit: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate one negative-rule RDR run.

    error_coverage means the fraction of positive false positives that the
    simulated negative rules correctly hit.  rule_precision means the fraction
    of all simulated rule hits that are truly model errors.  Bad hits are drawn
    from true positives, where a negative-rule refinement would hurt recall.
    """
    validate_probability("coverage", coverage)
    validate_probability("precision", precision)
    if precision <= 0.0 and coverage > 0.0:
        raise ValueError("precision must be > 0 when coverage is positive")

    rng = np.random.default_rng(seed)
    pos = label_value(pos_label)
    true_labels = label_series(df[true_col])
    pred_labels = label_series(df[pred_col])

    good_pool = df.index[(pred_labels == pos) & (true_labels != pos)]
    bad_pool = df.index[true_labels == pos]

    n_good_target = int(len(good_pool) * coverage)
    if precision == 1.0:
        n_bad_target = 0
    elif n_good_target == 0:
        n_bad_target = 0
    else:
        n_bad_target = int(n_good_target * (1.0 - precision) / precision)

    good_hits = sample_indices(good_pool, n_good_target, rng)
    bad_hits = sample_indices(bad_pool, n_bad_target, rng)
    hit_indices = np.concatenate([good_hits, bad_hits])

    if len(bad_hits) < n_bad_target:
        warnings.warn(
            f"bad_pool has only {len(bad_pool)} rows; requested {n_bad_target}, sampled {len(bad_hits)}.",
            RuntimeWarning,
        )
    if len(good_hits) < n_good_target:
        warnings.warn(
            f"good_pool has only {len(good_pool)} rows; requested {n_good_target}, sampled {len(good_hits)}.",
            RuntimeWarning,
        )

    out = df.copy()
    out["pred_before_rdr"] = out[pred_col]
    out["pred_after_rdr"] = out[pred_col]
    out["sim_rule_hit"] = False
    out["sim_rule_correct"] = False
    out["sim_rule_wrong"] = False

    if len(hit_indices) > 0:
        out.loc[hit_indices, "pred_after_rdr"] = neg_label
        out.loc[hit_indices, "sim_rule_hit"] = True
    if len(good_hits) > 0:
        out.loc[good_hits, "sim_rule_correct"] = True
    if len(bad_hits) > 0:
        out.loc[bad_hits, "sim_rule_wrong"] = True

    active_score_col = score_col if score_col and score_col in out.columns else None
    if active_score_col:
        out["score_before_rdr"] = out[active_score_col]
        out["score_after_rdr"] = pd.to_numeric(out[active_score_col], errors="coerce")
        if len(hit_indices) > 0:
            out.loc[hit_indices, "score_after_rdr"] = score_after_hit

    before_metrics = compute_metrics(
        out,
        true_col=true_col,
        pred_col="pred_before_rdr",
        pos_label=pos_label,
        score_col="score_before_rdr" if active_score_col else None,
    )
    after_metrics = compute_metrics(
        out,
        true_col=true_col,
        pred_col="pred_after_rdr",
        pos_label=pos_label,
        score_col="score_after_rdr" if active_score_col else None,
    )

    empirical_precision = (
        float(len(good_hits) / len(hit_indices)) if len(hit_indices) > 0 else math.nan
    )
    summary: dict[str, float] = {
        "coverage": coverage,
        "precision": precision,
        "seed": seed,
        "num_samples": len(out),
        "num_positive_false_positives": len(good_pool),
        "sim_good_hits": len(good_hits),
        "sim_bad_hits": len(bad_hits),
        "sim_total_hits": len(hit_indices),
        "empirical_rule_precision": empirical_precision,
    }
    add_prefixed_metrics(summary, before_metrics, after_metrics)
    return out, summary


def format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_grid(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float]] = []
    active_score_col = args.score_col if args.score_col in df.columns else None
    if args.score_col and active_score_col is None:
        warnings.warn(
            f"score column '{args.score_col}' not found; continuing without score outputs.",
            RuntimeWarning,
        )

    for coverage in args.coverage_list:
        for precision in args.precision_list:
            for seed_offset in range(args.num_seeds):
                run_seed = args.seed + seed_offset
                simulated_df, summary = simulate_one_run(
                    df=df,
                    coverage=coverage,
                    precision=precision,
                    seed=run_seed,
                    true_col=args.true_col,
                    pred_col=args.pred_col,
                    pos_label=args.pos_label,
                    neg_label=args.neg_label,
                    score_col=active_score_col,
                    score_after_hit=args.score_after_hit,
                )
                results.append(summary)

                if args.save_predictions:
                    cov_name = format_float_for_filename(coverage)
                    prec_name = format_float_for_filename(precision)
                    pred_path = predictions_dir / (
                        f"predictions_cov{cov_name}_prec{prec_name}_seed{run_seed}.csv"
                    )
                    simulated_df.to_csv(pred_path, index=False)

    return pd.DataFrame(results)


def summarize_results(summary_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary_path = output_dir / "simulated_rdr_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    numeric_cols = [
        col
        for col in summary_df.columns
        if col not in {"coverage", "precision", "seed"}
        and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    mean_df = summary_df.groupby(["coverage", "precision"], dropna=False)[numeric_cols].agg(
        ["mean", "std"]
    )
    mean_df.columns = [f"{col}_{stat}" for col, stat in mean_df.columns]
    mean_df = mean_df.reset_index()

    mean_path = output_dir / "simulated_rdr_summary_mean_std.csv"
    mean_df.to_csv(mean_path, index=False)
    return mean_df


def main() -> None:
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    for coverage in args.coverage_list:
        validate_probability("coverage", coverage)
    for precision in args.precision_list:
        validate_probability("precision", precision)

    df = load_predictions(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = run_grid(args, df)
    summarize_results(summary_df, output_dir)

    print(f"Wrote summary: {output_dir / 'simulated_rdr_summary.csv'}")
    print(f"Wrote mean/std summary: {output_dir / 'simulated_rdr_summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
