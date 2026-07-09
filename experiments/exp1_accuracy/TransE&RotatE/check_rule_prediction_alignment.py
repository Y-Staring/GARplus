from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path(
    r"D:\CodeWork\python\GAR+\GARplus\experiments\exp1_accuracy\TransE&RotatE"
    r"\predictions\DDA_10pct\predictions_with_original_ids.csv"
)
RULE_PAIRS_PATH = Path(
    r"D:\CodeWork\python\GAR+\GARplus\enumeration-discovery\processed"
    r"\rule_positive_negative_pairs_0707.csv"
)

RULE_HEAD_COL = "chemical_index"
RULE_TAIL_COL = "disease_index"
RULE_LABEL_COL = "predicted_label"

PRED_COLUMN_CANDIDATES = [
    ("head_old", "tail_old"),
    ("head", "tail"),
]


def norm_value(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def norm_series(series: pd.Series) -> pd.Series:
    return series.map(norm_value)


def pair_set(df: pd.DataFrame, head_col: str, tail_col: str) -> set[tuple[str, str]]:
    heads = norm_series(df[head_col])
    tails = norm_series(df[tail_col])
    return {
        (head, tail)
        for head, tail in zip(heads, tails)
        if head and tail
    }


def describe_ids(name: str, pairs: set[tuple[str, str]]) -> None:
    heads = {head for head, _tail in pairs}
    tails = {tail for _head, tail in pairs}
    print(f"\n[{name}]")
    print(f"pairs={len(pairs)} unique_heads={len(heads)} unique_tails={len(tails)}")
    print(f"head sample={sorted(heads)[:10]}")
    print(f"tail sample={sorted(tails)[:10]}")


def print_overlap(pred_pairs: set[tuple[str, str]], rule_pairs: set[tuple[str, str]], label: str) -> None:
    reversed_pred_pairs = {(tail, head) for head, tail in pred_pairs}
    overlap = pred_pairs & rule_pairs
    reverse_overlap = reversed_pred_pairs & rule_pairs

    pred_heads = {head for head, _tail in pred_pairs}
    pred_tails = {tail for _head, tail in pred_pairs}
    rule_heads = {head for head, _tail in rule_pairs}
    rule_tails = {tail for _head, tail in rule_pairs}

    print("\n" + "=" * 80)
    print(f"Prediction columns: {label}")
    print("=" * 80)
    print(f"prediction pairs        : {len(pred_pairs)}")
    print(f"rule pairs              : {len(rule_pairs)}")
    print(f"direct pair overlap     : {len(overlap)} ({len(overlap) / len(rule_pairs):.4%} of rules)")
    print(f"reverse pair overlap    : {len(reverse_overlap)} ({len(reverse_overlap) / len(rule_pairs):.4%} of rules)")
    print(f"head id overlap         : {len(pred_heads & rule_heads)} / rule_heads={len(rule_heads)}")
    print(f"tail id overlap         : {len(pred_tails & rule_tails)} / rule_tails={len(rule_tails)}")
    print(f"cross head-tail overlap : pred_heads∩rule_tails={len(pred_heads & rule_tails)}, pred_tails∩rule_heads={len(pred_tails & rule_heads)}")

    missing_rules = list(rule_pairs - pred_pairs)
    hit_rules = list(overlap)
    print(f"direct hit sample       : {hit_rules[:10]}")
    print(f"missing rule sample     : {missing_rules[:10]}")


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    rules = pd.read_csv(RULE_PAIRS_PATH)

    print(f"predictions path: {PREDICTIONS_PATH}")
    print(f"rules path      : {RULE_PAIRS_PATH}")
    print(f"prediction rows : {len(predictions)}")
    print(f"rule rows       : {len(rules)}")
    print(f"prediction cols : {predictions.columns.tolist()}")
    print(f"rule cols       : {rules.columns.tolist()}")

    rules[RULE_LABEL_COL] = rules[RULE_LABEL_COL].astype(str).str.strip().str.lower()
    for rule_label in ("positive", "negative"):
        subset = rules[rules[RULE_LABEL_COL] == rule_label]
        rule_pairs = pair_set(subset, RULE_HEAD_COL, RULE_TAIL_COL)
        describe_ids(f"rules:{rule_label}", rule_pairs)
        for head_col, tail_col in PRED_COLUMN_CANDIDATES:
            if head_col not in predictions.columns or tail_col not in predictions.columns:
                continue
            pred_pairs = pair_set(predictions, head_col, tail_col)
            print_overlap(pred_pairs, rule_pairs, f"{head_col},{tail_col} vs rules:{rule_label}")

    all_rule_pairs = pair_set(rules, RULE_HEAD_COL, RULE_TAIL_COL)
    describe_ids("rules:all", all_rule_pairs)
    for head_col, tail_col in PRED_COLUMN_CANDIDATES:
        if head_col not in predictions.columns or tail_col not in predictions.columns:
            continue
        pred_pairs = pair_set(predictions, head_col, tail_col)
        describe_ids(f"predictions:{head_col},{tail_col}", pred_pairs)
        print_overlap(pred_pairs, all_rule_pairs, f"{head_col},{tail_col} vs rules:all")


if __name__ == "__main__":
    main()
