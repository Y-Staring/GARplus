from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from apply_flip_garplus_rules import (
    Rule,
    instance_satisfies,
    load_instances,
    load_rules,
    load_subgraph,
)
from garplus_demo_runner import augment_graph_structural_features


def load_negative_rules(path: Path, min_confidence: float, min_lift: float) -> list[Rule]:
    rules = load_rules(path, min_confidence, allow_empty=True)
    return [rule for rule in rules if rule.label == 0 and rule.lift >= min_lift]


def apply_negative_rules(
    targets: pd.DataFrame,
    graph,
    instances,
    rules: list[Rule],
    min_matched_rules: int,
    min_negative_score: float,
) -> pd.DataFrame:
    by_pattern: dict[int, list[Rule]] = defaultdict(list)
    for rule in rules:
        by_pattern[rule.pattern_id].append(rule)

    outputs = []
    for row in targets.itertuples(index=False):
        noisy_label = int(row.noisy_label)
        pair = (int(row.src_big), int(row.dst_big))
        matched_lines: set[int] = set()
        matched_patterns: set[int] = set()
        negative_score = 0.0
        best_confidence = 0.0
        best_lift = 0.0

        # Only label-2 rows can contain the targeted 0 -> 2 corruption.
        if noisy_label == 2:
            for pattern_id, pattern_instances in instances.get(pair, {}).items():
                for rule in by_pattern.get(pattern_id, []):
                    if any(instance_satisfies(graph, item, rule.antecedent) for item in pattern_instances):
                        matched_lines.add(rule.index)
                        matched_patterns.add(pattern_id)
                        negative_score += rule.score
                        best_confidence = max(best_confidence, rule.confidence)
                        best_lift = max(best_lift, rule.lift)

        change = (
            noisy_label == 2
            and len(matched_lines) >= min_matched_rules
            and negative_score >= min_negative_score
        )
        corrected_label = 0 if change else noisy_label
        output = row._asdict()
        output.update(
            {
                "corrected_label": corrected_label,
                "changed_by_negative_rule": int(change),
                "negative_score": negative_score,
                "best_rule_confidence": best_confidence,
                "best_rule_lift": best_lift,
                "matched_rule_count": len(matched_lines),
                "matched_pattern_count": len(matched_patterns),
                "matched_rule_lines": ";".join(map(str, sorted(matched_lines))),
                "matched_pattern_ids": ";".join(map(str, sorted(matched_patterns))),
            }
        )
        outputs.append(output)
    return pd.DataFrame(outputs)


def summarize(result: pd.DataFrame, usable_rules: int) -> dict[str, object]:
    noisy = result.noisy_label.astype(int)
    original = result.original_label.astype(int)
    corrected = result.corrected_label.astype(int)
    flipped = result.flipped_flag.astype(int) == 1
    changed = result.changed_by_negative_rule.astype(int) == 1
    targeted_noise = flipped & original.eq(0) & noisy.eq(2)
    correct_changes = changed & original.eq(0)
    false_changes = changed & ~original.eq(0)

    changed_count = int(changed.sum())
    targeted_count = int(targeted_noise.sum())
    correct_count = int(correct_changes.sum())
    return {
        "rows": len(result),
        "usable_negative_rules": usable_rules,
        "candidate_label2_rows": int(noisy.eq(2).sum()),
        "targeted_0_to_2_noise": targeted_count,
        "changed_2_to_0": changed_count,
        "corrected_0_to_2": correct_count,
        "false_positive_changes": int(false_changes.sum()),
        "rule_precision": correct_count / changed_count if changed_count else 0.0,
        "targeted_noise_recall": correct_count / targeted_count if targeted_count else 0.0,
        "clean_label2_damaged": int((changed & ~flipped & original.eq(2)).sum()),
        "correct_before": int(noisy.eq(original).sum()),
        "correct_after": int(corrected.eq(original).sum()),
        "net_correct_gain": int(corrected.eq(original).sum() - noisy.eq(original).sum()),
        "training_label_accuracy_before": float(noisy.eq(original).mean()),
        "training_label_accuracy_after": float(corrected.eq(original).mean()),
        "oracle_used_for_decision": False,
        "evaluation_only_columns": ["flipped_flag", "original_label"],
    }


def write_per_rule_summary(result: pd.DataFrame, output_path: Path) -> None:
    counts: Counter[int] = Counter()
    correct: Counter[int] = Counter()
    false_positive: Counter[int] = Counter()
    changed = result[result.changed_by_negative_rule.astype(int) == 1]
    for row in changed.itertuples(index=False):
        lines = [int(item) for item in str(row.matched_rule_lines).split(";") if item]
        is_correct = int(row.original_label) == 0
        for line in lines:
            counts[line] += 1
            if is_correct:
                correct[line] += 1
            else:
                false_positive[line] += 1
    rows = []
    for line in sorted(counts):
        total = counts[line]
        rows.append(
            {
                "rule_line": line,
                "changed_rows": total,
                "corrected_0_to_2": correct[line],
                "false_positive_changes": false_positive[line],
                "evaluation_precision": correct[line] / total if total else 0.0,
            }
        )
    pd.DataFrame(
        rows,
        columns=(
            "rule_line",
            "changed_rows",
            "corrected_0_to_2",
            "false_positive_changes",
            "evaluation_precision",
        ),
    ).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply consequent-label-0 GARplus rules to current label-2 edges without oracle decisions."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-lift", type=float, default=1.50)
    parser.add_argument("--min-matched-rules", type=int, default=1)
    parser.add_argument("--min-negative-score", type=float, default=0.0)
    args = parser.parse_args()
    if args.min_matched_rules < 1:
        parser.error("--min-matched-rules must be at least 1")

    rules = load_negative_rules(
        args.result_dir / "deduped_rules.txt",
        args.min_confidence,
        args.min_lift,
    )
    instances = load_instances(
        args.result_dir / "pattern_instances.jsonl",
        {rule.pattern_id for rule in rules},
    )
    graph = load_subgraph(str(args.result_dir / "subgraph_edges_noisy.csv"))
    augment_graph_structural_features(graph)
    targets = pd.read_csv(args.result_dir / "target_edges_mapped.csv")
    result = apply_negative_rules(
        targets,
        graph,
        instances,
        rules,
        args.min_matched_rules,
        args.min_negative_score,
    )

    result.to_csv(args.result_dir / "negative_targeted_predictions.csv", index=False)
    result[["src", "corrected_label", "dst"]].to_csv(
        args.result_dir / "train_negative_targeted_denoised.txt",
        sep="\t",
        header=False,
        index=False,
    )
    result[result.changed_by_negative_rule.astype(int) == 1].to_csv(
        args.result_dir / "exported_negative_edges.csv",
        index=False,
    )
    write_per_rule_summary(result, args.result_dir / "per_rule_negative_summary.csv")

    summary = summarize(result, len(rules))
    summary.update(
        {
            "min_confidence": args.min_confidence,
            "min_lift": args.min_lift,
            "min_matched_rules": args.min_matched_rules,
            "min_negative_score": args.min_negative_score,
        }
    )
    (args.result_dir / "negative_targeted_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
