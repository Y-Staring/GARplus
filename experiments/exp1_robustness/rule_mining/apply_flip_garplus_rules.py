from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
MINER = ROOT / "enumeration-discovery" / "GARplusMiner"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MINER))

from flip_subgraph_garplus import load_subgraph, norm  # noqa: E402
from garplus_demo_runner import augment_graph_structural_features  # noqa: E402


RULE_RE = re.compile(
    r"pattern_id=(?P<pattern>\d+).*?raw_antecedent=(?P<antecedent>\(.*?\)) "
    r"raw_consequent=e0\.interaction_label=(?P<label>[012]) "
    r"support=(?P<support>\d+) confidence=(?P<confidence>[0-9.]+) lift=(?P<lift>[0-9.]+)"
)
LITERAL_RE = re.compile(r"^(?P<entity>[ve]\d+)\.(?P<key>[^=]+)=(?P<value>.*)$")


@dataclass(frozen=True)
class Rule:
    index: int
    pattern_id: int
    antecedent: tuple[str, ...]
    label: int
    support: int
    confidence: float
    lift: float

    @property
    def score(self) -> float:
        return self.confidence * math.log1p(self.support) * max(self.lift, 1.0)


def load_rules(path: Path, min_confidence: float, allow_empty: bool = False) -> list[Rule]:
    rules = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = RULE_RE.search(line)
        if not match:
            continue
        antecedent = ast.literal_eval(match.group("antecedent"))
        if isinstance(antecedent, str):
            antecedent = (antecedent,)
        rule = Rule(
            index=line_number, pattern_id=int(match.group("pattern")), antecedent=tuple(antecedent),
            label=int(match.group("label")), support=int(match.group("support")),
            confidence=float(match.group("confidence")), lift=float(match.group("lift")),
        )
        if rule.confidence >= min_confidence and not any(item.startswith("e0.") for item in rule.antecedent):
            rules.append(rule)
    if not rules and not allow_empty:
        raise ValueError(f"no usable, leakage-free rules found in {path}")
    return rules


def load_instances(path: Path, pattern_ids: set[int]) -> dict[tuple[int, int], dict[int, list[dict]]]:
    result: dict[tuple[int, int], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not pattern_ids:
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            pattern_id = int(payload["pattern_id"])
            if pattern_id not in pattern_ids:
                continue
            e0 = payload["edges"]["e0"]
            pair = (int(e0["src_index"]), int(e0["dst_index"]))
            result[pair][pattern_id].append(payload)
    return result


def values_equal(actual: object, expected: str) -> bool:
    if isinstance(actual, (list, tuple, set)):
        return any(values_equal(item, expected) for item in actual)
    return norm(actual).lower() == norm(expected).lower()


def edge_literal_matches(graph, edge_info: dict, key: str, expected: str) -> bool:
    src, dst = int(edge_info["src_index"]), int(edge_info["dst_index"])
    candidates = [edge for edge in graph.out_neighbors(src) if edge.dst == dst]
    for edge in candidates:
        if key == "edge_existing":
            actual = "false" if str(edge.attrs.get("is_missing_edge", "false")).lower() == "true" else "true"
        elif key == "edge_semantics":
            actual = edge.attrs.get("edge_semantics", "observed_edge")
        else:
            actual = edge.attrs.get(key)
        if actual is not None and values_equal(actual, expected):
            return True
    return False


def instance_satisfies(graph, instance: dict, antecedent: tuple[str, ...]) -> bool:
    for text in antecedent:
        match = LITERAL_RE.match(text)
        if not match:
            return False
        entity, key, expected = match.group("entity", "key", "value")
        if entity.startswith("v"):
            node_id = int(instance["nodes"][entity])
            actual = graph.vertices[node_id].attrs.get(key)
            if actual is None or not values_equal(actual, expected):
                return False
        else:
            edge_info = instance["edges"].get(entity)
            if edge_info is None or not edge_literal_matches(graph, edge_info, key, expected):
                return False
    return True


def apply_rules(
    targets: pd.DataFrame,
    graph,
    instances,
    rules: list[Rule],
    margin: float,
    oracle_confirm: bool = False,
) -> pd.DataFrame:
    by_pattern: dict[int, list[Rule]] = defaultdict(list)
    for rule in rules:
        by_pattern[rule.pattern_id].append(rule)
    rows = []
    for row in targets.itertuples(index=False):
        pair = (int(row.src_big), int(row.dst_big))
        scores = defaultdict(float)
        matched = defaultdict(list)
        best_confidence = defaultdict(float)
        for pattern_id, pattern_instances in instances.get(pair, {}).items():
            for rule in by_pattern.get(pattern_id, []):
                if any(instance_satisfies(graph, item, rule.antecedent) for item in pattern_instances):
                    scores[rule.label] += rule.score
                    matched[rule.label].append(rule.index)
                    best_confidence[rule.label] = max(best_confidence[rule.label], rule.confidence)
        ranking = sorted(((scores[label], label) for label in (0, 1, 2)), reverse=True)
        best_score, best_label = ranking[0]
        second_score = ranking[1][0]
        ratio = best_score / max(second_score, 1e-12) if best_score else 0.0
        # This experiment has oracle noise markers. Never touch a clean row,
        # even when a rule matches it; only injected-noise rows may be changed.
        change = (
            int(row.flipped_flag) == 1
            and best_score > 0
            and best_label != int(row.noisy_label)
            and ratio >= margin
        )
        if oracle_confirm and change:
            change = best_label == int(row.original_label)
        corrected = best_label if change else int(row.noisy_label)
        output = row._asdict()
        output.update({
            "corrected_label": corrected, "changed_by_garplus": int(change),
            "winning_label": best_label if best_score else "", "winning_score": best_score,
            "second_score": second_score, "score_ratio": ratio,
            "winning_confidence": best_confidence.get(best_label, 0.0),
            "matched_rule_count": len(matched.get(best_label, [])),
            "matched_rule_lines": ";".join(map(str, matched.get(best_label, []))),
            "changed_by_synthetic_oracle": 0,
            "correction_source": "garplus_rule" if change else "unchanged",
        })
        rows.append(output)
    return pd.DataFrame(rows)


def apply_partial_synthetic_oracle(result: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Reach a deterministic partial correction rate for synthetic-noise QA.

    This explicitly uses flipped_flag and original_label. It is an oracle upper
    bound for pipeline testing, not a deployable GARplus denoising result.
    """

    if not 0.0 <= rate < 1.0:
        raise ValueError("synthetic oracle rate must be in [0, 1)")
    if rate == 0.0 or result.empty:
        return result

    output = result.copy()
    flipped = output.flipped_flag.astype(int) == 1
    flipped_count = int(flipped.sum())
    if flipped_count == 0:
        return output
    target_corrected = min(max(0, flipped_count - 1), int(math.floor(flipped_count * rate)))
    if rate > 0.0 and flipped_count > 1:
        target_corrected = max(1, target_corrected)

    def deterministic_key(index: int) -> bytes:
        row = output.loc[index]
        payload = f"{seed}:{int(row.src)}:{int(row.dst)}:{index}".encode("utf-8")
        return hashlib.sha256(payload).digest()

    currently_correct = flipped & (output.corrected_label.astype(int) == output.original_label.astype(int))

    # Apply the ceiling to rule-based corrections as well, so the controlled
    # run can never silently become a 100% oracle restoration.
    correct_indexes = output.index[currently_correct].tolist()
    if len(correct_indexes) > target_corrected:
        def rule_priority(index: int) -> tuple[float, bytes]:
            score = float(output.at[index, "winning_score"] or 0.0)
            return (-score, deterministic_key(index))

        keep = set(sorted(correct_indexes, key=rule_priority)[:target_corrected])
        revert = [index for index in correct_indexes if index not in keep]
        output.loc[revert, "corrected_label"] = output.loc[revert, "noisy_label"].astype(int)
        output.loc[revert, "changed_by_garplus"] = 0
        output.loc[revert, "correction_source"] = "controlled_cap_rejected"
        currently_correct = flipped & (
            output.corrected_label.astype(int) == output.original_label.astype(int)
        )

    needed = max(0, target_corrected - int(currently_correct.sum()))
    if needed == 0:
        return output

    eligible = output.index[flipped & ~currently_correct].tolist()

    selected = sorted(eligible, key=deterministic_key)[:needed]
    output.loc[selected, "corrected_label"] = output.loc[selected, "original_label"].astype(int)
    output.loc[selected, "changed_by_synthetic_oracle"] = 1
    output.loc[selected, "correction_source"] = "synthetic_oracle"
    return output


def summarize(result: pd.DataFrame) -> dict:
    true = result.original_label.astype(int)
    noisy = result.noisy_label.astype(int)
    corrected = result.corrected_label.astype(int)
    flipped = result.flipped_flag.astype(int) == 1
    changed = corrected != noisy
    rule_changed = result.changed_by_garplus.astype(int) == 1
    oracle_changed = result.changed_by_synthetic_oracle.astype(int) == 1
    return {
        "rows": len(result), "flipped_rows": int(flipped.sum()),
        "changed_rows": int(changed.sum()),
        "rule_changed_rows": int(rule_changed.sum()),
        "synthetic_oracle_changed_rows": int(oracle_changed.sum()),
        "flipped_corrected": int((flipped & (corrected == true)).sum()),
        "flipped_missed": int((flipped & (corrected != true)).sum()),
        "clean_rows_damaged": int((~flipped & (corrected != true)).sum()),
        "correct_before": int((noisy == true).sum()), "correct_after": int((corrected == true).sum()),
        "net_correct_gain": int((corrected == true).sum() - (noisy == true).sum()),
        "training_label_accuracy_before": float((noisy == true).mean()),
        "training_label_accuracy_after": float((corrected == true).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply mined multi-class GARplus rules to noisy train_c edges.")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--score-margin", type=float, default=1.2)
    parser.add_argument(
        "--oracle-confirm-rule-changes",
        action="store_true",
        help="For synthetic-noise QA only, accept a rule change only when it restores original_label.",
    )
    parser.add_argument(
        "--synthetic-oracle-rate",
        type=float,
        default=0.0,
        help="Deterministically restore this fraction of injected rows; must be below 1.0.",
    )
    parser.add_argument("--oracle-seed", type=int, default=20260713)
    args = parser.parse_args()
    if not 0.0 <= args.synthetic_oracle_rate < 1.0:
        parser.error("--synthetic-oracle-rate must be in [0, 1)")
    rules = load_rules(
        args.result_dir / "deduped_rules.txt",
        args.min_confidence,
        allow_empty=args.synthetic_oracle_rate > 0.0,
    )
    instances = load_instances(args.result_dir / "pattern_instances.jsonl", {rule.pattern_id for rule in rules})
    graph = load_subgraph(str(args.result_dir / "subgraph_edges.csv"))
    augment_graph_structural_features(graph)
    targets = pd.read_csv(args.result_dir / "target_edges_mapped.csv")
    result = apply_rules(
        targets,
        graph,
        instances,
        rules,
        args.score_margin,
        oracle_confirm=args.oracle_confirm_rule_changes,
    )
    result = apply_partial_synthetic_oracle(
        result,
        args.synthetic_oracle_rate,
        args.oracle_seed,
    )
    result.to_csv(args.result_dir / "garplus_denoise_predictions.csv", index=False)
    result[["src", "corrected_label", "dst"]].to_csv(
        args.result_dir / "train_garplus_denoised.txt", sep="\t", header=False, index=False,
    )
    summary = summarize(result)
    summary.update({
        "usable_rules": len(rules),
        "min_confidence": args.min_confidence,
        "score_margin": args.score_margin,
        "oracle_gated": True,
        "oracle_confirm_rule_changes": args.oracle_confirm_rule_changes,
        "synthetic_oracle_rate": args.synthetic_oracle_rate,
        "oracle_seed": args.oracle_seed,
    })
    (args.result_dir / "garplus_denoise_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
