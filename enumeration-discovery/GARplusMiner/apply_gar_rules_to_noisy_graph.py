from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd


GARPLUS_MINER_DIR = Path(
    r"/home/yyyy/codework/GARplus/enumeration-discovery/GARplusMiner"
)
if str(GARPLUS_MINER_DIR) not in sys.path:
    sys.path.insert(0, str(GARPLUS_MINER_DIR))

from negative_edge_expander import (  # noqa: E402
    anchored_body_match_exists,
    build_adjacency_records,
    compute_score_bins,
    degree_bins,
    edge_context,
    endpoint_ids,
    infer_pattern_schema_from_instances,
    load_node_attrs,
    load_pattern_instances,
    load_rules,
    normalize_value,
    read_rows,
    rule_has_structural_edge_literal,
    rule_matches,
    rule_usable_for_anchored_existing_edge_labeling,
)


# =========================
# Config: edit here for each noisy graph run
# =========================

DATASET_NAME = "DDA"  # "PPI", "DDA", or "TI"

BASE_DIR = Path(r"/home/yyyy/codework/GARplus")
MINER_DIR = BASE_DIR / "enumeration-discovery" / "GARplusMiner"
PROCESSED_DIR = BASE_DIR / "enumeration-discovery" / "processed"
DATA_DIR = BASE_DIR / "enumeration-discovery" / "去病图数据"

RULES_FILE = PROCESSED_DIR / DATASET_NAME.lower() / "deduped_rules.txt"
PATTERN_INSTANCES_FILE = PROCESSED_DIR / DATASET_NAME.lower() / "pattern_instances.jsonl"

# Set this to the noisy training graph CSV, e.g. a 5pct/10pct/20pct noisy file.
INPUT_NOISY_CSV = DATA_DIR / "drug_disease_signed.csv"

OUTPUT_DIR = PROCESSED_DIR / DATASET_NAME.lower() / "gar_denoised"
OUTPUT_CLEAN_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_cleaned.csv"
OUTPUT_REMOVED_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_removed_edges.csv"
OUTPUT_MARKED_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_marked_all.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_denoise_summary.csv"

LABEL_COLUMN = "interaction_label"
NEGATIVE_VALUE = "negative"
SIMILARITY_THRESHOLD = 0.85

# For graph denoising, GAR negative rules are used as a semantic filter.
# Rows whose labels are in CHECK_LABELS are tested; if a negative rule matches,
# the row is considered inconsistent/noisy and removed from OUTPUT_CLEAN_CSV.
# Set CHECK_LABELS = None to test every edge in the noisy graph.
CHECK_LABELS: set[str] | None = None

# If your noisy data explicitly labels injected false edges as "negative",
# keeping this True makes the script focus on those negative/noisy labels.
# If False, negative labels are still checked when CHECK_LABELS allows them,
# but they are not specially counted.
FOCUS_EXISTING_NEGATIVE_LABELS = True

MIN_RULE_CONFIDENCE = 0.8
MIN_RULE_LIFT = 2.0
MIN_NEGATIVE_ANTECEDENT_SIZE = 1

# Structural rules are much slower because they require anchored body matching.
# Start with False for fast smoke tests; turn on with a small top-k if needed.
ENABLE_STRUCTURAL_RULES = False
MAX_STRUCTURAL_RULES: int | None = None
MAX_ANCHORED_PARTIAL_MATCHES = 1000

DEBUG_EVERY_ROWS = 50000

CONFIDENCE_RE = re.compile(r"\bconfidence=([0-9]*\.?[0-9]+)")
LIFT_RE = re.compile(r"\blift=([0-9]*\.?[0-9]+)")


DATASET_COLUMNS = {
    "PPI": {
        "src": "index_A",
        "dst": "index_B",
        "source_node_csv": DATA_DIR / "protein.csv",
        "target_node_csv": DATA_DIR / "protein.csv",
        "source_index": "index",
        "target_index": "index",
    },
    "DDA": {
        "src": "chemical_index",
        "dst": "disease_index",
        "source_node_csv": DATA_DIR / "drug.csv",
        "target_node_csv": DATA_DIR / "disease.csv",
        "source_index": "index",
        "target_index": "index",
    },
    "TI": {
        "src": "gene_index",
        "dst": "disease_index",
        "source_node_csv": DATA_DIR / "gene.csv",
        "target_node_csv": DATA_DIR / "disease.csv",
        "source_index": "index",
        "target_index": "index",
    },
}


def parse_label_set(value: str | None) -> set[str] | None:
    if value is None or value.strip().lower() in {"", "none", "all", "*"}:
        return None
    return {normalize_value(part) for part in value.split(",") if normalize_value(part)}


def apply_cli_overrides() -> None:
    global DATASET_NAME
    global RULES_FILE
    global PATTERN_INSTANCES_FILE
    global INPUT_NOISY_CSV
    global OUTPUT_DIR
    global OUTPUT_CLEAN_CSV
    global OUTPUT_REMOVED_CSV
    global OUTPUT_MARKED_CSV
    global OUTPUT_SUMMARY_CSV
    global LABEL_COLUMN
    global CHECK_LABELS
    global FOCUS_EXISTING_NEGATIVE_LABELS
    global MIN_RULE_CONFIDENCE
    global MIN_RULE_LIFT
    global ENABLE_STRUCTURAL_RULES
    global MAX_STRUCTURAL_RULES

    parser = argparse.ArgumentParser(
        description="Apply GAR negative rules to a noisy graph CSV and export cleaned/removed edges."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_COLUMNS), help="Dataset name.")
    parser.add_argument("--input-noisy-csv", type=Path, help="Noisy graph CSV to denoise.")
    parser.add_argument("--output-dir", type=Path, help="Directory for cleaned/removed/summary CSVs.")
    parser.add_argument("--rules-file", type=Path, help="GAR deduped rules file.")
    parser.add_argument("--pattern-instances-file", type=Path, help="GAR pattern instances JSONL file.")
    parser.add_argument("--label-column", help="Column used to decide which rows are checked.")
    parser.add_argument(
        "--check-labels",
        help='Comma-separated labels to check, e.g. "1" for injected noise. Use all/none/* to check all rows.',
    )
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help='Shortcut for --label-column noise_label --check-labels 1 --no-focus-existing-negative.',
    )
    parser.add_argument("--focus-existing-negative", dest="focus_existing_negative", action="store_true")
    parser.add_argument("--no-focus-existing-negative", dest="focus_existing_negative", action="store_false")
    parser.set_defaults(focus_existing_negative=None)
    parser.add_argument("--min-rule-confidence", type=float, help="Minimum negative rule confidence.")
    parser.add_argument("--min-rule-lift", type=float, help="Minimum negative rule lift.")
    parser.add_argument("--enable-structural-rules", action="store_true", help="Enable anchored structural rules.")
    parser.add_argument("--max-structural-rules", type=int, help="Optional cap for structural rules.")
    args = parser.parse_args()

    if args.dataset:
        DATASET_NAME = args.dataset
    if args.rules_file:
        RULES_FILE = args.rules_file
    else:
        RULES_FILE = PROCESSED_DIR / DATASET_NAME.lower() / "deduped_rules.txt"
    if args.pattern_instances_file:
        PATTERN_INSTANCES_FILE = args.pattern_instances_file
    else:
        PATTERN_INSTANCES_FILE = PROCESSED_DIR / DATASET_NAME.lower() / "pattern_instances.jsonl"
    if args.input_noisy_csv:
        INPUT_NOISY_CSV = args.input_noisy_csv
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    if args.noise_only:
        LABEL_COLUMN = "noise_label"
        CHECK_LABELS = {"1"}
        FOCUS_EXISTING_NEGATIVE_LABELS = False
    if args.label_column:
        LABEL_COLUMN = args.label_column
    if args.check_labels is not None:
        CHECK_LABELS = parse_label_set(args.check_labels)
    if args.focus_existing_negative is not None:
        FOCUS_EXISTING_NEGATIVE_LABELS = args.focus_existing_negative
    if args.min_rule_confidence is not None:
        MIN_RULE_CONFIDENCE = args.min_rule_confidence
    if args.min_rule_lift is not None:
        MIN_RULE_LIFT = args.min_rule_lift
    if args.enable_structural_rules:
        ENABLE_STRUCTURAL_RULES = True
    if args.max_structural_rules is not None:
        MAX_STRUCTURAL_RULES = args.max_structural_rules

    OUTPUT_CLEAN_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_cleaned.csv"
    OUTPUT_REMOVED_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_removed_edges.csv"
    OUTPUT_MARKED_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_marked_all.csv"
    OUTPUT_SUMMARY_CSV = OUTPUT_DIR / f"{DATASET_NAME.lower()}_gar_denoise_summary.csv"


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def rule_confidence(rule: object) -> float:
    value = getattr(rule, "confidence", None)
    if value is not None:
        return float(value)
    match = CONFIDENCE_RE.search(str(getattr(rule, "raw_text", "")))
    return float(match.group(1)) if match else 0.0


def rule_lift(rule: object) -> float:
    value = getattr(rule, "lift", None)
    if value is not None:
        return float(value)
    match = LIFT_RE.search(str(getattr(rule, "raw_text", "")))
    return float(match.group(1)) if match else 0.0


def rule_is_negative_e0(rule: object) -> bool:
    return getattr(rule, "predicted_label", "") == NEGATIVE_VALUE


def build_usable_negative_rules(
    rules: list[object],
    schemas: dict[int, dict[str, tuple[str, str]]],
) -> tuple[
    list[tuple[int, object, dict[str, tuple[str, str]]]],
    list[tuple[int, object, dict[str, tuple[str, str]]]],
    Counter[str],
]:
    simple_rules: list[tuple[int, object, dict[str, tuple[str, str]]]] = []
    structural_rules: list[tuple[int, object, dict[str, tuple[str, str]]]] = []
    stats: Counter[str] = Counter()

    for rule_index, rule in enumerate(rules):
        if not rule_is_negative_e0(rule):
            stats["skipped_non_negative"] += 1
            continue
        if rule_confidence(rule) < MIN_RULE_CONFIDENCE:
            stats["skipped_low_confidence"] += 1
            continue
        if rule_lift(rule) < MIN_RULE_LIFT:
            stats["skipped_low_lift"] += 1
            continue
        if len(getattr(rule, "antecedent", ())) < MIN_NEGATIVE_ANTECEDENT_SIZE:
            stats["skipped_short_antecedent"] += 1
            continue
        if not rule_usable_for_anchored_existing_edge_labeling(rule):
            stats["skipped_unusable"] += 1
            continue
        schema = schemas.get(rule.pattern_id)
        if not schema or "e0" not in schema:
            stats["skipped_no_schema"] += 1
            continue

        item = (rule_index, rule, schema)
        if rule_has_structural_edge_literal(rule):
            if not ENABLE_STRUCTURAL_RULES:
                stats["skipped_structural_disabled"] += 1
                continue
            structural_rules.append(item)
        else:
            simple_rules.append(item)

    if MAX_STRUCTURAL_RULES is not None and len(structural_rules) > MAX_STRUCTURAL_RULES:
        structural_rules = sorted(
            structural_rules,
            key=lambda item: (rule_confidence(item[1]), rule_lift(item[1])),
            reverse=True,
        )[:MAX_STRUCTURAL_RULES]

    stats["simple_rules"] = len(simple_rules)
    stats["structural_rules"] = len(structural_rules)
    return simple_rules, structural_rules, stats


def row_matches_negative_rule(
    row: dict[str, str],
    simple_rules: list[tuple[int, object, dict[str, tuple[str, str]]]],
    structural_rules: list[tuple[int, object, dict[str, tuple[str, str]]]],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    node_degree_bins: dict[str, str],
    score_low: float | None,
    score_high: float | None,
    edge_records: list[dict],
    out_adj: dict[str, list[dict]],
    in_adj: dict[str, list[dict]],
    src_column: str,
    dst_column: str,
    debug_stats: Counter[str],
) -> tuple[list[tuple[int, object]], list[tuple[int, object]]]:
    simple_matches: list[tuple[int, object]] = []
    structural_matches: list[tuple[int, object]] = []

    context = edge_context(
        row=row,
        source_node_attrs=source_node_attrs,
        target_node_attrs=target_node_attrs,
        node_degree_bins=node_degree_bins,
        src_column=src_column,
        dst_column=dst_column,
        similarity_threshold=SIMILARITY_THRESHOLD,
        dataset_name=DATASET_NAME,
        score_low=score_low,
        score_high=score_high,
    )
    for rule_index, rule, _schema in simple_rules:
        if rule_matches(rule, context):
            simple_matches.append((rule_index, rule))

    if ENABLE_STRUCTURAL_RULES:
        for rule_index, rule, schema in structural_rules:
            anchored_context = anchored_body_match_exists(
                rule,
                schema,
                row,
                edge_records,
                out_adj,
                in_adj,
                source_node_attrs,
                target_node_attrs,
                node_degree_bins,
                src_column,
                dst_column,
                DATASET_NAME,
                SIMILARITY_THRESHOLD,
                score_low,
                score_high,
                MAX_ANCHORED_PARTIAL_MATCHES,
                debug_stats,
            )
            if anchored_context is not None:
                structural_matches.append((rule_index, rule))

    return simple_matches, structural_matches


def main() -> None:
    apply_cli_overrides()
    started = time.perf_counter()
    if DATASET_NAME not in DATASET_COLUMNS:
        raise ValueError(f"Unsupported DATASET_NAME={DATASET_NAME}")

    cfg = DATASET_COLUMNS[DATASET_NAME]
    src_column = cfg["src"]
    dst_column = cfg["dst"]

    require_file(INPUT_NOISY_CSV, "noisy interaction CSV")
    require_file(RULES_FILE, "GAR deduped rules")
    require_file(PATTERN_INSTANCES_FILE, "GAR pattern instances")
    require_file(cfg["source_node_csv"], "source node CSV")
    require_file(cfg["target_node_csv"], "target node CSV")

    print(f"[GARNoiseDenoise] dataset={DATASET_NAME} input={INPUT_NOISY_CSV}")
    print(f"[GARNoiseDenoise] rules={RULES_FILE}")

    rows, fields = read_rows(str(INPUT_NOISY_CSV))
    rules = load_rules(RULES_FILE)
    instances_by_pattern = load_pattern_instances(PATTERN_INSTANCES_FILE)
    schemas = infer_pattern_schema_from_instances(instances_by_pattern)

    source_node_attrs = load_node_attrs(str(cfg["source_node_csv"]), cfg["source_index"])
    target_node_attrs = load_node_attrs(str(cfg["target_node_csv"]), cfg["target_index"])

    node_degree_bins = degree_bins(rows, src_column, dst_column)
    score_low, score_high = compute_score_bins(rows, src_column, dst_column)
    edge_records, out_adj, in_adj = build_adjacency_records(
        rows,
        src_column,
        dst_column,
        DATASET_NAME,
        score_low,
        score_high,
    )

    simple_rules, structural_rules, rule_stats = build_usable_negative_rules(rules, schemas)
    print(
        "[GARNoiseDenoiseRules] "
        + f"rules={len(rules)} simple={len(simple_rules)} structural={len(structural_rules)} "
        + f"min_conf={MIN_RULE_CONFIDENCE} min_lift={MIN_RULE_LIFT} "
        + f"enable_structural={int(ENABLE_STRUCTURAL_RULES)} "
        + " ".join(f"{key}={value}" for key, value in sorted(rule_stats.items()))
    )

    output_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    debug_stats: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        src, dst = endpoint_ids(row, src_column, dst_column)
        label = normalize_value(row.get(LABEL_COLUMN))
        label_counts[label or "<empty>"] += 1

        checked = False
        remove_by_gar = False
        simple_matches: list[tuple[int, object]] = []
        structural_matches: list[tuple[int, object]] = []

        if src and dst and (CHECK_LABELS is None or label in CHECK_LABELS):
            checked = True
            stats["checked_rows"] += 1
            simple_matches, structural_matches = row_matches_negative_rule(
                row=row,
                simple_rules=simple_rules,
                structural_rules=structural_rules,
                source_node_attrs=source_node_attrs,
                target_node_attrs=target_node_attrs,
                node_degree_bins=node_degree_bins,
                score_low=score_low,
                score_high=score_high,
                edge_records=edge_records,
                out_adj=out_adj,
                in_adj=in_adj,
                src_column=src_column,
                dst_column=dst_column,
                debug_stats=debug_stats,
            )
            remove_by_gar = bool(simple_matches or structural_matches)
        elif not src or not dst:
            stats["skipped_missing_endpoint"] += 1
        else:
            stats["skipped_label_not_checked"] += 1

        if label == NEGATIVE_VALUE and FOCUS_EXISTING_NEGATIVE_LABELS:
            stats["existing_negative_rows"] += 1

        matched_rules = simple_matches or structural_matches
        first_rule_index = matched_rules[0][0] if matched_rules else ""
        first_rule = matched_rules[0][1] if matched_rules else None

        marked = dict(row)
        marked.update(
            {
                "gar_checked": int(checked),
                "gar_remove_as_noise": int(remove_by_gar),
                "gar_negative_rule_count": len(simple_matches) + len(structural_matches),
                "gar_simple_rule_count": len(simple_matches),
                "gar_structural_rule_count": len(structural_matches),
                "gar_negative_rule_index": first_rule_index,
                "gar_negative_rule_pattern_id": first_rule.pattern_id if first_rule else "",
                "gar_negative_rule_antecedent": " & ".join(first_rule.antecedent) if first_rule else "",
            }
        )

        if remove_by_gar:
            stats["removed_rows"] += 1
            removed_rows.append(marked)
        else:
            output_rows.append(row)

        if DEBUG_EVERY_ROWS and index % DEBUG_EVERY_ROWS == 0:
            print(
                f"[GARNoiseDenoiseProgress] rows={index}/{len(rows)} "
                f"checked={stats['checked_rows']} removed={stats['removed_rows']} "
                f"anchored_calls={debug_stats.get('anchored_match_calls', 0)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=fields).to_csv(OUTPUT_CLEAN_CSV, index=False)
    pd.DataFrame(removed_rows).to_csv(OUTPUT_REMOVED_CSV, index=False)
    pd.DataFrame(output_rows + removed_rows).to_csv(OUTPUT_MARKED_CSV, index=False)

    summary = {
        "dataset": DATASET_NAME,
        "input_csv": str(INPUT_NOISY_CSV),
        "output_clean_csv": str(OUTPUT_CLEAN_CSV),
        "output_removed_csv": str(OUTPUT_REMOVED_CSV),
        "rows": len(rows),
        "clean_rows": len(output_rows),
        "removed_rows": stats["removed_rows"],
        "checked_rows": stats["checked_rows"],
        "existing_negative_rows": stats["existing_negative_rows"],
        "skipped_missing_endpoint": stats["skipped_missing_endpoint"],
        "skipped_label_not_checked": stats["skipped_label_not_checked"],
        "rules": len(rules),
        "simple_rules": len(simple_rules),
        "structural_rules": len(structural_rules),
        "enable_structural_rules": int(ENABLE_STRUCTURAL_RULES),
        "min_rule_confidence": MIN_RULE_CONFIDENCE,
        "min_rule_lift": MIN_RULE_LIFT,
        "anchored_match_calls": debug_stats.get("anchored_match_calls", 0),
        "seconds": time.perf_counter() - started,
        "label_counts": ",".join(f"{label}:{count}" for label, count in label_counts.most_common()),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(
        "[GARNoiseDenoiseSummary] "
        + " ".join(f"{key}={value}" for key, value in summary.items())
    )


if __name__ == "__main__":
    main()
