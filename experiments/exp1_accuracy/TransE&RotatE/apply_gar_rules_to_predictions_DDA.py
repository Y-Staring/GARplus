from __future__ import annotations

import sys
import time
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


REFINEMENT_DIR = Path(__file__).resolve().parents[1] / "deductive_refinement"
if str(REFINEMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REFINEMENT_DIR))

from negative_edge_expander import (  # noqa: E402
    anchored_body_match_exists,
    build_adjacency_records,
    compute_score_bins,
    degree_bins,
    edge_context,
    infer_pattern_schema_from_instances,
    load_node_attrs,
    load_pattern_instances,
    load_rules,
    read_rows,
    rule_has_structural_edge_literal,
    rule_matches,
    rule_usable_for_anchored_existing_edge_labeling,
)


# =========================
# Config
# =========================

PREDICTIONS_ROOT = Path(
    r"/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/predictions(1)/predictions/"
)
PREDICTION_FILE_NAME = "predictions_with_original_ids.csv"
# Only process folders whose names start with these prefixes.  This script is
# wired to DDA graph/rules, so TI folders must not be mixed into this run.
PREDICTION_GROUP_PREFIXES = ("DDA",)
# Noisy prediction folders use names such as DDA_5pct / DDA_10pct /
# DDA_rotate_20pct.  Deductive refinement evaluates clean ML predictions, so
# skip pct folders by default.
SKIP_PREDICTION_GROUP_CONTAINS = ("pct",)

RULES_FILE = Path(
    r"/home/yyyy/codework/GARplus/enumeration-discovery/processed/dda/deduped_rules.txt"
)
PATTERN_INSTANCES_FILE = Path(
    r"/home/yyyy/codework/GARplus/enumeration-discovery/processed/dda/pattern_instances.jsonl"
)


DATA_DIR = Path(
    r"/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据"
)
GRAPH_EDGE_CSV = DATA_DIR / "drug_disease_signed.csv"
DRUG_CSV = DATA_DIR / "drug.csv"
DISEASE_CSV = DATA_DIR / "disease.csv"

OUTPUT_DIR = Path(
    r"/home/yyyy/codework/GARplus/experiments/exp1_accuracy/TransE&RotatE/modified_prediction"
)
OUTPUT_REFINED = OUTPUT_DIR / "DDA_all_predictions_schemeA_gar_refined.csv"
OUTPUT_CHANGED = OUTPUT_DIR / "DDA_all_predictions_schemeA_gar_refined_changed.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "DDA_all_predictions_schemeA_gar_refined_summary.csv"

PRED_HEAD_COL = "head"
PRED_TAIL_COL = "tail"
TRUE_COL = "true_relation"
PRED_COL = "pred_relation"
SCORE_COL = "confidence"

SRC_COLUMN = "chemical_index"
DST_COLUMN = "disease_index"
DATASET_NAME = "DDA"
SIMILARITY_THRESHOLD = 0.85
MAX_ANCHORED_PARTIAL_MATCHES = 1000
EARLY_STOP_PER_LABEL = True
DEBUG_EVERY = 1000
MIN_RULE_CONFIDENCE = 0.8
MIN_RULE_LIFT = 2.0
NEGATIVE_REFINEMENT_ONLY = True
REQUIRE_STRUCTURAL_NEGATIVE_RULE = False
MIN_NEGATIVE_ANTECEDENT_SIZE = 1
MAX_ROWS_PER_FILE = 1000
# Speed controls.  Structural rules call anchored_body_match_exists(), which is
# much slower than row-local simple-rule matching.  Keep this off for smoke
# tests; turn it on only after simple negative filtering looks promising.
ENABLE_STRUCTURAL_RULES = False
MAX_STRUCTURAL_RULES = None
# In negative-only deductive refinement, only ML-positive rows can be changed.
# Skipping other rows avoids wasted rule matching.
SKIP_UNCHANGEABLE_ROWS = True

NO_EDGE = 0
POS_EDGE = 1
NEG_EDGE = 2
LABELS = [NO_EDGE, POS_EDGE, NEG_EDGE]
PN_LABELS = [POS_EDGE, NEG_EDGE]
# Deductive refinement is a filter over predicted associations.  When a
# predicted-positive edge is contradicted by a negative GAR rule, reject it as
# no-edge instead of forcing it into the supervised negative class.
REFINEMENT_REJECT_LABEL = NO_EDGE
CONFIDENCE_RE = re.compile(r"\bconfidence=([0-9]*\.?[0-9]+)")
LIFT_RE = re.compile(r"\blift=([0-9]*\.?[0-9]+)")


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def rule_confidence(rule: object) -> float:
    value = getattr(rule, "confidence", None)
    if value is not None:
        return float(value)
    match = CONFIDENCE_RE.search(str(getattr(rule, "raw_text", "")))
    if match:
        return float(match.group(1))
    return 0.0


def rule_lift(rule: object) -> float:
    value = getattr(rule, "lift", None)
    if value is not None:
        return float(value)
    match = LIFT_RE.search(str(getattr(rule, "raw_text", "")))
    if match:
        return float(match.group(1))
    return 0.0


def discover_prediction_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing predictions root: {root}")
    files = sorted(path for path in root.rglob(PREDICTION_FILE_NAME) if path.is_file())
    if PREDICTION_GROUP_PREFIXES:
        prefixes = tuple(prefix.lower() for prefix in PREDICTION_GROUP_PREFIXES)
        files = [
            path
            for path in files
            if path.parent.name.lower().startswith(prefixes)
        ]
    if SKIP_PREDICTION_GROUP_CONTAINS:
        needles = tuple(value.lower() for value in SKIP_PREDICTION_GROUP_CONTAINS)
        files = [
            path
            for path in files
            if not any(needle in path.parent.name.lower() for needle in needles)
        ]
    if not files:
        raise FileNotFoundError(
            f"No {PREDICTION_FILE_NAME} found under predictions root: {root}"
        )
    return files


def norm_id(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def make_candidate_row(row: pd.Series) -> dict[str, str]:
    candidate = {
        SRC_COLUMN: norm_id(row[PRED_HEAD_COL]),
        DST_COLUMN: norm_id(row[PRED_TAIL_COL]),
    }
    candidate["pred_relation"] = str(row.get(PRED_COL, ""))
    candidate["true_relation"] = str(row.get(TRUE_COL, ""))
    if SCORE_COL in row:
        candidate[SCORE_COL] = str(row.get(SCORE_COL, ""))
    return candidate


def pn_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mask = y_true.isin(PN_LABELS)
    yt = y_true.loc[mask]
    yp = y_pred.loc[mask]
    if yt.empty:
        return {
            "pn_accuracy": 0.0,
            "pn_macro_precision": 0.0,
            "pn_macro_recall": 0.0,
            "pn_macro_f1": 0.0,
            "positive_f1": 0.0,
            "negative_f1": 0.0,
        }
    return {
        "pn_accuracy": accuracy_score(yt, yp),
        "pn_macro_precision": precision_score(
            yt, yp, labels=PN_LABELS, average="macro", zero_division=0
        ),
        "pn_macro_recall": recall_score(
            yt, yp, labels=PN_LABELS, average="macro", zero_division=0
        ),
        "pn_macro_f1": f1_score(yt, yp, labels=PN_LABELS, average="macro", zero_division=0),
        "positive_f1": f1_score(yt == POS_EDGE, yp == POS_EDGE, zero_division=0),
        "negative_f1": f1_score(yt == NEG_EDGE, yp == NEG_EDGE, zero_division=0),
    }


def multiclass_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print("\n" + title)
    print("-" * len(title))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def build_summary(result: pd.DataFrame, base: dict[str, object]) -> dict[str, object]:
    before_pn = pn_metrics(result[TRUE_COL], result[PRED_COL])
    after_pn = pn_metrics(result[TRUE_COL], result["refined_pred_relation"])
    before_multi = multiclass_metrics(result[TRUE_COL], result[PRED_COL])
    after_multi = multiclass_metrics(result[TRUE_COL], result["refined_pred_relation"])

    summary: dict[str, object] = {
        **base,
        "rows": len(result),
        "gar_positive_hits": int(result["gar_positive_hit"].sum()),
        "gar_negative_hits": int(result["gar_negative_hit"].sum()),
        "gar_conflict_hits": int(result["gar_conflict_hit"].sum()),
        "changed_by_gar": int(result["changed_by_gar"].sum()),
    }
    for key, value in before_pn.items():
        summary[f"before_{key}"] = value
    for key, value in after_pn.items():
        summary[f"after_{key}"] = value
        summary[f"delta_{key}"] = value - before_pn[key]
    for key, value in before_multi.items():
        summary[f"before_multi_{key}"] = value
    for key, value in after_multi.items():
        summary[f"after_multi_{key}"] = value
        summary[f"delta_multi_{key}"] = value - before_multi[key]
    return summary


def refine_predictions(
    predictions: pd.DataFrame,
    prediction_file: Path,
    prediction_group: str,
    simple_rules: list[tuple[int, object, dict]],
    structural_rules: list[tuple[int, object, dict]],
    source_node_attrs: dict,
    target_node_attrs: dict,
    node_degree_bins: dict,
    score_low: float,
    score_high: float,
    edge_records: list[dict],
    out_adj: dict,
    in_adj: dict,
    debug_stats: Counter[str],
    started: float,
) -> pd.DataFrame:
    required_columns = {PRED_HEAD_COL, PRED_TAIL_COL, TRUE_COL, PRED_COL}
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise ValueError(f"{prediction_file} missing required columns: {missing}")

    predictions = predictions.copy()
    predictions[TRUE_COL] = predictions[TRUE_COL].astype(int)
    predictions[PRED_COL] = predictions[PRED_COL].astype(int)

    rows_out = []
    for i, pred_row in predictions.iterrows():
        candidate_row = make_candidate_row(pred_row)
        src = candidate_row[SRC_COLUMN]
        dst = candidate_row[DST_COLUMN]
        original_pred = int(pred_row[PRED_COL])

        pos_matches = []
        neg_matches = []
        should_match_rules = bool(src and dst)
        if (
            SKIP_UNCHANGEABLE_ROWS
            and NEGATIVE_REFINEMENT_ONLY
            and original_pred != POS_EDGE
        ):
            should_match_rules = False

        if should_match_rules:
            simple_context = edge_context(
                row=candidate_row,
                source_node_attrs=source_node_attrs,
                target_node_attrs=target_node_attrs,
                node_degree_bins=node_degree_bins,
                src_column=SRC_COLUMN,
                dst_column=DST_COLUMN,
                similarity_threshold=SIMILARITY_THRESHOLD,
                dataset_name=DATASET_NAME,
                score_low=score_low,
                score_high=score_high,
            )
            for rule_index, rule, _schema in simple_rules:
                if rule_matches(rule, simple_context):
                    if rule.predicted_label == "positive":
                        pos_matches.append((rule_index, rule))
                    elif rule.predicted_label == "negative":
                        neg_matches.append((rule_index, rule))
                    if EARLY_STOP_PER_LABEL and pos_matches and neg_matches:
                        break

            if (
                ENABLE_STRUCTURAL_RULES
                and not (EARLY_STOP_PER_LABEL and pos_matches and neg_matches)
            ):
                for rule_index, rule, schema in structural_rules:
                    if EARLY_STOP_PER_LABEL:
                        if rule.predicted_label == "positive" and pos_matches:
                            continue
                        if rule.predicted_label == "negative" and neg_matches:
                            continue
                    context = anchored_body_match_exists(
                        rule,
                        schema,
                        candidate_row,
                        edge_records,
                        out_adj,
                        in_adj,
                        source_node_attrs,
                        target_node_attrs,
                        node_degree_bins,
                        SRC_COLUMN,
                        DST_COLUMN,
                        DATASET_NAME,
                        SIMILARITY_THRESHOLD,
                        score_low,
                        score_high,
                        MAX_ANCHORED_PARTIAL_MATCHES,
                        debug_stats,
                    )
                    if context is not None:
                        if rule.predicted_label == "positive":
                            pos_matches.append((rule_index, rule))
                        elif rule.predicted_label == "negative":
                            neg_matches.append((rule_index, rule))
                        if EARLY_STOP_PER_LABEL and pos_matches and neg_matches:
                            break

        refined = original_pred
        # Deductive refinement: accept an ML-predicted association only when no
        # negative GAR rule derives a contradiction.  Rejected associations are
        # mapped to no-edge, not to the supervised negative class.
        if NEGATIVE_REFINEMENT_ONLY:
            if original_pred == POS_EDGE and neg_matches and not pos_matches:
                refined = REFINEMENT_REJECT_LABEL
        else:
            if neg_matches and not pos_matches:
                refined = NEG_EDGE
            elif pos_matches and not neg_matches:
                refined = POS_EDGE

        output = pred_row.to_dict()
        output.update(
            {
                "prediction_group": prediction_group,
                "prediction_file": str(prediction_file),
                "prediction_relpath": str(prediction_file.relative_to(PREDICTIONS_ROOT)),
                "schemeA_src": src,
                "schemeA_dst": dst,
                "gar_positive_hit": int(bool(pos_matches)),
                "gar_negative_hit": int(bool(neg_matches)),
                "gar_conflict_hit": int(bool(pos_matches and neg_matches)),
                "gar_positive_rule_index": pos_matches[0][0] if pos_matches else "",
                "gar_negative_rule_index": neg_matches[0][0] if neg_matches else "",
                "gar_positive_rule_pattern_id": pos_matches[0][1].pattern_id if pos_matches else "",
                "gar_negative_rule_pattern_id": neg_matches[0][1].pattern_id if neg_matches else "",
                "refined_pred_relation": refined,
                "changed_by_gar": int(refined != int(pred_row[PRED_COL])),
            }
        )
        rows_out.append(output)

        if (i + 1) % DEBUG_EVERY == 0:
            print(
                f"[Progress] group={prediction_group} rows={i + 1}/{len(predictions)} "
                f"pos_hits={sum(r['gar_positive_hit'] for r in rows_out)} "
                f"neg_hits={sum(r['gar_negative_hit'] for r in rows_out)} "
                f"changed={sum(r['changed_by_gar'] for r in rows_out)} "
                f"anchored_calls={debug_stats.get('anchored_match_calls', 0)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows_out)


def main() -> None:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    require_file(RULES_FILE, "GAR deduped rules")
    require_file(PATTERN_INSTANCES_FILE, "GAR pattern instances")
    require_file(GRAPH_EDGE_CSV, "DDA graph edge CSV")
    require_file(DRUG_CSV, "drug node CSV")
    require_file(DISEASE_CSV, "disease node CSV")

    prediction_files = discover_prediction_files(PREDICTIONS_ROOT)
    print(f"[Discover] prediction_files={len(prediction_files)} root={PREDICTIONS_ROOT}")
    for path in prediction_files:
        print(f"  - {path.relative_to(PREDICTIONS_ROOT)}")

    print("[Load] DDA graph rows")
    graph_rows, _ = read_rows(str(GRAPH_EDGE_CSV))

    print("[Load] rules and schemas")
    rules = load_rules(RULES_FILE)
    instances_by_pattern = load_pattern_instances(PATTERN_INSTANCES_FILE)
    schemas = infer_pattern_schema_from_instances(instances_by_pattern)

    print("[Load] node attrs")
    source_node_attrs = load_node_attrs(str(DRUG_CSV), "index")
    target_node_attrs = load_node_attrs(str(DISEASE_CSV), "index")

    print("[Build] graph indexes")
    node_degree_bins = degree_bins(graph_rows, SRC_COLUMN, DST_COLUMN)
    score_low, score_high = compute_score_bins(graph_rows, SRC_COLUMN, DST_COLUMN)
    edge_records, out_adj, in_adj = build_adjacency_records(
        graph_rows,
        SRC_COLUMN,
        DST_COLUMN,
        DATASET_NAME,
        score_low,
        score_high,
    )

    print("[Build] usable rules")
    simple_rules = []
    structural_rules = []
    skipped_no_schema = 0
    skipped_unusable = 0
    skipped_low_confidence = 0
    skipped_low_lift = 0
    skipped_negative_shape = 0
    skipped_structural_disabled = 0
    for rule_index, rule in enumerate(rules):
        if rule_confidence(rule) < MIN_RULE_CONFIDENCE:
            skipped_low_confidence += 1
            continue
        if rule_lift(rule) < MIN_RULE_LIFT:
            skipped_low_lift += 1
            continue
        if not rule_usable_for_anchored_existing_edge_labeling(rule):
            skipped_unusable += 1
            continue
        schema = schemas.get(rule.pattern_id)
        if not schema or "e0" not in schema:
            skipped_no_schema += 1
            continue
        is_structural = rule_has_structural_edge_literal(rule)
        if rule.predicted_label == "negative":
            if REQUIRE_STRUCTURAL_NEGATIVE_RULE and not is_structural:
                skipped_negative_shape += 1
                continue
            if len(rule.antecedent) < MIN_NEGATIVE_ANTECEDENT_SIZE:
                skipped_negative_shape += 1
                continue
        item = (rule_index, rule, schema)
        if is_structural:
            if not ENABLE_STRUCTURAL_RULES:
                skipped_structural_disabled += 1
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
    print(
        f"rules={len(rules)} simple={len(simple_rules)} structural={len(structural_rules)} "
        f"min_rule_confidence={MIN_RULE_CONFIDENCE} min_rule_lift={MIN_RULE_LIFT} "
        f"enable_structural_rules={int(ENABLE_STRUCTURAL_RULES)} "
        f"skipped_low_confidence={skipped_low_confidence} "
        f"skipped_low_lift={skipped_low_lift} "
        f"skipped_negative_shape={skipped_negative_shape} "
        f"skipped_structural_disabled={skipped_structural_disabled} "
        f"skipped_unusable={skipped_unusable} skipped_no_schema={skipped_no_schema}"
    )

    all_results = []
    summary_rows = []
    total_debug_stats: Counter[str] = Counter()

    for prediction_file in prediction_files:
        group_started = time.perf_counter()
        prediction_group = prediction_file.parent.name
        print(f"\n[Process] group={prediction_group} file={prediction_file}")
        predictions = pd.read_csv(prediction_file)
        if MAX_ROWS_PER_FILE is not None:
            predictions = predictions.head(MAX_ROWS_PER_FILE).copy()
            print(f"[Limit] group={prediction_group} max_rows_per_file={MAX_ROWS_PER_FILE}")
        group_debug_stats: Counter[str] = Counter()
        result = refine_predictions(
            predictions=predictions,
            prediction_file=prediction_file,
            prediction_group=prediction_group,
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
            debug_stats=group_debug_stats,
            started=started,
        )
        total_debug_stats.update(group_debug_stats)
        all_results.append(result)

        group_summary = build_summary(
            result,
            {
                "prediction_group": prediction_group,
                "prediction_file": str(prediction_file),
                "rules": len(rules),
                "simple_rules": len(simple_rules),
                "structural_rules": len(structural_rules),
                "min_rule_confidence": MIN_RULE_CONFIDENCE,
                "min_rule_lift": MIN_RULE_LIFT,
                "negative_refinement_only": int(NEGATIVE_REFINEMENT_ONLY),
                "refinement_reject_label": REFINEMENT_REJECT_LABEL,
                "require_structural_negative_rule": int(REQUIRE_STRUCTURAL_NEGATIVE_RULE),
                "min_negative_antecedent_size": MIN_NEGATIVE_ANTECEDENT_SIZE,
                "max_rows_per_file": MAX_ROWS_PER_FILE,
                "prediction_group_prefixes": "|".join(PREDICTION_GROUP_PREFIXES),
                "skip_prediction_group_contains": "|".join(SKIP_PREDICTION_GROUP_CONTAINS),
                "enable_structural_rules": int(ENABLE_STRUCTURAL_RULES),
                "skip_unchangeable_rows": int(SKIP_UNCHANGEABLE_ROWS),
                "anchored_match_calls": group_debug_stats.get("anchored_match_calls", 0),
                "seconds": time.perf_counter() - group_started,
            },
        )
        summary_rows.append(group_summary)
        print(
            f"[GroupSummary] group={prediction_group} rows={group_summary['rows']} "
            f"changed={group_summary['changed_by_gar']} "
            f"after_pn_f1={group_summary['after_pn_macro_f1']:.4f} "
            f"delta_pn_f1={group_summary['delta_pn_macro_f1']:.4f}"
        )

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_REFINED, index=False)
    combined[combined["changed_by_gar"] == 1].to_csv(OUTPUT_CHANGED, index=False)

    overall_summary = build_summary(
        combined,
        {
            "prediction_group": "__ALL__",
            "prediction_file": str(PREDICTIONS_ROOT),
            "rules": len(rules),
            "simple_rules": len(simple_rules),
            "structural_rules": len(structural_rules),
            "min_rule_confidence": MIN_RULE_CONFIDENCE,
            "min_rule_lift": MIN_RULE_LIFT,
            "negative_refinement_only": int(NEGATIVE_REFINEMENT_ONLY),
            "refinement_reject_label": REFINEMENT_REJECT_LABEL,
            "require_structural_negative_rule": int(REQUIRE_STRUCTURAL_NEGATIVE_RULE),
            "min_negative_antecedent_size": MIN_NEGATIVE_ANTECEDENT_SIZE,
            "max_rows_per_file": MAX_ROWS_PER_FILE,
            "prediction_group_prefixes": "|".join(PREDICTION_GROUP_PREFIXES),
            "skip_prediction_group_contains": "|".join(SKIP_PREDICTION_GROUP_CONTAINS),
            "enable_structural_rules": int(ENABLE_STRUCTURAL_RULES),
            "skip_unchangeable_rows": int(SKIP_UNCHANGEABLE_ROWS),
            "anchored_match_calls": total_debug_stats.get("anchored_match_calls", 0),
            "seconds": time.perf_counter() - started,
        },
    )
    summary_df = pd.DataFrame([overall_summary, *summary_rows])
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print_metrics("Overall before PN", pn_metrics(combined[TRUE_COL], combined[PRED_COL]))
    print_metrics(
        "Overall after PN",
        pn_metrics(combined[TRUE_COL], combined["refined_pred_relation"]),
    )
    print_metrics("Overall before multiclass", multiclass_metrics(combined[TRUE_COL], combined[PRED_COL]))
    print_metrics(
        "Overall after multiclass",
        multiclass_metrics(combined[TRUE_COL], combined["refined_pred_relation"]),
    )

    print("\n[Summary]")
    for key, value in overall_summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved combined refined predictions: {OUTPUT_REFINED}")
    print(f"Saved combined changed predictions: {OUTPUT_CHANGED}")
    print(f"Saved per-group and overall summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
