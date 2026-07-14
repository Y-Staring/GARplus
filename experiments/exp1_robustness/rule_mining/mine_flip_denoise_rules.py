from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text


FLIP_TO_ORIGINAL = {0: None, 1: 0, 2: 1, 3: 0, 4: 2, 5: 1, 6: 2}
DATASET = {
    "dda": {"source_type": "drug", "target_type": "disease"},
    "ti": {"source_type": "gene", "target_type": "disease"},
}


def read_train(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["src", "noisy_label", "dst", "flipped_flag", "flip_type"],
    )
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    invalid = sorted(set(frame.flip_type) - set(FLIP_TO_ORIGINAL))
    if invalid:
        raise ValueError(f"unknown flip_type values: {invalid}")
    expected_flag = (frame.flip_type != 0).astype(int)
    if not np.array_equal(frame.flipped_flag.to_numpy(), expected_flag.to_numpy()):
        raise ValueError("flipped_flag and flip_type are inconsistent")
    frame["original_label"] = frame.noisy_label
    changed = frame.flip_type != 0
    frame.loc[changed, "original_label"] = frame.loc[changed, "flip_type"].map(FLIP_TO_ORIGINAL)
    return frame


def normalized_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def map_to_big_graph(
    frame: pd.DataFrame, node_map_path: Path, big_nodes_path: Path,
    source_type: str, target_type: str,
) -> pd.DataFrame:
    local = pd.read_csv(node_map_path)
    if set(local.columns) < {"node_id", "old_index"}:
        raise ValueError(f"bad local node map: {node_map_path}")
    local_map = dict(zip(local.node_id.astype(int), local.old_index.map(normalized_id)))
    missing = (set(frame.src) | set(frame.dst)) - set(local_map)
    if missing:
        raise ValueError(f"{len(missing)} local node ids have no old_index mapping")

    nodes = pd.read_csv(big_nodes_path, dtype=str)
    lookup = {
        (str(row.node_type).strip().lower(), normalized_id(row.node_id)): int(row.node_index)
        for row in nodes.itertuples(index=False)
    }
    frame = frame.copy()
    # prepare_data's node ids were made by concatenating all source ids followed
    # by target-only ids.  Its synthetic class-0 sampler then uses arbitrary
    # node pairs, so column position cannot be used to recover node type.
    typed_edges_path = node_map_path.with_name("edges_labeled_with_reason.csv")
    if not typed_edges_path.exists():
        raise FileNotFoundError(f"needed to recover the source/target node boundary: {typed_edges_path}")
    typed_src = pd.read_csv(typed_edges_path, usecols=["src"])["src"]
    source_block = int(pd.to_numeric(typed_src, errors="raise").max()) + 1
    if source_block <= 0 or source_block >= len(local):
        raise ValueError(f"invalid inferred source-node block size: {source_block}")

    def big_index(local_id: int) -> int | None:
        old_id = local_map[int(local_id)]
        node_type = source_type if int(local_id) < source_block else target_type
        return lookup.get((node_type, old_id))

    frame["src_old_index"] = frame.src.map(local_map)
    frame["dst_old_index"] = frame.dst.map(local_map)
    frame["src_node_type"] = np.where(frame.src < source_block, source_type, target_type)
    frame["dst_node_type"] = np.where(frame.dst < source_block, source_type, target_type)
    frame["src_big_index"] = frame.src.map(big_index)
    frame["dst_big_index"] = frame.dst.map(big_index)
    if frame[["src_big_index", "dst_big_index"]].isna().any().any():
        bad = frame[frame.src_big_index.isna() | frame.dst_big_index.isna()].head(10)
        raise ValueError(f"old_index could not be mapped into the big graph:\n{bad}")
    frame[["src_big_index", "dst_big_index"]] = frame[["src_big_index", "dst_big_index"]].astype(int)
    return frame


def graph_features(frame: pd.DataFrame, edges_path: Path) -> tuple[pd.DataFrame, list[str]]:
    edges = pd.read_csv(edges_path, usecols=["relation", "x_index", "y_index"])
    edges["x_index"] = pd.to_numeric(edges.x_index, errors="coerce")
    edges["y_index"] = pd.to_numeric(edges.y_index, errors="coerce")
    edges = edges.dropna(subset=["x_index", "y_index"])
    edges[["x_index", "y_index"]] = edges[["x_index", "y_index"]].astype(int)

    out_degree = edges.x_index.value_counts()
    in_degree = edges.y_index.value_counts()
    relations = sorted(edges.relation.dropna().astype(str).unique())
    out_by_rel = {r: g.x_index.value_counts() for r, g in edges.groupby("relation")}
    in_by_rel = {r: g.y_index.value_counts() for r, g in edges.groupby("relation")}

    result = pd.DataFrame(index=frame.index)
    result["src_out_degree"] = frame.src_big_index.map(out_degree).fillna(0)
    result["src_in_degree"] = frame.src_big_index.map(in_degree).fillna(0)
    result["dst_out_degree"] = frame.dst_big_index.map(out_degree).fillna(0)
    result["dst_in_degree"] = frame.dst_big_index.map(in_degree).fillna(0)
    for relation in relations:
        safe = relation.replace(" ", "_").replace("-", "_")
        result[f"src_out_{safe}"] = frame.src_big_index.map(out_by_rel[relation]).fillna(0)
        result[f"src_in_{safe}"] = frame.src_big_index.map(in_by_rel[relation]).fillna(0)
        result[f"dst_out_{safe}"] = frame.dst_big_index.map(out_by_rel[relation]).fillna(0)
        result[f"dst_in_{safe}"] = frame.dst_big_index.map(in_by_rel[relation]).fillna(0)
    result = np.log1p(result.astype(float))
    return result, list(result.columns)


def mine_and_evaluate(
    frame: pd.DataFrame, features: pd.DataFrame, feature_names: list[str],
    confidence: float, max_depth: int, min_leaf: int, folds: int, seed: int,
) -> tuple[pd.DataFrame, DecisionTreeClassifier, dict]:
    y = frame.original_label.to_numpy()
    noisy = frame.noisy_label.to_numpy()
    prediction = np.full(len(frame), -1, dtype=int)
    probability = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(features, y):
        model = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf,
            class_weight="balanced", random_state=seed,
        )
        model.fit(features.iloc[train_index], y[train_index])
        probs = model.predict_proba(features.iloc[test_index])
        best = probs.argmax(axis=1)
        prediction[test_index] = model.classes_[best]
        probability[test_index] = probs[np.arange(len(test_index)), best]

    change = (probability >= confidence) & (prediction != noisy)
    corrected = noisy.copy()
    corrected[change] = prediction[change]
    output = frame.copy()
    output["rule_prediction_oof"] = prediction
    output["rule_confidence_oof"] = probability
    output["changed_by_rule_oof"] = change.astype(int)
    output["corrected_label_oof"] = corrected
    output["was_noisy_correct"] = (noisy == y).astype(int)
    output["is_correct_after_rule"] = (corrected == y).astype(int)

    flipped = frame.flipped_flag.to_numpy() == 1
    summary = {
        "rows": int(len(frame)),
        "flipped_rows": int(flipped.sum()),
        "changed_rows": int(change.sum()),
        "flipped_corrected": int((change & flipped & (corrected == y)).sum()),
        "flipped_missed": int((flipped & (corrected != y)).sum()),
        "clean_rows_damaged": int((~flipped & (corrected != y)).sum()),
        "correct_before": int((noisy == y).sum()),
        "correct_after": int((corrected == y).sum()),
        "net_correct_gain": int((corrected == y).sum() - (noisy == y).sum()),
        "accuracy_before": float((noisy == y).mean()),
        "accuracy_after": float((corrected == y).mean()),
    }
    final_model = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf,
        class_weight="balanced", random_state=seed,
    ).fit(features, y)
    return output, final_model, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine interpretable big-graph rules and denoise train_c labels.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET))
    parser.add_argument("--train-c", required=True, type=Path)
    parser.add_argument("--node-map", required=True, type=Path)
    parser.add_argument("--big-nodes", required=True, type=Path)
    parser.add_argument("--big-edges", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-leaf", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = DATASET[args.dataset]
    frame = read_train(args.train_c)
    frame = map_to_big_graph(frame, args.node_map, args.big_nodes, **cfg)
    features, names = graph_features(frame, args.big_edges)
    output, model, summary = mine_and_evaluate(
        frame, features, names, args.confidence, args.max_depth, args.min_leaf, args.folds, args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "denoise_predictions.csv", index=False)
    output[["src", "corrected_label_oof", "dst"]].to_csv(
        args.output_dir / "train_denoised_oof.txt", sep="\t", index=False, header=False,
    )
    (args.output_dir / "rules.txt").write_text(
        export_text(model, feature_names=names, decimals=3), encoding="utf-8",
    )
    summary.update(vars(args))
    summary = {key: str(value) if isinstance(value, Path) else value for key, value in summary.items()}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
