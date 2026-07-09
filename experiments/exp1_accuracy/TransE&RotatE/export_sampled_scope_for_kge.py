from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path("/home/yyyy/codework/GARplus")
DATA_DIR = BASE_DIR / "enumeration-discovery" / "\u53bb\u75c5\u56fe\u6570\u636e"

DATASETS = {
    "DDA": {
        "edge_csv": DATA_DIR / "drug_disease_signed.csv",
        "src_column": "chemical_index",
        "dst_column": "disease_index",
        "label_column": "interaction_label",
        "src_prefix": "drug",
        "dst_prefix": "disease",
    },
    "TI": {
        "edge_csv": DATA_DIR / "gene_disease_signed.csv",
        "src_column": "gene_index",
        "dst_column": "disease_index",
        "label_column": "interaction_label",
        "src_prefix": "gene",
        "dst_prefix": "disease",
    },
}

LABEL_TO_RELATION = {
    "neutral": 0,
    "unknown": 0,
    "positive": 1,
    "negative": 2,
}

RELATION_NAMES = {
    0: "no_edge",
    1: "positive",
    2: "negative",
}


def norm_id(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def read_sampled_e0_pairs(path: Path, edge_key: str) -> tuple[set[tuple[str, str]], set[str], set[str], Counter]:
    pairs: set[tuple[str, str]] = set()
    src_nodes: set[str] = set()
    dst_nodes: set[str] = set()
    stats: Counter = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["jsonl_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json_rows"] += 1
                continue
            edge = (record.get("edges") or {}).get(edge_key) or {}
            src = norm_id(edge.get("src_index", edge.get("src")))
            dst = norm_id(edge.get("dst_index", edge.get("dst")))
            if not src or not dst:
                stats["missing_e0_pair_rows"] += 1
                continue
            pairs.add((src, dst))
            src_nodes.add(src)
            dst_nodes.add(dst)

    stats["unique_e0_pairs"] = len(pairs)
    stats["sampled_src_nodes"] = len(src_nodes)
    stats["sampled_dst_nodes"] = len(dst_nodes)
    return pairs, src_nodes, dst_nodes, stats


def choose_pair_labels(
    edge_csv: Path,
    src_column: str,
    dst_column: str,
    label_column: str,
    sampled_pairs: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], str], Counter, pd.DataFrame]:
    if not edge_csv.exists():
        raise FileNotFoundError(f"Missing edge CSV: {edge_csv}")
    df = pd.read_csv(edge_csv, dtype=str, keep_default_na=False)
    required = {src_column, dst_column, label_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{edge_csv} is missing required columns: {missing}")

    df["_src_norm"] = df[src_column].map(norm_id)
    df["_dst_norm"] = df[dst_column].map(norm_id)
    df["_pair"] = list(zip(df["_src_norm"], df["_dst_norm"]))
    scoped = df[df["_pair"].isin(sampled_pairs)].copy()

    labels_by_pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for pair, label in zip(scoped["_pair"], scoped[label_column]):
        label = str(label).strip().lower()
        labels_by_pair[pair][label] += 1

    # Prefer actionable signed labels over neutral rows when duplicate evidence exists.
    priority = {"negative": 3, "positive": 2, "neutral": 1, "unknown": 0}
    pair_labels: dict[tuple[str, str], str] = {}
    stats: Counter = Counter()
    for pair, counts in labels_by_pair.items():
        stats["raw_rows_in_scope"] += sum(counts.values())
        if len(counts) > 1:
            stats["pairs_with_multiple_labels"] += 1
        label = max(counts, key=lambda key: (priority.get(key, -1), counts[key], key))
        pair_labels[pair] = label
        stats[f"pair_label_{label}"] += 1

    stats["unique_pairs_with_source_label"] = len(pair_labels)
    stats["sampled_pairs_without_source_label"] = len(sampled_pairs - set(pair_labels))
    return pair_labels, stats, scoped


def make_entity_id(src: str, dst: str, src_prefix: str, dst_prefix: str, namespace_entities: bool) -> tuple[str, str]:
    if namespace_entities:
        return f"{src_prefix}:{src}", f"{dst_prefix}:{dst}"
    return src, dst


def build_triples(
    pair_labels: dict[tuple[str, str], str],
    include_labels: set[str],
    src_prefix: str,
    dst_prefix: str,
    namespace_entities: bool,
) -> tuple[list[dict], Counter]:
    triples: list[dict] = []
    stats: Counter = Counter()
    for (src, dst), label in sorted(pair_labels.items()):
        if label not in include_labels:
            stats[f"excluded_label_{label}"] += 1
            continue
        relation = LABEL_TO_RELATION.get(label)
        if relation is None:
            stats[f"unknown_label_{label}"] += 1
            continue
        head, tail = make_entity_id(src, dst, src_prefix, dst_prefix, namespace_entities)
        triples.append(
            {
                "head": head,
                "tail": tail,
                "relation": relation,
                "relation_name": RELATION_NAMES[relation],
                "source_id": src,
                "target_id": dst,
                "true_label": label,
            }
        )
        stats[f"included_label_{label}"] += 1
    stats["included_triples"] = len(triples)
    return triples, stats


def sample_no_edges(
    src_nodes: set[str],
    dst_nodes: set[str],
    existing_pairs: set[tuple[str, str]],
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    src_list = sorted(src_nodes)
    dst_list = sorted(dst_nodes)
    capacity = len(src_list) * len(dst_list) - len(existing_pairs)
    target = min(count, max(0, capacity))
    sampled: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(10000, target * 50)
    while len(sampled) < target and attempts < max_attempts:
        attempts += 1
        pair = (rng.choice(src_list), rng.choice(dst_list))
        if pair in existing_pairs or pair in sampled:
            continue
        sampled.add(pair)
    if len(sampled) < target:
        for src in src_list:
            for dst in dst_list:
                pair = (src, dst)
                if pair not in existing_pairs and pair not in sampled:
                    sampled.add(pair)
                    if len(sampled) >= target:
                        break
            if len(sampled) >= target:
                break
    return sorted(sampled)


def split_triples(triples: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_relation: dict[int, list[dict]] = defaultdict(list)
    for triple in triples:
        by_relation[int(triple["relation"])].append(triple)

    splits = {"train": [], "valid": [], "test": []}
    for _, group in sorted(by_relation.items()):
        group = list(group)
        rng.shuffle(group)
        n_total = len(group)
        if n_total == 1:
            splits["train"].extend(group)
            continue
        n_train = max(1, int(n_total * 0.8))
        n_valid = max(1, int(n_total * 0.1)) if n_total >= 3 else 0
        if n_train + n_valid >= n_total:
            n_train = n_total - 1
            n_valid = 0
        splits["train"].extend(group[:n_train])
        splits["valid"].extend(group[n_train : n_train + n_valid])
        splits["test"].extend(group[n_train + n_valid :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def write_openke(output_dir: Path, triples: list[dict], splits: dict[str, list[dict]]) -> dict[str, int]:
    openke_dir = output_dir / "openke"
    openke_dir.mkdir(parents=True, exist_ok=True)
    entities = sorted({row["head"] for row in triples} | {row["tail"] for row in triples})
    relations = sorted({int(row["relation"]) for row in triples})
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: relation for relation in relations}

    with (openke_dir / "entity2id.txt").open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{len(entity2id)}\n")
        for entity, idx in entity2id.items():
            handle.write(f"{entity}\t{idx}\n")

    with (openke_dir / "relation2id.txt").open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{len(relation2id)}\n")
        for relation in relations:
            handle.write(f"{RELATION_NAMES[relation]}\t{relation2id[relation]}\n")

    for split_name, rows in splits.items():
        with (openke_dir / f"{split_name}2id.txt").open("w", encoding="utf-8", newline="") as handle:
            handle.write(f"{len(rows)}\n")
            for row in rows:
                handle.write(f"{entity2id[row['head']]} {entity2id[row['tail']]} {relation2id[int(row['relation'])]}\n")

    return {"openke_entities": len(entity2id), "openke_relations": len(relations)}


def write_nbfnet(output_dir: Path, splits: dict[str, list[dict]]) -> None:
    nbfnet_dir = output_dir / "nbfnet"
    nbfnet_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        with (nbfnet_dir / f"{split_name}.txt").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            for row in rows:
                writer.writerow([row["head_id"], row["relation"], row["tail_id"]])


def write_metadata(output_dir: Path, triples: list[dict], splits: dict[str, list[dict]], stats: Counter) -> None:
    rows = []
    for row in triples:
        rows.append(
            {
                "head": row["head"],
                "tail": row["tail"],
                "head_id": row["head_id"],
                "tail_id": row["tail_id"],
                "relation": row["relation"],
                "relation_name": row["relation_name"],
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "true_label": row["true_label"],
                "is_generated_no_edge": row.get("is_generated_no_edge", False),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "sampled_scope_triples.csv", index=False)

    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(candidate_dir / "sampled_prediction_candidates.csv", index=False)

    split_rows = []
    for split_name, group in splits.items():
        for row in group:
            split_rows.append(
                {
                    "split": split_name,
                    "head_id": row["head_id"],
                    "tail_id": row["tail_id"],
                    "relation": row["relation"],
                    "relation_name": row["relation_name"],
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "true_label": row["true_label"],
                    "is_generated_no_edge": row.get("is_generated_no_edge", False),
                }
            )
    pd.DataFrame(split_rows).to_csv(output_dir / "split_assignments.csv", index=False)

    with (output_dir / "export_summary.txt").open("w", encoding="utf-8") as handle:
        for key in sorted(stats):
            handle.write(f"{key}={stats[key]}\n")


def add_nbfnet_ids(triples: list[dict]) -> dict[str, int]:
    entities = sorted({row["head"] for row in triples} | {row["tail"] for row in triples})
    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    for row in triples:
        row["head_id"] = entity2id[row["head"]]
        row["tail_id"] = entity2id[row["tail"]]
    return entity2id


def write_nbfnet_node_edge_tables(output_dir: Path, triples: list[dict], entity2id: dict[str, int]) -> None:
    nbfnet_dir = output_dir / "nbfnet"
    node_rows = [{"id": idx, "name": entity} for entity, idx in sorted(entity2id.items(), key=lambda item: item[1])]
    pd.DataFrame(node_rows).to_csv(nbfnet_dir / "node.csv", index=False)
    edge_rows = [
        {
            "src": row["head_id"],
            "dst": row["tail_id"],
            "label": row["relation"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "true_label": row["true_label"],
        }
        for row in triples
    ]
    pd.DataFrame(edge_rows).to_csv(nbfnet_dir / "edge_update.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GAR sampled scope triples for TransE/RotatE and NBFNet.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--pattern-instances-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-key", default="e0")
    parser.add_argument("--include-labels", default="positive,negative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--namespace-entities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--add-no-edge-samples",
        choices=["none", "match-positive", "match-signed"],
        default="match-positive",
        help="Optionally add sampled-scope non-edges as relation 0 for NBFNet-style three-class training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DATASETS[args.dataset]
    include_labels = {item.strip().lower() for item in args.include_labels.split(",") if item.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampled_pairs, src_nodes, dst_nodes, sampled_stats = read_sampled_e0_pairs(args.pattern_instances_file, args.edge_key)
    pair_labels, source_stats, _ = choose_pair_labels(
        cfg["edge_csv"],
        cfg["src_column"],
        cfg["dst_column"],
        cfg["label_column"],
        sampled_pairs,
    )
    triples, triple_stats = build_triples(
        pair_labels,
        include_labels,
        cfg["src_prefix"],
        cfg["dst_prefix"],
        args.namespace_entities,
    )

    existing_pairs = set(pair_labels)
    positive_count = sum(1 for row in triples if row["true_label"] == "positive")
    signed_count = sum(1 for row in triples if row["true_label"] in {"positive", "negative"})
    no_edge_count = 0
    if args.add_no_edge_samples == "match-positive":
        no_edge_count = positive_count
    elif args.add_no_edge_samples == "match-signed":
        no_edge_count = signed_count
    no_edge_pairs = sample_no_edges(src_nodes, dst_nodes, existing_pairs, no_edge_count, args.seed)
    for src, dst in no_edge_pairs:
        head, tail = make_entity_id(src, dst, cfg["src_prefix"], cfg["dst_prefix"], args.namespace_entities)
        triples.append(
            {
                "head": head,
                "tail": tail,
                "relation": 0,
                "relation_name": "no_edge",
                "source_id": src,
                "target_id": dst,
                "true_label": "no_edge",
                "is_generated_no_edge": True,
            }
        )

    stats = Counter()
    stats.update(sampled_stats)
    stats.update(source_stats)
    stats.update(triple_stats)
    stats["generated_no_edge_triples"] = len(no_edge_pairs)
    stats["final_triples"] = len(triples)

    entity2id = add_nbfnet_ids(triples)
    splits = split_triples(triples, args.seed)
    for split_name, rows in splits.items():
        stats[f"{split_name}_triples"] = len(rows)
        for row in rows:
            stats[f"{split_name}_relation_{row['relation']}"] += 1

    openke_stats = write_openke(args.output_dir, triples, splits)
    stats.update(openke_stats)
    write_nbfnet(args.output_dir, splits)
    write_nbfnet_node_edge_tables(args.output_dir, triples, entity2id)
    write_metadata(args.output_dir, triples, splits, stats)

    print(f"[SampledScopeExport] dataset={args.dataset} output_dir={args.output_dir}")
    for key in sorted(stats):
        print(f"[SampledScopeExport] {key}={stats[key]}")


if __name__ == "__main__":
    main()
