"""Batch-train TI/DDA/PPI GNN edge classifiers over built ablation datasets.

The model and data-preparation code are a command-line version of the logic in
``experiments/exp1_accuracy/TI_test/{GAT,HGT,RGCN}_DDKG.ipynb``. By default it
trains on each variant's ``gar_augmented_edges.csv`` created by
``build_ablation_gnn_edges.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import traceback
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import HGTConv
from dgl.nn import GraphConv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_REPO_ROOT = Path("/home/yyyy/codework/GARplus")
DEFAULT_DATASETS = ("TI",)
DEFAULT_VARIANTS = (
    "full",
    "wo_order_embedding",
    "wo_bayesian_pruning",
    "wo_logicgar",
    "wo_neuralgar",
    "logicgar_only",
    "neuralgar_only",
)
DEFAULT_EDGE_FILES = ("gar_augmented_edges.csv",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-train GAT/HGT/RGCN on ablation GNN edge CSVs.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=("TI", "DDA", "PPI"))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--edge-files", nargs="+", default=list(DEFAULT_EDGE_FILES))
    parser.add_argument("--models", nargs="+", default=["gat"], choices=("gat", "hgt", "rgcn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    return parser.parse_args()


def reset_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EdgeDataset(Dataset):
    def __init__(self, src: np.ndarray, dst: np.ndarray, label: np.ndarray):
        self.src = torch.as_tensor(src, dtype=torch.long)
        self.dst = torch.as_tensor(dst, dtype=torch.long)
        self.label = torch.as_tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.src[idx], self.dst[idx], self.label[idx]


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads=num_heads, feat_drop=0, attn_drop=0)
        self.layer2 = GATConv(hidden_dim, out_dim, num_heads=1, feat_drop=dropout, attn_drop=dropout)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = self.layer1(g, feat)
        h = h.mean(dim=1)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(g, h)
        return h.squeeze(1)


class EdgeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([emb[src], emb[dst]], dim=1))


class GATModel(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.encoder = GATEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_heads=num_heads)
        self.decoder = EdgeClassifier(out_dim)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(g, feat)
        return self.decoder(emb, src, dst)


class HGTEncoder(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_ntypes = 1
        self.num_etypes = 1
        self.dropout = dropout
        self.layer1 = HGTConv(
            in_dim,
            hidden_dim,
            num_heads,
            self.num_ntypes,
            self.num_etypes,
            dropout=dropout,
            use_norm=True,
        )
        self.layer2 = HGTConv(
            hidden_dim * num_heads,
            out_dim,
            1,
            self.num_ntypes,
            self.num_etypes,
            dropout=dropout,
            use_norm=True,
        )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        ntype = torch.zeros(g.num_nodes(), dtype=torch.long, device=feat.device)
        etype = torch.zeros(g.num_edges(), dtype=torch.long, device=feat.device)
        h = self.layer1(g, feat, ntype, etype)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(g, h, ntype, etype)
        return h.squeeze(1)


class HGTModel(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 32, out_dim: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = HGTEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = EdgeClassifier(out_dim)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(g, feat)
        return self.decoder(emb, src, dst)


class RGCNEncoder(nn.Module):
    def __init__(self, in_feats: int = 16, h_feats: int = 32, out_feats: int = 32):
        super().__init__()
        self.layer1 = GraphConv(in_feats, h_feats)
        self.layer2 = GraphConv(h_feats, out_feats)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.layer1(g, feat))
        return self.layer2(g, h)


class RGCNModel(nn.Module):
    def __init__(self, in_dim: int = 16):
        super().__init__()
        self.encoder = RGCNEncoder(in_dim, 32, 32)
        self.decoder = EdgeClassifier(32, 64, 3)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(g, feat)
        return self.decoder(emb, src, dst)


def create_model(model_name: str) -> nn.Module:
    if model_name == "gat":
        return GATModel(in_dim=16, hidden_dim=32, out_dim=32, num_heads=4)
    if model_name == "hgt":
        return HGTModel(in_dim=16, hidden_dim=32, out_dim=32, num_heads=4)
    if model_name == "rgcn":
        return RGCNModel(in_dim=16)
    raise ValueError(f"Unsupported model: {model_name}")


def sample_non_edges_fixed(num_nodes: int, forbidden_src: np.ndarray, forbidden_dst: np.ndarray, num_samples: int) -> list[tuple[int, int]]:
    existing = set()
    for raw_s, raw_d in zip(forbidden_src, forbidden_dst):
        s, d = int(raw_s), int(raw_d)
        if s > d:
            s, d = d, s
        existing.add((s, d))

    samples: set[tuple[int, int]] = set()
    max_attempts = max(10_000, num_samples * 200)
    attempts = 0
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        s = int(np.random.randint(0, num_nodes))
        d = int(np.random.randint(0, num_nodes))
        if s == d:
            continue
        key = (s, d) if s < d else (d, s)
        if key not in existing:
            samples.add((s, d))

    if len(samples) < num_samples:
        for s in range(num_nodes):
            for d in range(s + 1, num_nodes):
                if (s, d) not in existing and (s, d) not in samples:
                    samples.add((s, d))
                    if len(samples) == num_samples:
                        break
            if len(samples) == num_samples:
                break

    if len(samples) != num_samples:
        raise ValueError(f"Could only sample {len(samples)}/{num_samples} label-0 non-edges")
    return list(samples)


def prepare_data(node_csv: Path, edge_csv: Path, seed: int) -> tuple[dgl.DGLGraph, EdgeDataset, EdgeDataset, int, dict[str, int]]:
    nodes = pd.read_csv(node_csv)
    edges = pd.read_csv(edge_csv)
    required = {"src", "dst", "label"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"{edge_csv} missing columns: {sorted(missing)}")

    num_nodes = len(nodes)
    edges = edges[["src", "dst", "label"]].copy()
    edges["src"] = edges["src"].astype(int)
    edges["dst"] = edges["dst"].astype(int)
    edges["label"] = edges["label"].astype(int)

    df_pos = edges[edges["label"] == 1]
    df_neg = edges[edges["label"] == 2]
    if df_pos.empty or df_neg.empty:
        raise ValueError(f"{edge_csv} needs both label=1 and label=2 rows")

    reset_seed(seed)
    generated_l0 = sample_non_edges_fixed(
        num_nodes=num_nodes,
        forbidden_src=edges["src"].values,
        forbidden_dst=edges["dst"].values,
        num_samples=len(df_pos),
    )
    src_l0 = np.array([pair[0] for pair in generated_l0], dtype=int)
    dst_l0 = np.array([pair[1] for pair in generated_l0], dtype=int)
    lbl_l0 = np.zeros(len(generated_l0), dtype=int)

    all_src = np.concatenate([edges["src"].values, src_l0])
    all_dst = np.concatenate([edges["dst"].values, dst_l0])
    all_lbl = np.concatenate([edges["label"].values, lbl_l0])

    all_idx = np.arange(len(all_lbl))
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, shuffle=True, random_state=seed)
    graph_edges_mask = np.isin(all_idx, train_idx) & (all_lbl != 0)
    g_src = all_src[graph_edges_mask]
    g_dst = all_dst[graph_edges_mask]
    graph = dgl.graph((g_src, g_dst), num_nodes=num_nodes)
    graph = dgl.add_edges(graph, g_dst, g_src)
    graph = dgl.add_self_loop(graph)

    stats = {
        "num_nodes": num_nodes,
        "label0": int(len(lbl_l0)),
        "label1": int(len(df_pos)),
        "label2": int(len(df_neg)),
        "total_samples": int(len(all_lbl)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
    }
    train_dataset = EdgeDataset(all_src[train_idx], all_dst[train_idx], all_lbl[train_idx])
    val_dataset = EdgeDataset(all_src[val_idx], all_dst[val_idx], all_lbl[val_idx])
    return graph, train_dataset, val_dataset, num_nodes, stats


def multi_class_pr_auc(labels: np.ndarray, probs: np.ndarray, num_classes: int = 3) -> float:
    prcs = []
    for cls in range(num_classes):
        y_true = (labels == cls).astype(int)
        if y_true.sum() == 0:
            continue
        prcs.append(average_precision_score(y_true, probs[:, cls]))
    return float(np.mean(prcs)) if prcs else 0.0


def evaluate(model: nn.Module, loader: DataLoader, graph: dgl.DGLGraph, feat: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for src, dst, label in loader:
            src, dst, label = src.to(device), dst.to(device), label.to(device)
            logits = model(graph, feat, src, dst)
            total_loss += float(criterion(logits, label).item())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(label.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    labels = np.asarray(all_labels)
    preds = np.asarray(all_preds)
    probs = np.asarray(all_probs)
    metrics = {
        "val_loss": total_loss,
        "acc": accuracy_score(labels, preds),
        "pre": precision_score(labels, preds, average="macro", zero_division=0),
        "rec": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro"),
        "auc": roc_auc_score(labels, probs, multi_class="ovr"),
        "prc": multi_class_pr_auc(labels, probs, num_classes=3),
    }
    per_class_rec = recall_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    per_class_pre = precision_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    metrics.update(
        label2_recall=float(per_class_rec[2]),
        label2_precision=float(per_class_pre[2]),
    )
    return {key: float(value) for key, value in metrics.items()}


def train_model(
    node_csv: Path,
    edge_csv: Path,
    model_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, list[float]], dict[str, float], dict[str, int]]:
    reset_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    graph, train_ds, val_ds, num_nodes, data_stats = prepare_data(node_csv, edge_csv, args.seed)
    graph = graph.to(device)

    node_emb = nn.Embedding(num_nodes, 16).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)
    model = create_model(model_name).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(node_emb.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "acc": [],
        "pre": [],
        "rec": [],
        "f1": [],
        "auc": [],
        "prc": [],
        "label2_recall": [],
        "label2_precision": [],
    }

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}",
            ncols=100,
            leave=False,
            disable=args.quiet_progress,
        )
        for src, dst, label in iterator:
            src, dst, label = src.to(device), dst.to(device), label.to(device)
            logits = model(graph, node_emb.weight, src, dst)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, graph, node_emb.weight, device)
        history["train_loss"].append(total_loss)
        for key, value in metrics.items():
            history[key].append(value)
        print(
            f"[Epoch {epoch:02d}] Train Loss={total_loss:.4f} | "
            f"Val Loss={metrics['val_loss']:.4f} | Acc={metrics['acc']:.4f} | "
            f"Pre={metrics['pre']:.4f} | Rec={metrics['rec']:.4f} | "
            f"F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f} | PRC={metrics['prc']:.4f} | "
            f"L2Rec={metrics['label2_recall']:.4f} | L2Pre={metrics['label2_precision']:.4f}",
            flush=True,
        )

    final = {key: values[-1] for key, values in history.items() if values}
    final["device"] = str(device)
    return history, final, data_stats


def run_one(
    dataset: str,
    variant: str,
    edge_name: str,
    model_name: str,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    node_csv = input_dir / "unified_node.csv"
    edge_csv = input_dir / edge_name
    result_stem = edge_name.removesuffix(".csv")
    result_json = output_dir / model_name / dataset.lower() / variant / f"{result_stem}_seed{args.seed}.json"
    row: dict[str, object] = {
        "dataset": dataset,
        "variant": variant,
        "model": model_name,
        "edge_file": edge_name,
        "status": "ok",
        "node_csv": str(node_csv),
        "edge_csv": str(edge_csv),
        "result_json": str(result_json),
        "error": "",
    }
    if not node_csv.is_file():
        row.update(status="missing_node_csv", error=f"missing {node_csv}")
        return row
    if not edge_csv.is_file():
        row.update(status="missing_edge_csv", error=f"missing {edge_csv}")
        return row
    if result_json.exists() and not args.overwrite:
        row.update(status="skipped_existing_result")
        return row

    print(f"[TrainAblationGNN] model={model_name} dataset={dataset} variant={variant} edge={edge_name}", flush=True)
    try:
        history, final, data_stats = train_model(node_csv, edge_csv, model_name, args)
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "dataset": dataset,
            "variant": variant,
            "model": model_name,
            "edge_file": edge_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "node_csv": str(node_csv),
            "edge_csv": str(edge_csv),
            "data_stats": data_stats,
            "final": final,
            "history": history,
        }
        result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row.update(data_stats)
        row.update({f"final_{key}": value for key, value in final.items()})
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        print(row["traceback"], flush=True)
    return row


def write_summary(rows: list[dict[str, object]], output_root: Path, seed: int) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"train_summary_seed{seed}.csv"
    preferred = [
        "dataset",
        "variant",
        "model",
        "edge_file",
        "status",
        "label0",
        "label1",
        "label2",
        "final_train_loss",
        "final_val_loss",
        "final_acc",
        "final_pre",
        "final_rec",
        "final_f1",
        "final_auc",
        "final_prc",
        "final_label2_recall",
        "final_label2_precision",
        "result_json",
        "error",
    ]
    fieldnames = [key for key in preferred if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fieldnames and key != "traceback":
                fieldnames.append(key)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    return summary_path


def write_variant_model_means(rows: list[dict[str, object]], output_root: Path, seed: int) -> Path:
    mean_path = output_root / f"train_summary_seed{seed}_variant_model_mean.csv"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        with mean_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "variant", "model", "n_runs"])
            writer.writeheader()
        return mean_path

    df = pd.DataFrame(ok_rows)
    group_cols = ["dataset", "variant", "model"]
    preferred_metrics = [
        "label0",
        "label1",
        "label2",
        "final_train_loss",
        "final_val_loss",
        "final_acc",
        "final_pre",
        "final_rec",
        "final_f1",
        "final_auc",
        "final_prc",
        "final_label2_recall",
        "final_label2_precision",
    ]
    metric_cols = [col for col in preferred_metrics if col in df.columns]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mean_df = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="n_runs")
    mean_df = counts.merge(mean_df, on=group_cols, how="left")
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")
    return mean_path


def write_variant_means(rows: list[dict[str, object]], output_root: Path, seed: int) -> Path:
    mean_path = output_root / f"train_summary_seed{seed}_variant_mean.csv"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        with mean_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "variant", "n_runs", "n_models", "models"])
            writer.writeheader()
        return mean_path

    df = pd.DataFrame(ok_rows)
    group_cols = ["dataset", "variant"]
    preferred_metrics = [
        "label0",
        "label1",
        "label2",
        "final_train_loss",
        "final_val_loss",
        "final_acc",
        "final_pre",
        "final_rec",
        "final_f1",
        "final_auc",
        "final_prc",
        "final_label2_recall",
        "final_label2_precision",
    ]
    metric_cols = [col for col in preferred_metrics if col in df.columns]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mean_df = df.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    counts = df.groupby(group_cols, dropna=False).size().reset_index(name="n_runs")
    models = (
        df.groupby(group_cols, dropna=False)["model"]
        .agg(lambda values: ",".join(sorted({str(value) for value in values})))
        .reset_index(name="models")
    )
    models["n_models"] = models["models"].apply(lambda value: len([item for item in value.split(",") if item]))
    mean_df = counts.merge(models, on=group_cols, how="left").merge(mean_df, on=group_cols, how="left")
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")
    return mean_path


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    input_root = (
        Path(args.input_root).expanduser().resolve()
        if args.input_root
        else repo_root / "experiments" / "ablation_gnn_batch" / "built_edges"
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else repo_root / "experiments" / "ablation_gnn_batch" / "gnn_results"
    )

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for variant in args.variants:
            input_dir = input_root / dataset.lower() / variant
            for edge_name in args.edge_files:
                for model_name in args.models:
                    rows.append(run_one(dataset, variant, edge_name, model_name, input_dir, output_root, args))

    summary_path = write_summary(rows, output_root, args.seed)
    mean_path = write_variant_model_means(rows, output_root, args.seed)
    variant_mean_path = write_variant_means(rows, output_root, args.seed)
    print(f"[Done] summary={summary_path}", flush=True)
    print(f"[Done] variant_model_mean={mean_path}", flush=True)
    print(f"[Done] variant_mean={variant_mean_path}", flush=True)


if __name__ == "__main__":
    main()
