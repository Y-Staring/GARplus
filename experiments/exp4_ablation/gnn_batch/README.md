# Ablation GNN Batch

This folder contains two scripts for testing GNN performance with negative
edges exported from GAR+ ablation rules.

## 1. Build edge CSVs

```bash
cd /home/yyyy/codework/GARplus
python experiments/ablation_gnn_batch/build_ablation_gnn_edges.py \
  --datasets TI \
  --variants full logicgar_only neuralgar_only wo_order_embedding \
  --negative-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_negative_only \
  --negative-file-name negative_edges_only_candidate_non_edges.csv \
  --negative-sampling-strategy all \
  --gar-only \
  --overwrite
```

For each variant, this writes GAR-only GNN inputs:

```text
experiments/ablation_gnn_batch/built_edges/ti/<variant>/
  gar_augmented_edges.csv
  unified_node.csv
  dataset_stats.json
```

Omit `--gar-only` only when the baseline and LLM edge files are also needed.

## 2. Batch train GAT / HGT / RGCN

```bash
cd /home/yyyy/codework/GARplus
python experiments/ablation_gnn_batch/train_ablation_gnn_batch.py \
  --datasets TI \
  --models gat hgt rgcn \
  --edge-files gar_augmented_edges.csv \
  --epochs 20 \
  --seed 42 \
  --overwrite
```

Results are written to:

```text
experiments/ablation_gnn_batch/gnn_results/
  train_summary_seed42.csv
  train_summary_seed42_variant_model_mean.csv
  train_summary_seed42_variant_mean.csv
  gat/ti/<variant>/gar_augmented_edges_seed42.json
  hgt/ti/<variant>/gar_augmented_edges_seed42.json
  rgcn/ti/<variant>/gar_augmented_edges_seed42.json
```

`train_summary_seed42_variant_mean.csv` averages the final metrics across
GAT/HGT/RGCN for each ablation variant.

To also compare the generated baseline and LLM settings:

```bash
python experiments/ablation_gnn_batch/train_ablation_gnn_batch.py \
  --datasets TI \
  --models gat hgt rgcn \
  --edge-files baseline_edges.csv llm_augmented_edges.csv gar_augmented_edges.csv \
  --epochs 20 \
  --seed 42 \
  --overwrite
```

## 3. Negative-edge export scope

When exporting ablation negative edges, use
`enumeration-discovery/GARplusMiner/batch_expand_ablation_negative_edges.py`.
The `--scope-input-csv` option overrides only the edge table scanned by the
negative-edge expander. This is useful for PPI, where the signed graph may have
all rows labeled as positive/negative and therefore no eligible unlabeled rows:

```bash
cd /home/yyyy/codework/GARplus/enumeration-discovery/GARplusMiner
python batch_expand_ablation_negative_edges.py \
  --datasets PPI \
  --variants full logicgar_only neuralgar_only \
  --ablation-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_results \
  --output-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_negative_only \
  --scope-input-csv /home/yyyy/codework/GARplus/enumeration-discovery/去病图数据/protein_protein.csv \
  --mode anchored_existing_edge_labeling \
  --overwrite
```
