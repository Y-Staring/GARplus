# Exp-4: Ablation study

## Experiment entrypoints

1. Mine every component variant:
   `python experiments/exp4_ablation/miner/run_ablation_study.py --help`
2. Export rule-derived negative edges:
   `python experiments/exp4_ablation/negative_edges/batch_expand_ablation_negative_edges.py --help`
3. Build GNN inputs:
   `python experiments/exp4_ablation/gnn_batch/build_ablation_gnn_edges.py --help`
4. Train/evaluate GAT, HGT, or RGCN:
   `python experiments/exp4_ablation/gnn_batch/train_ablation_gnn_batch.py --help`

## Experiment results

- Mining: `miner/ablation_results/<dataset>/<variant>/ablation_summary.csv`,
  rules, instances, and logs.
- Expanded edges: `negative_edges/ablation_negative_edges_only/`.
- GNN inputs and metric summaries: paths selected through the `gnn_batch`
  command-line options; see `gnn_batch/README.md`.

## Paper destination

Fill Table `tab-ablation-study`, one row per variant (`full`,
`wo_order_embedding`, `wo_bayesian_pruning`, `wo_logicgar`, `wo_neuralgar`,
`logicgar_only`, `neuralgar_only`). Keep mining time/rule counts separate from
downstream model Accuracy/Precision/Recall/F1/AUC/AP.

