# GARplus Engineering Map

## Overall Workflow

1. Sampling preparation
   - Dataset configs live in `enumeration-discovery/GARplusMiner/{ppi,dda,ti}_demo.py`.
   - Default full runs load sampled `.pt` graphs through `sampled_graph_loader`.
   - `wo_order_embedding` disables sampled/order-embedding input and reads the full CSV graph.

2. Rule discovery
   - Entry point: `enumeration-discovery/GARplusMiner/run_ablation_study.py`.
   - Main runner: `enumeration-discovery/GARplusMiner/garplus_demo_runner.py`.
   - Pattern mining uses `GraphSpawn` and emits `[VSpawnStats]` counters.
   - Pattern pruning uses `PatternBayesianNetwork`; logs `[PatternBN]`.
   - Predicate generation is performed during rule mining over matched pattern instances.
   - Predicate pruning uses predicate BNs; logs `[PredicateBN]`.

3. Support and confidence
   - Rule support/confidence/lift are computed inside the rule-mining stage in `garplus_demo_runner.py`.
   - Default full runs verify support on the sampled graph when `global_match_scope="sampled"`.
   - Use `global_match_scope="original"` to rematch pattern support on the original CSV graph.
   - `wo_order_embedding` reads the original CSV graph as its working graph, so its sampled scope is already the full graph.

4. Negative-edge export
   - Entry point: `enumeration-discovery/GARplusMiner/batch_expand_ablation_negative_edges.py`.
   - Core expander: `enumeration-discovery/GARplusMiner/negative_edge_expander.py`.
   - Default input CSVs are signed large graphs under `enumeration-discovery/去病图数据/*_signed.csv`.
   - Use `--scope-input-csv` to scan an unlabeled large graph, such as PPI `protein_protein.csv`, while keeping node CSVs and rule files unchanged.

5. Refinement / GNN follow-up
   - Negative edges are converted into GNN edge files by `experiments/ablation_gnn_batch/build_ablation_gnn_edges.py`.
   - `--gar-only` writes only `gar_augmented_edges.csv`, `unified_node.csv`, and `dataset_stats.json`.
   - GAT/HGT/RGCN batch tests run through `experiments/ablation_gnn_batch/train_ablation_gnn_batch.py`.
   - Training summaries include per-run metrics, per-variant/per-model means, and per-variant means averaged across GAT/HGT/RGCN.

## Current Experiment Entrypoints

- Ablation mining:
  `python enumeration-discovery/GARplusMiner/run_ablation_study.py --dataset TI --ablation full`
- Negative-edge export:
  `python enumeration-discovery/GARplusMiner/batch_expand_ablation_negative_edges.py --datasets TI --mode anchored_existing_edge_labeling`
- GAR-only GNN edge build:
  `python experiments/ablation_gnn_batch/build_ablation_gnn_edges.py --datasets TI --gar-only`
- GNN model batch:
  `python experiments/ablation_gnn_batch/train_ablation_gnn_batch.py --datasets TI --models gat hgt rgcn`

## Recent Experiment Notes

- TI GAR-only edge construction succeeded for `full` and `logicgar_only` with 5079 exported GAR negatives each.
- TI `neuralgar_only` and `wo_order_embedding` candidate-non-edge CSVs contained only headers, so no GNN GAR dataset could be built for those exports.
- PPI signed-graph anchored export produced no rows because all rows were filtered before rule matching: positive rows were skipped as existing positive and negative rows as existing negative.
- PPI negative-edge export should use `--scope-input-csv` with an unlabeled PPI edge table when the intended scope is the original unlabeled large graph.

## Useful Output Files

- `ablation_results/<dataset>/<variant>/run.log`: raw mining log.
- `ablation_results/<dataset>/<variant>/ablation_summary.csv`: runtime, VSpawn, PatternBN, PredicateBN, pattern instance, and rule counts.
- `experiments/ablation_gnn_batch/ablation_negative_only/<dataset>/<variant>/*.csv`: exported negative edges.
- `experiments/ablation_gnn_batch/built_edges/<dataset>/<variant>/`: GNN-ready GAR edge files.
- `experiments/ablation_gnn_batch/gnn_results/`: GAT/HGT/RGCN result JSON and CSV summaries.
