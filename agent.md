# GARplus Engineering Map and Experiment Workflow

This file is the working map for the GARplus experiment codebase. It combines
the implementation map, the current ablation/test workflow, and the concrete
parameters that should be checked before running experiments.

## 1. End-to-End Workflow

```text
sampling preparation
  -> rule discovery
       -> pattern mining / VSpawn
       -> pattern BN pruning
       -> predicate generation
       -> predicate BN pruning
       -> support / confidence / lift computation
  -> negative-edge export from discovered rules
  -> refinement or GNN downstream tests
```

Main code locations:

- Sampling and dataset configs:
  `enumeration-discovery/GARplusMiner/{ppi,dda,ti}_demo.py`
- Ablation runner:
  `enumeration-discovery/GARplusMiner/run_ablation_study.py`
- Main mining runner:
  `enumeration-discovery/GARplusMiner/garplus_demo_runner.py`
- Negative-edge export:
  `enumeration-discovery/GARplusMiner/batch_expand_ablation_negative_edges.py`
  and `enumeration-discovery/GARplusMiner/negative_edge_expander.py`
- GNN follow-up:
  `experiments/ablation_gnn_batch/`

## 2. Sampling and Graph Scope

Default full runs use sampled `.pt` graphs:

```text
PPI: processed/ppi/ppi_selected.pt
DDA: processed/dda/dda_selected.pt
TI : processed/ti/ti_selected.pt
```

Dataset CSVs are the signed large graphs under the data directory:

```text
PPI: protein_protein_signed.csv
DDA: drug_disease_signed.csv
TI : gene_disease_signed.csv
```

Important scope behavior:

- `use_sampled_pt_graph=True` means the working graph is the sampled `.pt` graph.
- `global_match_scope="sampled"` means global rematch/support verification stays on the current working graph.
- `global_match_scope="original"` means rematch uses `verification_graph_loader` over the original CSV graph.
- `wo_order_embedding` sets `use_sampled_pt_graph=False`, `max_rows=None`, and uses the CSV/verification graph loader. Its working graph is therefore the original CSV graph even though its `global_match_scope` remains `"sampled"`.

## 3. Core Mining Parameters

`GarplusRunConfig` defaults in `garplus_demo_runner.py`:

```text
use_sampled_pt_graph=True
mode=decision-tree
fp_growth_max_itemset_size=3
decision_tree_max_depth=3
y_key=e0.interaction_label
pattern_support=5
min_support=50
min_confidence=0.6
min_value_support_count=20
max_radius=4
max_add_edge=4
node_max_add_edge=4
max_multi_support=10000
global_rematch_patterns=True
global_vspawn_instances=False
global_match_scope=sampled
rule_coverage_scope=none
enable_sampled_frequent_patterns=True
sampled_frequent_min_graph_support=5
sampled_frequent_print_limit=20
inject_sampled_frequent_patterns=True
sampled_frequent_pattern_limit=8
enable_pattern_bn=True
tau_p=0.5
pattern_bn_min_keep_per_spawn_node=1
pattern_bn_frequent_prior_weight=0.25
enable_predicate_bn=True
tau_x=0.5
predicate_bn_min_keep_features=8
predicate_bn_max_parent_features=12
predicate_bn_feature_score=bic
```

Ablation runner override in `run_ablation_study.py`:

```text
MAX_PATTERN_NODES=4
print_rule_limit=0
print_deduped_rule_limit=20
pattern_extension_debug=False
debug_match_expansion=True
debug_transaction_cost=True
debug_sample_matches=0
```

## 4. Dataset-Specific Settings

### PPI

Configured in `ppi_demo.py`:

```text
mode=fp-growth
interaction_csv_path=protein_protein_signed.csv
node_csv_path=protein.csv
sampled_pt_path=processed/ppi/ppi_selected.pt
force_edge_label=candidate_interaction
edge_label_column=Experimental System
pattern_bn_relative_tau=0.5
pattern_bn_top_k_per_spawn_node=4
pattern_bn_min_keep_per_spawn_node=1
tau_p=0.0
tau_x=0.05
predicate_bn_top_k_features=24
predicate_bn_min_keep_features=6
predicate_bn_max_parent_features=16
global_rematch_patterns=True
global_match_scope=sampled (default)
global_rematch_max_pattern_edges=3
max_radius=2
max_add_edge=2
max_saved_instances_per_pattern=100000
```

PPI signed graph issue for negative-edge export:

```text
protein_protein_signed.csv rows are already positive/negative.
Default export skips existing positive and existing negative rows.
Use --scope-input-csv with unlabeled protein_protein.csv when the intended
negative-edge search scope is the original unlabeled PPI graph.
```

### DDA

Configured in `dda_demo.py`:

```text
mode=decision-tree
decision_tree_max_depth=4
interaction_csv_path=drug_disease_signed.csv
sampled_pt_path=processed/dda/dda_selected.pt
force_edge_label=drug_disease
edge_label_column=EdgeLabel
include_edge_existing_target=True
pattern_bn_relative_tau=0.5
pattern_bn_top_k_per_spawn_node=4
pattern_bn_min_keep_per_spawn_node=1
tau_p=0.0
tau_x=0.05
predicate_bn_top_k_features=24
predicate_bn_min_keep_features=6
predicate_bn_max_parent_features=16
predicate_bn_focus_targets=(negative, positive)
global_match_scope=sampled (default)
max_radius=2
max_add_edge=2
```

### TI

Configured in `ti_demo.py`:

```text
mode=fp-growth
fp_growth_max_itemset_size=4
interaction_csv_path=gene_disease_signed.csv
sampled_pt_path=processed/ti/ti_selected.pt
force_edge_label=gene_disease
edge_label_column=EdgeLabel
include_edge_existing_target=False
pattern_bn_relative_tau=0.5
pattern_bn_top_k_per_spawn_node=4
pattern_bn_min_keep_per_spawn_node=1
tau_p=0.0
tau_x=0.05
predicate_bn_top_k_features=24
predicate_bn_min_keep_features=6
predicate_bn_max_parent_features=16
predicate_bn_focus_targets=(negative, positive)
global_match_scope=sampled
max_radius=3
max_add_edge=2
```

## 5. Ablation Variants

Entry point:

```bash
python enumeration-discovery/GARplusMiner/run_ablation_study.py \
  --dataset TI \
  --ablation full \
  --max-pattern-nodes 4
```

Supported variants:

```text
full
wo_order_embedding
wo_bayesian_pruning
wo_logicgar
wo_neuralgar
logicgar_only
neuralgar_only
```

Variant definitions:

- `full`: dataset default config plus ablation runner common settings.
- `wo_order_embedding`: disables sampled/order-embedding graph input and reads the full CSV graph:
  ```text
  use_sampled_pt_graph=False
  max_rows=None
  csv_graph_loader=csv_graph_loader or verification_graph_loader
  inject_sampled_frequent_patterns=False
  enable_sampled_frequent_patterns=False
  global_match_scope=sampled
  ```
- `wo_bayesian_pruning`: disables both BN pruners:
  ```text
  enable_pattern_bn=False
  enable_predicate_bn=False
  ```
- `logicgar_only` and `wo_neuralgar`: keep structural and symbolic predicates, drop ML/similarity/equivalence predicates:
  ```text
  ml_predicates.enabled=False
  ignored_predicate_key_tokens += (ml_, similarity, equivalence)
  kept_predicate_key_tokens=()
  filter_degree_predicates=True
  ```
- `neuralgar_only` and `wo_logicgar`: keep only ML-like predicates:
  ```text
  kept_predicate_key_tokens=(ml_, similarity, equivalence)
  filter_degree_predicates=True
  ```

## 6. Mining Outputs and Paper Metrics

Each run writes:

```text
ablation_results/<dataset>/<variant>/run.log
ablation_results/<dataset>/<variant>/ablation_summary.csv
ablation_results/<dataset>/<variant>/deduped_rules.txt
ablation_results/<dataset>/<variant>/pattern_instances.jsonl
```

`ablation_summary.csv` fields come from regex parsing of `run.log`:

```text
wall_seconds
rule_mining_seconds
patterns_mined
pattern_instances
raw_rules
deduped_rules
positive_rules
negative_rules
vspawn_candidates_seen
vspawn_bn_pruned
vspawn_duplicate_pruned
vspawn_constraint_pruned
vspawn_no_match_pruned
vspawn_support_pruned
pattern_bn_seen
pattern_bn_kept
pattern_bn_pruned
predicate_bn_seen
predicate_bn_kept
predicate_bn_pruned
```

For paper text:

- Runtime increase for `wo_order_embedding`:
  `wo_order_embedding.wall_seconds / full.wall_seconds`
- Candidate verification-space factor for order embedding:
  preferably `wo_order_embedding.pattern_instances / full.pattern_instances`.
- Candidate extension-space factor for BN pruning:
  `wo_bayesian_pruning.vspawn_candidates_seen / full.vspawn_candidates_seen`.
- Pattern BN pruning rate:
  `full.pattern_bn_pruned / full.pattern_bn_seen`.
- Combined BN pruning rate:
  `(pattern_bn_pruned + predicate_bn_pruned) / (pattern_bn_seen + predicate_bn_seen)`.
- Rule recall is not directly emitted by the current ablation summary; compute it by comparing `deduped_rules.txt` sets, for example:
  `|rules_full intersect rules_without_bayesian| / |rules_without_bayesian|`.

## 7. Negative-Edge Export Workflow

Entry point:

```bash
cd /home/yyyy/codework/GARplus/enumeration-discovery/GARplusMiner
python batch_expand_ablation_negative_edges.py \
  --datasets TI \
  --variants full logicgar_only neuralgar_only wo_order_embedding \
  --ablation-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_results \
  --output-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_negative_only \
  --mode anchored_existing_edge_labeling \
  --overwrite
```

Important export parameters:

```text
--mode anchored_existing_edge_labeling
--mode candidate_non_edges
--allow-positive-relabel
--allow-existing-negative-relabel
--only-labels ...
--scope-input-csv <custom edge scope>
```

Default scope before `--scope-input-csv`:

```text
PPI: enumeration-discovery/去病图数据/protein_protein_signed.csv
DDA: enumeration-discovery/去病图数据/drug_disease_signed.csv
TI : enumeration-discovery/去病图数据/gene_disease_signed.csv
```

PPI unlabeled-scope export:

```bash
python batch_expand_ablation_negative_edges.py \
  --datasets PPI \
  --variants full logicgar_only neuralgar_only \
  --ablation-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_results \
  --output-root /home/yyyy/codework/GARplus/experiments/ablation_gnn_batch/ablation_negative_only \
  --scope-input-csv /home/yyyy/codework/GARplus/enumeration-discovery/去病图数据/protein_protein.csv \
  --mode anchored_existing_edge_labeling \
  --overwrite
```

Success check:

```text
checked_rows > 0
matched_rows > 0
exported_rows > 0
```

If `checked_rows=0`, inspect:

```text
skipped_positive
skipped_existing_negative
skipped_label_not_allowed
```

## 8. GAR-Only GNN Test Workflow

Build GAR-only GNN edges:

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

GAR-only outputs:

```text
experiments/ablation_gnn_batch/built_edges/<dataset>/<variant>/
  gar_augmented_edges.csv
  unified_node.csv
  dataset_stats.json
```

Train GAT/HGT/RGCN:

```bash
python experiments/ablation_gnn_batch/train_ablation_gnn_batch.py \
  --datasets TI \
  --variants full logicgar_only \
  --models gat hgt rgcn \
  --edge-files gar_augmented_edges.csv \
  --epochs 20 \
  --seed 42 \
  --overwrite
```

Training parameters:

```text
models=(gat, hgt, rgcn)
epochs=20
seed=42
batch_size=64
val_batch_size=256
lr=1e-3
device=cuda if available else cpu
node embedding dim=16
hidden dim=32
out dim=32
GAT/HGT heads=4
label 0 = sampled non-edge
label 1 = positive edge
label 2 = GAR negative edge
label-0 count = label-1 count
train/val split = 80/20, random_state=seed
```

GNN result files:

```text
experiments/ablation_gnn_batch/gnn_results/train_summary_seed42.csv
experiments/ablation_gnn_batch/gnn_results/train_summary_seed42_variant_model_mean.csv
experiments/ablation_gnn_batch/gnn_results/train_summary_seed42_variant_mean.csv
experiments/ablation_gnn_batch/gnn_results/<model>/<dataset>/<variant>/gar_augmented_edges_seed42.json
```

`train_summary_seed42_variant_mean.csv` averages final metrics across GAT/HGT/RGCN for each variant.

## 9. Current Experiment Notes

- TI GAR-only edge construction succeeded for `full` and `logicgar_only` with 5079 exported GAR negatives each.
- TI `neuralgar_only` and `wo_order_embedding` candidate-non-edge exports contained only headers, so no GAR-only GNN dataset could be built from those files.
- PPI signed-graph anchored export produced no rows because all rows were filtered before matching: positives were skipped as existing positives and negatives as existing negatives.
- PPI should be re-exported with `--scope-input-csv` pointing to the unlabeled `protein_protein.csv` if the intended scope is the original unlabeled large graph.

## 10. Remote Runtime

Remote host:

```text
ssh -i $USERPROFILE/.ssh/id_ed25519 admin@192.168.121.55
```

WSL repo:

```text
/home/yyyy/codework/GARplus
```

Conda environment used for DGL/GNN scripts:

```text
/home/yyyy/anaconda3/bin/conda run -n gnn python ...
```
