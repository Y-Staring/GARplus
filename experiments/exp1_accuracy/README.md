# Exp-1: Accuracy and deductive refinement

## Experiment entrypoints

- GNN accuracy: execute the GAT/HGT/RGCN notebooks under `PPI_test/`,
  `DDA_test/`, and `TI_test/` after building their three edge CSVs.
- TransE/RotatE: follow `TransE&RotatE/README.md`.
- Noise/refinement: run
  `python experiments/exp1_robustness/deductive_refinement/apply_gar_rules_to_noisy_graph.py`.
- Noise-ratio batch probe: run `gar_denoise_noise_only_probe.py OUTPUT_DIR DDA TI`.
- Counterfactual diagnostic only (not a paper measurement):
  `simulate_rule_deductive_refinement.py`.

## Experiment results

- Model prediction scripts/notebooks produce baseline and GAR-enhanced
  Accuracy, Precision, Recall, F1, AUC and AP.
- `../exp1_robustness/deductive_refinement/` produces refined edge CSVs and rule-hit summaries.
- The simulation output must never be copied into a paper table as a real result.

## Paper destination

- `tab:exp1-accuracy`: baseline vs GAR+ model metrics on PPI/DDA/TI.
- `tab-CGARs`: gains from negated edges and GAR+.
- `tab-deductive-refinement`: Precision/Recall/F1 before and after refinement.
- `tab-noise-robustness`: metrics at the configured graph-noise ratios.
