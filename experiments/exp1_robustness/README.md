# Exp-1: Robustness under label noise

This block owns noise generation and the noise-ratio evaluation inputs used by
Table `tab-noise-robustness`. Accuracy experiments remain in
`../exp1_accuracy/`. All noise-specific code is kept here: dataset generation
in `noise_generation/`, GAR+ mining on flipped data in `rule_mining/`, and the
earlier graph-refinement pipeline in `deductive_refinement/`.

## Quick handoff map

| Part | Run entry | Result location | Paper mapping |
|---|---|---|---|
| Generate label noise | `noise_generation/prepare_label_flip_noise.py` | `NBFNet/data_updated/my_dataset_*_noise_{5,10,20}pct/` | Inputs for `tab-noise-robustness` |
| Mine GAR+ on flipped data | `rule_mining/flip_subgraph_garplus.py` or `run_all_flip_garplus.sh` | Selected `--output-dir`: rules, instances and subgraph summary | Rule source for `tab-noise-robustness`; optionally validate `tab-rule-statistics` |
| Apply mined rules | `rule_mining/apply_flip_garplus_rules.py` | Same result directory: predictions, corrected train file and JSON summary | Intermediate refinement result; retrain models before filling the table |
| Negative-targeted variant | `rule_mining/apply_negative_targeted_garplus_rules.py` | Same result directory: `negative_targeted_*` files | Robustness ablation/diagnostic, not a replacement for model metrics |
| Earlier graph refinement | `deductive_refinement/apply_gar_rules_to_noisy_graph.py` | Chosen `--output-dir`: cleaned, removed, marked and summary CSVs | `tab-deductive-refinement` and robustness diagnostics |
| Simulated RDR | `deductive_refinement/simulate_rule_deductive_refinement.py` | `outputs/simulated_rdr/` by default | Counterfactual only; never fill a real-result table |

## Experiment entrypoint

From the repository root:

```bash
python experiments/exp1_robustness/noise_generation/prepare_label_flip_noise.py
```

For data stored elsewhere:

```bash
python experiments/exp1_robustness/noise_generation/prepare_label_flip_noise.py \
  --data-root /path/to/NBFNet/data_updated
```

The detailed Chinese method description is
`noise_generation/noise_generation_guide_zh.md`.

## Generated data

For both `my_dataset_ti` and `my_dataset_dda`, the script creates cumulative
label-flip variants at 5%, 10%, and 20%. Only `train.txt` changes; `valid.txt`
and `test.txt` are copied unchanged. The 10% flip set contains the 5% set, and
the 20% set contains the 10% set.

The output directories are written beside the clean inputs:

- `my_dataset_<dataset>_noise_5pct`
- `my_dataset_<dataset>_noise_10pct`
- `my_dataset_<dataset>_noise_20pct`

## Experiment result and paper destination

This script produces noisy datasets, not model metrics. Run the Exp-1 NBFNet
and GAR+ refinement evaluation on each generated directory. Record Accuracy,
Precision, Recall, and F1 for each noise ratio in Table
`tab-noise-robustness`. Do not copy the script's flip-count log into the table;
it is only a dataset-integrity check.

## Mine GAR+ rules on flipped data

The remote-host version from 2026-07-13 is preserved in `rule_mining/`.
`flip_subgraph_garplus.py` recovers flipped training edges, maps them into the
large graph, extracts a local subgraph, and invokes the complete GAR+ miner.
`apply_flip_garplus_rules.py` then applies leakage-free multi-class rules to
the noisy labels. A targeted negative-rule variant is also provided.

Remote-style batch entrypoints:

```bash
GARPLUS_ROOT=/home/yyyy/codework/GARplus \
GARPLUS_PYTHON=/home/yyyy/anaconda3/envs/digress/bin/python \
bash experiments/exp1_robustness/rule_mining/run_all_flip_garplus.sh

bash experiments/exp1_robustness/rule_mining/run_all_negative_targeted_garplus.sh
```

Required `train_c` and node-mapping inputs copied from the remote host are in
`rule_mining/mappings/`. Outputs contain mined rules, pattern instances,
denoised predictions, and JSON summaries; these feed Table
`tab-noise-robustness` after downstream model evaluation is rerun.
