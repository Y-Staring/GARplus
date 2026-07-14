# Paper experiment map

This directory follows `sec-expt.tex` in `D:\Research\data\GARs+\GARpm`.
Run commands from the repository root. Each experiment block has its own
`README.md` with the entrypoint, generated artifacts, and paper destination.

| Paper block | Directory | Paper result |
|---|---|---|
| Exp-1: Accuracy | `exp1_accuracy/` | Table `tab:exp1-accuracy`, `tab-CGARs`, `tab-deductive-refinement`, `tab-noise-robustness` |
| Exp-1: Robustness | `exp1_robustness/` | Noise inputs and results for Table `tab-noise-robustness` |
| Exp-2: Scalability and Efficiency | `exp2_scalability_efficiency/` | Figure `fig-performance-evaluation` and its `n`, `sigma`, `delta`, `tau_P`, `tau_X`, scalability panels |
| Exp-3: Explainability and Case Study | `exp3_explainability_case_study/` | Table `tab:rule-quality`; case-study pattern `fig-exp3-pattern` |
| Exp-4: Ablation Study | `exp4_ablation/` | Table `tab-ablation-study` |

`enumeration-discovery/GARplusMiner/` contains only the reusable mining
pipeline and dataset adapters. Batch sweeps, refinement probes, model training,
plotting, and ablation drivers belong here.
