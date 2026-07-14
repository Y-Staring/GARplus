# Exp-2: Scalability and efficiency

## Experiment entrypoints

- Pattern size `n`: run the dataset script in `pattern_size/`.
- Support/confidence: run the dataset script in `sigma_confidence/`.
- BN thresholds `tau_P` and `tau_X`: run
  `python experiments/exp2_scalability_efficiency/tau_sensitivity/run_tau_sensitivity.py --help`.
- Cluster example: `tau_sensitivity/run_tau_dda.slurm`.

Set `GARPLUS_PATTERN_SIZE_RESULT_DIR` when a pattern-size output directory is
needed outside the default local folder.

## Experiment results

- Pattern-size CSV: `garplus_<dataset>_pattern_size_timing.csv`, plus one log,
  rule file, and pattern-instance file per `n`.
- Sigma/confidence CSV:
  `sigma_confidence/batch_results/sigma_confidence_<dataset>/garplus_<dataset>_sigma_confidence_timing.csv`.
- Tau CSV: `tau_sensitivity/tau_sensitivity_results/tau_sensitivity_summary.csv`.

## Paper destination

These results fill `fig-performance-evaluation`: the `n`, `sigma`, `tau_P`,
and `tau_X` runtime/rule-count/coverage panels. The paper also lists `delta`
and overall scalability panels; no dedicated `delta` or graph-scale runner is
present yet, so those panels must not be filled from another sweep.

