# Exp-3: Explainability and case study

## Experiment entrypoints

1. Generate prediction rows with the Exp-1 refinement/matching scripts. The
   input CSV must contain `pred_label`, `gar_positive_hit`, and
   `gar_negative_hit` (or pass alternative column names below).
2. Aggregate explanation coverage:
   `python experiments/exp3_explainability_case_study/compute_explanation_coverage.py INPUTS --output summary.csv --dataset DATASET`

Use `--help` for column-name and grouping options. No separate Exp-3 matcher is
kept: the former wrapper referenced a missing
`validate_sampled_refinement.py`, so it was removed instead of documenting a
non-runnable entrypoint.

## Experiment results

`summary.csv` separates positive-rule support, negative-rule rejection, and
negative-prediction explanations. This separation avoids treating every
rule-rejected prediction as automatically explained.

## Paper destination

- Coverage/quality aggregates: Table `tab:rule-quality`.
- Selected human-readable rules and graph patterns: case-study Figure
  `fig-exp3-pattern` and the accompanying Exp-3 text.
