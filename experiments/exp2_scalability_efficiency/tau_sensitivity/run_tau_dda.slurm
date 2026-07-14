#!/bin/bash
#SBATCH -p AMD_9654
#SBATCH -J tau_dda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -o /home/yangsiyi10504/GARplusMiner/tau_log/tau_dda_%j.log
#SBATCH -e /home/yangsiyi10504/GARplusMiner/tau_log/tau_dda_error_%j.log

set -euo pipefail

RUN_DIR="/home/yangsiyi10504/GARplusMiner"
DATASET="DDA"
MAX_PATTERN_NODES="4"
RULE_COVERAGE_SCOPE="sampled"
RULE_COVERAGE_MAX_INSTANCES="none"

TAU_P_VALUES=(0.0 0.05 0.10 0.20 0.30 0.50)
TAU_X_VALUES=(0.0 0.02 0.05 0.10 0.20 0.30)

mkdir -p "$RUN_DIR/tau_log"
cd "$RUN_DIR"

echo "[TauSlurm] dataset=$DATASET max_pattern_nodes=$MAX_PATTERN_NODES coverage=$RULE_COVERAGE_SCOPE"
echo "[TauSlurm] start=$(date)"

for VALUE in "${TAU_P_VALUES[@]}"; do
  echo
  echo "[TauSlurm] running sweep=tau_p value=$VALUE"
  python run_tau_sensitivity.py \
    --dataset "$DATASET" \
    --sweep tau_p \
    --value "$VALUE" \
    --max-pattern-nodes "$MAX_PATTERN_NODES" \
    --rule-coverage-scope "$RULE_COVERAGE_SCOPE" \
    --rule-coverage-max-instances "$RULE_COVERAGE_MAX_INSTANCES"
done

for VALUE in "${TAU_X_VALUES[@]}"; do
  echo
  echo "[TauSlurm] running sweep=tau_x value=$VALUE"
  python run_tau_sensitivity.py \
    --dataset "$DATASET" \
    --sweep tau_x \
    --value "$VALUE" \
    --max-pattern-nodes "$MAX_PATTERN_NODES" \
    --rule-coverage-scope "$RULE_COVERAGE_SCOPE" \
    --rule-coverage-max-instances "$RULE_COVERAGE_MAX_INSTANCES"
done

echo
echo "[TauSlurm] done=$(date)"
echo "[TauSlurm] results are under $RUN_DIR/tau_sensitivity_results/dda"
