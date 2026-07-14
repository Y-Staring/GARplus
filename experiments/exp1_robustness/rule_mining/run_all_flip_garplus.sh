#!/usr/bin/env bash
set -u

ROOT="${GARPLUS_ROOT:-/home/yyyy/codework/GARplus}"
DENOISE="$ROOT/experiments/exp1_robustness/rule_mining"
PY="${GARPLUS_PYTHON:-/home/yyyy/anaconda3/envs/digress/bin/python}"
DATA="$ROOT/enumeration-discovery/data"
ATTR="$ROOT/enumeration-discovery/去病图数据"
RUN_ID="${GARPLUS_DENOISE_RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
OUT="${GARPLUS_DENOISE_OUT:-$DENOISE/garplus_flip_results_controlled_$RUN_ID}"
export OUT
mkdir -p "$OUT/logs"
echo "Result root: $OUT"

run_one() {
  local dataset="$1" pct="$2" train_c="$3" node_map="$4"
  local result="$OUT/${dataset}_${pct}pct"
  local log="$OUT/logs/${dataset}_${pct}pct.log"
  if [[ -e "$result" ]]; then
    echo "Refusing to overwrite existing result directory: $result" >&2
    return 98
  fi
  mkdir -p "$result"
  echo "[$(date '+%F %T')] START dataset=$dataset pct=$pct" | tee "$log"
  "$PY" "$DENOISE/flip_subgraph_garplus.py" \
    --dataset "$dataset" \
    --train-c "$train_c" \
    --node-map "$node_map" \
    --big-nodes "$DATA/node.csv" \
    --big-edges "$DATA/edges.csv" \
    --attribute-data-dir "$ATTR" \
    --output-dir "$result" \
    --hops 1 \
    --overfit-noise \
    --enable-bn >>"$log" 2>&1
  mine_status=$?
  if [[ $mine_status -eq 0 ]]; then
    "$PY" "$DENOISE/apply_flip_garplus_rules.py" \
      --result-dir "$result" \
      --min-confidence 0.75 \
      --score-margin 1.2 \
      --oracle-confirm-rule-changes \
      --synthetic-oracle-rate 0.70 \
      --oracle-seed 20260713 >>"$log" 2>&1
    apply_status=$?
  else
    apply_status=99
  fi
  echo "[$(date '+%F %T')] END dataset=$dataset pct=$pct mine=$mine_status apply=$apply_status" | tee -a "$log"
}

for dataset in ti dda; do
  for pct in 5 10 20; do
    if [[ "$dataset" == ti ]]; then
      node_map="$DENOISE/mappings/ti/node_labeled.csv"
    else
      node_map="$DENOISE/mappings/node_labeled.csv"
    fi
    run_one "$dataset" "$pct" "$DENOISE/mappings/${dataset}${pct}_train_c.txt" "$node_map"
  done
done

"$PY" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['OUT'])
rows = []
for path in sorted(root.glob('*_*pct/garplus_denoise_summary.json')):
    row = json.loads(path.read_text())
    row['experiment'] = path.parent.name
    rows.append(row)
(root / 'all_denoise_summaries.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(rows, indent=2))
PY
