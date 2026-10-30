#!/bin/bash
# task158: run the terrain arms serially (Mac is shared with other agents; one job at a time).
set -uo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-artifacts/terrainmoe/terrain_sample.npz}
OUT=${2:-artifacts/terrainmoe/runs}
SEEDS=${3:-1,2}
ARMS=${4:-joint64,joint128,experts,hard_moe,film,soft_moe}
IFS=, read -ra seeds <<< "$SEEDS"
IFS=, read -ra arms <<< "$ARMS"
for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    echo "[$(date '+%F %T')] start $arm seed=$seed"
    python3 scripts/train_terrain_arms.py --data "$DATA" --out "$OUT" --arm "$arm" --seed "$seed" --epochs 12 --device mps \
      || echo "[$(date '+%F %T')] FAILED $arm seed=$seed"
  done
done
echo "[$(date '+%F %T')] all done"
