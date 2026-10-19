#!/bin/bash
# Pause-gate acceptance: a GPU worker waits at a game boundary while the pause file exists.
set -u
cd /home/newbiexvwu/pvz-cuda
OUT=artifacts/cuda/codex_r66/pause_accept
PAUSE=/tmp/r66_gpu_pause
rm -rf "$OUT"; mkdir -p "$OUT"
touch "$PAUSE"
echo "$(date -Is) pause file created: $PAUSE"
PVZ_GPU_PAUSE_FILE=$PAUSE scripts/mixed_dagger_round.sh \
  --student $HOME/PvZ-search/artifacts/student/v28/model.pt \
  --task day_l7_z17_train --seed-offset 14096 --games 2 \
  --frames $HOME/PvZ-search/artifacts/student/frames --frames-label r66pause \
  --out "$OUT" --cpu-workers 1 --gpu-workers 1 > "$OUT/driver.log" 2>&1 &
DRIVER=$!
for i in $(seq 1 60); do
  if grep -q "GPU collection paused" "$OUT/logs/gpu.log" 2>/dev/null; then echo "$(date -Is) gate observed"; break; fi
  sleep 2
done
echo "$(date -Is) sampling GPU while paused"
for i in 1 2 3 4 5 6; do
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr '\n' ' '
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ','
  echo "  $(date -Is)"
  sleep 5
done
echo "$(date -Is) removing pause file"
rm -f "$PAUSE"
wait $DRIVER
echo "$(date -Is) driver rc=$?"
cat "$OUT/summary.json" 2>/dev/null
ls "$OUT/games.jsonl" 2>/dev/null && wc -l "$OUT/games.jsonl"
