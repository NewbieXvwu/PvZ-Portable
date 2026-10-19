#!/bin/bash
set -u
cd /home/newbiexvwu/pvz-cuda
PY=$HOME/.venvs/ml/bin/python3
R=$HOME/.cache/pvz-research-resources
S=$HOME/PvZ-search
OUT=/home/newbiexvwu/pvz-cuda/artifacts/cuda/codex_r66
mkdir -p $OUT
run() {
  name=$1; poll=$2
  mkdir -p $OUT/$name/frames
  echo "$(date -Is) start $name poll=$poll"
  PVZ_CUDA_POLL_US=$poll $PY scripts/search_play.py \
    --rollout-policy $S/artifacts/student/v28/model.pt \
    --value $S/artifacts/value_net/v3/model.pt \
    --static-value $S/artifacts/hard/value_branch1/model.pt \
    --manifest $S/artifacts/stage4/day_l7_hard.json --task-id day_l7_z20_eval \
    --student-candidates 2 --verify-samples 9 --dagger-beta 0 \
    --seeds 1 --workers 1 --seed-offset 0 \
    --executable /home/newbiexvwu/pvz-cuda/build/cmake-no-cuda/pvz-portable \
    --resource-dir $R --measure-cpu --rollout-backend cuda --cuda-speculative-halving \
    --cuda-lanes 128 --cuda-gather-ms 80 \
    --cuda-binary /home/newbiexvwu/pvz-cuda/build/rollout_cuda_poll \
    --frames-label scheduling --frames $OUT/$name/frames \
    --out $OUT/$name > $OUT/$name.log 2>&1
  echo "$(date -Is) done $name rc=$?"
}
run poll0 0
run poll100 100
echo "$(date -Is) ALL DONE"
