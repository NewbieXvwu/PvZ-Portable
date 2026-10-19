#!/bin/bash
set -u
cd /home/newbiexvwu/pvz-cuda
BIN=build/rollout_cuda_poll
NET=artifacts/cuda/codex_r66/v28.pvzrn
PAIRS=artifacts/cuda/codex_r65/pairs-18.txt
ROOTS=artifacts/cuda/corpus_l7_z15_v9/roots.cudaroot
OUT=/tmp/r66_batch; mkdir -p $OUT
run() {
  tag=$1; poll=$2
  PVZ_CUDA_POLL_US=$poll /usr/bin/time -f "wall=%e user=%U sys=%S" -o $OUT/$tag.time     $BIN --roots $ROOTS --net $NET --pairs $PAIRS --lanes 18 --horizon 2500 --max-actions 3000     --out $OUT/$tag.bin --leaf-out $OUT/$tag.leaf > $OUT/$tag.log 2>&1
  echo "$(date -Is) $tag poll=$poll $(cat $OUT/$tag.time)"
}
for round in 1 2; do
  for poll in 0 20 50 100 200; do run "r${round}_p${poll}" $poll; done
done
echo "$(date -Is) BATCH DONE"
sha256sum $OUT/*.bin | awk '{print $1}' | sort | uniq -c
sha256sum $OUT/*.leaf | awk '{print $1}' | sort | uniq -c
