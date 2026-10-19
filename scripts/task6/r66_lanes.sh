#!/bin/bash
set -u
cd /home/newbiexvwu/pvz-cuda
BIN=build/rollout_cuda_poll
NET=artifacts/cuda/codex_r66/v28.pvzrn
ROOTS=artifacts/cuda/corpus_l7_z15_v9/roots.cudaroot
OUT=/tmp/r66_lanes; mkdir -p $OUT
# pairs files with 18/36/72/144 roots: reuse pairs-18 and extend by cycling its lines
for n in 18 36 72 144; do
  if [ ! -f $OUT/pairs-$n.txt ]; then
    ~/.venvs/ml/bin/python3 - $n $OUT/pairs-$n.txt <<'PY'
import sys
n = int(sys.argv[1]); out = open(sys.argv[2], "w")
base = [l for l in open("/home/newbiexvwu/pvz-cuda/artifacts/cuda/codex_r65/pairs-18.txt").read().splitlines() if l.strip()]
for i in range(n):
    row = base[i % len(base)].split()
    row[0] = str(i)          # unique root index per lane
    row[1] = str(1700000000 + i)
    out.write(" ".join(row) + "\n")
PY
  fi
done
for round in 1 2 3; do
  for n in 18 36 72 144; do
    PVZ_CUDA_POLL_US=100 /usr/bin/time -f "wall=%e user=%U sys=%S" -o $OUT/r${round}_n${n}.time       $BIN --roots $ROOTS --net $NET --pairs $OUT/pairs-$n.txt --lanes $n --horizon 2500 --max-actions 3000       --out $OUT/r${round}_n${n}.bin --leaf-out $OUT/r${round}_n${n}.leaf > $OUT/r${round}_n${n}.log 2>&1
    echo "$(date -Is) lanes=$n $(cat $OUT/r${round}_n${n}.time)"
  done
done
echo "$(date -Is) LANES DONE"
