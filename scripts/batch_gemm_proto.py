#!/usr/bin/env python3
"""Batching prototype for the search rollouts (task35 step 4; design input for task33).

Every decision branches into "candidate actions x verification samples" continuations, and
each continuation runs the student network on its own.  The convolutions are im2col +
sgemm of shape (M=cells=60) x (N=width=64) x (K=features), one call per continuation per
layer.  Those are small GEMMs: the weight panel is repacked on every call, and the kernel
runs far below peak.

The rollouts of one decision are independent but share the weights, so they can advance in
lockstep and hand the network a batch: stacking B continuations along the M axis turns B
calls of (60 x 64 x K) into one call of (60B x 64 x K) with the same weights.  This script
measures that trade with the same BLAS the engine links (numpy's OpenBLAS), for the three
conv layers of the deployed student (dw4i_s1: 101 input channels, width 64, kernels 3x3,
3x3, 1x5; the first layer is sparse, ~20 of 101 channels are non-empty in a real position).

It is a prototype only: nothing in the engine uses it.  The numbers are the ceiling for
"batch the rollouts" -- the real design also has to keep the lockstep scheduling and the
per-continuation RNG streams, which task33's single-process collector owns.

    python3 scripts/batch_gemm_proto.py
"""

from __future__ import annotations

import os
import time

# The engine runs the simulator with OPENBLAS_NUM_THREADS=1 (python/pvz_env.py); measuring
# with the default thread count measures thread thrash on a machine that is also running the
# collection queue, not the GEMM.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

CELLS = 60
WIDTH = 64
# (K, label): the three trunk layers for a real position (layer 1 sparse: ~20 channels).
LAYERS = [(9 * 20, "conv1 3x3 sparse(20ch)"), (9 * 64, "conv2 3x3"), (5 * 64, "conv3 1x5")]
BATCHES = (1, 2, 4, 9)


def time_gemm(k: int, m: int, n: int, repeats: int) -> float:
    """Seconds per call of (m x n x k) sgemm, best of five passes."""
    a = np.random.default_rng(0).standard_normal((k, m), dtype=np.float32)
    b = np.random.default_rng(1).standard_normal((k, n), dtype=np.float32)
    c = np.empty((m, n), dtype=np.float32)
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(repeats):
            np.matmul(a.T, b, out=c)
        best = min(best, (time.perf_counter() - start) / repeats)
    return best


def main() -> None:
    print(f"{'layer':26s} {'batch':>5s} {'us/rollout':>11s} {'us/call':>9s} {'GFLOP/s':>8s} {'speedup':>8s}")
    total = {batch: 0.0 for batch in BATCHES}
    for k, label in LAYERS:
        base = None
        for batch in BATCHES:
            m = CELLS * batch
            repeats = max(20, 300 // batch)
            seconds = time_gemm(k, m, WIDTH, repeats)
            flops = 2.0 * m * WIDTH * k
            per_rollout = seconds / batch
            base = per_rollout if base is None else base
            total[batch] += per_rollout
            print(f"{label:26s} {batch:5d} {per_rollout * 1e6:11.1f} {seconds * 1e6:9.1f} "
                  f"{flops / seconds / 1e9:8.1f} {base / per_rollout:7.2f}x")
    print()
    print(f"{'all three layers':26s} {'1':>5s} {total[1] * 1e6:11.1f}")
    for batch in BATCHES[1:]:
        print(f"{'':26s} {batch:5d} {total[batch] * 1e6:11.1f} {total[1] / total[batch]:7.2f}x")


if __name__ == "__main__":
    main()
