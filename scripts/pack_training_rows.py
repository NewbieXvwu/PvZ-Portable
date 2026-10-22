"""Pack episode archives into one sequential training file (task33 §7 item 2).

Why: a training run today decompresses 25 M rows x 13.8 KB (345 GB) out of 21 056 per-game
frames.npz files and then compresses every row again into CSR (461 B/row) -- on every run, because
neither the decompressed rows nor the CSR are kept. Measured: 48 598 rows/s with --cache-dir and 8
precompute workers (1.28 M rows in 32 s; 13.6 M rows ~ 280 s).

This writes what the trainer wants, once: rows in archive order, boards and scalars as CSR
(uint16 element index, float16/float32 value, int64 row offset), the labels, and a per-episode
index so a split can still be taken by seed. Reading it back is one sequential pass with no
decompression and no np.nonzero.

  python3 scripts/pack_training_rows.py --data DIR --labels a,b --out FILE.npz [--wait-keep 1.0]
  python3 scripts/pack_training_rows.py --verify FILE.npz --data DIR --labels a,b

The format is additive: nothing reads it yet, and the old archives stay the source of truth.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))


def _paths(data: Path, labels: set[str]):
    from pvz_frame_store import frame_paths
    return [path for path in frame_paths(data) if path.parent.name in labels]


def pack(args) -> None:
    from train_rollout_policy import CompactBoards, load
    started = time.perf_counter()
    # One code path for the numbers: whatever load() builds in memory is what gets written.
    loaded = load(args.data, set(args.labels.split(",")), wait_keep=args.wait_keep)
    boards, scalars, plant, weight, packet, cell, values, seed_of = loaded[:8]
    boards = boards.finish()
    scalars = scalars.finish()
    elapsed = time.perf_counter() - started
    rows = len(plant)
    print(f"load: {rows} rows in {elapsed:.1f} s = {rows/elapsed:.0f} rows/s", flush=True)
    started = time.perf_counter()
    arrays = dict(rows=np.int64(rows),
                  board_indices=boards.indices, board_values=boards.values, board_offsets=boards.offsets,
                  board_shape=np.asarray(boards.shape[1:], np.int64),
                  scalar_indices=scalars.indices, scalar_values=scalars.values, scalar_offsets=scalars.offsets,
                  scalar_shape=np.asarray(scalars.shape[1:], np.int64),
                  plant=plant, weight=weight, packet=packet, cell=cell, values=values,
                  seed=np.asarray(seed_of, np.int64),
                  labels=np.asarray(sorted(set(args.labels.split(",")))))
    np.savez(args.out, **arrays)
    print(f"write: {args.out} {args.out.stat().st_size/1e6:.0f} MB in {time.perf_counter()-started:.1f} s", flush=True)


def verify(args) -> None:
    from train_rollout_policy import load
    started = time.perf_counter()
    with np.load(args.out, allow_pickle=False) as stored:
        boards_shape = tuple(stored["board_shape"])
        rows = int(stored["rows"])
        read = {name: stored[name] for name in stored.files}
    read_s = time.perf_counter() - started
    print(f"read pack: {rows} rows in {read_s:.2f} s = {rows/read_s:.0f} rows/s", flush=True)
    loaded = load(args.data, set(args.labels.split(",")), wait_keep=args.wait_keep)
    boards, scalars, plant, weight, packet, cell, values, seed_of = loaded[:8]
    boards = boards.finish()
    scalars = scalars.finish()
    same = {
        "rows": rows == len(plant),
        "board_indices": np.array_equal(read["board_indices"], boards.indices),
        "board_values": np.array_equal(read["board_values"], boards.values),
        "board_offsets": np.array_equal(read["board_offsets"], boards.offsets),
        "scalar_indices": np.array_equal(read["scalar_indices"], scalars.indices),
        "scalar_values": np.array_equal(read["scalar_values"], scalars.values),
        "plant": np.array_equal(read["plant"], plant),
        "weight": np.array_equal(read["weight"], weight),
        "packet": np.array_equal(read["packet"], packet),
        "cell": np.array_equal(read["cell"], cell),
        "values": np.array_equal(read["values"], values),
        "seed": np.array_equal(read["seed"], np.asarray(seed_of, np.int64)),
        "shape": tuple(read["board_shape"]) == boards_shape,
    }
    for name, ok in same.items():
        print(f"  {name:16s} {'same' if ok else 'DIFFERENT'}")
    if not all(same.values()):
        raise SystemExit("pack does not reproduce the loader")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", required=True, help="comma separated <label> directory names")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-keep", type=float, default=1.0)
    parser.add_argument("--verify", action="store_true", help="compare the pack with load() instead of writing")
    args = parser.parse_args()
    (verify if args.verify else pack)(args)


if __name__ == "__main__":
    main()
