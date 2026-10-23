"""Where the student trainer's memory goes, per row, before anything is changed.

Loads a deterministic subset of archives through the *production* `load()` (same flags as
`run_dw4_study.py`), then accounts for every component: the CSR board store, the CSR scalar store,
each per-row label tensor, the identity packet block, cell candidates, and the Python objects that
`load()` keeps only to make its masks (`keep`, `seed_of`) plus the index tensors `main()` builds.

Usage:
  python3 scripts/measure_training_memory.py --data artifacts/hard_mac/frames \
      --labels z17dag34b_fast_search,... --archives 200 --cache-dir /tmp/pvz-mem-probe/cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import torch  # noqa: E402

import train_rollout_policy as T  # noqa: E402
from train_rollout_policy import PacketStore  # noqa: E402


def current_rss() -> int:
    """Resident bytes right now (macOS via ps, Linux via /proc)."""
    import os
    import subprocess
    if Path("/proc/self/status").exists():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out.strip()) * 1024


def peak_rss() -> int:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024      # macOS reports bytes, Linux KiB


def subset(paths, count):
    if count is None or count >= len(paths):
        return paths
    step = len(paths) / count
    return [paths[int(i * step)] for i in range(count)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--archives", type=int, default=200)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--wait-keep", type=float, default=0.1)
    p.add_argument("--packet-head", default="identity")
    p.add_argument("--cell-soft-weight", type=float, default=3.0)
    p.add_argument("--obs", default="v2")
    p.add_argument("--json", type=Path)
    p.add_argument("--extrapolate", type=int, default=15_000_000)
    args = p.parse_args()

    labels = set(args.labels.split(","))
    all_paths = [q for q in T.frame_paths(args.data) if q.parent.name in labels]
    paths = subset(all_paths, args.archives)
    T.frame_paths = lambda data, _paths=paths: list(_paths)      # this measurement only
    frames_per_archive = []
    for path in paths:
        cached = T.cache_episode(path, args.cache_dir, args.obs)
        with np.load(cached, allow_pickle=False) as stored:
            frames_per_archive.append(len(stored["hints"]))
    frames = sum(frames_per_archive)

    baseline = current_rss()
    started = current_rss()
    loaded = T.load(args.data, labels, args.wait_keep, {},
                    args.cache_dir, False, args.obs, 1,
                    cells=bool(args.cell_soft_weight),
                    aux_horizon=0, row_observer=None,
                    packet_head=args.packet_head, read_cache_dir=None)
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    packet_info = loaded[9] if args.packet_head == "identity" else None
    after_load = current_rss()

    rows = len(plant)

    def tensor_bytes(t):
        if t is None:
            return None
        return int(t.nbytes) if isinstance(t, PacketStore) else int(t.numel() * t.element_size())

    components = {
        "boards.indices": int(boards.indices.nbytes),
        "boards.values": int(boards.values.nbytes),
        "boards.offsets": int(boards.offsets.nbytes),
        "scalars.indices": int(scalars.indices.nbytes),
        "scalars.values": int(scalars.values.nbytes),
        "scalars.offsets": int(scalars.offsets.nbytes),
        "plant": tensor_bytes(plant),
        "packet": tensor_bytes(packet),
        "cell": tensor_bytes(cell),
        "weight": tensor_bytes(weight),
        "values": tensor_bytes(values),
        "cands": tensor_bytes(cands),
        "packet_info": tensor_bytes(packet_info),
        "seed_of (python list)": sys.getsizeof(seed_of),
        "keep masks (python bool lists)": frames * 8,
    }

    # main()'s index tensors: is_val from a python bool list, then nonzero + randperm.
    import random
    seeds = sorted(set(seed_of)); random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: len(seeds) // 10])
    is_val = torch.tensor([s in val_seeds for s in seed_of])
    train_idx = torch.nonzero(~is_val).squeeze(1).to(torch.int32)
    perm = train_idx[torch.randperm(len(train_idx))]
    after_index = current_rss()
    components["is_val + train_idx + perm"] = int(is_val.numel() * is_val.element_size()
                                                 + train_idx.numel() * train_idx.element_size()
                                                 + perm.numel() * perm.element_size())

    total = sum(components.values())
    per_row = {k: v / rows for k, v in components.items()}
    scale = args.extrapolate / rows
    report = {
        "archives": len(paths),
        "frames": frames,
        "rows": rows,
        "retained_fraction_of_frames": rows / frames,
        "board_nnz_per_row": boards.nnz / rows,
        "board_cells": int(np.prod(boards.shape[1:])),
        "scalar_nnz_per_row": scalars.nnz / rows,
        "components_bytes": components,
        "components_bytes_per_row": per_row,
        "components_gib_at_extrapolation": {k: v * scale / 2**30 for k, v in components.items()},
        "total_gib_now": total / 2**30,
        "total_gib_at_extrapolation": total * scale / 2**30,
        "rss_mib": {"baseline": baseline / 2**20, "after_load": after_load / 2**20,
                    "after_index": after_index / 2**20, "peak": peak_rss() / 2**20,
                    "load_delta": (after_load - baseline) / 2**20},
        "rows_with_search_targets": int((~torch.isnan(values)).any(1).sum()),
        "rows_with_two_candidates": int(((~torch.isnan(values)).sum(1) >= 2).sum()),
        "planted_rows": int((plant > 0.5).sum()),
        "cands_rows_nonempty": int((~torch.isnan(cands[:, :, 2])).any(1).sum()) if cands is not None else None,
        "dtypes": {"boards.values": str(boards.values.dtype), "boards.indices": str(boards.indices.dtype),
                   "boards.offsets": str(boards.offsets.dtype), "scalars.values": str(scalars.values.dtype),
                   "plant": str(plant.dtype), "packet": str(packet.dtype), "cell": str(cell.dtype),
                   "values": str(values.dtype), "cands": str(cands.dtype) if cands is not None else None,
                   "packet_info": str(packet_info.dtype) if packet_info is not None else None},
    }
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
