"""Before/after table for the compact row store: same archives, same seed, same steps, bitwise weights.

Runs the production training command once per memory variant (default, --compact-data, --data-memmap,
both), recording each run peak resident set, wall time and the sha256 of every weight tensor. The gate
from the task is that all variants end with the same weights, bit for bit (task32).

Usage:
  python3 scripts/bench_compact_training.py --data artifacts/hard_mac/frames --labels a,b --archives 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import torch  # noqa: E402

VARIANTS = {"default": [], "compact": ["--compact-data"],
            "memmap": ["--data-memmap"], "compact+memmap": ["--compact-data", "--data-memmap"]}


def subset(data: Path, labels: set, archives: int, destination: Path) -> Path:
    """A deterministic every-k-th archive tree of symlinks, so every run reads the same rows."""
    from pvz_frame_store import frame_paths
    paths = [path for path in frame_paths(data) if path.parent.name in labels]
    if archives and archives < len(paths):
        step = len(paths) / archives
        paths = [paths[int(index * step)] for index in range(archives)]
    root = destination / "data"
    if root.exists():
        return root
    for path in paths:
        target = root / path.parent.parent.name / path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        for original in (path, path.parent / "meta.json"):
            if original.is_file():
                (target / original.name).symlink_to(original.resolve())
    return root


def fingerprint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {name: hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in checkpoint["state_dict"].items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--archives", type=int, default=0)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--steps", type=int, default=0, help="optimizer steps (0 = one full epoch)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", type=Path)
    p.add_argument("--variants", default=",".join(VARIANTS))
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    labels = args.labels.split(",")
    data = subset(args.data, set(labels), args.archives, args.work)
    common = ["--data", str(data), "--labels", ",".join(labels),
              "--packet-balance", ".5", "--pool", "meanmax", "--wait-keep", ".1", "--obs", "v2",
              "--cache-dir", str(args.work / "cache"), "--spread-weight", "8", "--spread-cap", "10",
              "--seed", str(args.seed), "--device", args.device, "--lr-schedule", "cosine",
              "--cell-soft-weight", "3", "--row-pool", "--packet-head", "identity",
              "--precompute-workers", "8", "--epochs", "1"]
    if args.steps:
        common += ["--max-steps", str(args.steps)]
    report = {"archives": args.archives or "all", "seed": args.seed, "device": args.device, "runs": {}}
    reference = None
    for name in args.variants.split(","):
        out = args.work / ("run_" + name.replace("+", "_"))
        if out.exists():
            shutil.rmtree(out)
        command = ["/usr/bin/time", "-l", sys.executable, str(ROOT / "scripts/train_rollout_policy.py"),
                   *common, "--out", str(out), *VARIANTS[name]]
        started = time.time()
        finished = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if finished.returncode:
            raise SystemExit(name + " failed:" + finished.stdout[-3000:] + finished.stderr[-3000:])
        peak = int(re.search(r"([0-9]+)  maximum resident set size", finished.stderr).group(1))
        rows = int(re.search(r"([0-9]+) decisions", finished.stdout).group(1))
        seconds = float(re.search(r"wrote .* \(([0-9.]+)s\)", finished.stdout).group(1))
        memory = next(json.loads(line) for line in finished.stdout.splitlines()
                      if line.startswith("{\"data_memory\""))["data_memory"]
        weights = fingerprint(out / "model.pt")
        if reference is None:
            reference = weights
        differ = sorted(key for key, value in weights.items() if reference.get(key) != value)
        report["runs"][name] = {"rows": rows, "peak_rss_bytes": peak, "wall_seconds": round(seconds, 1),
                                "total_seconds": round(time.time() - started, 1), "data_memory": memory,
                                "weights_differ_from_default": differ, "weight_tensors": len(weights)}
        print(json.dumps({name: report["runs"][name]}), flush=True)
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
