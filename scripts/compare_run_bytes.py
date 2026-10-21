"""Are two search_play runs the same run, byte for byte?

The runs are compared on everything a training archive holds -- every row of games.jsonl, every
array of every frames.npz, the archived frame JSON and meta.json -- with wall-clock fields left
out, because those differ between any two runs of the same arm and are not part of a label.

    python scripts/compare_run_bytes.py artifacts/hard_mac/runA artifacts/hard_mac/runB

Both arguments are --out directories of scripts/search_play.py.  The archive directory is read
from each run's config.json, so a run collected without --frames is compared on games.jsonl
alone.  Exit code 1 means something differs; the first differing field is printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TIMING = frozenset({"seconds", "cuda_seconds"})


def strip(value):
    """Drop wall-clock fields at any depth (the search writes one into each verified frame)."""
    if isinstance(value, dict):
        return {key: strip(item) for key, item in value.items() if key not in TIMING}
    if isinstance(value, list):
        return [strip(item) for item in value]
    return value


def games(run: Path) -> dict:
    rows = {}
    for line in (run / "games.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["arm"], row["seed"])] = strip(row)
    return rows


def archives(run: Path) -> dict:
    config = json.loads((run / "config.json").read_text())
    root = Path(config.get("frames") or "")
    label = config.get("frames_label") or "play"
    if not root.is_dir():
        return {}
    return {path.parent.relative_to(root): path
            for path in sorted(root.glob(f"*/{label}_*/frames.npz"))}


def compare_games(a: dict, b: dict) -> int:
    problems = 0
    for key in sorted(set(a) ^ set(b)):
        print(f"games.jsonl: only in one run: {key}")
        problems += 1
    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            differing = {name: (a[key].get(name), b[key].get(name))
                         for name in a[key] if a[key].get(name) != b[key].get(name)}
            print(f"games.jsonl: {key} differs: {differing}")
            problems += 1
    print(f"games.jsonl: {len(a)} rows compared")
    return problems


def compare_archive(left: Path, right: Path) -> int:
    problems = 0
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            print(f"{left.parent.name}: arrays differ: {sorted(set(a.files) ^ set(b.files))}")
            return 1
        for name in a.files:
            x, y = a[name], b[name]
            if name != "frames":
                if not np.array_equal(x, y, equal_nan=np.issubdtype(x.dtype, np.floating)):
                    print(f"{left.parent.name}: array {name} differs")
                    problems += 1
                continue
            frames_a = [strip(json.loads(line)) for line in x.tobytes().splitlines() if line.strip()]
            frames_b = [strip(json.loads(line)) for line in y.tobytes().splitlines() if line.strip()]
            if len(frames_a) != len(frames_b):
                print(f"{left.parent.name}: {len(frames_a)} vs {len(frames_b)} frames")
                problems += 1
            for index, (frame_a, frame_b) in enumerate(zip(frames_a, frames_b)):
                if frame_a != frame_b:
                    differing = {key: (frame_a.get(key), frame_b.get(key))
                                 for key in frame_a if frame_a.get(key) != frame_b.get(key)}
                    print(f"{left.parent.name}: frame {index} differs: {differing}")
                    problems += 1
                    break
    meta_a = json.loads((left.parent / "meta.json").read_text())
    meta_b = json.loads((right.parent / "meta.json").read_text())
    if strip(meta_a) != strip(meta_b):
        print(f"{left.parent.name}: meta.json differs")
        problems += 1
    return problems


def compare(left: Path, right: Path) -> int:
    problems = compare_games(games(left), games(right))
    first, second = archives(left), archives(right)
    for key in sorted(set(first) ^ set(second)):
        print(f"archives: only in one run: {key}")
        problems += 1
    for key in sorted(set(first) & set(second)):
        problems += compare_archive(first[key], second[key])
    print(f"archives: {len(first)} compared")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=Path, help="--out directory of the first run")
    parser.add_argument("right", type=Path, help="--out directory of the second run")
    args = parser.parse_args()
    problems = compare(args.left, args.right)
    print("identical" if not problems else f"{problems} differences")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
