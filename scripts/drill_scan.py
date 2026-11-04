"""List episode archives worth mining for drills (task235).

``drill_mine.py`` takes an explicit archive list; this script discovers the archives
under the three production roots without following symlinks and deduplicates by real
path (the curriculum pipeline keeps symlinks into ``frames_v2``).  It reads only each
archive's small ``meta.json`` and writes a KB/MB JSON list -- never the frames.

  python3 scripts/drill_scan.py --levels 31-34,36-40 --out artifacts/task235/archives.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

DEFAULT_ROOTS = [
    ("framesv2", "~/PvZ-search/artifacts/student/frames_v2"),
    ("llmimport", "~/PvZ-search/.claude/worktrees/llmimport/artifacts/task105/import/sha256"),
    ("pipeline", "~/pvz-logs/curriculum_state.pipeline/frames"),
]


def parse_levels(text: str) -> set[int]:
    levels = set()
    for part in (piece.strip() for piece in text.split(",")):
        if not part:
            continue
        low, _, high = part.partition("-")
        start, stop = int(low), int(high or low)
        if stop < start:
            raise ValueError(f"invalid level range: {part}")
        levels.update(range(start, stop + 1))
    return levels


def meta_level(meta: dict) -> int:
    return int(((meta.get("task") or {}).get("level")
                or (meta.get("environment_task") or {}).get("level")
                or (meta.get("context") or {}).get("level") or 0))


def scan(root: str, name: str, levels: set[int], seen: dict, skipped: Counter) -> None:
    if not os.path.isdir(root):
        skipped[f"missing_root:{name}"] += 1
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        if "frames.npz" not in filenames:
            continue
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.isfile(meta_path):
            skipped["no_meta"] += 1
            dirnames[:] = []
            continue
        try:
            meta = json.loads(Path(meta_path).read_text())
        except (OSError, ValueError):
            skipped["bad_meta"] += 1
            dirnames[:] = []
            continue
        if meta.get("demo_archive"):
            skipped["demo_continuation"] += 1
            dirnames[:] = []
            continue
        level = meta_level(meta)
        if level not in levels:
            skipped["level"] += 1
            dirnames[:] = []
            continue
        target = os.path.realpath(directory)
        if target in seen:
            skipped["duplicate_realpath"] += 1
            dirnames[:] = []
            continue
        task = meta.get("task") or {}
        seen[target] = dict(source_root=name, archive=os.path.relpath(directory, root),
                            path=target, level=level, seed=task.get("seed"),
                            task_id=task.get("id"), label_source=meta.get("label_source"),
                            verified=bool(meta.get("verified")),
                            multiplier=task.get("zombie_count_multiplier"),
                            deck=task.get("deck"), frames=meta.get("frame_count"))
        dirnames[:] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--levels", default="31-34,36-40")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    levels = parse_levels(args.levels)
    roots = ([(name, path) for name, path in
              (entry.partition("=")[::2] for entry in args.root)]
             if args.root else DEFAULT_ROOTS)
    roots = [(name, str(Path(path).expanduser())) for name, path in roots]
    seen, skipped = {}, Counter()
    for name, root in roots:
        scan(root, name, levels, seen, skipped)
    items = sorted(seen.values(), key=lambda item: (item["level"], item["source_root"], item["archive"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"total": len(items),
                      "by_level": dict(sorted(Counter(item["level"] for item in items).items())),
                      "by_source": dict(sorted(Counter(item["source_root"] for item in items).items())),
                      "skipped": dict(skipped)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
