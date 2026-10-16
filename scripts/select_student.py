"""Pick the student that plays best, not the one that imitates best (learn/06: same data, another training seed,
same held-out imitation loss, 140 fewer wins out of 1024).

Every candidate plays alone (C++, greedy) on selection seeds that are not evaluation seeds -- by default the
unused tail of the 1.4x and 1.5x training tasks -- and the one with the most wins is copied to --out.

Usage:
  python scripts/select_student.py --candidates 'artifacts/student/d12s*/model.pt' --out artifacts/student/v12
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, help="comma-separated paths or globs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", default=str(ROOT / "artifacts/stage4/day_l7_hard.json"))
    p.add_argument("--sets", default="day_l7_z14_train:768:256,day_l7_z15_train:384:256",
                   help="task:offset:count,... (keep them apart from evaluation seeds and from collection seeds)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--value", default=str(ROOT / "artifacts/value_net/v2/model.pt"), help="unused by the fast arm")
    p.add_argument("--resource-dir")
    args = p.parse_args()
    paths = sorted({q for pattern in args.candidates.split(",") for q in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no candidates")
    work = args.out / "selection"
    scores = {}
    for i, path in enumerate(paths):
        wins = {}
        for spec in args.sets.split(","):
            task, offset, count = spec.split(":")
            out = work / f"c{i}_{task}"
            cmd = [sys.executable, str(ROOT / "scripts/search_play.py"), "--rollout-policy", path, "--value", args.value,
                   "--manifest", args.manifest, "--task-id", task, "--seed-offset", offset, "--seeds", count,
                   "--arms", "fast", "--workers", str(args.workers), "--out", str(out)]
            if args.resource_dir:
                cmd += ["--resource-dir", args.resource_dir]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wins[task] = json.loads((out / "summary.json").read_text())["fast"]["won"]
        scores[path] = {"wins": wins, "total": sum(wins.values())}
        print(json.dumps({"candidate": path, **scores[path]}), flush=True)
    best = max(paths, key=lambda q: scores[q]["total"])
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out / "model.pt")
    (args.out / "selection.json").write_text(json.dumps({"best": best, "sets": args.sets, "scores": scores}, indent=1))
    print(f"best {best} -> {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
