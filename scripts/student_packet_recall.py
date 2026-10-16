"""Does a student imitate each card of its teacher? Per packet: how many of the teacher's plant decisions the
student also calls a plant, and how many it gives the same packet. Rows and the validation split are the ones
train_rollout_policy.py uses (seeds shuffled with Random(0), first 10% held out over ``--split-labels``).

Usage:
  python scripts/student_packet_recall.py --data artifacts/student/frames --labels z13wide_fast_search \
      --split-labels u84_search_search,s1_fast_search,deep_fast_search,z13wide_fast_search \
      --models artifacts/student/v3all/model.pt,artifacts/student/v4/model.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_rollout_policy import features, load_student  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", required=True, help="teacher labels whose plant decisions are checked")
    p.add_argument("--split-labels", required=True, help="the student's training labels (for the seed split)")
    p.add_argument("--models", required=True)
    args = p.parse_args()
    split = set(args.split_labels.split(","))
    seeds = sorted({int(q.parent.parent.name.split("_")[-1]) for q in args.data.glob("*/*/frames.jsonl")
                    if q.parent.name in split})
    random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: len(seeds) // 10])
    rows = {"train": [], "val": []}
    for path in sorted(args.data.glob("*/*/frames.jsonl")):
        if path.parent.name not in args.labels.split(","):
            continue
        part = "val" if int(path.parent.parent.name.split("_")[-1]) in val_seeds else "train"
        frames = [json.loads(line) for line in path.open()]
        for i in range(1, len(frames)):
            a = frames[i].get("label") or frames[i].get("action")
            if a and a.get("type") == "plant" and not frames[i].get("action_rejected"):
                rows[part].append((*features(frames[i - 1], frames[i - 2] if i >= 2 else None), a["packet"]))
    for model in args.models.split(","):
        net = load_student(model)
        for part, rs in rows.items():
            if not rs:
                continue
            b = torch.stack([r[0] for r in rs]).float(); s = torch.stack([r[1] for r in rs]); pk = torch.tensor([r[2] for r in rs])
            with torch.inference_mode():
                out = net(b, s)
            called = out["plant"] > 0; same = out["packet"].argmax(1) == pk
            table = {int(k): {"n": int((pk == k).sum()), "plant": round(float(called[pk == k].float().mean()), 3),
                              "packet": round(float(same[pk == k].float().mean()), 3)} for k in pk.unique()}
            print(json.dumps({"model": model, "split": part, "rows": len(rs), "per_packet": table}), flush=True)


if __name__ == "__main__":
    main()
