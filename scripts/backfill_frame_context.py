"""Add the task context (level, zombie-count multiplier, zombie roster) to archives written before
search_play recorded it, so scripts/train_value_net.py --context can use them.

The multiplier comes from the archive's label (directory name), the roster from one reset of the level.
Archives that already have a context are left alone.

Usage:
  python scripts/backfill_frame_context.py --data artifacts/student/frames --level 7 \
      --multiplier u84_search_search=1,s1_fast_search=1,deep_fast_search=1,z13wide_fast_search=1.3,z15wide_fast_search=1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402
from pvz_tasks import profile_for_deck  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--multiplier", required=True, help="label=multiplier,...")
    p.add_argument("--deck", default="0,1,2,3,4,5,7")
    p.add_argument("--resource-dir", default="/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
    args = p.parse_args()
    multiplier = {k: float(v) for k, v in (kv.split("=") for kv in args.multiplier.split(","))}
    deck = tuple(int(x) for x in args.deck.split(","))
    with EventWaitEnv(Path(args.resource_dir)) as env:
        obs, _ = env.reset(deck=deck, task=TaskSpec(level=args.level, seed=0, playthrough=2,
                                                    profile=profile_for_deck(deck)))
        roster = obs["loadout_context"]["zombie_roster"]
    done = 0
    for meta_path in sorted(args.data.glob("*/*/meta.json")):
        label = meta_path.parent.name
        meta = json.loads(meta_path.read_text())
        if label not in multiplier or "context" in meta:
            continue
        meta["context"] = {"level": args.level, "multiplier": multiplier[label], "roster": roster}
        meta_path.write_text(json.dumps(meta))
        done += 1
    print(f"level {args.level} roster {roster}: added context to {done} archives")


if __name__ == "__main__":
    main()
