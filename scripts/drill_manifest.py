"""Turn a mined drill bank into a task manifest with ``drill_start`` cells (task235).

The bank records logical roots plus archive-relative paths.  This script resolves the
real files, reads each archive's ``meta.json`` once, and writes one manifest task per
drill.  The emitted ``drill_start.archive`` is absolute: the production archives stay
where the desktop collector wrote them and are never copied into git.

Usage (production paths are the defaults):
  python3 scripts/drill_manifest.py --bank artifacts/task235/drill_bank.json \
      --out configs/task235_drills.json --future-seed-start 53000000

For a Mac end-to-end run point the roots at a local staging copy:
  python3 scripts/drill_manifest.py --bank ... --out artifacts/task235/drills.json \
      --archive-root framesv2=artifacts/task235/desktop/.../frames_v2 \
      --archive-root llmimport=artifacts/task235/desktop/.../sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

PRODUCTION_ROOTS = {
    "framesv2": Path("/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2"),
    "llmimport": Path("/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport/"
                      "artifacts/task105/import/sha256"),
}


def parse_roots(values):
    roots = dict(PRODUCTION_ROOTS)
    for entry in values:
        name, sep, path = entry.partition("=")
        if not sep:
            raise SystemExit("--archive-root needs NAME=PATH")
        roots[name] = Path(path).expanduser().resolve()
    return roots


def task_id_for(drill: dict) -> str:
    digest = hashlib.sha1(f"{drill['id']}|{drill['decision']}".encode()).hexdigest()[:8]
    trigger = drill.get("trigger", {}).get("name", "zombie")
    return f"drill235_l{drill['level']}_{trigger}_{digest}"


def task_for(drill: dict, meta: dict, archive: Path, *, future_seed_start: int,
             future_seed_count: int, horizon: int | None) -> dict:
    task = meta.get("task") or {}
    environment = meta.get("environment_task") or {}
    level = int(task.get("level") or environment.get("level") or drill["level"])
    if level != int(drill["level"]):
        raise ValueError(f"{drill['id']}: archive level {level} != bank level {drill['level']}")
    seed = task.get("seed")
    if seed is None:
        raise ValueError(f"{drill['id']}: archive meta has no task seed")
    deck = task.get("deck")
    conveyor = environment.get("conveyor_start")
    if not isinstance(deck, list) or (not deck and conveyor != "original"):
        raise ValueError(f"{drill['id']}: archive meta has no deck")
    fog = environment.get("fog_observation")
    if fog is None and drill.get("terrain") == "fog":
        fog = "human_v2"
    definition = dict(
        task_id=task_id_for(drill), level=level, seeds=[int(seed)],
        deck=list(deck),
        zombie_count_multiplier=float(
            task.get("zombie_count_multiplier", environment.get("zombie_count_multiplier", 1.0))),
        **({"fog_observation": fog} if fog else {}),
        **({"first_aid_owned": bool(task.get("first_aid_owned"))}
           if task.get("first_aid_owned") else {}),
        **({"wave_cap": environment["wave_cap"]} if environment.get("wave_cap") else {}),
        **({"preplanted": [list(p) for p in environment["preplanted"]]}
           if environment.get("preplanted") else {}),
        **({"conveyor_start": environment["conveyor_start"]}
           if environment.get("conveyor_start") else {}),
        drill_start=dict(
            archive=str(archive),
            decisions=[int(drill["decision"])], index=0,
            future_seed_start=int(future_seed_start),
            future_seed_count=int(future_seed_count),
            horizon=int(horizon or drill.get("horizon") or 1200),
            criterion=dict(drill.get("criterion") or {}),
            targets=[dict(target) for target in drill.get("targets") or []],
        ),
    )
    return definition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--archive-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--future-seed-start", type=int, default=53000000)
    parser.add_argument("--future-seed-count", type=int, default=256)
    parser.add_argument("--horizon", type=int, help="override every drill's horizon")
    parser.add_argument("--ids", type=Path, help="JSON list of drill ids to emit (default: all)")
    parser.add_argument("--report", type=Path, help="write the bank/calibration summary here")
    args = parser.parse_args()
    roots = parse_roots(args.archive_root)
    bank = json.loads(args.bank.read_text())
    drills = bank.get("drills") or []
    if args.ids:
        wanted = set(json.loads(args.ids.read_text()))
        drills = [drill for drill in drills if drill["id"] in wanted]
    if not drills:
        parser.error("no drills selected")
    tasks, counts = [], {}
    for index, drill in enumerate(drills):
        root = roots.get(drill["source_root"])
        if root is None:
            parser.error(f"unknown source root {drill['source_root']!r}")
        archive = root / drill["archive"]
        meta = json.loads((archive / "meta.json").read_text())
        start = args.future_seed_start + index * args.future_seed_count
        definition = task_for(drill, meta, archive, future_seed_start=start,
                              future_seed_count=args.future_seed_count, horizon=args.horizon)
        definition["drill_meta"] = dict(id=drill["id"], terrain=drill["terrain"],
                                        trigger=drill.get("trigger"),
                                        source_root=drill["source_root"],
                                        archive=drill["archive"],
                                        decision=drill["decision"],
                                        teacher=drill.get("teacher"),
                                        coverage_only=bool(drill.get("coverage_only")))
        tasks.append(definition)
        for target in drill.get("targets") or []:
            key = f"{drill['level']}|{target['name']}"
            counts[key] = counts.get(key, 0) + 1
    document = dict(schema="pvz-drill-manifest-v1", task="task235",
                    seed_block=[args.future_seed_start,
                                args.future_seed_start + len(tasks) * args.future_seed_count],
                    future_seed_count=args.future_seed_count,
                    counts=dict(drills=len(tasks), by_level_type=dict(sorted(counts.items()))),
                    tasks=tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    if args.report:
        args.report.write_text(json.dumps(
            dict(drills=len(tasks), seed_block=document["seed_block"],
                 by_level_type=document["counts"]["by_level_type"],
                 sources=bank.get("sources"), calibration=bank.get("calibration")),
            ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(drills=len(tasks), seed_block=document["seed_block"],
                          by_level_type=document["counts"]["by_level_type"]),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
