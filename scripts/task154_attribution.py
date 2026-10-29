"""task154 step 4: one-variable attribution for the teacher's losses on unseen decks.

Six decks the teacher lost 8/8 with (task129, 2026-10-27) spanning day / night / pool / fog / roof.
Every arm replays the same seeds, so arms are paired:

  base          v5 leaves, horizon 2500, samples 3, per-card 1, student-candidates 2
  value         same, but leaves scored by the value net retrained on recent qc frames
  continuation  horizon 4000, samples 5 (longer and more rollouts per candidate)
  width         per-card 2, student-candidates 4

    python3 scripts/task154_attribution.py manifest --out artifacts/task154/attribution
    python3 scripts/task154_attribution.py run --dir artifacts/task154/attribution --arm base
    python3 scripts/task154_attribution.py summarise --dir artifacts/task154/attribution
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pvz_constants  # noqa: E402

PIPELINE = Path("/home/newbiexvwu/pvz-logs/curriculum_state.pipeline")
STUDENT = PIPELINE / "round_000036/student/model.pt"
STATIC = Path("/home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt")
VALUE = Path("/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt")
RECENT = ROOT / "artifacts/task154/value_recent.pt"
EXECUTABLE = Path("/home/newbiexvwu/PvZ-search/.claude/worktrees/prodint16/build16/pvz-portable")
SEED_BLOCK = 154000000
SEEDS_PER_DECK = 8

# (level, deck, source) — teacher-loss decks from task129 (2026-10-27) and task148 (2026-10-29).
# Four are near-misses (the teacher won 1/8 with them), four are 0/8, so an arm that helps has
# somewhere to show it.  L42 task148 d1 is the one roof deck the teacher already won 4/8 with.
DECKS = [
    (42, [1, 2, 14, 15, 17, 30, 32, 33, 35, 41], "task148_l42_roof_r2_d1", 148001000),
    (9, [1, 41, 0, 8, 15, 30, 7, 6, 35], "deck129_v5_l09_d06", 129001500),
    (24, [34, 35, 16, 32, 26, 31, 41, 1], "deck129_v5_l24_d06", 129003500),
    (26, [2, 39, 35, 16, 19, 1, 4, 3, 18, 15], "deck129_v5_l26_d02", 129004100),
    (37, [21, 25, 17, 16, 18, 9, 32, 8, 34, 46], "deck129_v5_l37_d04", 129006300),
    (14, [36, 6, 32, 14, 5, 29, 9, 7, 20, 13], "deck129_v5_l14_d02", 129002100),
    (42, [3, 14, 9, 1, 34, 23, 33, 35, 0], "deck129_v5_l42_d01", 129007000),
    (48, [23, 33, 17, 1, 7, 26, 39], "deck129_v5_l48_d01", 129008000),
]

ARMS = {
    "base": {},
    "value": {"value": RECENT},
    "continuation": {"horizon": 4000, "samples": 5},
    "width": {"per_card": 2, "student_candidates": 4},
}


def terrain_of(level: int) -> str:
    return {"BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night", "BACKGROUND_3_POOL": "pool",
            "BACKGROUND_4_FOG": "fog", "BACKGROUND_5_ROOF": "roof",
            "BACKGROUND_6_BOSS": "boss"}[pvz_constants.background_for_level(level)]


def manifest(directory: Path) -> Path:
    tasks = []
    for index, (level, deck, source, seed_start) in enumerate(DECKS):
        tasks.append({
            "task_id": f"task154_l{level:02d}_d{index}",
            "level": level,
            "deck": deck,
            "terrain": terrain_of(level),
            "zombie_count_multiplier": 1.0,
            "playthrough": 2,
            "seeds": [seed_start + offset for offset in range(SEEDS_PER_DECK)],
            "source": source,
        })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "tasks": tasks}, indent=1))
    return path


def run(directory: Path, arm: str, workers: int, only: list[str] | None) -> None:
    path = directory / "manifest.json"
    tasks = json.loads(path.read_text())["tasks"]
    extra = ARMS[arm]
    for task in tasks:
        if only and task["task_id"] not in only:
            continue
        out = directory / "runs" / arm / task["task_id"]
        if (out / "summary.json").exists():
            print(f"skip {arm} {task['task_id']}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ROOT / "scripts/search_play.py"),
            "--rollout-policy", str(directory / "models/student.pt"),
            "--value", str(extra.get("value", directory / "models/value.pt")),
            "--static-value", str(directory / "models/static.pt"),
            "--manifest", str(path), "--task-id", task["task_id"],
            "--seed-list", ",".join(str(seed) for seed in task["seeds"]),
            "--arms", "fast_search", "--workers", str(workers),
            "--verify-samples", "9", "--student-candidates",
            str(extra.get("student_candidates", 2)),
            "--per-card", str(extra.get("per_card", 1)),
            "--samples", str(extra.get("samples", 3)),
            "--horizon", str(extra.get("horizon", 2500)),
            "--executable", str(EXECUTABLE),
            "--resource-dir", "/home/newbiexvwu/.cache/pvz-research-resources",
            "--out", str(out),
        ]
        print(" ".join(command), flush=True)
        started = time.time()
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
        (out / "stdout.log").write_text(result.stdout + result.stderr)
        print(f"{arm} {task['task_id']} rc={result.returncode} {time.time() - started:.0f}s",
              flush=True)
        if result.returncode != 0:
            raise SystemExit(f"search_play failed for {arm} {task['task_id']}")


def summarise(directory: Path) -> None:
    tasks = json.loads((directory / "manifest.json").read_text())["tasks"]
    table: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        table[arm] = {}
        for task in tasks:
            summary_path = directory / "runs" / arm / task["task_id"] / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            row = summary.get("fast_search") or {}
            table[arm][task["task_id"]] = {
                "level": task["level"], "terrain": task["terrain"], "deck": task["deck"],
                "won": row.get("won"), "games": row.get("games"),
                "mean_final_wave": row.get("mean_final_wave"),
                "mean_seconds": row.get("mean_seconds"),
                "overrides_per_game": row.get("overrides_per_game"),
                "searched_per_game": row.get("searched_per_game"),
            }
    totals = {}
    for arm, rows in table.items():
        won = sum(row["won"] or 0 for row in rows.values())
        games = sum(row["games"] or 0 for row in rows.values())
        totals[arm] = {"won": won, "games": games,
                       "rate": (won / games) if games else None}
    report = {"arms": table, "totals": totals}
    (directory / "results.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(totals, indent=1))
    header = f"{'task':<18}{'terrain':<8}" + "".join(f"{arm:>14}" for arm in ARMS)
    print(header)
    for task in tasks:
        line = f"{task['task_id']:<18}{task['terrain']:<8}"
        for arm in ARMS:
            row = table[arm].get(task["task_id"])
            line += f"{(str(row['won']) + '/' + str(row['games'])) if row else '-':>14}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("manifest", "run", "summarise"))
    parser.add_argument("--dir", type=Path, default=ROOT / "artifacts/task154/attribution")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arm", default="base")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    directory = args.out or args.dir
    if args.action == "manifest":
        print(manifest(directory))
    elif args.action == "run":
        run(directory, args.arm, args.workers,
            [name for name in (args.only or "").split(",") if name])
    else:
        summarise(directory)


if __name__ == "__main__":
    main()
