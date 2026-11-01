"""Offline A/B harness for the teacher's upgrade decisions (task214b).

Runs one cold-start teacher game per (parameter group, seed) by invoking the existing
``scripts/search_play.py`` entry point with an inline task definition (``--task-json``).
There is no second game loop here: this script only builds the task, calls that entry
point, and summarizes its ``games.jsonl``.

Example (Mac smoke, one game):
  python scripts/teacher_upgrade_ab.py \
    --level 29 --deck 1,16,35,10,42,39,44,30,21,17 --seeds 3237972681 \
    --student /tmp/task214-models/student/model.pt \
    --value /tmp/task214-models/v5/model.pt \
    --static-value /tmp/task214-models/static/model.pt \
    --group default \
    --group first2:--halving-first-samples 2 \
    --group first2rel:--halving-first-samples 2 --margin-relative 0.1 \
    --executable ~/pvz-mac-collect/build/pvz-portable \
    --workers 1 --timeout 3600 --out /tmp/task214b-smoke

The large panel belongs on the desktop; the Mac smoke should stay at one seed.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "scripts" / "search_play.py"
UPGRADE_TYPES = tuple(range(40, 48))
KEY_TYPES = (22, 35, 39) + UPGRADE_TYPES


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--deck", required=True, help="comma-separated seed types, e.g. 1,16,35,10,42,39,44,30,21,17")
    p.add_argument("--seeds", required=True, help="comma-separated game seeds")
    p.add_argument("--group", action="append", default=[],
                   help="NAME[:search_play args]; repeat for each arm. No --group means one 'default' arm. "
                        "Example: first2:--halving-first-samples 2 --margin-relative 0.1")
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--value", type=Path, required=True)
    p.add_argument("--static-value", type=Path, required=True)
    p.add_argument("--value-roof", type=Path)
    p.add_argument("--gpt", type=Path, help="GPT behaviour-clone prior for fog/roof levels")
    p.add_argument("--executable", type=Path, default=Path.home() / "pvz-mac-collect/build/pvz-portable")
    p.add_argument("--resource-dir", type=Path, default=Path.home() / ".cache/pvz-research-resources")
    p.add_argument("--workers", type=int, default=1, help="game workers inside search_play (smoke: 1)")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task214b_ab")
    p.add_argument("--python", default=sys.executable, help="Python that can import torch and the repo modules")
    p.add_argument("--timeout", type=float, default=0, help="per-arm timeout in seconds; 0 = none")
    p.add_argument("--keep-going", action="store_true", help="continue after an arm fails")
    return p


def _groups(values):
    groups = []
    seen = set()
    for value in values or ["default"]:
        name, sep, extra = value.partition(":")
        name = name.strip()
        if not name or name in seen:
            raise SystemExit(f"group names must be unique and non-empty: {value!r}")
        seen.add(name)
        groups.append((name, shlex.split(extra) if sep else []))
    return groups


def _ints(text, what):
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"{what} must be comma-separated integers: {text!r}") from exc
    if not values:
        raise SystemExit(f"{what} is empty")
    return values


def _task(level, deck, seeds):
    return {
        "task_id": f"task214b_L{level}",
        "level": level,
        "deck": list(deck),
        "seeds": list(seeds),
        "playthrough": 2,
        "zombie_count_multiplier": 1.0,
        "wave_cap": None,
        "preplanted": [],
        "first_aid_owned": False,
    }


def _command(args, group_args, task, out_dir):
    command = [args.python, str(SEARCH)]
    # The production teacher defaults (task214's 8-game panel): the student's top-2 plants and
    # nine verification futures.  A group may override either one below; the fixed arguments
    # after the group win when it tries to override the task, the models or the worker width.
    command += ["--student-candidates", "2", "--verify-samples", "9"]
    command += list(group_args)
    command += [
        "--rollout-policy", str(args.student),
        "--value", str(args.value),
        "--static-value", str(args.static_value),
        "--manifest", "/dev/null",
        "--task-json", json.dumps(task, separators=(",", ":")),
        "--task-id", task["task_id"],
        "--seed-list", ",".join(str(seed) for seed in task["seeds"]),
        "--seeds", str(len(task["seeds"])),
        "--arms", "fast_search",
        "--workers", str(args.workers),
        "--initial-workers", str(args.workers),
        "--out", str(out_dir),
        "--resource-dir", str(args.resource_dir),
        "--executable", str(args.executable),
    ]
    if args.value_roof:
        command += ["--value-roof", str(args.value_roof)]
    if args.gpt:
        command += ["--rollout-policy-gpt", str(args.gpt)]
    return command


def _read_games(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        planted = {str(key): int(value) for key, value in (raw.get("planted_types") or {}).items()}
        upgrade_planted = {str(seed): planted.get(str(seed), 0) for seed in UPGRADE_TYPES}
        key_planted = {str(seed): planted.get(str(seed), 0) for seed in KEY_TYPES}
        rows.append({
            "seed": raw.get("seed"),
            "won": bool(raw.get("won")),
            "final_wave": raw.get("final_wave"),
            "final_tick": raw.get("final_tick"),
            "seconds": raw.get("seconds"),
            "upgrade_planted": upgrade_planted,
            "upgrade_seen": [int(seed) for seed, count in upgrade_planted.items() if count],
            "key_planted": key_planted,
        })
    return rows


def main():
    args = parser()
    parsed = args.parse_args()
    if parsed.level < 1 or parsed.level > 50:
        raise SystemExit("--level must be 1..50")
    deck = _ints(parsed.deck, "--deck")
    if not 6 <= len(deck) <= 10 or len(set(deck)) != len(deck):
        raise SystemExit("--deck must contain 6..10 distinct seed types")
    seeds = _ints(parsed.seeds, "--seeds")
    for path, name in ((parsed.student, "--student"), (parsed.value, "--value"),
                       (parsed.static_value, "--static-value")):
        if not path.is_file():
            raise SystemExit(f"{name} not found: {path}")
    if not parsed.executable.is_file():
        raise SystemExit(f"--executable not found: {parsed.executable}")
    groups = _groups(parsed.group)

    run_dir = parsed.out / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    task = _task(parsed.level, deck, seeds)
    summary = {"run": str(run_dir), "level": parsed.level, "deck": deck, "seeds": seeds, "groups": {}}
    failed = []
    print(f"run {run_dir}: L{parsed.level} deck={deck} seeds={seeds} groups={[name for name, _ in groups]}")
    for name, group_args in groups:
        out_dir = run_dir / name
        command = _command(parsed, group_args, task, out_dir)
        print(f"[{name}] {' '.join(shlex.quote(part) for part in command)}", flush=True)
        started = time.time()
        try:
            completed = subprocess.run(command, text=True, capture_output=True,
                                       timeout=parsed.timeout or None, check=False)
        except subprocess.TimeoutExpired:
            failed.append(name)
            print(f"[{name}] TIMEOUT after {time.time() - started:.0f}s", flush=True)
            if not parsed.keep_going:
                break
            continue
        results = out_dir / "games.jsonl"
        if completed.returncode != 0 or not results.is_file():
            failed.append(name)
            print(f"[{name}] FAILED rc={completed.returncode} after {time.time() - started:.0f}s", flush=True)
            tail = (completed.stderr or completed.stdout or "").splitlines()[-30:]
            print("\n".join(tail), flush=True)
            if not parsed.keep_going:
                break
            continue
        rows = _read_games(results)
        summary["groups"][name] = {"args": group_args, "seconds": round(time.time() - started, 1),
                                   "games": rows}
        print(f"[{name}] {len(rows)} game(s) in {time.time() - started:.0f}s", flush=True)
        for row in rows:
            seen = ",".join(str(seed) for seed in row["upgrade_seen"]) or "-"
            print(f"  seed={row['seed']} won={int(row['won'])} wave={row['final_wave']} "
                  f"tick={row['final_tick']} upgrades={seen} key={row['key_planted']}", flush=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"summary {run_dir / 'summary.json'}")
    if failed:
        raise SystemExit(f"failed arms: {failed}")


if __name__ == "__main__":
    main()
