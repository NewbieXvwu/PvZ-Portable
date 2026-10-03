"""Score the hand-written scripted policy on a full acceptance manifest.

The scripted teacher in ``scripts/scripted_baseline.py`` is the only policy in
this repo known to beat full level 7, but every recorded run of it used four
seeds per probe. Four wins out of four has a 95% interval of [0.51, 1.00], so
those records cannot distinguish "solves the game" from "wins half the time".
This script answers the question the records leave open, on the exact frozen
manifest the acceptance thresholds are defined against.

It also fixes a deck bug in the original entry point: ``scripted_baseline``
hard-codes ``DECK = (0,1,2,3,4,5)``, which matches only two of the six
acceptance tasks (level 7 day and day 8). Here each task uses its own deck.

No neural network, no PPO, no GPU. Read-only with respect to every artifact it
does not write.

Usage::

    python scripts/score_scripted_acceptance.py \
        --manifest experiments/t7/full_level7_v1/acceptance.json \
        --workers 8 \
        --output artifacts/t5/perf/scripted_acceptance_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from scripted_baseline import choose  # noqa: E402

DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
MAX_ACTIONS = 4000

_ENV = None


def _simulator_fingerprint() -> str:
    return hashlib.sha256((ROOT / "build/pvz-portable").read_bytes()).hexdigest()


def _get_env(resource_dir: str):
    global _ENV
    if _ENV is None:
        _ENV = PvZEnv(resource_dir, headless=True)
    return _ENV


def _one_case(job: tuple) -> dict:
    (resource_dir, level, deck, multiplier, wave_cap, preplanted, playthrough,
     seed, max_actions) = job
    env = _get_env(resource_dir)
    spec = TaskSpec(level=level, seed=seed, playthrough=playthrough,
                    zombie_count_multiplier=multiplier, wave_cap=wave_cap,
                    preplanted=tuple(tuple(p) for p in preplanted))
    observation, _ = env.reset(deck=list(deck), task=spec)
    actions = 0
    mowers = 0
    info: dict = {}
    while not observation["terminal"] and actions < max_actions:
        action = choose(observation)
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            observation, _, done, _, info = env.step({"type": "wait", "ticks": 60})
        mowers += int((info.get("events") or {}).get("mower_triggered", 0) or 0)
        actions += 1
        if done:
            break
    result = int(observation["result"])
    planted = Counter(plant["type"] for plant in observation["plants"])
    return {"seed": seed, "won": result == 1, "result": result,
            "terminal": bool(observation["terminal"]),
            "truncated": not bool(observation["terminal"]),
            "wave": observation["wave"], "tick": observation["tick"],
            "actions": actions, "mowers": mowers,
            "plants_left": dict(planted), "sun_left": observation["sun"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    p.add_argument("--output", default="artifacts/t5/perf/scripted_acceptance_v1.json")
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    thresholds = manifest.get("thresholds") or {}

    jobs, meta = [], []
    for task in manifest["tasks"]:
        seeds = list(task.get("seeds") or [])
        meta.append({"task_id": task["task_id"], "terrain": task.get("terrain"),
                     "level": task["level"], "wave_cap": task.get("wave_cap"),
                     "deck": list(task["deck"]), "seeds": len(seeds)})
        for seed in seeds:
            jobs.append((args.resource_dir, task["level"], list(task["deck"]),
                         float(task.get("zombie_count_multiplier", 1.0)),
                         task.get("wave_cap"), list(task.get("preplanted") or []),
                         int(task.get("playthrough", 2)), int(seed), args.max_actions))

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_one_case, jobs, chunksize=1):
            results.append(result)
    elapsed = round(time.perf_counter() - started, 1)

    cursor, rows = 0, []
    for entry in meta:
        batch = results[cursor:cursor + entry["seeds"]]
        cursor += entry["seeds"]
        wins = sum(r["won"] for r in batch)
        rows.append({**entry, "wins": wins, "n": len(batch),
                     "win_rate": wins / len(batch) if batch else None,
                     "truncated": sum(r["truncated"] for r in batch),
                     "mean_terminal_wave": round(sum(r["wave"] or 0 for r in batch) / len(batch), 2) if batch else None,
                     "mean_mowers_triggered": round(sum(r["mowers"] for r in batch) / len(batch), 2) if batch else None,
                     "mean_sun_left": round(sum(r["sun_left"] for r in batch) / len(batch), 1) if batch else None,
                     "episodes": batch})

    level7 = next((r for r in rows if r["task_id"] == "level7_development"), None)
    terrains = [r for r in rows if r["task_id"] != "level7_development"]
    terrain_rates = [r["win_rate"] for r in terrains if r["win_rate"] is not None]
    macro = sum(terrain_rates) / len(terrain_rates) if terrain_rates else None
    weakest = min(terrain_rates) if terrain_rates else None

    payload = {
        "schema_version": 1,
        "policy": "scripted_baseline.choose (5-priority rule list; each task uses its own deck)",
        "simulator_sha256": _simulator_fingerprint(),
        "manifest": args.manifest, "max_actions": args.max_actions,
        "workers": args.workers, "seconds": elapsed,
        "purpose": "Score the only known winning policy on the frozen acceptance set; "
                   "existing records cover 4 seeds per probe, which cannot separate a solver from a coin flip.",
        "scope": "No neural network, no PPO, no GPU. Read-only. This is a teacher-quality probe, "
                 "not a change to any threshold or any running training candidate.",
        "thresholds": thresholds,
        "level7_development_win_rate": level7["win_rate"] if level7 else None,
        "five_terrain_macro_pass_rate": macro,
        "weakest_task_pass_rate": weakest,
        "tasks": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(jobs)} cases in {elapsed}s -> {out}", flush=True)
    print(f"simulator_sha256 {payload['simulator_sha256']}", flush=True)
    print(f"level7_development {payload['level7_development_win_rate']}  "
          f"five_terrain_macro {macro}  weakest {weakest}", flush=True)
    for row in rows:
        print(f"  {row['task_id']:22s} {str(row['terrain']):6s} L{row['level']:<3} "
              f"{row['wins']:>4}/{row['n']:<4} trunc={row['truncated']:<3} "
              f"终波{row['mean_terminal_wave']:<6} 割草机{row['mean_mowers_triggered']:<5} "
              f"剩余阳光{row['mean_sun_left']}", flush=True)


if __name__ == "__main__":
    main()
