"""Measure how much of a task pool a policy that never acts can already win.

A win rate only means something relative to a floor. On short PvZ tasks the
floor is high: lawnmowers absorb the first zombies, so a policy that only
waits clears most 1/3/5-wave tasks without planting a single thing. Any task
where this baseline wins is a task whose win rate carries no policy
information, and it should not be used as a progress metric or weighted in a
curriculum.

This script measures that floor. It runs no neural network at all -- the
policy is "wait forever" -- so it needs no GPU, only the real simulator.

Usage::

    python scripts/donothing_baseline.py \
        --manifest experiments/t5/multicap_course_v1/train.json \
        --manifest experiments/t7/bridge_level7_v1/train.json \
        --seeds 8 --workers 8 \
        --output artifacts/t5/perf/donothing_baseline_v1.json

Reads each task's own frozen seed list, so the numbers are on the same seeds
the project already evaluates on. ``--seeds`` is a screen, not the full 64;
treat it as a screen and widen it before making an irreversible decision.

The output records ``simulator_sha256``. Two baselines are only comparable when
that matches: the 2026-10-02 mower/intro-initialisation fix moved the floor, so
an old-build floor and a current-build win rate must not be subtracted from each
other. The same reason makes this file reproducible by construction -- verified
2026-10-02: two full runs of all 88 task entries and of the 35-entry evaluation
manifest disagreed on 0 entries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import policy_env  # noqa: E402

DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
# wait_mode=events keeps the native clock/metadata path identical to training.
DONOTHING_CONFIG = {"wait_mode": "events", "wait_mask": "progress_v1", "input_flags": 7}
WAIT_ACTION = {"type": "wait", "ticks": 300, "until": "timeout"}

_ENV = None


def _simulator_fingerprint() -> str:
    """SHA256 of the simulator binary, the same identity training records.

    Two baselines are only comparable if this matches: the 2026-10-02 mower /
    intro-initialisation fix changed the floor, so an old-build floor and a
    current-build win rate cannot be subtracted from each other.
    """
    return hashlib.sha256((ROOT / "build/pvz-portable").read_bytes()).hexdigest()


def _get_env(resource_dir: str):
    """One simulator per worker process; reset() re-tasks it per episode."""
    global _ENV
    if _ENV is None:
        _ENV = policy_env(DONOTHING_CONFIG, resource_dir=resource_dir, headless=True)
    return _ENV


def _one_episode(job: tuple) -> dict:
    resource_dir, level, deck, multiplier, wave_cap, preplanted, playthrough, seed, max_actions = job
    env = _get_env(resource_dir)
    spec = TaskSpec(level=level, seed=seed, playthrough=playthrough,
                    zombie_count_multiplier=multiplier, wave_cap=wave_cap,
                    preplanted=tuple(tuple(p) for p in preplanted))
    observation, _ = env.reset(deck=deck, task=spec)
    decisions = 0
    for _ in range(max_actions):
        observation, _, done, _, _ = env.step(WAIT_ACTION)
        decisions += 1
        if done:
            break
    return {"won": observation.get("result") == 1,
            "terminal_wave": observation.get("wave"),
            "ticks": observation.get("tick"), "decisions": decisions}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-actions", type=int, default=800)
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    p.add_argument("--output", default="artifacts/t5/perf/donothing_baseline_v1.json")
    args = p.parse_args()

    jobs, meta = [], []
    for manifest in args.manifest:
        tasks = json.loads(Path(manifest).read_text(encoding="utf-8"))["tasks"]
        for task in tasks:
            seeds = list(task.get("seeds") or [])[: args.seeds]
            meta.append({"manifest": manifest, "task_id": task["task_id"],
                         "terrain": task.get("terrain"), "level": task["level"],
                         "wave_cap": task.get("wave_cap"),
                         "preplanted": bool(task.get("preplanted")),
                         "seeds": len(seeds)})
            for seed in seeds:
                jobs.append((args.resource_dir, task["level"], list(task["deck"]),
                             float(task.get("zombie_count_multiplier", 1.0)),
                             task.get("wave_cap"), list(task.get("preplanted") or []),
                             int(task.get("playthrough", 2)), int(seed), args.max_actions))

    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(_one_episode, jobs, chunksize=1):
            results.append(result)
    elapsed = round(time.perf_counter() - started, 1)

    cursor, rows = 0, []
    for entry in meta:
        batch = results[cursor:cursor + entry["seeds"]]
        cursor += entry["seeds"]
        wins = sum(r["won"] for r in batch)
        rows.append({**entry, "baseline_wins": wins, "n": len(batch),
                     "baseline_win_rate": wins / len(batch) if batch else None,
                     "mean_terminal_wave": round(sum(r["terminal_wave"] or 0 for r in batch) / len(batch), 1) if batch else None})

    payload = {
        "schema_version": 1,
        "policy": "donothing (wait 300 ticks forever, never plants or shovels)",
        "purpose": "floor for every win rate; a task the floor wins carries no policy information",
        "simulator_sha256": _simulator_fingerprint(),
        "seeds_per_task": args.seeds, "max_actions": args.max_actions,
        "workers": args.workers, "seconds": elapsed,
        "scope": "screen on each task's own frozen seeds, not the full 64-seed acceptance",
        "tasks": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{len(jobs)} episodes in {elapsed}s -> {out}", flush=True)
    print(f"simulator_sha256 {payload['simulator_sha256']}", flush=True)
    for row in sorted(rows, key=lambda r: (r["manifest"], r["baseline_win_rate"] or 0)):
        flag = "空(无信息)" if (row["baseline_win_rate"] or 0) >= 0.75 else (
            "有区分度" if (row["baseline_win_rate"] or 0) <= 0.25 else "弱")
        print(f"  {row['task_id']:24s} {str(row['terrain']):6s} L{row['level']:<3} "
              f"cap{str(row['wave_cap']):<5} aid={int(row['preplanted'])} "
              f"{row['baseline_wins']}/{row['n']}  {flag}", flush=True)


if __name__ == "__main__":
    main()
