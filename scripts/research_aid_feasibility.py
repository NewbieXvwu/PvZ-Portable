"""Wait/scripted feasibility probes for a frozen preplant manifest, never training data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv, TaskSpec
from pvz_seed_jobs import atomic_json


def choose(observation: dict) -> dict:
    """A deterministic public-observation probe; it never removes plants.

    Plant choices remain within the task deck. This routine establishes a feasible
    solution for a task; its actions and trajectories are not fed into PPO.
    """
    legal = observation["legal_actions"]["plants"]
    packets = observation["packets"]
    by_type: dict[int, list[dict]] = {}
    for item in legal:
        by_type.setdefault(packets[item["packet"]]["type"], []).append(item)
    active_rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"]})
    water_rows = {cell["row"] for cell in observation["cells"] if cell["row_type"] == 2}
    plants = [p for p in observation["plants"] if p.get("health", 0) > 0 and not p.get("squished")]
    is_night = observation["night"]
    economy, shooter = (9, 10) if is_night else (1, 0)
    shooter_col = 6 if observation["roof"] else (5 if is_night else 3)
    threats = {row: min((z["x"] for z in observation["zombies"]
                         if z["row"] == row and z["on_board"]), default=9999)
               for row in active_rows}
    counts = Counter((p["type"], p["row"]) for p in plants)
    occupied = {(p["type"], p["row"], p["col"]) for p in plants}

    def plant(kind: int, row: int, cols: tuple[int, ...]) -> dict | None:
        for col in cols:
            for option in by_type.get(kind, []):
                if (option["row"], option["col"]) == (row, col):
                    return {"type": "plant", **option}
            # Make a needed support using the same legal-action interface.
            support = 33 if observation["roof"] else (16 if row in water_rows else None)
            if support is not None and (support, row, col) not in occupied:
                for option in by_type.get(support, []):
                    if (option["row"], option["col"]) == (row, col):
                        return {"type": "plant", **option}
        return None

    threatened = sorted(active_rows, key=lambda row: (threats[row], counts[(shooter, row)], row))
    # Free short-range mushrooms form the immediate defence on night/fog decks.
    if 8 in by_type:
        for row in threatened:
            if counts[(8, row)] < 3:
                action = plant(8, row, (6, 7, 8))
                if action:
                    return action
    for row in threatened:
        if threats[row] < 650 and counts[(shooter, row)] < 2:
            action = plant(shooter, row, (shooter_col, shooter_col + 1))
            if action:
                return action
    # Build two economy plants per row, then complete offensive coverage.
    for row in sorted(active_rows, key=lambda row: (counts[(economy, row)], row)):
        if counts[(economy, row)] < 2:
            action = plant(economy, row, (0, 1))
            if action:
                return action
    for row in sorted(active_rows, key=lambda row: (counts[(shooter, row)], threats[row], row)):
        if counts[(shooter, row)] < 3:
            action = plant(shooter, row, (shooter_col, shooter_col + 1, shooter_col + 2))
            if action:
                return action
    return {"type": "wait", "ticks": 150}


def episode(env: PvZEnv, task: dict, seed: int, strategy: str, max_actions: int) -> dict:
    started = time.monotonic()
    spec = TaskSpec(level=task["level"], seed=seed, playthrough=task["playthrough"],
                    zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
                    preplanted=tuple(tuple(p) for p in task["preplanted"]))
    observation, _ = env.reset(deck=task["deck"], task=spec)
    if observation["sun"] != task["sun_start"]:
        raise ValueError("unexpected starting sun")
    if observation["wave_count"] != task["wave_cap"]:
        raise ValueError("requested cap differs from the real task wave count")
    initial_plants = [(p["type"], p["row"], p["col"]) for p in observation["plants"]]
    if not set(spec.preplanted).issubset(initial_plants):
        raise ValueError("requested preplants missing from reset")
    counts, events = Counter(), Counter()
    for index in range(max_actions):
        action = {"type": "wait", "ticks": 300} if strategy == "wait" else choose(observation)
        observation, _, done, _, info = env.step(action)
        if not info["ok"]:
            raise RuntimeError(f"illegal feasibility action: {task['task_id']} seed {seed} {action}")
        counts[action["type"]] += 1
        events.update(info["events"])
        if done:
            break
    return {"task_id": task["task_id"], "terrain": task["terrain"], "aid": task["aid"],
            "wave_cap": task["wave_cap"], "environment_seed": seed, "strategy": strategy,
            "won": observation["result"] == 1, "result": observation["result"],
            "terminated": bool(observation["terminal"]), "truncated": not observation["terminal"],
            "wave": observation["wave"], "wave_count": observation["wave_count"],
            "tick": observation["tick"], "actions": index + 1, "action_counts": dict(counts),
            "events": dict(events), "initial_plants": initial_plants,
            "seconds": time.monotonic() - started}


def wilson(won: int, count: int) -> list[float]:
    z = 1.959963984540054
    p, denominator = won / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * ((p * (1 - p) / count + z * z / (4 * count * count)) ** .5) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds-per-task", type=int, default=None,
                        help="initial inspection only; a subset cannot pass the full feasibility protocol")
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    config = json.loads(manifest_bytes)
    if config["purpose"] != "feasibility_only_no_training":
        raise ValueError("not a feasibility manifest")
    if args.seeds_per_task is not None and not 1 <= args.seeds_per_task <= 64:
        raise ValueError("inspection seed count must be 1..64")
    fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = args.output_dir / "protocol.json"
    scope = {"manifest_sha256": fingerprint, "seeds_per_task": args.seeds_per_task,
             "simulator_sha256": hashlib.sha256((ROOT / "build/pvz-portable").read_bytes()).hexdigest(),
             "env_sha256": hashlib.sha256((ROOT / "python/pvz_env.py").read_bytes()).hexdigest(),
             "main_pak_sha256": hashlib.sha256((args.resource_dir / "main.pak").read_bytes()).hexdigest(),
             "partner_xml_sha256": hashlib.sha256((args.resource_dir / "properties/partner.xml").read_bytes()).hexdigest(),
             "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if previous.exists() and json.loads(previous.read_text()) != scope:
        raise ValueError("cannot mix protocols in an existing probe directory")
    atomic_json(previous, scope)
    started = time.monotonic()
    rows, summaries = [], {}
    try:
        with PvZEnv(resource_dir=args.resource_dir) as env:
            for task in config["tasks"]:
                seeds = task["seeds"][:args.seeds_per_task] if args.seeds_per_task is not None else task["seeds"]
                for strategy in config["strategies"]:
                    records = []
                    for seed in seeds:
                        path = args.output_dir / "episodes" / f"{task['task_id']}_{strategy}_{seed}.json"
                        if path.exists():
                            record = json.loads(path.read_text())
                        else:
                            record = episode(env, task, seed, strategy, config["max_actions"])
                            atomic_json(path, record)
                        records.append(record)
                    rows.extend(records)
                    won = sum(r["won"] for r in records)
                    summary = {"count": len(records), "won": won, "win_rate": won / len(records),
                               "wilson_95": wilson(won, len(records)),
                               "truncated": sum(r["truncated"] for r in records),
                               "mean_tick": sum(r["tick"] for r in records) / len(records)}
                    summaries.setdefault(task["task_id"], {})[strategy] = summary
                    print(task["task_id"], strategy, f"{won}/{len(records)}", flush=True)
                atomic_json(args.output_dir / "progress.json", {"task_id": task["task_id"],
                            "completed_episodes": len(rows), "seconds": time.monotonic() - started})
    except Exception as error:
        atomic_json(args.output_dir / "failure.json", {"error_type": type(error).__name__,
                    "error": str(error), "completed_episodes": len(rows),
                    "seconds": time.monotonic() - started})
        raise
    selection = config["prospective_curriculum_screen"]
    candidates = [task_id for task_id, result in summaries.items()
                  if result["wait"]["count"] >= selection["minimum_seeds"]
                  and result["wait"]["win_rate"] < selection["maximum_wait_win_rate"]
                  and result["scripted"]["win_rate"] >= selection["minimum_scripted_win_rate"]
                  and result["scripted"]["win_rate"] - result["wait"]["win_rate"] >= selection["minimum_gap"]]
    atomic_json(args.output_dir / "results.json.gz", rows, compressed=True)
    atomic_json(args.output_dir / "summary.json", {"protocol": scope, "summaries": summaries,
                "prospective_curriculum_candidates": candidates, "screen": selection,
                "scope": "feasibility, not a learned-policy evaluation or T5 learning gate",
                "seconds": time.monotonic() - started, "episodes": len(rows)})


if __name__ == "__main__":
    main()
