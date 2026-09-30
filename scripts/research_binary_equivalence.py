"""Compare two simulator binaries on every decision of frozen feasibility episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv, TaskSpec
from pvz_seed_jobs import atomic_json
from research_aid_feasibility import choose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--reference-executable", type=Path, required=True)
    parser.add_argument("--candidate-executable", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds-per-task", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seeds_per_task <= 64:
        raise ValueError("seed count must be 1..64")
    config = json.loads(args.manifest.read_text())
    started, records, comparisons = time.monotonic(), [], 0
    with PvZEnv(resource_dir=args.resource_dir, executable=args.reference_executable.resolve()) as reference, \
         PvZEnv(resource_dir=args.resource_dir, executable=args.candidate_executable.resolve()) as candidate:
        for task in config["tasks"]:
            for strategy in config["strategies"]:
                for seed in task["seeds"][:args.seeds_per_task]:
                    spec = TaskSpec(level=task["level"], seed=seed, playthrough=task["playthrough"],
                                    zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
                                    preplanted=tuple(tuple(p) for p in task["preplanted"]))
                    expected = reference.reset(deck=task["deck"], task=spec)
                    actual = candidate.reset(deck=task["deck"], task=spec)
                    if actual != expected:
                        raise AssertionError(f"RESET mismatch: {task['task_id']}/{seed}")
                    comparisons += 1
                    observation = actual[0]
                    for index in range(config["max_actions"]):
                        action = {"type": "wait", "ticks": 300} if strategy == "wait" else choose(observation)
                        expected, actual = reference.step(action), candidate.step(action)
                        if actual != expected:
                            atomic_json(args.output.with_suffix(".failure.json"), {
                                "task_id": task["task_id"], "seed": seed, "strategy": strategy,
                                "decision": index, "action": action, "expected": expected, "actual": actual})
                            raise AssertionError(f"STEP mismatch: {task['task_id']}/{seed}/{index}")
                        comparisons += 1
                        observation = actual[0]
                        if observation["terminal"]:
                            break
                    records.append({"task_id": task["task_id"], "seed": seed, "strategy": strategy,
                                    "decisions": index + 1, "result": observation["result"],
                                    "tick": observation["tick"], "terminal": observation["terminal"]})
            atomic_json(args.output.with_suffix(".progress.json"), {
                "task_id": task["task_id"], "completed_episodes": len(records),
                "comparisons": comparisons, "seconds": time.monotonic() - started})
            print(task["task_id"], "matched episodes", len(records), flush=True)
    atomic_json(args.output, {"scope": "every public response, legal mask, event, reward, terminal and truncation tuple",
                             "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                             "reference_sha256": hashlib.sha256(args.reference_executable.read_bytes()).hexdigest(),
                             "candidate_sha256": hashlib.sha256(args.candidate_executable.read_bytes()).hexdigest(),
                             "episodes": len(records), "response_comparisons": comparisons,
                             "records": records, "seconds": time.monotonic() - started,
                             "gate_result": "pass"})


if __name__ == "__main__":
    main()
