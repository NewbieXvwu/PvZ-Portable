"""Audit the added public zombie ID against the unchanged native simulator."""
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
from pvz_agent_model import configure_torch_threads, observation_tokens
from pvz_env import PvZEnv, TaskSpec
from pvz_observation_features import require_public_fields
from pvz_seed_jobs import atomic_json
from research_aid_feasibility import choose


def without_added_ids(value):
    if isinstance(value, dict):
        return {key: ([{k: without_added_ids(v) for k, v in z.items() if k != "id"} for z in item]
                      if key == "zombies" else without_added_ids(item)) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [without_added_ids(item) for item in value]
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--reference-executable", type=Path, required=True)
    parser.add_argument("--candidate-executable", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds-per-task", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seeds_per_task <= 64 or args.output.exists():
        raise ValueError("require seeds 1..64 and a fresh evidence path")
    configure_torch_threads(1)
    manifest = json.loads(args.manifest.read_text())
    started, records, counts = time.monotonic(), [], Counter()

    def check_observation(obs: dict) -> None:
        require_public_fields(obs, 7)
        counts["observations"] += 1
        ids = {z["id"] for z in obs["zombies"]}
        counts["zombie_records"] += len(ids)
        counts["on_board_records"] += sum(z["on_board"] for z in obs["zombies"])
        for kind in ("plants", "projectiles"):
            for entity in obs[kind]:
                target = entity["target_zombie_id"]
                if target not in (0, -1):
                    counts[kind + "_resolved_targets" if target in ids else kind + "_unresolved_targets"] += 1
        if counts["observations"] % 97 == 1:
            tokens, metadata = observation_tokens(obs, 7)
            counts["encoded_observations"] += 1
            counts["max_tokens"] = max(counts["max_tokens"], len(tokens["kinds"]))
            for action in obs["legal_actions"]["plants"]:
                if action["row"] * 9 + action["col"] not in metadata["cell_tokens"]:
                    raise AssertionError("new input dropped a legal action cell")

    def compare(expected, actual, context: dict) -> None:
        counts["response_comparisons"] += 1
        if without_added_ids(actual) != without_added_ids(expected):
            atomic_json(args.output.with_suffix(".failure.json"), {
                "context": context, "expected": expected, "actual": actual})
            raise AssertionError("added public identity changed an existing native response")

    with PvZEnv(args.resource_dir, executable=args.reference_executable.resolve()) as reference, \
         PvZEnv(args.resource_dir, executable=args.candidate_executable.resolve()) as candidate:
        for task in manifest["tasks"]:
            for strategy in manifest["strategies"]:
                for seed in task["seeds"][:args.seeds_per_task]:
                    context = {"task_id": task["task_id"], "strategy": strategy, "seed": seed}
                    spec = TaskSpec(level=task["level"], seed=seed, playthrough=task["playthrough"],
                                    zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
                                    preplanted=tuple(tuple(p) for p in task["preplanted"]))
                    expected = reference.reset(deck=task["deck"], task=spec)
                    actual = candidate.reset(deck=task["deck"], task=spec)
                    compare(expected, actual, {**context, "operation": "RESET"})
                    obs, _ = actual
                    check_observation(obs)
                    snapshotted = False
                    for index in range(manifest["max_actions"]):
                        if not snapshotted and obs["tick"] >= 3000 and not obs["terminal"]:
                            parent = candidate.snapshot()
                            action = {"type": "wait", "ticks": 120}
                            first = candidate.step(action)
                            candidate.restore(parent)
                            replay = candidate.step(action)
                            if first != replay:
                                atomic_json(args.output.with_suffix(".failure.json"), {
                                    "context": {**context, "operation": "snapshot replay"},
                                    "first": first, "replay": replay})
                                raise AssertionError("public IDs or responses changed across snapshot replay")
                            check_observation(first[0])
                            counts["snapshot_replays"] += 1
                            candidate.restore(parent)
                            branch = candidate.branch_snapshot(parent, [action])[0]
                            if branch["observation"] != first[0] or branch["events"] != first[4]["events"]:
                                atomic_json(args.output.with_suffix(".failure.json"), {
                                    "context": {**context, "operation": "snapshot branch"},
                                    "first": first, "branch": branch})
                                raise AssertionError("public IDs or responses differ in batched snapshot branch")
                            counts["snapshot_branches"] += 1
                            candidate.restore(parent)
                            if branch.get("snapshot_id", 0):
                                candidate.release_snapshot(branch["snapshot_id"])
                            candidate.release_snapshot(parent)
                            snapshotted = True
                        action = {"type": "wait", "ticks": 300} if strategy == "wait" else choose(obs)
                        expected, actual = reference.step(action), candidate.step(action)
                        compare(expected, actual, {**context, "operation": "STEP", "decision": index, "action": action})
                        obs = actual[0]
                        check_observation(obs)
                        if obs["terminal"]:
                            break
                    if not obs["terminal"]:
                        raise AssertionError("native audit reached action limit without a true terminal")
                    records.append({**context, "actions": index + 1, "result": obs["result"], "tick": obs["tick"]})
            atomic_json(args.output.with_suffix(".progress.json"), {
                "task_id": task["task_id"], "episodes": len(records), "counts": dict(counts),
                "seconds": time.monotonic() - started})
            print(task["task_id"], "episodes", len(records), flush=True)
    atomic_json(args.output, {"schema_version": 1, "gate_result": "pass", "scope":
                             "native public-ID addition only; old responses exact after removing only zombie.id; CPU audit, no learning pass",
                             "manifest_sha256": digest(args.manifest), "seeds_per_task": args.seeds_per_task,
                             "reference_sha256": digest(args.reference_executable),
                             "candidate_sha256": digest(args.candidate_executable),
                             "episodes": len(records), "counts": dict(counts), "records": records,
                             "seconds": time.monotonic() - started})
    print("pass", len(records), "episodes", dict(counts), flush=True)


if __name__ == "__main__":
    main()
