"""Exercise real plant/projectile target IDs using isolated interface-only tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_agent_model import configure_torch_threads, observation_tokens, TOKEN_KINDS
from pvz_env import PvZEnv, TaskSpec, PlayerProfileContext
from pvz_seed_jobs import atomic_json
from research_observation_native_audit import without_added_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--reference-executable", type=Path, required=True)
    parser.add_argument("--candidate-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("keep existing evidence and choose a fresh path")
    configure_torch_threads(1)
    started, records = time.monotonic(), []
    with PvZEnv(args.resource_dir, executable=args.reference_executable.resolve()) as reference, \
         PvZEnv(args.resource_dir, executable=args.candidate_executable.resolve()) as candidate:
        for label in ("squash_plant", "cattail_projectile"):
            for seed in range(95000, 95004):
                if label == "squash_plant":
                    spec = TaskSpec(level=1, seed=seed, wave_cap=3,
                                    preplanted=tuple((17, row, 6) for row in range(5)))
                    deck = (0, 1, 2, 3, 4, 17)
                    kind, token_kind = "plants", TOKEN_KINDS["plant"]
                else:
                    spec = TaskSpec(level=21, seed=seed, wave_cap=3,
                                    profile=PlayerProfileContext(owned_upgrade_plants=(43,)),
                                    preplanted=((16, 2, 1), (43, 2, 1)))
                    deck = (0, 1, 2, 4, 16, 43)
                    kind, token_kind = "projectiles", TOKEN_KINDS["projectile"]
                expected, actual = reference.reset(deck=deck, task=spec), candidate.reset(deck=deck, task=spec)
                if without_added_ids(actual) != without_added_ids(expected):
                    raise AssertionError("target probe reset changed original native response")
                observations, found = 0, False
                while not actual[0]["terminal"] and actual[0]["tick"] < 30000:
                    action = {"type": "wait", "ticks": 30}
                    expected, actual = reference.step(action), candidate.step(action)
                    observations += 1
                    obs = actual[0]
                    if without_added_ids(actual) != without_added_ids(expected):
                        atomic_json(args.output.with_suffix(".failure.json"), {
                            "label": label, "seed": seed, "expected": expected, "actual": actual})
                        raise AssertionError("target probe step changed original native response")
                    ids = {z["id"] for z in obs["zombies"]}
                    entities = [e for e in obs[kind] if e["target_zombie_id"] in ids]
                    if not entities:
                        continue
                    tokens, _ = observation_tokens(obs, 7)
                    source_tokens = (tokens["kinds"] == token_kind).nonzero().flatten()
                    if sum(int(tokens["target_indices"][i]) >= 0 for i in source_tokens) != len(entities):
                        raise AssertionError("native resolved target disappeared during token encoding")
                    parent = candidate.snapshot()
                    first = candidate.step({"type": "wait", "ticks": 20})
                    candidate.restore(parent)
                    replay = candidate.step({"type": "wait", "ticks": 20})
                    if first != replay:
                        raise AssertionError("real target IDs changed across snapshot replay")
                    candidate.release_snapshot(parent)
                    records.append({"label": label, "seed": seed, "observations": observations,
                                    "tick": obs["tick"], "resolved_sources": len(entities),
                                    "first_target_observation": obs, "snapshot_replay_exact": True})
                    found = True
                    break
                if not found:
                    atomic_json(args.output.with_suffix(".failure.json"), {
                        "label": label, "seed": seed, "observations": observations, "last_response": actual})
                    raise AssertionError("fixed target fixture did not expose its intended native target")
                print(label, seed, "resolved", len(entities), "tick", obs["tick"], flush=True)
    atomic_json(args.output, {"schema_version": 1, "gate_result": "pass", "scope":
                             "eight interface-only fixtures outside frozen training/evaluation tasks; actual plant/projectile targets and snapshot replay; no learning",
                             "seed_range": [95000, 95003], "wait_ticks": 30, "maximum_ticks": 30000,
                             "reference_sha256": hashlib.sha256(args.reference_executable.read_bytes()).hexdigest(),
                             "candidate_sha256": hashlib.sha256(args.candidate_executable.read_bytes()).hexdigest(),
                             "records": records, "seconds": time.monotonic() - started})
    print("pass", len(records), "real target fixtures", flush=True)


if __name__ == "__main__":
    main()
