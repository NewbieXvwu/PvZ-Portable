"""Real-simulator regression: policy inputs must not depend on previous episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
from pvz_agent_model import observation_tokens, pack_tokens
from pvz_env import PvZEnv, TaskSpec
from pvz_seed_jobs import atomic_json
from train_pvz_ppo import _task_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]
    tasks = [task for task in tasks if task["wave_cap"] == 1 and task["zombie_count_multiplier"] == 1]
    records = []
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for task in tasks:
            for seed in task["seeds"]:
                env.reset(deck=[0, 1, 2, 3, 4, 5], task=TaskSpec(level=1, seed=60000, wave_cap=1))
                _, _, _, _, warm_info = env.step({"type": "wait", "ticks": 6000})
                observation, reset_info = env.reset(deck=task["deck"], task=_task_spec(task, seed))
                warmed_reset = pack_tokens(*observation_tokens(observation))
                observation, _, _, _, _ = env.step({"type": "wait", "ticks": 150})
                warmed = pack_tokens(*observation_tokens(observation))
                warmed_income = observation["sun_income_rate"]
                observation, clean_info = env.reset(deck=task["deck"], task=_task_spec(task, seed))
                clean_reset = pack_tokens(*observation_tokens(observation))
                observation, _, _, _, _ = env.step({"type": "wait", "ticks": 150})
                clean = pack_tokens(*observation_tokens(observation))
                equal = all(np.array_equal(warmed[key], clean[key]) and
                            np.array_equal(warmed_reset[key], clean_reset[key]) for key in warmed)
                zero_reset_events = all(value == 0 for value in reset_info["events"].values())
                records.append({"task_id": task["task_id"], "seed": seed,
                                "previous_sun_produced": warm_info["events"]["sun_produced"],
                                "reset_events": reset_info["events"],
                                "warmed_income_rate": warmed_income,
                                "clean_income_rate": observation["sun_income_rate"],
                                "identical_policy_inputs": equal, "zero_reset_events": zero_reset_events})
    passed = all(row["identical_policy_inputs"] and row["zero_reset_events"] and
                 row["previous_sun_produced"] > 0 for row in records)
    atomic_json(args.output, {"gate_result": "pass" if passed else "fail",
                              "sample_count": len(records), "terrain_count": len(tasks),
                              "scope": "all original cap1 seeds; warmed previous episode vs clean reset; first150ticks",
                              "records": records})
    print("pass" if passed else "fail", len(records), "target seeds", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
