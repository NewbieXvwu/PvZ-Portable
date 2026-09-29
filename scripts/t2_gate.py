"""Run the T2 wave_cap and preplanted integration gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from scripted_baseline import run  # noqa: E402

NO_SUNFLOWER_DECK = (0, 2, 3, 4, 5, 6)
PREPLANTED = ((1, 0, 0), (1, 1, 0))


def result_fields(result: dict) -> dict:
    return {key: result[key] for key in ("won", "terminal", "wave", "wave_count", "tick")}


def wave_table(env: PvZEnv, seed: int, wave_cap: int | None) -> list[list[int]]:
    task = TaskSpec(level=7, seed=seed, wave_cap=wave_cap)
    env.reset(deck=(0, 1, 2, 3, 4, 5), task=task)
    response = env._command("PRIV")
    if not response.get("ok") or response["observation"] is None:
        raise AssertionError("could not read generated zombie wave table")
    return response["observation"]["hidden"]["zombies_in_wave"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    args = parser.parse_args()

    results = {}
    with PvZEnv(args.resource_dir) as env:
        full_waves = wave_table(env, 30000, None)
        capped_waves = wave_table(env, 30000, 3)
        assert len(capped_waves) == 3 and capped_waves == full_waves[:3]
        results["wave_composition_consistency"] = {
            "seed": 30000,
            "full_wave_count": len(full_waves),
            "wave_cap": 3,
            "capped_waves": capped_waves,
            "matches_full_wave_prefix": True,
        }

        baseline = run(env, 30000, 7)
        assert baseline["won"] and baseline["wave_count"] == 30
        assert baseline["tick"] == 65401
        results["default_level_7_seed_30000"] = result_fields(baseline)

        capped_50 = run(env, 30000, 7, task=TaskSpec(level=7, seed=30000, wave_cap=50))
        assert capped_50["won"] and capped_50["wave_count"] == 30
        assert capped_50["tick"] == baseline["tick"]
        results["wave_cap_50_does_not_expand"] = result_fields(capped_50)

        short_runs = []
        for seed in (30000, 30001, 30002):
            result = run(env, seed, 7, task=TaskSpec(level=7, seed=seed, wave_cap=3))
            assert result["won"] and result["terminal"] and result["wave_count"] == 3
            assert result["tick"] < 30000
            short_runs.append({"seed": seed, **result_fields(result)})
        results["wave_cap_3"] = short_runs

        one_wave = run(env, 30000, 7, task=TaskSpec(level=7, seed=30000, wave_cap=1))
        assert one_wave["terminal"] and one_wave["wave_count"] == 1
        results["wave_cap_1"] = result_fields(one_wave)

        task = TaskSpec(level=7, seed=30000, preplanted=PREPLANTED)
        baseline_observation, _ = env.reset(deck=NO_SUNFLOWER_DECK, task=TaskSpec(level=7, seed=30000))
        planted_observation, _ = env.reset(deck=NO_SUNFLOWER_DECK, task=task)
        positions = [(plant["type"], plant["row"], plant["col"]) for plant in planted_observation["plants"]]
        expected_positions = [(1, 0, 0), (1, 1, 0)]
        assert positions == expected_positions
        assert planted_observation["sun"] == baseline_observation["sun"]
        assert planted_observation["packets"] == baseline_observation["packets"]
        results["preplanted_without_sunflower_card"] = {
            "plants": positions,
            "sun_before": baseline_observation["sun"],
            "sun_after_reset": planted_observation["sun"],
            "packet_types_unchanged": True,
        }

        planted_run = run(env, 30000, 7, task=task)
        assert planted_run["won"] == baseline["won"]
        assert [(plant["type"], plant["row"], plant["col"]) for plant in planted_run["initial_plants"]] == expected_positions
        results["preplanted_game"] = result_fields(planted_run)

        try:
            env.reset(task=TaskSpec(level=21, seed=30000, preplanted=((0, 2, 0),)))
        except ValueError as error:
            results["pool_peashooter_without_lilypad"] = {"reset_rejected": True, "error": str(error)}
        else:
            raise AssertionError("pool peashooter without a lily pad unexpectedly reset")

    print(json.dumps({"task_id": "T2", "gate_result": "pass", "results": results}, indent=2))


if __name__ == "__main__":
    main()
