"""Read scalar shard metadata to inspect the actual observed PPO reward signal.

No tensors, checkpoints or extra environment interactions are needed. Every
saved episode is retained in the report, including budget truncations. This is
a diagnostic of collected experience, not an estimate of optimal game values.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from train_pvz_ppo import add_advantages


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(a.mean()), "std": float(a.std()),
            "minimum": float(a.min()), "maximum": float(a.max())}


def inspect(run: Path) -> dict:
    started = time.monotonic()
    state = json.loads((run / "training_state.json").read_text())
    if state["status"] not in {"update_boundary_stop", "budget_complete"} or state["phase"] != "ready":
        raise ValueError("inspect a closed complete boundary, not a live writer")
    config = json.loads((run / "experiment_config.json").read_text())
    rows, actions, shovel_types = [], Counter(), Counter()
    for path in sorted(run.glob("runs/run_*/.seed_jobs/update_*/*/seed_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            # Leave all numeric feature arrays on disk; scalar rewards/actions
            # already live in the small metadata entry.
            payload = json.loads(archive["__manifest__"].tobytes())["value"]
        episode = payload["result"]
        steps = episode["transitions"]
        total, shaped, outcome = (math.fsum(step[key] for step in steps)
                                  for key in ("reward", "shaping_reward", "terminal_outcome"))
        discounted = float(episode["bootstrap_value"])
        mc_errors = []
        for step in reversed(steps):
            discounted = step["reward"] + step["discount"] * discounted
            mc_errors.append((discounted - step["value"]) ** 2)
        add_advantages([episode], config["ppo"]["gae_lambda"], config["reward"]["gamma"])
        action_counts, positive_shovels = Counter(), 0
        last_plant_packets: dict[tuple[int, int], int] = {}
        for step in steps:
            action = step["action"]
            actions[action["type"]] += 1
            action_counts[action["type"]] += 1
            cell = (action.get("row"), action.get("col"))
            if action["type"] == "plant":
                last_plant_packets[cell] = action["packet"]
            elif action["type"] == "shovel":
                shovel_types[str(last_plant_packets.pop(cell, "not_observed"))] += 1
                positive_shovels += step["shaping_reward"] > 1e-12
        rows.append({"shard": str(path.relative_to(run)), "update": payload["metadata"]["update"],
            "episode_id": episode["seed"], "environment_seed": episode["task_seed"],
            "task_id": episode["task_id"], "won": episode["won"],
            "terminated": episode["terminated"], "truncated": episode["truncated"],
            "wave": episode["wave"], "tick": episode["tick"], "decisions": len(steps),
            "action_duration_ticks": sum(step["action_duration_ticks"] for step in steps),
            "sum_reward": total, "sum_shaping": shaped, "terminal_outcome": outcome,
            "discounted_bootstrapped_return": discounted,
            "first_potential": steps[0]["potential"], "first_predicted_value": steps[0]["value"],
            "first_gae_value_target": steps[0]["return"],
            "mc_value_mse_before_update": math.fsum(mc_errors) / len(steps),
            "gae_value_mse_before_update": math.fsum(step["advantage"] ** 2 for step in steps) / len(steps),
            "all_discounts_one": all(step["discount"] == 1 for step in steps),
            "telescoping_error_if_natural_terminal_gamma1":
                abs(total - (outcome - config["reward"]["shaping_weight"] * steps[0]["potential"]))
                if episode["terminated"] and all(step["discount"] == 1 for step in steps) else None,
            "action_counts": dict(action_counts), "positive_shaping_shovels": positive_shovels})
    normal = [r for r in rows if r["terminated"] and not r["truncated"]]
    complete_inventory = (len(rows) == state["counters"]["episodes"]
        and len({r["episode_id"] for r in rows}) == len(rows)
        and sum(r["decisions"] for r in rows) == state["counters"]["decisions"]
        and sum(r["action_duration_ticks"] for r in rows) == state["counters"]["ticks"])
    errors = [r["telescoping_error_if_natural_terminal_gamma1"] for r in normal
              if r["telescoping_error_if_natural_terminal_gamma1"] is not None]
    return {"schema_version": 1, "status": "complete" if complete_inventory else "inventory_mismatch", "experiment_id": state["experiment_id"],
            "source_run": str(run), "reward_config": config["reward"],
            "snapshot": {key: state[key] for key in ("updates", "counters", "checkpoint")},
            "saved_inventory_matches_committed_counters": complete_inventory,
            "normal_terminal_count": len(normal), "normal_terminal_wins": sum(r["won"] for r in normal),
            "truncated_count": sum(r["truncated"] for r in rows),
            "natural_terminal_reward_sums": describe([r["sum_reward"] for r in normal]),
            "natural_terminal_first_potentials": describe([r["first_potential"] for r in normal]),
            "natural_terminal_episode_mc_value_mse": describe([r["mc_value_mse_before_update"] for r in normal]),
            "natural_terminal_episode_gae_value_mse": describe([r["gae_value_mse_before_update"] for r in normal]),
            "telescoping_error_gamma1": describe(errors), "action_counts": dict(actions),
            "shovel_last_observed_packet_counts": dict(shovel_types),
            "positive_shaping_shovels": sum(r["positive_shaping_shovels"] for r in rows),
            "interpretation": "Observed episode returns only; equal all-loss returns do not prove the environment unsolvable or RL impossible. Truncated rows retained separately; bootstraps are not natural wins. Packet tracking describes previously observed planted cards, not a native entity-identity audit.",
            "episodes": rows, "seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"},
                     ensure_ascii=False, indent=2))
    if not report["saved_inventory_matches_committed_counters"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
