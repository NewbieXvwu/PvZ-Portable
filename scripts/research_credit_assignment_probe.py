"""Compare saved GAE credit with full discounted episode returns, without rollout.

Read only closed scalar shard manifests. Include budget truncations and their
recorded bootstraps. Raw advantage signs alone are insufficient: PPO normalizes
all transitions in an update, so report that normalization as well. This is a
diagnostic of observed experience, not a causal test of another estimator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from pvz_value import DISCOUNT_REFERENCE_TICKS
from train_pvz_ppo import add_advantages
from research_terminal_signal_probe import describe


def summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "first_raw_positive": {k: sum(r[f"{k}_first_advantage"] > 0 for r in rows)
                               for k in ("gae", "mc")},
        "first_normalized_positive": {k: sum(r[f"{k}_normalized_first"] > 0 for r in rows)
                                      for k in ("gae", "mc")},
        "first_normalized_negative_to_positive": sum(
            r["gae_normalized_first"] <= 0 < r["mc_normalized_first"] for r in rows),
        **{k: describe([r[k] for r in rows]) for k in (
            "gae_first_advantage", "mc_first_advantage",
            "gae_normalized_first", "mc_normalized_first")},
        **{k: describe([r[k] for r in rows if r[k] is not None]) for k in (
            "terminal_reward_prefix_ticks", "terminal_gae_coefficient",
            "terminal_mc_coefficient", "terminal_lambda_attenuation")},
    }


def inspect(run: Path) -> dict:
    started = time.monotonic()
    state = json.loads((run / "training_state.json").read_text())
    if state["phase"] != "ready" or state["status"] not in {"update_boundary_stop", "budget_complete"}:
        raise ValueError("inspect a closed complete boundary, not a live writer")
    config = json.loads((run / "experiment_config.json").read_text())
    gamma, lam = config["reward"]["gamma"], config["ppo"]["gae_lambda"]
    rows, batches = [], defaultdict(lambda: {"gae": [], "mc": []})
    max_return_error, max_discount_error = 0.0, 0.0
    for path in sorted(run.glob("runs/run_*/.seed_jobs/update_*/*/seed_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(archive["__manifest__"].tobytes())["value"]
        episode, update = payload["result"], payload["metadata"]["update"]
        steps = episode["transitions"]
        returned, returns = float(episode["bootstrap_value"]), []
        for step in reversed(steps):
            returned = step["reward"] + step["discount"] * returned
            returns.append(returned)
        returns.reverse()
        add_advantages([episode], lam, gamma)
        gae = [s["advantage"] for s in steps]
        mc = [target - s["value"] for target, s in zip(returns, steps)]
        for name, values in (("gae", gae), ("mc", mc)):
            batches[update][name].extend(values)
        terminal_indices = [i for i, s in enumerate(steps) if s["terminal_outcome"]]
        natural = episode["terminated"] and not episode["truncated"]
        if natural and terminal_indices != [len(steps) - 1]:
            raise ValueError(f"unexpected natural terminal reward position: {path}")
        prefix_ticks = sum(s["action_duration_ticks"] for s in steps[:-1]) if natural else None
        ratio = prefix_ticks / DISCOUNT_REFERENCE_TICKS if natural else None
        rows.append({
            "shard": str(path.relative_to(run)), "update": update,
            "episode_id": episode["seed"], "environment_seed": episode["task_seed"],
            "task_id": episode["task_id"], "won": episode["won"],
            "terminated": episode["terminated"], "truncated": episode["truncated"],
            "decisions": len(steps),
            "ticks": sum(s["action_duration_ticks"] for s in steps),
            "bootstrap_value": episode["bootstrap_value"],
            "zero_tick_actions": sum(s["action_duration_ticks"] == 0 for s in steps),
            "gae_first_advantage": gae[0], "mc_first_advantage": mc[0],
            "terminal_reward_prefix_ticks": prefix_ticks,
            "terminal_gae_coefficient": (gamma * lam) ** ratio if natural else None,
            "terminal_mc_coefficient": gamma ** ratio if natural else None,
            "terminal_lambda_attenuation": lam ** ratio if natural else None,
        })
        add_advantages([episode], 1.0, gamma)
        max_return_error = max(max_return_error,
            max(abs(s["return"] - target) for s, target in zip(steps, returns)))
        max_discount_error = max(max_discount_error, max(
            abs(s["discount"] - gamma ** (s["action_duration_ticks"] / DISCOUNT_REFERENCE_TICKS))
            for s in steps))
    normalization = {}
    for update, estimators in batches.items():
        normalization[update] = {}
        for name, values in estimators.items():
            values = torch.tensor(values, dtype=torch.float32)
            normalization[update][name] = (
                values.mean(), values.std(unbiased=False).clamp_min(1e-6))
    for row in rows:
        for name in ("gae", "mc"):
            mean, std = normalization[row["update"]][name]
            first = torch.tensor(row[f"{name}_first_advantage"], dtype=torch.float32)
            row[f"{name}_normalized_first"] = float((first - mean) / std)
    inventory_matches = (len(rows) == state["counters"]["episodes"]
        and len({r["episode_id"] for r in rows}) == len(rows)
        and sum(r["decisions"] for r in rows) == state["counters"]["decisions"]
        and sum(r["ticks"] for r in rows) == state["counters"]["ticks"])
    natural = [r for r in rows if r["terminated"] and not r["truncated"]]
    valid = inventory_matches and max_return_error < 1e-10 and max_discount_error < 1e-12
    provenance = json.loads((run / "provenance.json").read_text())
    return {
        "schema_version": 1, "status": "complete" if valid else "mismatch_preserved",
        "source_run": str(run), "experiment_id": state["experiment_id"],
        "simulator_sha256": provenance["fingerprints"]["simulator"],
        "snapshot": {k: state[k] for k in ("updates", "counters", "checkpoint")},
        "observed_estimator": {"gamma": gamma, "gae_lambda": lam,
                               "reference_ticks": DISCOUNT_REFERENCE_TICKS},
        "saved_inventory_matches_committed_counters": inventory_matches,
        "lambda1_full_return_max_abs_error": max_return_error,
        "recorded_discount_max_abs_error": max_discount_error,
        "normal_terminal_count": len(natural),
        "normal_wins": summarize([r for r in natural if r["won"]]),
        "normal_losses": summarize([r for r in natural if not r["won"]]),
        "budget_truncations": summarize([r for r in rows if r["truncated"]]),
        "per_task": {task: {
            "wins": summarize([r for r in natural if r["task_id"] == task and r["won"]]),
            "losses": summarize([r for r in natural if r["task_id"] == task and not r["won"]])}
            for task in sorted({r["task_id"] for r in rows})},
        "seconds": time.monotonic() - started,
        "scope": "Saved closed experience only. No PPO, new rollout, or checkpoint changes. CPU float32 per-update normalization mirrors train_update; this is not a CUDA bitwise comparison. MC recomputation includes all saved bootstrap truncations.",
        "interpretation": "Terminal coefficients are the direct contribution of the last reward/TD residual to the first estimate, excluding the last action's duration. GAE also carries intervening TD signals: attenuation or negative early advantages do not prove a bug. Lambda1 removes intermediate critic bootstraps, but has higher sampling variance; its policy learning quality remains untested.",
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    report = inspect(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {k: v for k, v in report.items() if k != "episodes"}
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "seconds", "snapshot",
        "lambda1_full_return_max_abs_error", "normal_terminal_count")}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
