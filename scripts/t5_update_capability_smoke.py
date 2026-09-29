"""Compare one measured PPO update with its initialization on frozen held-out tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import GameplayModelV1, configure_torch_threads  # noqa: E402
from pvz_env import PvZEnv  # noqa: E402
import t4_capability_profile  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--updated-state", type=Path, required=True)
    parser.add_argument("--seeds-per-task", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds_per_task < 1:
        parser.error("seeds-per-task must be positive")

    _, heldout = trainer._task_family()
    gate_tasks, _ = trainer._heldout_tasks(heldout)
    configure_torch_threads(1)
    torch.manual_seed(0)
    initial = GameplayModelV1().eval()
    updated = GameplayModelV1().eval()
    updated.load_state_dict(torch.load(args.updated_state, map_location="cpu", weights_only=True))
    models = {"initial": initial, "updated": updated}
    rows = []
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for task in gate_tasks:
            for seed in task["seeds"][:args.seeds_per_task]:
                pair = {name: t4_capability_profile.run_episode(
                    env, task, seed, "checkpoint", model,
                ) for name, model in models.items()}
                rows.append({"task_id": task["task_id"], "seed": seed, **pair})

    summaries = {}
    for name in models:
        subset = [row[name] for row in rows]
        summaries[name] = {
            "sample_count": len(subset),
            "passes": sum(record["won"] for record in subset),
            "pass_rate": sum(record["won"] for record in subset) / len(subset),
            "terminal_wave_histogram": dict(Counter(str(record["terminal_wave"]) for record in subset)),
            "per_task_pass_rate": {
                task["task_id"]: sum(row[name]["won"] for row in rows if row["task_id"] == task["task_id"])
                / sum(row["task_id"] == task["task_id"] for row in rows)
                for task in gate_tasks if any(row["task_id"] == task["task_id"] for row in rows)
            },
        }
    report = {
        "evaluation": "paired deterministic smoke sample; not a full capability gate",
        "heldout_manifest": str(trainer.HELDOUT_PATH.relative_to(ROOT)),
        "seeds_per_task": args.seeds_per_task,
        "task_count": len(gate_tasks),
        "updated_state": str(args.updated_state),
        "summaries": summaries,
        "paired_records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
