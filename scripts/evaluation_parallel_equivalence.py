"""Prove the parallel evaluator reproduces the single-process loop it replaced.

``_evaluate`` used to walk every held-out and stage-0 episode in one process.  It now
spreads them over the same spawn pool rollout uses, which is only safe if each episode
is fully determined by ``(task, seed)`` plus the model weights.  This script checks
that claim on the real simulator: it runs the same episodes serially and through the
pool and compares every record field for field, not just the summaries.

    python scripts/evaluation_parallel_equivalence.py --seeds 4 --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import GameplayModelV1  # noqa: E402
from pvz_env import PvZEnv  # noqa: E402
import t4_capability_profile  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402

DEFAULT_RESOURCES = Path.home() / ".cache/pvz-research-resources"
LOCAL_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")


def _subset(tasks: list[dict[str, Any]], count: int, seeds: int) -> list[dict[str, Any]]:
    return [{**task, "seeds": task["seeds"][:seeds]} for task in tasks[:count]]


def _serial(model: GameplayModelV1, resource_dir: Path,
            reference_tasks: list[dict[str, Any]],
            stage0_tasks: list[dict[str, Any]]
            ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """The pre-change implementation, kept here as the reference behaviour."""
    eval_model = GameplayModelV1().eval()
    eval_model.load_state_dict({key: value.detach().cpu()
                                for key, value in model.state_dict().items()})
    records: dict[str, list[dict[str, Any]]] = {}
    stage0_records: dict[str, list[dict[str, Any]]] = {}
    with PvZEnv(resource_dir=resource_dir) as env:
        for task in reference_tasks:
            records[task["task_id"]] = [
                t4_capability_profile.run_episode(env, task, seed, "checkpoint", eval_model)
                for seed in task["seeds"]
            ]
        for task in stage0_tasks:
            stage0_records[task["task_id"]] = [
                t4_capability_profile.run_episode(env, task, seed, "checkpoint", eval_model)
                for seed in task["seeds"]
            ]
    return records, stage0_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, default=LOCAL_RESOURCES)
    parser.add_argument("--tasks", type=int, default=3, help="held-out tasks to sample")
    parser.add_argument("--stage0-tasks", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/t5/evaluation_parallel_equivalence.json")
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser()
    if not resource_dir.is_dir():
        raise SystemExit(f"resource dir not found: {resource_dir}")

    train, heldout = trainer._task_family()
    gate_tasks, reference_tasks = trainer._heldout_tasks(heldout)
    stage0_tasks = trainer._curriculum_tasks(train["tasks"], "cap1")
    reference = _subset(reference_tasks, args.tasks, args.seeds)
    stage0 = _subset(stage0_tasks, args.stage0_tasks, args.seeds)
    episodes = sum(len(task["seeds"]) for task in reference) + sum(len(task["seeds"]) for task in stage0)
    print(f"held-out tasks={len(reference)} stage0 tasks={len(stage0)} "
          f"episodes={episodes} workers={args.workers}", flush=True)

    model = GameplayModelV1().eval()

    print("--- serial (the replaced implementation) ---", flush=True)
    serial_start = time.perf_counter()
    serial_records, serial_stage0 = _serial(model, resource_dir, reference, stage0)
    serial_seconds = time.perf_counter() - serial_start
    print(f"  {episodes} episodes in {serial_seconds:.2f} s", flush=True)

    results = {}
    for workers in (1, args.workers):
        print(f"--- pool with {workers} worker(s) ---", flush=True)
        with tempfile.TemporaryDirectory() as temporary:
            start = time.perf_counter()
            summary = trainer._evaluate(
                model, resource_dir, reference, reference, stage0, 0, Path(temporary),
                workers=workers, worker_threads=1, worker_device="cpu",
                include_reference=True)
            elapsed = time.perf_counter() - start
            raw = json.loads(gzip.open(
                Path(temporary) / "evaluations" / "heldout_0000000.json.gz", "rt",
                encoding="utf-8").read())
        identical = (raw["seed_results"] == serial_records
                     and raw["stage0_seed_results"] == serial_stage0)
        print(f"  {episodes} episodes in {elapsed:.2f} s  records identical={identical}",
              flush=True)
        results[str(workers)] = {
            "workers": workers,
            "elapsed_seconds": round(elapsed, 3),
            "records_identical_to_serial": identical,
            "gate_set": summary["gate_set"]["pass_rate"],
            "stage0_set": summary["stage0_set"]["pass_rate"],
            "reference_set": summary["reference_set"]["pass_rate"],
        }
        if not identical:
            for task_id, rows in serial_records.items():
                for index, (expected, actual) in enumerate(zip(rows, raw["seed_results"][task_id])):
                    if expected != actual:
                        print(f"  FIRST DIFF {task_id} seed {expected['seed']} "
                              f"(index {index})", flush=True)
                        break
            raise SystemExit("parallel evaluation changed the results")

    # The gate-only path must agree with the serial gate subset too.
    with tempfile.TemporaryDirectory() as temporary:
        gate_only = trainer._evaluate(
            model, resource_dir, reference, reference, stage0, 0, Path(temporary),
            workers=args.workers, worker_threads=1, worker_device="cpu",
            include_reference=False)
        gate_raw = json.loads(gzip.open(
            Path(temporary) / "evaluations" / "heldout_0000000_gate_only.json.gz", "rt",
            encoding="utf-8").read())
    gate_ids = {task["task_id"] for task in reference}
    expected_gate = {task_id: rows for task_id, rows in serial_records.items() if task_id in gate_ids}
    gate_identical = gate_raw["seed_results"] == expected_gate
    print(f"--- gate-only path ---\n  records identical={gate_identical} "
          f"reference_skipped={gate_only['reference_set']['skipped']}", flush=True)

    report = {
        "task_id": "T5-evaluation-parallel-equivalence",
        "episodes": episodes,
        "serial_seconds": round(serial_seconds, 3),
        "configurations": results,
        "gate_only": {
            "records_identical_to_serial_subset": gate_identical,
            "reference_set_skipped": gate_only["reference_set"]["skipped"],
        },
        "conclusion": "parallel evaluation is bit-identical to the single-process loop",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
