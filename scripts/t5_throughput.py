"""Measure full T5 PPO rollout throughput with spawn-based environment workers."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import GameplayModelV1, configure_torch_threads  # noqa: E402
from pvz_common import sha256_file  # noqa: E402
from pvz_env import PvZEnv  # noqa: E402
from pvz_seed_jobs import atomic_json  # noqa: E402
from train_pvz_ppo import collect_task_episode  # noqa: E402
from check_task_manifests import check_manifests  # noqa: E402
import t4_capability_profile  # noqa: E402

TRAIN_PATH = ROOT / "artifacts/task_family/train.json"
HELDOUT_PATH = ROOT / "artifacts/task_family/heldout.json"
T4_GATE_PATH = ROOT / "gates/T4.json"
DEFAULT_RESOURCE_DIR = Path.home() / "Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
_MODEL: GameplayModelV1 | None = None
_ENV: PvZEnv | None = None
_TASKS: list[dict[str, Any]] = []
_SHARD_DIR: Path | None = None
_MAX_ACTIONS = 4000


def _init_worker(resource_dir: str, state_dict: dict[str, torch.Tensor],
                 tasks: list[dict[str, Any]], shard_dir: str, max_actions: int) -> None:
    global _MODEL, _ENV, _TASKS, _SHARD_DIR, _MAX_ACTIONS
    configure_torch_threads(1)
    _MODEL = GameplayModelV1().eval()
    _MODEL.load_state_dict(state_dict)
    _ENV = PvZEnv(resource_dir=resource_dir)
    _TASKS = tasks
    _SHARD_DIR = Path(shard_dir)
    _MAX_ACTIONS = max_actions
    Finalize(None, _close_worker, exitpriority=10)


def _close_worker() -> None:
    if _ENV is not None:
        _ENV.close()


def _rollout(job: tuple[int, int, int, int]) -> dict[str, Any]:
    job_id, task_index, task_seed, action_seed = job
    if _MODEL is None or _ENV is None or _SHARD_DIR is None:
        raise RuntimeError("rollout worker was not initialized")
    torch.manual_seed(action_seed)
    episode = collect_task_episode(
        _MODEL, _ENV, _TASKS[task_index], task_seed, job_id, _MAX_ACTIONS,
    )
    persistence_started = time.perf_counter()
    atomic_json(_SHARD_DIR / f"seed_{job_id}.json.gz", episode, compressed=True)
    return {
        "task_id": episode["task_id"],
        "won": episode["won"],
        "seconds": episode["seconds"],
        **episode["profile_seconds"],
        "persistence": time.perf_counter() - persistence_started,
    }


def _jobs(tasks: list[dict[str, Any]], start_id: int):
    rng = random.Random(503)
    job_id = start_id
    while True:
        task_index = rng.randrange(len(tasks))
        task = tasks[task_index]
        yield (
            job_id,
            task_index,
            rng.choice(task["seeds"]),
            rng.randrange(1, 2**31),
        )
        job_id += 1


def _measure(workers: int, seconds: float, tasks: list[dict[str, Any]], state_dict: dict[str, torch.Tensor],
             resource_dir: Path, output_dir: Path, max_actions: int) -> dict[str, Any]:
    shard_dir = output_dir / f"workers_{workers}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    job_stream = iter(_jobs(tasks, workers * 1_000_000))
    deadline = started + seconds
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(resource_dir), state_dict, tasks, str(shard_dir), max_actions),
    ) as pool:
        pending = {pool.submit(_rollout, next(job_stream)) for _ in range(workers)}
        while pending:
            remaining = deadline - time.perf_counter()
            done, pending = wait(pending, timeout=max(0.0, remaining), return_when=FIRST_COMPLETED)
            if not done:
                break
            for future in done:
                completed.append(future.result())
                if time.perf_counter() < deadline:
                    pending.add(pool.submit(_rollout, next(job_stream)))
        elapsed_before_shutdown = time.perf_counter() - started
        completed.extend(future.result() for future in pending)
    elapsed = time.perf_counter() - started
    count = len(completed)
    means = {
        name: sum(item[name] for item in completed) / count if count else None
        for name in ("model", "environment", "privileged_state", "persistence")
    }
    total_rate = count * 3600.0 / elapsed if elapsed else 0.0
    return {
        "workers": workers,
        "torch_threads_per_worker": 1,
        "target_seconds": seconds,
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_before_shutdown_seconds": round(elapsed_before_shutdown, 3),
        "completed_episodes": count,
        "episodes_per_hour": round(total_rate, 3),
        "episodes_per_hour_per_worker": round(total_rate / workers, 3),
        "mean_episode_seconds": round(sum(item["seconds"] for item in completed) / count, 6) if count else None,
        "mean_profile_seconds": {key: None if value is None else round(value, 6)
                                  for key, value in means.items()},
        "wins": sum(item["won"] for item in completed),
        "completed_task_counts": {
            task["task_id"]: sum(item["task_id"] == task["task_id"] for item in completed)
            for task in tasks
        },
        "shards_path": str(shard_dir.resolve().relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, default=Path(os.environ.get("PVZ_RESOURCE_DIR", DEFAULT_RESOURCE_DIR)))
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--max-actions", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/t5/throughput.json")
    args = parser.parse_args()
    if args.minutes <= 0 or args.max_actions < 1:
        parser.error("minutes and max-actions must be positive")

    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    check_manifests(train, heldout)
    t4_gate = json.loads(T4_GATE_PATH.read_text(encoding="utf-8"))
    if sha256_file(HELDOUT_PATH) != t4_gate["metrics"]["manifests"]["heldout_sha256"]:
        raise RuntimeError("held-out manifest no longer matches the frozen T4 evidence")

    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_hash = t4_capability_profile._state_sha256(state_dict)
    expected_hash = t4_gate["metrics"]["checkpoint"]["state_sha256"]
    if state_hash != expected_hash:
        raise RuntimeError(f"random initialization mismatch: expected {expected_hash}, got {state_hash}")

    tasks = train["tasks"]
    workers = (8, 9, 10, 1)
    seconds_per_configuration = args.minutes * 60.0 / len(workers)
    overall_started = datetime.now(timezone.utc)
    run_id = overall_started.strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output.parent / f"throughput_shards_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for count in workers:
        results.append(_measure(
            count, seconds_per_configuration, tasks, state_dict,
            args.resource_dir.expanduser().resolve(), output_dir, args.max_actions,
        ))
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    single_core = next(item for item in results if item["workers"] == 1)
    parallel = [item for item in results if item["workers"] >= 8]
    result = {
        "schema_version": 1,
        "task_id": "T5-throughput",
        "started_at": overall_started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes_requested": args.minutes,
        "resource_dir": str(args.resource_dir.expanduser().resolve()),
        "manifests": {
            "train_sha256": sha256_file(TRAIN_PATH),
            "heldout_sha256": sha256_file(HELDOUT_PATH),
        },
        "policy": {
            "kind": "T4_random_initialized_checkpoint",
            "seed": 0,
            "state_sha256": state_hash,
            "training_episodes": 0,
        },
        "configurations": results,
        "single_core_episodes_per_hour": single_core["episodes_per_hour"],
        "single_core_threshold_met": single_core["episodes_per_hour"] >= 5000,
        "selected_parallel_workers": max(parallel, key=lambda item: item["episodes_per_hour"])["workers"],
        "protected_assets_unmodified": True,
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "single_core_episodes_per_hour": result["single_core_episodes_per_hour"],
        "single_core_threshold_met": result["single_core_threshold_met"],
        "selected_parallel_workers": result["selected_parallel_workers"],
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
