"""Benchmark the exact resumable rollout-pool path used by formal T5 updates."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import GameplayModelV1, configure_torch_threads  # noqa: E402
from pvz_seed_jobs import run_seed_jobs  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402


def _process_tree_rss_mb() -> float:
    pending = [os.getpid()]
    seen: set[int] = set()
    total_kib = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return total_kib / 1024.0


def _measure(workers: int, threads: int, device: str, episodes: int, max_actions: int,
             state_dict: dict[str, torch.Tensor],
             assignments: dict[int, dict[str, Any]], resource_dir: Path,
             shard_dir: Path | None = None) -> dict[str, Any]:
    job_ids = list(range(episodes))
    metadata = {
        "benchmark": "T5 formal rollout batch",
        "workers": workers,
        "threads": threads,
        "device": device,
        "max_actions": max_actions,
        "assignments": {
            str(job_id): {
                "task_id": value["task"]["task_id"],
                "task_seed": value["task_seed"],
                "action_seed": value["action_seed"],
            }
            for job_id, value in assignments.items()
        },
    }
    peak_rss_mb = [_process_tree_rss_mb()]
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_sampling.wait(0.25):
            peak_rss_mb[0] = max(peak_rss_mb[0], _process_tree_rss_mb())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    temporary_context = (tempfile.TemporaryDirectory(prefix="pvz-t5-rollout-batch-")
                         if shard_dir is None else nullcontext(str(shard_dir)))
    with temporary_context as temporary:
        output_path = Path(temporary)
        output_path.mkdir(parents=True, exist_ok=True)
        sampler.start()
        started = time.perf_counter()
        try:
            results = run_seed_jobs(
                job_ids,
                output_path,
                metadata,
                trainer._rollout_worker,
                workers=workers,
                initializer=trainer._init_worker,
                initargs=(str(resource_dir), state_dict, assignments, max_actions, threads, device),
                label=f"benchmark {device}:{workers}x{threads}",
            )
        finally:
            elapsed = time.perf_counter() - started
            stop_sampling.set()
            sampler.join()
            peak_rss_mb[0] = max(peak_rss_mb[0], _process_tree_rss_mb())
        names = ("model", "environment", "critic_inputs", "tokenization")
        return {
            "device": device,
            "workers": workers,
            "torch_threads_per_worker": threads,
            "episodes": len(results),
            "elapsed_seconds": round(elapsed, 3),
            "episodes_per_hour": round(len(results) * 3600 / elapsed, 1),
            "mean_episode_seconds": round(sum(result["seconds"] for result in results) / len(results), 4),
            "mean_profile_seconds": {
                name: round(sum(result["profile_seconds"][name] for result in results) / len(results), 5)
                for name in names
            },
            "peak_process_tree_rss_mb": round(peak_rss_mb[0], 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-actions", type=int, default=4000)
    parser.add_argument("--configurations", default="cpu:12x1,cpu:18x1,cpu:20x1")
    parser.add_argument("--keep-shards", type=Path,
                        help="retain collected episode shards here, grouped by worker configuration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_actions < 1:
        parser.error("episodes and max-actions must be positive")
    configurations = []
    for text in args.configurations.split(","):
        device, shape = text.split(":", 1)
        workers, threads = (int(value) for value in shape.lower().split("x", 1))
        if device not in ("cpu", "cuda") or workers < 1 or threads < 1:
            parser.error(f"invalid configuration {text!r}")
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA configuration requested but unavailable")
        configurations.append((device, workers, threads))

    train = json.loads(trainer.TRAIN_PATH.read_text(encoding="utf-8"))
    tasks = train["tasks"]
    rng = random.Random(503)
    assignments = {}
    for job_id in range(args.episodes):
        task = rng.choice(tasks)
        assignments[job_id] = {
            "task": task,
            "task_seed": rng.choice(task["seeds"]),
            "action_seed": rng.randrange(1, 2**31),
        }
    configure_torch_threads(1)
    torch.manual_seed(0)
    model = GameplayModelV1().eval()
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    started_at = datetime.now(timezone.utc).isoformat()
    results = []
    for device, workers, threads in configurations:
        shard_dir = (None if args.keep_shards is None else
                     args.keep_shards / f"{device}_{workers}workers_{threads}threads")
        results.append(_measure(
            workers, threads, device, args.episodes, args.max_actions,
            state_dict, assignments, args.resource_dir.expanduser().resolve(), shard_dir,
        ))
    selected = max(results, key=lambda result: result["episodes_per_hour"])
    report = {
        "started_at": started_at,
        "episodes_per_batch": args.episodes,
        "max_actions": args.max_actions,
        "configurations": results,
        "selected_configuration": selected,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
