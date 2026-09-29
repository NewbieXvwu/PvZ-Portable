"""Benchmark recurrent PPO updates on collected, dtype-preserving rollout shards."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import GameplayModelV1, configure_torch_threads, resolve_device  # noqa: E402
from pvz_seed_jobs import read_numpy  # noqa: E402
from train_pvz_ppo import add_advantages, train_update  # noqa: E402


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(_deep_size(key, seen) + _deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        total += sum(_deep_size(item, seen) for item in value)
    elif isinstance(value, np.ndarray):
        total = max(total, value.nbytes)
    return total


def _configs(value: str) -> list[tuple[int, int, str]]:
    result = []
    for item in value.split(","):
        sequence, chunks, precision = item.strip().split(":", 2)
        result.append((int(sequence), int(chunks), precision))
    return result


def _memory_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _optimizer_steps(episodes: list[dict[str, Any]], sequence_length: int,
                     minibatch_chunks: int, ppo_epochs: int) -> int:
    layer_counts: list[int] = []
    for episode in episodes:
        chunks = (len(episode["transitions"]) + sequence_length - 1) // sequence_length
        while len(layer_counts) < chunks:
            layer_counts.append(0)
        for layer in range(chunks):
            layer_counts[layer] += 1
    return ppo_epochs * sum(
        (count + minibatch_chunks - 1) // minibatch_chunks for count in layer_counts)


def _run(data_dir: Path, count: int, device: torch.device, sequence_length: int,
         minibatch_chunks: int, precision: str, ppo_epochs: int,
         learning_rate: float, attention_backend: str,
         save_state: Path | None = None) -> dict[str, Any]:
    paths = sorted(data_dir.glob("seed_*.npz"))[:count]
    if len(paths) < count:
        raise ValueError(f"need {count} episode shards in {data_dir}, found {len(paths)}")
    episodes = []
    for path in paths:
        stored = read_numpy(path)
        episodes.append(stored["result"] if "result" in stored else stored)
    optimizer_steps = _optimizer_steps(episodes, sequence_length, minibatch_chunks, ppo_epochs)
    add_advantages(episodes, gae_lambda=0.95)
    torch.manual_seed(0)
    random.seed(0)
    model = GameplayModelV1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    before_rss = _memory_mb()
    amp_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in ("fp32", "tf32", "bf16", "fp16"):
        raise ValueError(f"unsupported precision: {precision}")
    autocast = (torch.autocast("cuda", dtype=amp_dtypes[precision])
                if precision in amp_dtypes and device.type == "cuda" else nullcontext())
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    started = time.perf_counter()
    try:
        with autocast:
            losses = train_update(
                model, episodes, optimizer, device, ppo_epochs, sequence_length,
                0.2, 0.5, 0.01, minibatch_chunks=minibatch_chunks,
                attention_backend=attention_backend,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if save_state is not None:
            save_state.parent.mkdir(parents=True, exist_ok=True)
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, save_state)
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "sequence_length": sequence_length,
            "minibatch_chunks": minibatch_chunks,
            "precision": precision,
            "status": "oom",
            "episodes": count,
        }
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    elapsed = time.perf_counter() - started
    result = {
        "sequence_length": sequence_length,
        "minibatch_chunks": minibatch_chunks,
        "precision": precision,
        "attention_backend": attention_backend,
        "status": "ok",
        "episodes": count,
        "transitions": sum(len(episode["transitions"]) for episode in episodes),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": round(elapsed, 3),
        "transitions_per_second": round(
            sum(len(episode["transitions"]) for episode in episodes) / elapsed, 1),
        "episode_shard_bytes": sum(path.stat().st_size for path in paths),
        "episode_object_bytes": sum(_deep_size(episode) for episode in episodes),
        "learning_rate": learning_rate,
        "rss_before_update_mb": None if before_rss is None else round(before_rss, 1),
        "rss_after_update_mb": None if _memory_mb() is None else round(_memory_mb(), 1),
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
        if device.type == "cuda" else None,
        "peak_cuda_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1)
        if device.type == "cuda" else None,
        "policy_loss": losses["policy_loss"],
        "value_loss": losses["value_loss"],
        "entropy": losses["entropy"],
        "gradient_norm": losses["gradient_norm"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--attention-backend", choices=("dense", "flex", "auto"), default="dense")
    parser.add_argument("--save-state", type=Path,
                        help="save the updated state dict; requires exactly one configuration")
    parser.add_argument("--configurations", default="64:1:fp32,64:16:fp32,16:16:fp32,16:32:fp32,16:16:bf16,16:16:fp16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.ppo_epochs < 1:
        parser.error("episodes and ppo-epochs must be positive")
    device = resolve_device(args.device)
    configure_torch_threads(1)
    configurations = _configs(args.configurations)
    if args.save_state and len(configurations) != 1:
        parser.error("--save-state requires exactly one configuration")
    warm_sequence, warm_chunks, warm_precision = configurations[0]
    warmup = _run(args.data_dir, min(args.episodes, 4), device, warm_sequence,
                  warm_chunks, warm_precision, 1, args.learning_rate, args.attention_backend)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = [
        _run(args.data_dir, args.episodes, device, sequence, chunks, precision,
             args.ppo_epochs, args.learning_rate, args.attention_backend,
             args.save_state)
        for sequence, chunks, precision in configurations
    ]
    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "episodes": args.episodes,
        "ppo_epochs": args.ppo_epochs,
        "learning_rate": args.learning_rate,
        "attention_backend": args.attention_backend,
        "warmup": {key: warmup.get(key) for key in (
            "sequence_length", "minibatch_chunks", "precision", "elapsed_seconds", "status")},
        "data_dir": str(args.data_dir.resolve()),
        "configurations": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
