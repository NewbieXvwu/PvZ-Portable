"""Compare resumable JSON and dtype-preserving NPZ trajectory shards."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_agent_model import GameplayModelV1, legal_summary, observation_tokens, pack_tokens  # noqa: E402
from pvz_seed_jobs import atomic_json, atomic_numpy, read_numpy  # noqa: E402
from train_pvz_ppo import _jsonable  # noqa: E402


def deep_size(value: Any, seen: set[int] | None = None) -> int:
    seen = seen if seen is not None else set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(key, seen) + deep_size(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(deep_size(item, seen) for item in value)
    return size


def compact_episode(source: dict[str, Any]) -> dict[str, Any]:
    model = GameplayModelV1()
    result = {key: value for key, value in source.items() if key != "transitions"}
    transitions = []
    for old in source["transitions"]:
        observation = old["observation"]
        tensors, metadata = observation_tokens(observation)
        transitions.append({
            "decision_index": old["decision_index"],
            "tokens": pack_tokens(tensors, metadata),
            "wave": observation["wave"],
            "legal": legal_summary(observation["legal_actions"]),
            "previous_action": old["previous_action"],
            "elapsed_since_previous_observation": old["elapsed_since_previous_observation"],
            "events": old["events"],
            "critic_extra": model.privileged_extra(old["privileged_state"], observation["wave"]),
            "action": old["action"],
            "log_prob": old["log_prob"],
            "value": old["value"],
            "potential": old.get("potential", 0.0),
            "action_duration_ticks": old["action_duration_ticks"],
            "shaping_reward": old.get("shaping_reward", 0.0),
            "terminal_outcome": old.get("terminal_outcome", 0.0),
            "reward": old.get("reward", 0.0),
        })
    result["transitions"] = transitions
    return result


def restore_json_arrays(episode: dict[str, Any]) -> dict[str, Any]:
    for transition in episode["transitions"]:
        packed = transition["tokens"]
        for name, dtype in (("ids", np.int64), ("features", np.float32),
                            ("cell_index", np.int64), ("packet_ids", np.int64),
                            ("packet_index", np.int64)):
            packed[name] = np.asarray(packed[name], dtype=dtype)
        transition["legal"]["packets"] = tuple(transition["legal"]["packets"])
        transition["legal"]["plant_mask"] = tuple(transition["legal"]["plant_mask"])
    return episode


def time_one(function: Callable[[], Any], repetitions: int) -> tuple[float, Any]:
    samples = []
    value = None
    for _ in range(repetitions):
        started = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), value


def run_format(directory: Path, episode: dict[str, Any], fmt: str, repetitions: int) -> dict[str, Any]:
    path = directory / f"sample.{fmt}"
    loaded_json_bytes = 0

    def roundtrip() -> dict[str, Any]:
        nonlocal loaded_json_bytes
        if fmt == "json.gz":
            atomic_json(path, _jsonable(episode), compressed=True)
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                decoded = json.load(stream)
            loaded_json_bytes = deep_size(decoded)
            return restore_json_arrays(decoded)
        compressed = fmt == "npz.deflate"
        atomic_numpy(path, episode, compressed=compressed)
        return read_numpy(path)

    seconds, result = time_one(roundtrip, repetitions)
    return {
        "format": fmt,
        "roundtrip_median_seconds": seconds,
        "file_bytes": path.stat().st_size,
        "decoded_json_object_bytes_before_array_restore": loaded_json_bytes if fmt == "json.gz" else None,
        "restored_object_bytes": deep_size(result),
        "transitions": len(result["transitions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="legacy JSON.GZ rollout shard containing full observation and privileged_state")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    with gzip.open(args.source, "rt", encoding="utf-8") as stream:
        episode = compact_episode(json.load(stream))
    with tempfile.TemporaryDirectory(prefix="pvz-trajectory-storage-") as temporary:
        directory = Path(temporary)
        results = [run_format(directory, episode, fmt, args.repetitions)
                   for fmt in ("json.gz", "npz.deflate", "npz.store")]
    report = {
        "source": str(args.source),
        "source_episode_id": episode.get("seed"),
        "repetitions": args.repetitions,
        "transitions": len(episode["transitions"]),
        "in_memory_compact_bytes": deep_size(episode),
        "formats": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
