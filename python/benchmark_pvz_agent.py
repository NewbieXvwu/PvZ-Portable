"""Evaluate teacher and GameplayModel checkpoints on a frozen seed set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from pvz_agent import GameplayModelV0, predict_action, teacher_action
from pvz_env import PlayerProfileContext, PvZEnv, TaskSpec


LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = ROOT / "artifacts" / "gameplay-v0" / "benchmark_seeds.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout)
        return {"git_sha": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}


def read_seed_set(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("level") != LEVEL or data.get("playthrough") != 1:
        raise ValueError("frozen seed file must specify Adventure I 1-7, playthrough 1")
    if "seeds" in data:
        seeds = [int(seed) for seed in data["seeds"]]
    else:
        first = int(data["first_seed"])
        seeds = list(range(first, first + int(data["count"])))
    if len(seeds) < 256 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("frozen evaluation seed file must contain at least 256 unique seeds")
    return seeds


def checkpoint_model(path: Path, device: torch.device) -> tuple[GameplayModelV0, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GameplayModelV0().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def run_episode(env: PvZEnv, seed: int, model: GameplayModelV0 | None,
                clear_hidden: bool) -> dict[str, Any]:
    task = TaskSpec(level=LEVEL, seed=seed, playthrough=1, profile=PlayerProfileContext())
    observation, _ = env.reset(deck=DECK, task=task)
    hidden = None
    previous_action = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    event_totals = Counter()
    actions = 0
    started = time.perf_counter()
    while not observation["terminal"] and actions < 2000:
        if model is None:
            action = teacher_action(observation)
        else:
            with torch.inference_mode():
                action, next_hidden, _ = predict_action(
                    model, observation, None if clear_hidden else hidden,
                    previous_action, delta_ticks, events,
                )
            hidden = next_hidden
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"illegal action on seed {seed}: {action}")
        for key, value in info["events"].items():
            if isinstance(value, (int, float)):
                event_totals[key] += value
        previous_action = action
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        events = info["events"]
        actions += 1
        if done:
            break
    if not observation["terminal"]:
        raise RuntimeError(f"benchmark episode exceeded 2000 decisions on seed {seed}")
    return {
        "seed": seed, "won": observation["result"] == 1, "result": observation["result"],
        "terminal": bool(observation["terminal"]), "wave": observation["wave"],
        "wave_count": observation["wave_count"], "tick": observation["tick"],
        "actions": actions, "plants_eaten": int(event_totals["plants_eaten"]),
        "mower_triggers": int(event_totals["mower_triggered"]),
        "seconds": round(time.perf_counter() - started, 4),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    wins = sum(record["won"] for record in records)
    rate = wins / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    margin = z * ((rate * (1 - rate) / count + z * z / (4 * count * count)) ** 0.5) / denominator
    losses = [record for record in records if not record["won"]]
    mean = lambda key: sum(record[key] for record in records) / count
    return {
        "count": count, "wins": wins, "win_rate": rate,
        "wilson_95": [max(0.0, center - margin), min(1.0, center + margin)],
        "failure_wave_distribution": dict(sorted(Counter(str(record["wave"]) for record in losses).items(),
                                                  key=lambda item: int(item[0]))),
        "mean_actions": mean("actions"), "mean_plants_eaten": mean("plants_eaten"),
        "plants_eaten_total": sum(record["plants_eaten"] for record in records),
        "mower_triggers_total": sum(record["mower_triggers"] for record in records),
        "mean_mower_triggers": mean("mower_triggers"),
        "wall_seconds_total": sum(record["seconds"] for record in records),
        "wall_seconds_mean": mean("seconds"),
    }


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, filename = value.partition("=")
    if not separator or not label or not filename:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    return label, Path(filename).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, default=[])
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--clear-hidden", action="store_true")
    parser.add_argument("--no-relation-bias", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if not args.teacher and not args.checkpoint:
        parser.error("select --teacher and/or --checkpoint LABEL=PATH")
    labels = (["teacher"] if args.teacher else []) + [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels):
        parser.error("policy labels must be unique")

    seeds = read_seed_set(args.seeds)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(1)
    policies: list[tuple[str, GameplayModelV0 | None, dict[str, Any]]] = []
    if args.teacher:
        policies.append(("teacher", None, {}))
    for label, path in args.checkpoint:
        model, checkpoint = checkpoint_model(path, device)
        if args.no_relation_bias:
            for layer in model.encoder:
                layer.attention.relation_bias_enabled = False
        policies.append((label, model, {"sha256": sha256_file(path),
                                        "provenance": checkpoint.get("provenance")}))

    all_records = {}
    resource_metadata = None
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for label, model, checkpoint_info in policies:
            records = []
            for index, seed in enumerate(seeds, start=1):
                record = run_episode(env, seed, model, args.clear_hidden)
                records.append(record)
                if index % 16 == 0 or index == len(seeds):
                    print(f"{label} {index}/{len(seeds)} wins={sum(row['won'] for row in records)}",
                          flush=True)
            if resource_metadata is None and env.episode:
                resource_metadata = {"resource_sha256": env.episode["resource_sha256"],
                                     "properties_partner_sha256": env.episode["properties_partner_sha256"]}
            all_records[label] = {"checkpoint": checkpoint_info, "metrics": summarize(records),
                                  "episodes": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        **git_metadata(), "level": LEVEL, "playthrough": 1, "deck": list(DECK),
        "profile": "Adventure I, six slots, no store items",
        "seed_file": str(args.seeds.resolve()), "seed_file_sha256": sha256_file(args.seeds),
        "seed_count": len(seeds), "seed_range": [min(seeds), max(seeds)],
        "device": str(device), "clear_hidden": args.clear_hidden,
        "relation_bias": not args.no_relation_bias,
        "command": {"argv": sys.argv, "resource": args.resource_dir},
        "resources": resource_metadata, "policies": all_records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
