"""Evaluate simulator search and GameplayModel checkpoints on frozen seed sets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from pvz_agent_model import GameplayModelV1, MODEL_ARCHITECTURE_VERSION, predict_action, resolve_device
from pvz_env import ENV_PROTOCOL_VERSION, PlayerProfileContext, PvZEnv, TaskSpec
from pvz_search import SearchTeacher
from pvz_search_value import load_search_value
from pvz_seed_sets import DEFAULT_TEST_SEEDS, read_seed_set
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS

LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)
ROOT = Path(__file__).resolve().parent.parent


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


def checkpoint_model(path: Path, device: torch.device) -> tuple[GameplayModelV1, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    provenance = checkpoint["provenance"]
    if (checkpoint["model_architecture_version"] != MODEL_ARCHITECTURE_VERSION
            or checkpoint.get("value_semantics") != VALUE_SEMANTICS
            or provenance.get("search_label_version") != SEARCH_LABEL_VERSION
            or provenance.get("protocol_version") != ENV_PROTOCOL_VERSION
            or provenance["observation_version"] != 2
            or provenance["task_version"] != 2):
        raise ValueError("checkpoint does not match the current model/search semantics")
    model = GameplayModelV1().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def run_episode(
    env: PvZEnv,
    seed: int,
    model: GameplayModelV1 | None,
    searcher: SearchTeacher | None,
    clear_hidden: bool,
    zombie_count_multiplier: float,
    replay_dir: Path,
    policy_label: str,
    level: int,
    deck: tuple[int, ...],
) -> dict[str, Any]:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)
    reset_started = time.perf_counter()
    observation, _ = env.reset(deck=deck, task=task)
    reset_seconds = time.perf_counter() - reset_started
    hidden = None
    previous_action = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    totals = Counter()
    actions = 0
    ticks_advanced = 0
    search_simulations = 0
    search_margins: list[float] = []
    search_entropies: list[float] = []
    search_candidate_counts: list[int] = []
    search_elapsed: list[int] = []
    started = time.perf_counter()
    while not observation["terminal"] and actions < 2000:
        if searcher is not None:
            advice = searcher.advice(observation)
            action = advice.action
            search_simulations += advice.simulation_count
            if advice.best_second_margin is not None:
                search_margins.append(advice.best_second_margin)
            if advice.search_policy:
                search_entropies.append(-sum(p * math.log(max(p, 1e-12)) for p in advice.search_policy))
                search_candidate_counts.append(len(advice.search_policy))
            search_elapsed.append(advice.search_elapsed_ticks)
        else:
            if model is None:
                raise RuntimeError("benchmark policy has neither model nor searcher")
            with torch.inference_mode():
                action, hidden, _ = predict_action(
                    model,
                    observation,
                    None if clear_hidden else hidden,
                    previous_action,
                    delta_ticks,
                    events,
                )
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"illegal action on seed {seed}: {action}")
        for key, value in info["events"].items():
            if isinstance(value, (int, float)):
                totals[key] += value
        previous_action = action
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        ticks_advanced += delta_ticks
        events = info["events"]
        actions += 1
        if done:
            break
    if not observation["terminal"]:
        raise RuntimeError(f"benchmark episode exceeded 2000 decisions on seed {seed}")
    safe_label = "".join(character if character.isalnum() or character in "-_" else "_" for character in policy_label)
    replay_id = f"{safe_label}_seed_{seed}.jsonl.gz"
    env.save_replay(replay_dir / replay_id)
    elapsed_seconds = max(time.perf_counter() - started, 1e-9)
    record = {
        "seed": seed,
        "replay_id": replay_id,
        "won": observation["result"] == 1,
        "result": observation["result"],
        "terminal": bool(observation["terminal"]),
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "tick": observation["tick"],
        "actions": actions,
        "ticks_advanced": ticks_advanced,
        "reset_seconds": round(reset_seconds, 4),
        "ticks_per_second": round(ticks_advanced / elapsed_seconds, 2),
        "actions_per_second": round(actions / elapsed_seconds, 2),
        "plants_eaten": int(totals["plants_eaten"]),
        "mower_triggers": int(totals["mower_triggered"]),
        "seconds": round(elapsed_seconds, 4),
    }
    if searcher is not None:
        record.update({
            "search_simulations": search_simulations,
            "search_mean_simulations_per_decision": search_simulations / max(actions, 1),
            "search_mean_margin": sum(search_margins) / max(len(search_margins), 1),
            "search_mean_entropy": sum(search_entropies) / max(len(search_entropies), 1),
            "search_mean_candidate_count": sum(search_candidate_counts) / max(len(search_candidate_counts), 1),
            "search_mean_elapsed_ticks": sum(search_elapsed) / max(len(search_elapsed), 1),
        })
    return record


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
    summary = {
        "count": count,
        "wins": wins,
        "win_rate": rate,
        "wilson_95": [max(0.0, center - margin), min(1.0, center + margin)],
        "failure_wave_distribution": dict(sorted(
            Counter(str(record["wave"]) for record in losses).items(), key=lambda item: int(item[0])
        )),
        "mean_actions": mean("actions"),
        "mean_plants_eaten": mean("plants_eaten"),
        "mean_ticks_per_second": mean("ticks_per_second"),
        "mean_actions_per_second": mean("actions_per_second"),
        "mean_reset_seconds": mean("reset_seconds"),
        "plants_eaten_total": sum(record["plants_eaten"] for record in records),
        "mower_triggers_total": sum(record["mower_triggers"] for record in records),
        "mean_mower_triggers": mean("mower_triggers"),
        "wall_seconds_total": sum(record["seconds"] for record in records),
        "wall_seconds_mean": mean("seconds"),
    }
    for key in (
        "search_mean_simulations_per_decision",
        "search_mean_margin",
        "search_mean_entropy",
        "search_mean_candidate_count",
        "search_mean_elapsed_ticks",
    ):
        if key in records[0]:
            summary[key] = mean(key)
    if "search_simulations" in records[0]:
        summary["search_simulations_total"] = sum(record["search_simulations"] for record in records)
    return summary


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, filename = value.partition("=")
    if not separator or not label or not filename:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    return label, Path(filename).expanduser().resolve()


def _checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--seeds", type=Path, default=DEFAULT_TEST_SEEDS)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=lambda value: tuple(map(int, value.split(","))), default=DECK)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, default=[])
    parser.add_argument("--search-value", type=Path)
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-horizon-ticks", type=int, default=900)
    parser.add_argument("--search-simulation-budget", type=int, default=256)
    parser.add_argument("--clear-hidden", action="store_true")
    parser.add_argument("--no-relation-bias", action="store_true")
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if not args.search_value and not args.checkpoint:
        parser.error("select --search-value and/or --checkpoint LABEL=PATH")
    if min(args.search_width, args.search_candidates, args.search_horizon_ticks,
           args.search_simulation_budget) < 1:
        parser.error("search parameters must be positive")
    if not 1.0 <= args.zombie_count_multiplier <= 10.0:
        parser.error("--zombie-count-multiplier must be from 1 to 10")
    labels = (["search_teacher"] if args.search_value else []) + [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        parser.error("policy labels must be unique")

    seeds = read_seed_set(args.seeds, args.level)
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    policies: list[tuple[str, GameplayModelV1 | None, Any, dict[str, Any]]] = []
    if args.search_value:
        value_model, value_checkpoint = load_search_value(args.search_value, device)
        policies.append((
            "search_teacher",
            None,
            value_model,
            {"sha256": sha256_file(args.search_value), "checkpoint": _checkpoint_metadata(value_checkpoint)},
        ))
    for label, path in args.checkpoint:
        model, checkpoint = checkpoint_model(path, device)
        if args.no_relation_bias:
            for layer in model.encoder:
                layer.attention.relation_bias_enabled = False
        policies.append((label, model, None, {
            "sha256": sha256_file(path),
            "provenance": checkpoint.get("provenance"),
        }))

    all_records: dict[str, Any] = {}
    resource_metadata = None
    replay_dir = args.output.parent / f"{args.output.stem}_replays"
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for label, model, value_model, metadata in policies:
            searcher = SearchTeacher(
                env,
                value_model=value_model,
                beam_width=args.search_width,
                candidate_limit=args.search_candidates,
                horizon_ticks=args.search_horizon_ticks,
                simulation_budget=args.search_simulation_budget,
            ) if value_model is not None else None
            records: list[dict[str, Any]] = []
            for index, seed in enumerate(seeds, start=1):
                record = run_episode(
                    env, seed, model, searcher, args.clear_hidden, args.zombie_count_multiplier,
                    replay_dir, label, args.level, args.deck,
                )
                records.append(record)
                if index % 16 == 0 or index == len(seeds):
                    print(f"{label} {index}/{len(seeds)} wins={sum(row['won'] for row in records)}", flush=True)
            if resource_metadata is None and env.episode:
                resource_metadata = {
                    "resource_sha256": env.episode["resource_sha256"],
                    "properties_partner_sha256": env.episode["properties_partner_sha256"],
                }
            all_records[label] = {"checkpoint": metadata, "metrics": summarize(records), "episodes": records}
        worker_rss_bytes = None
        if env._process is not None:
            try:
                rss_kib = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(env._process.pid)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                worker_rss_bytes = int(rss_kib) * 1024
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        **git_metadata(),
        "protocol_version": ENV_PROTOCOL_VERSION,
        "level": args.level,
        "playthrough": 2,
        "deck": args.deck,
        "zombie_count_multiplier": args.zombie_count_multiplier,
        "profile": "Adventure-II, six slots, no store items",
        "seed_file": str(args.seeds.resolve()),
        "seed_file_sha256": sha256_file(args.seeds),
        "seed_count": len(seeds),
        "seed_range": [min(seeds), max(seeds)],
        "device": str(device),
        "clear_hidden": args.clear_hidden,
        "replay_directory": str(replay_dir.resolve()),
        "relation_bias": not args.no_relation_bias,
        "command": {"argv": sys.argv, "resource": args.resource_dir},
        "resources": resource_metadata,
        "worker_rss_bytes": worker_rss_bytes,
        "policies": all_records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
