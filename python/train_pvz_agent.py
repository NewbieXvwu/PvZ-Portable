"""Collect simulator-search episodes, train GameplayModel-v1, and run model-only games."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from pvz_agent_model import (GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG,
                             predict_action, resolve_device)
from pvz_search import SearchTeacher
from pvz_imitation import train
from pvz_training_artifacts import provenance, save_checkpoint
from pvz_value import SEARCH_LABEL_VERSION, VALUE_GAMMA, VALUE_SEMANTICS
from pvz_env import PlayerProfileContext, PvZEnv, TaskSpec


LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)


def parse_seeds(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def contiguous_seeds(start: int, count: int) -> list[int]:
    return list(range(start, start + count))


def validate_seed_sets(train_seeds: list[int], dagger_seeds: list[int], eval_seeds: list[int]) -> None:
    train, dagger, evaluation = set(train_seeds), set(dagger_seeds), set(eval_seeds)
    if len(train) != len(train_seeds) or len(dagger) != len(dagger_seeds) or len(evaluation) != len(eval_seeds):
        raise ValueError("search, DAgger, and evaluation seed sets must each contain unique seeds")
    if train & dagger:
        raise ValueError("search and DAgger seed sets must be disjoint")
    if evaluation & (train | dagger):
        raise ValueError("evaluation seeds must be disjoint from search and DAgger seeds")


def collect_episode(env: PvZEnv, seed: int, replay_dir: Path, max_actions: int = 2000,
                    level: int = LEVEL, deck: tuple[int, ...] = DECK,
                    zombie_count_multiplier: float = 1.0, search_width: int = 3,
                    search_candidates: int = 8, search_horizon_ticks: int = 900,
                    search_simulation_budget: int = 64) -> dict[str, Any]:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)
    observation, _ = env.reset(deck=deck, task=task)
    steps: list[dict[str, Any]] = []
    delta_ticks = 0
    events: dict[str, Any] = {}
    searcher = SearchTeacher(env, beam_width=search_width, candidate_limit=search_candidates,
                            horizon_ticks=search_horizon_ticks, simulation_budget=search_simulation_budget)
    for decision_index in range(max_actions):
        advice = searcher.advice(observation, delta_ticks=delta_ticks, events=events)
        action = advice.action
        steps.append(_search_labels({"decision_index": decision_index, "observation": observation, "action": action,
                                      "delta_ticks": delta_ticks, "events": events}, advice))
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"search selected an illegal action on seed {seed}: {action}")
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        events = info["events"]
        if done:
            won = observation["result"] == 1
            for step in steps:
                step["episode_outcome"] = 1 if won else -1
            replay_id = f"search_seed_{seed}.jsonl.gz"
            env.save_replay(replay_dir / replay_id)
            return {"seed": seed, "replay_id": replay_id, "observation_version": 2, "task_version": 2,
                    "search_label_version": SEARCH_LABEL_VERSION, "steps": steps, "won": won,
                    "result": observation["result"], "tick": observation["tick"],
                    "wave": observation["wave"], "wave_count": observation["wave_count"]}
    raise RuntimeError(f"search episode exceeded {max_actions} decisions on seed {seed}")


def collect_dagger_episode(model: GameplayModelV1, env: PvZEnv, seed: int,
                           replay_dir: Path, max_actions: int = 2000, level: int = LEVEL,
                           deck: tuple[int, ...] = DECK, zombie_count_multiplier: float = 1.0,
                           search_width: int = 3, search_candidates: int = 8,
                           search_horizon_ticks: int = 900, search_simulation_budget: int = 64) -> dict[str, Any]:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)
    observation, _ = env.reset(deck=deck, task=task)
    steps: list[dict[str, Any]] = []
    hidden = None
    previous_action = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    model.eval()
    searcher = SearchTeacher(env, model=model, beam_width=search_width, candidate_limit=search_candidates,
                            horizon_ticks=search_horizon_ticks, simulation_budget=search_simulation_budget)
    for decision_index in range(max_actions):
        advice = searcher.advice(observation, hidden, previous_action, delta_ticks, events)
        label = advice.action
        steps.append(_search_labels({"decision_index": decision_index, "observation": observation, "action": label,
                                      "previous_action": previous_action, "delta_ticks": delta_ticks,
                                      "events": events}, advice))
        with torch.inference_mode():
            action, hidden, _ = predict_action(model, observation, hidden, previous_action, delta_ticks, events)
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"model selected an illegal action on seed {seed}: {action}")
        previous_action = action
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        events = info["events"]
        if done:
            break
    if not observation["terminal"]:
        raise RuntimeError(f"DAgger episode exceeded {max_actions} decisions on seed {seed}")
    won = observation["result"] == 1
    for step in steps:
        step["episode_outcome"] = 1 if won else -1
    replay_id = f"dagger_seed_{seed}.jsonl.gz"
    env.save_replay(replay_dir / replay_id)
    return {"seed": seed, "replay_id": replay_id, "observation_version": 2, "task_version": 2,
            "search_label_version": SEARCH_LABEL_VERSION, "steps": steps, "won": won,
            "result": observation["result"], "tick": observation["tick"],
            "wave": observation["wave"], "wave_count": observation["wave_count"]}


def _search_labels(step: dict[str, Any], advice: Any) -> dict[str, Any]:
    if advice.search_policy is None:
        step.update({"candidate_actions": [], "search_values": [], "search_policy": []})
    else:
        step.update({"candidate_actions": [action for action, _ in advice.candidates],
                     "search_values": [value for _, value in advice.candidates],
                     "search_policy": advice.search_policy})
    step.update({"best_action": advice.action, "best_second_margin": advice.best_second_margin,
                 "search_elapsed_ticks": advice.search_elapsed_ticks,
                 "simulation_count": advice.simulation_count, "terminal_outcome": advice.terminal_outcome})
    return step


def write_episodes(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(episodes, stream, separators=(",", ":"), ensure_ascii=False)


def read_episodes(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def evaluate(model: GameplayModelV1, env: PvZEnv, seeds: list[int], device: torch.device,
             max_actions: int = 2000, level: int = LEVEL, deck: tuple[int, ...] = DECK,
             zombie_count_multiplier: float = 1.0) -> list[dict[str, Any]]:
    model.eval()
    records = []
    for seed in seeds:
        task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                        zombie_count_multiplier=zombie_count_multiplier)
        observation, _ = env.reset(deck=deck, task=task)
        hidden = None
        previous_action = None
        delta_ticks = 0
        events: dict[str, Any] = {}
        actions = 0
        started = time.time()
        while not observation["terminal"] and actions < max_actions:
            with torch.inference_mode():
                action, hidden, _ = predict_action(model, observation, hidden, previous_action, delta_ticks, events)
            observation, _, done, _, info = env.step(action)
            if not info.get("ok"):
                raise RuntimeError(f"model selected an illegal action on seed {seed}: {action}")
            previous_action = action
            delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
            events = info["events"]
            actions += 1
            if done:
                break
        if not observation["terminal"]:
            raise RuntimeError(f"evaluation exceeded {max_actions} decisions on seed {seed}")
        record = {
            "seed": seed, "won": observation["result"] == 1, "result": observation["result"],
            "wave": observation["wave"], "wave_count": observation["wave_count"],
            "tick": observation["tick"], "actions": actions, "seconds": round(time.time() - started, 2),
        }
        records.append(record)
        print(f"model seed={seed} won={record['won']} wave={record['wave']}/{record['wave_count']} "
              f"tick={record['tick']} actions={actions}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "artifacts" / "adventure2_level7")
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=parse_seeds, default=list(DECK))
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--dagger-seed-start", type=int, default=10000)
    parser.add_argument("--dagger-episodes", type=int, default=64)
    parser.add_argument("--eval-seeds", default="30000,30001,30002,30003")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-horizon-ticks", type=int, default=900)
    parser.add_argument("--search-simulation-budget", type=int, default=64)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if min(args.max_actions, args.train_episodes, args.dagger_episodes, args.dagger_rounds,
           args.search_width, args.search_candidates, args.search_horizon_ticks, args.search_simulation_budget) < 1:
        parser.error("episode counts and search parameters must be positive")
    if min(args.train_seed_start, args.dagger_seed_start) < 0:
        parser.error("seed starts must be non-negative")

    eval_seeds = parse_seeds(args.eval_seeds)
    train_seeds = contiguous_seeds(args.train_seed_start, args.train_episodes)
    all_dagger_seeds = contiguous_seeds(args.dagger_seed_start, args.dagger_episodes * args.dagger_rounds)
    validate_seed_sets(train_seeds, all_dagger_seeds, eval_seeds)

    torch.manual_seed(17)
    random.seed(17)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.output_dir / "replays"
    data_path = args.output_dir / "search_trajectories.json.gz"

    if not args.train_only:
        episodes = []
        with PvZEnv(resource_dir=args.resource_dir) as env:
            for seed in train_seeds:
                episode = collect_episode(
                    env, seed, replay_dir, args.max_actions, args.level, tuple(args.deck),
                    args.zombie_count_multiplier, args.search_width, args.search_candidates,
                    args.search_horizon_ticks, args.search_simulation_budget,
                )
                episodes.append(episode)
                print(f"search seed={seed} won={episode['won']} wave={episode['wave']}/{episode['wave_count']} "
                      f"tick={episode['tick']} steps={len(episode['steps'])}", flush=True)
        write_episodes(data_path, episodes)
        print(f"saved {sum(len(e['steps']) for e in episodes)} demonstrations to {data_path}", flush=True)
        search_metadata = provenance(args, {"search": data_path}, train_seeds, [], device)
        (args.output_dir / "search_trajectory_metadata.json").write_text(
            json.dumps({"provenance": search_metadata}, indent=2) + "\n", encoding="utf-8")
        if args.collect_only:
            return
    else:
        episodes = read_episodes(data_path)
        if any(episode.get("observation_version") != 2 or episode.get("task_version") != 2
               or episode.get("search_label_version") != SEARCH_LABEL_VERSION for episode in episodes):
            raise ValueError("training trajectories do not match the current search schema; recollect them")
        train_seeds = [int(episode["seed"]) for episode in episodes]
        validate_seed_sets(train_seeds, all_dagger_seeds, eval_seeds)

    model = GameplayModelV1().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"training on {device}; parameters={parameter_count:,}; architecture={MODEL_CONFIG}", flush=True)
    losses, plant_weight = train(model, episodes, args.epochs, device)
    bc_metadata = provenance(args, {"search": data_path}, train_seeds, [], device)
    bc_checkpoint = args.output_dir / "gameplay_model_v1_bc.pt"
    save_checkpoint(bc_checkpoint, model, bc_metadata, plant_action_weight=plant_weight,
                    training_seeds=[episode["seed"] for episode in episodes],
                    parameter_count=parameter_count, epochs=args.epochs, losses=losses)
    (args.output_dir / "gameplay_model_v1_bc.json").write_text(
        json.dumps({"checkpoint": bc_checkpoint.name, "provenance": bc_metadata,
                    "training_seeds": [episode["seed"] for episode in episodes], "losses": losses},
                   indent=2) + "\n", encoding="utf-8")

    dagger_path = args.output_dir / "dagger_search_trajectories.json.gz"
    dagger_episodes: list[dict[str, Any]] = []
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for dagger_round in range(args.dagger_rounds):
            round_start = args.dagger_seed_start + dagger_round * args.dagger_episodes
            round_seeds = contiguous_seeds(round_start, args.dagger_episodes)
            new_episodes = [collect_dagger_episode(
                model, env, seed, replay_dir, args.max_actions, args.level, tuple(args.deck),
                args.zombie_count_multiplier, args.search_width, args.search_candidates,
                args.search_horizon_ticks, args.search_simulation_budget,
            ) for seed in round_seeds]
            dagger_episodes.extend(new_episodes)
            episodes.extend(new_episodes)
            write_episodes(dagger_path, dagger_episodes)
            for episode in new_episodes:
                print(f"dagger round={dagger_round + 1} seed={episode['seed']} won={episode['won']} "
                      f"wave={episode['wave']}/{episode['wave_count']} tick={episode['tick']} "
                      f"steps={len(episode['steps'])}", flush=True)
            losses, plant_weight = train(model, episodes, args.epochs, device)
        results = evaluate(model, env, eval_seeds, device, args.max_actions,
                           args.level, tuple(args.deck), args.zombie_count_multiplier)

    checkpoint = args.output_dir / "gameplay_model_v1.pt"
    final_metadata = provenance(args, {"search": data_path, "dagger_search": dagger_path},
                                train_seeds, all_dagger_seeds, device)
    save_checkpoint(checkpoint, model, final_metadata, plant_action_weight=plant_weight,
                    training_seeds=[episode["seed"] for episode in episodes],
                    parameter_count=parameter_count, epochs=args.epochs, losses=losses,
                    evaluation=results)
    summary = {
        "model": checkpoint.name, "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "protocol_version": 1, "observation_version": 2, "task_version": 2,
        "search_label_version": SEARCH_LABEL_VERSION, "value_range": [-1, 1],
        "value_gamma": VALUE_GAMMA, "value_semantics": VALUE_SEMANTICS,
        "level": args.level, "playthrough": 2, "deck": args.deck,
        "zombie_count_multiplier": args.zombie_count_multiplier, "device": str(device),
        "parameter_count": parameter_count, "training_episodes": len(episodes),
        "training_steps": sum(len(episode["steps"]) for episode in episodes),
        "training_seeds": [episode["seed"] for episode in episodes], "plant_action_weight": plant_weight,
        "losses": losses, "provenance": final_metadata,
        "evaluation": results, "wins": sum(record["won"] for record in results),
        "evaluation_count": len(results),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
