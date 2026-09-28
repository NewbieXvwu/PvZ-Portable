"""Collect search trajectories, train an independent search value, then distill GameplayModel-v1."""

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

from pvz_agent_model import GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG, predict_action, resolve_device
from pvz_env import ENV_PROTOCOL_VERSION, PlayerProfileContext, PvZEnv, TaskSpec
from pvz_imitation import train
from pvz_search import SearchTeacher
from pvz_search_value import SearchValueModel, load_search_value, save_search_value, train_search_value
from pvz_seed_sets import DEFAULT_DEV_SEEDS, DEFAULT_TEST_SEEDS, read_seed_set
from pvz_training_artifacts import provenance, save_checkpoint, sha256_file
from pvz_value import SEARCH_LABEL_VERSION, VALUE_GAMMA, VALUE_SEMANTICS

LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)


def contiguous_seeds(start: int, count: int) -> list[int]:
    return list(range(start, start + count))


def validate_seed_sets(**groups: list[int]) -> None:
    named = {name: list(values) for name, values in groups.items()}
    for name, values in named.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{name} seed set contains duplicates")
    names = list(named)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = set(named[left]) & set(named[right])
            if overlap:
                raise ValueError(f"{left} and {right} seed sets overlap: {sorted(overlap)[:8]}")


def _task(seed: int, level: int, zombie_count_multiplier: float) -> TaskSpec:
    return TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)


def _search_labels(step: dict[str, Any], advice: Any) -> dict[str, Any]:
    step.update({
        "candidate_actions": [action for action, _ in advice.candidates],
        "search_values": [value for _, value in advice.candidates],
        "search_policy": advice.search_policy,
        "best_action": advice.action,
        "best_second_margin": advice.best_second_margin,
        "search_elapsed_ticks": advice.search_elapsed_ticks,
        "simulation_count": advice.simulation_count,
        "terminal_outcome": advice.terminal_outcome,
    })
    return step


def collect_search_episode(
    env: PvZEnv,
    seed: int,
    replay_dir: Path,
    value_model: SearchValueModel | None,
    max_actions: int,
    level: int,
    deck: tuple[int, ...],
    zombie_count_multiplier: float,
    search_width: int,
    search_candidates: int,
    search_horizon_ticks: int,
    search_simulation_budget: int,
    replay_prefix: str,
) -> dict[str, Any]:
    observation, _ = env.reset(deck=deck, task=_task(seed, level, zombie_count_multiplier))
    searcher = SearchTeacher(env, value_model=value_model, beam_width=search_width,
                            candidate_limit=search_candidates, horizon_ticks=search_horizon_ticks,
                            simulation_budget=search_simulation_budget)
    steps: list[dict[str, Any]] = []
    delta_ticks = 0
    events: dict[str, Any] = {}
    for decision_index in range(max_actions):
        advice = searcher.advice(observation)
        action = advice.action
        steps.append(_search_labels({
            "decision_index": decision_index,
            "observation": observation,
            "action": action,
            "delta_ticks": delta_ticks,
            "events": events,
        }, advice))
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"search selected an illegal action on seed {seed}: {action}")
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        events = info["events"]
        if done:
            won = observation["result"] == 1
            for step in steps:
                step["episode_outcome"] = 1 if won else -1
            replay_id = f"{replay_prefix}_seed_{seed}.jsonl.gz"
            env.save_replay(replay_dir / replay_id)
            return {
                "seed": seed,
                "replay_id": replay_id,
                "observation_version": 2,
                "task_version": 2,
                "search_label_version": SEARCH_LABEL_VERSION,
                "steps": steps,
                "won": won,
                "result": observation["result"],
                "tick": observation["tick"],
                "wave": observation["wave"],
                "wave_count": observation["wave_count"],
            }
    raise RuntimeError(f"search episode exceeded {max_actions} decisions on seed {seed}")


def collect_dagger_episode(
    model: GameplayModelV1,
    search_value: SearchValueModel,
    env: PvZEnv,
    seed: int,
    replay_dir: Path,
    max_actions: int,
    level: int,
    deck: tuple[int, ...],
    zombie_count_multiplier: float,
    search_width: int,
    search_candidates: int,
    search_horizon_ticks: int,
    search_simulation_budget: int,
) -> dict[str, Any]:
    observation, _ = env.reset(deck=deck, task=_task(seed, level, zombie_count_multiplier))
    searcher = SearchTeacher(env, value_model=search_value, beam_width=search_width,
                            candidate_limit=search_candidates, horizon_ticks=search_horizon_ticks,
                            simulation_budget=search_simulation_budget)
    steps: list[dict[str, Any]] = []
    hidden = None
    previous_action = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    model.eval()
    for decision_index in range(max_actions):
        advice = searcher.advice(observation)
        steps.append(_search_labels({
            "decision_index": decision_index,
            "observation": observation,
            "action": advice.action,
            "previous_action": previous_action,
            "delta_ticks": delta_ticks,
            "events": events,
        }, advice))
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
    return {
        "seed": seed,
        "replay_id": replay_id,
        "observation_version": 2,
        "task_version": 2,
        "search_label_version": SEARCH_LABEL_VERSION,
        "steps": steps,
        "won": won,
        "result": observation["result"],
        "tick": observation["tick"],
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
    }


def write_episodes(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(episodes, stream, separators=(",", ":"), ensure_ascii=False)


def read_episodes(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def validate_episode_schema(episodes: list[dict[str, Any]]) -> None:
    if any(episode.get("observation_version") != 2 or episode.get("task_version") != 2
           or episode.get("search_label_version") != SEARCH_LABEL_VERSION for episode in episodes):
        raise ValueError("training trajectories do not match the current search schema; recollect them")


def _search_value_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def evaluate(model: GameplayModelV1, env: PvZEnv, seeds: list[int], device: torch.device,
             max_actions: int, level: int, deck: tuple[int, ...],
             zombie_count_multiplier: float) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(deck=deck, task=_task(seed, level, zombie_count_multiplier))
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
        records.append({
            "seed": seed,
            "won": observation["result"] == 1,
            "result": observation["result"],
            "wave": observation["wave"],
            "wave_count": observation["wave_count"],
            "tick": observation["tick"],
            "actions": actions,
            "seconds": round(time.time() - started, 2),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "artifacts" / "adventure2_level7")
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=lambda text: [int(value) for value in text.split(",") if value],
                        default=list(DECK))
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--train-seed-start", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--dagger-seed-start", type=int, default=10000)
    parser.add_argument("--dagger-episodes", type=int, default=64)
    parser.add_argument("--value-bootstrap-seed-start", type=int, default=20000)
    parser.add_argument("--value-bootstrap-episodes", type=int, default=32)
    parser.add_argument("--value-refinement-seed-start", type=int, default=21000)
    parser.add_argument("--value-refinement-episodes", type=int, default=32)
    parser.add_argument("--dev-seeds", type=Path, default=DEFAULT_DEV_SEEDS)
    parser.add_argument("--test-seeds", type=Path, default=DEFAULT_TEST_SEEDS)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--value-epochs", type=int, default=8)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-horizon-ticks", type=int, default=900)
    parser.add_argument("--search-simulation-budget", type=int, default=256)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    positive = (
        args.max_actions, args.train_episodes, args.dagger_episodes, args.dagger_rounds,
        args.value_bootstrap_episodes, args.value_refinement_episodes, args.epochs, args.value_epochs,
        args.search_width, args.search_candidates, args.search_horizon_ticks, args.search_simulation_budget,
    )
    if min(positive) < 1:
        parser.error("episode counts, epochs, and search parameters must be positive")
    if min(args.train_seed_start, args.dagger_seed_start, args.value_bootstrap_seed_start,
           args.value_refinement_seed_start) < 0:
        parser.error("seed starts must be non-negative")
    if not 1.0 <= args.zombie_count_multiplier <= 10.0:
        parser.error("--zombie-count-multiplier must be from 1 to 10")

    train_seeds = contiguous_seeds(args.train_seed_start, args.train_episodes)
    dagger_seeds = contiguous_seeds(args.dagger_seed_start, args.dagger_episodes * args.dagger_rounds)
    bootstrap_seeds = contiguous_seeds(args.value_bootstrap_seed_start, args.value_bootstrap_episodes)
    refinement_seeds = contiguous_seeds(args.value_refinement_seed_start, args.value_refinement_episodes)
    dev_seeds = read_seed_set(args.dev_seeds, args.level, "development")
    test_seeds = read_seed_set(args.test_seeds, args.level, "final_test")
    validate_seed_sets(
        train=train_seeds,
        dagger=dagger_seeds,
        value_bootstrap=bootstrap_seeds,
        value_refinement=refinement_seeds,
        development=dev_seeds,
        final_test=test_seeds,
    )

    random.seed(17)
    torch.manual_seed(17)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.output_dir / "replays"
    data_path = args.output_dir / "search_trajectories.json.gz"
    bootstrap_path = args.output_dir / "search_value_bootstrap.json.gz"
    refinement_path = args.output_dir / "search_value_refinement.json.gz"
    search_value_path = args.output_dir / "search_value_v1.pt"

    if args.train_only:
        episodes = read_episodes(data_path)
        validate_episode_schema(episodes)
        train_seeds = [int(episode["seed"]) for episode in episodes]
        search_value, loaded_search_value_checkpoint = load_search_value(search_value_path, device)
        checkpoint_bootstrap = [int(seed) for seed in loaded_search_value_checkpoint.get("bootstrap_seeds", [])]
        checkpoint_refinement = [int(seed) for seed in loaded_search_value_checkpoint.get("refinement_seeds", [])]
        if not checkpoint_bootstrap or not checkpoint_refinement:
            raise ValueError("search-value checkpoint is missing its bootstrap/refinement seed provenance")
        validate_seed_sets(
            train=train_seeds,
            dagger=dagger_seeds,
            value_bootstrap=checkpoint_bootstrap,
            value_refinement=checkpoint_refinement,
            development=dev_seeds,
            final_test=test_seeds,
        )
        search_value_summary = _search_value_metadata(loaded_search_value_checkpoint)
    else:
        with PvZEnv(resource_dir=args.resource_dir) as env:
            bootstrap = [
                collect_search_episode(
                    env, seed, replay_dir, None, args.max_actions, args.level, tuple(args.deck),
                    args.zombie_count_multiplier, args.search_width, args.search_candidates,
                    args.search_horizon_ticks, args.search_simulation_budget, "value_bootstrap",
                )
                for seed in bootstrap_seeds
            ]
            write_episodes(bootstrap_path, bootstrap)
            search_value = SearchValueModel().to(device)
            bootstrap_losses = train_search_value(search_value, bootstrap, args.value_epochs, device)

            refinement = [
                collect_search_episode(
                    env, seed, replay_dir, search_value, args.max_actions, args.level, tuple(args.deck),
                    args.zombie_count_multiplier, args.search_width, args.search_candidates,
                    args.search_horizon_ticks, args.search_simulation_budget, "value_refinement",
                )
                for seed in refinement_seeds
            ]
            write_episodes(refinement_path, refinement)
            refinement_losses = train_search_value(search_value, bootstrap + refinement, args.value_epochs, device)
            save_search_value(
                search_value_path,
                search_value,
                bootstrap_seeds=bootstrap_seeds,
                refinement_seeds=refinement_seeds,
                bootstrap_losses=bootstrap_losses,
                refinement_losses=refinement_losses,
            )
            search_value_summary = {
                "bootstrap_seeds": bootstrap_seeds,
                "refinement_seeds": refinement_seeds,
                "bootstrap_losses": bootstrap_losses,
                "refinement_losses": refinement_losses,
            }

            episodes = [
                collect_search_episode(
                    env, seed, replay_dir, search_value, args.max_actions, args.level, tuple(args.deck),
                    args.zombie_count_multiplier, args.search_width, args.search_candidates,
                    args.search_horizon_ticks, args.search_simulation_budget, "search",
                )
                for seed in train_seeds
            ]
        write_episodes(data_path, episodes)
        if args.collect_only:
            return

    search_value_sha256 = sha256_file(search_value_path)
    model = GameplayModelV1().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    losses, plant_weight = train(model, episodes, args.epochs, device)
    bc_metadata = provenance(args, {"search": data_path}, train_seeds, [], device)
    bc_checkpoint = args.output_dir / "gameplay_model_v1_bc.pt"
    save_checkpoint(
        bc_checkpoint,
        model,
        bc_metadata,
        plant_action_weight=plant_weight,
        training_seeds=[episode["seed"] for episode in episodes],
        parameter_count=parameter_count,
        epochs=args.epochs,
        losses=losses,
        search_value_checkpoint=search_value_path.name,
        search_value_sha256=search_value_sha256,
    )

    dagger_path = args.output_dir / "dagger_search_trajectories.json.gz"
    dagger_episodes: list[dict[str, Any]] = []
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for dagger_round in range(args.dagger_rounds):
            round_start = args.dagger_seed_start + dagger_round * args.dagger_episodes
            round_seeds = contiguous_seeds(round_start, args.dagger_episodes)
            new_episodes = [
                collect_dagger_episode(
                    model, search_value, env, seed, replay_dir, args.max_actions, args.level, tuple(args.deck),
                    args.zombie_count_multiplier, args.search_width, args.search_candidates,
                    args.search_horizon_ticks, args.search_simulation_budget,
                )
                for seed in round_seeds
            ]
            dagger_episodes.extend(new_episodes)
            episodes.extend(new_episodes)
            write_episodes(dagger_path, dagger_episodes)
            losses, plant_weight = train(model, episodes, args.epochs, device)
        results = evaluate(
            model, env, dev_seeds, device, args.max_actions, args.level,
            tuple(args.deck), args.zombie_count_multiplier,
        )

    checkpoint = args.output_dir / "gameplay_model_v1.pt"
    final_metadata = provenance(
        args,
        {"search": data_path, "dagger_search": dagger_path},
        train_seeds,
        dagger_seeds,
        device,
    )
    save_checkpoint(
        checkpoint,
        model,
        final_metadata,
        plant_action_weight=plant_weight,
        training_seeds=[episode["seed"] for episode in episodes],
        parameter_count=parameter_count,
        epochs=args.epochs,
        losses=losses,
        development_evaluation=results,
        search_value_checkpoint=search_value_path.name,
        search_value_sha256=search_value_sha256,
    )
    summary = {
        "model": checkpoint.name,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "protocol_version": ENV_PROTOCOL_VERSION,
        "observation_version": 2,
        "task_version": 2,
        "search_label_version": SEARCH_LABEL_VERSION,
        "value_range": [-1, 1],
        "value_gamma": VALUE_GAMMA,
        "value_semantics": VALUE_SEMANTICS,
        "search_value_checkpoint": search_value_path.name,
        "search_value_sha256": search_value_sha256,
        "search_value_training": search_value_summary,
        "level": args.level,
        "deck": args.deck,
        "training_episodes": len(episodes),
        "training_steps": sum(len(episode["steps"]) for episode in episodes),
        "development_seed_file": str(args.dev_seeds),
        "development_seed_sha256": sha256_file(args.dev_seeds),
        "final_test_seed_file": str(args.test_seeds),
        "final_test_seed_sha256": sha256_file(args.test_seeds),
        "development_evaluation": results,
        "development_wins": sum(record["won"] for record in results),
        "development_count": len(results),
        "losses": losses,
        "provenance": final_metadata,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
