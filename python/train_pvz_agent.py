"""Collect rule-teacher episodes, train GameplayModel-v1, and run model-only games."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from pvz_agent import (GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG, SearchTeacher,
                       behavior_cloning_loss, predict_action, resolve_device)
from pvz_env import PlayerProfileContext, PvZEnv, TaskSpec


LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)


def parse_seeds(text: str) -> list[int]:
    return [int(value) for value in text.split(",") if value.strip()]


def collect_episode(env: PvZEnv, seed: int, replay_dir: Path, max_actions: int = 2000,
                    level: int = LEVEL, deck: tuple[int, ...] = DECK,
                    zombie_count_multiplier: float = 1.0, search_width: int = 4,
                    search_depth: int = 3, search_candidates: int = 4) -> dict[str, Any]:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)
    observation, _ = env.reset(deck=deck, task=task)
    steps: list[dict[str, Any]] = []
    delta_ticks = 0
    events: dict[str, Any] = {}
    teacher = SearchTeacher(env, beam_width=search_width, depth=search_depth,
                            candidate_limit=search_candidates)
    for decision_index in range(max_actions):
        advice = teacher.advice(observation, delta_ticks=delta_ticks, events=events)
        action = advice.action
        steps.append(_teacher_labels({"decision_index": decision_index, "observation": observation, "action": action,
                                      "delta_ticks": delta_ticks, "events": events}, advice))
        before = observation
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"teacher selected an illegal action on seed {seed}: {action}")
        teacher.record_action(before, action)
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        events = info["events"]
        if done:
            won = observation["result"] == 1
            for step in steps:
                step["episode_outcome"] = int(won)
            replay_id = f"teacher_seed_{seed}.jsonl.gz"
            env.save_replay(replay_dir / replay_id)
            return {"seed": seed, "replay_id": replay_id, "observation_version": 2, "task_version": 2,
                    "teacher_label_version": 1, "steps": steps, "won": won, "result": observation["result"],
                    "tick": observation["tick"], "wave": observation["wave"], "wave_count": observation["wave_count"]}
    raise RuntimeError(f"teacher episode exceeded {max_actions} decisions on seed {seed}")


def collect_dagger_episode(model: GameplayModelV1, env: PvZEnv, seed: int, device: torch.device,
                           replay_dir: Path, max_actions: int = 2000, level: int = LEVEL,
                           deck: tuple[int, ...] = DECK, zombie_count_multiplier: float = 1.0,
                           search_width: int = 4, search_depth: int = 3,
                           search_candidates: int = 4) -> dict[str, Any]:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)
    observation, _ = env.reset(deck=deck, task=task)
    steps: list[dict[str, Any]] = []
    hidden = None
    previous_action = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    model.eval()
    teacher = SearchTeacher(env, model=model, beam_width=search_width, depth=search_depth,
                            candidate_limit=search_candidates)
    for decision_index in range(max_actions):
        advice = teacher.advice(observation, hidden, previous_action, delta_ticks, events)
        label = advice.action
        steps.append(_teacher_labels({"decision_index": decision_index, "observation": observation, "action": label,
                                      "previous_action": previous_action, "delta_ticks": delta_ticks,
                                      "events": events}, advice))
        with torch.inference_mode():
            action, hidden, _ = predict_action(model, observation, hidden, previous_action, delta_ticks, events)
        before = observation
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"model selected an illegal action on seed {seed}: {action}")
        teacher.record_action(before, action)
        previous_action = action
        delta_ticks = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        events = info["events"]
        if done:
            break
    if not observation["terminal"]:
        raise RuntimeError(f"DAgger episode exceeded {max_actions} decisions on seed {seed}")
    for step in steps:
        step["episode_outcome"] = int(observation["result"] == 1)
    replay_id = f"dagger_seed_{seed}.jsonl.gz"
    env.save_replay(replay_dir / replay_id)
    return {"seed": seed, "replay_id": replay_id, "observation_version": 2, "task_version": 2,
            "teacher_label_version": 1, "steps": steps, "won": observation["result"] == 1, "result": observation["result"],
            "tick": observation["tick"], "wave": observation["wave"], "wave_count": observation["wave_count"]}


def _teacher_labels(step: dict[str, Any], advice: Any) -> dict[str, Any]:
    if advice.search_policy is None:
        step.update({"candidate_actions": [], "search_values": [], "search_policy": []})
    else:
        step.update({"candidate_actions": [action for action, _ in advice.candidates],
                     "search_values": [value for _, value in advice.candidates],
                     "search_policy": advice.search_policy})
    step.update({"best_action": advice.action, "best_second_margin": advice.best_second_margin,
                 "search_depth": advice.search_depth, "simulation_count": advice.simulation_count,
                 "terminal_outcome": advice.terminal_outcome, "search_value": advice.search_value})
    return step


def write_episodes(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(episodes, stream, separators=(",", ":"), ensure_ascii=False)


def read_episodes(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args: argparse.Namespace, data_paths: dict[str, Path], train_seeds: list[int],
               dagger_seeds: list[int], device: torch.device) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                 capture_output=True, text=True).stdout.strip()
        git_dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                        capture_output=True, text=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_dirty = None, None
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    cli = vars(args).copy()
    cli["output_dir"] = str(args.output_dir)
    cli["resolved_device"] = str(device)
    return {
        "git_sha": git_sha, "git_dirty": git_dirty,
        "observation_version": 2, "task_version": 2,
        "command": {"argv": list(sys.argv), "arguments": cli},
        "trajectory_sha256": {name: sha256_file(path) for name, path in data_paths.items()},
        "resource_sha256": {
            "main.pak": sha256_file(resource_dir / "main.pak"),
            "properties/partner.xml": sha256_file(resource_dir / "properties" / "partner.xml"),
        },
        "random_seeds": {"python": 17, "torch": 17, "teacher_episodes": train_seeds,
                         "dagger_episodes": dagger_seeds},
        "model_config": MODEL_CONFIG,
    }


def save_checkpoint(path: Path, model: GameplayModelV1, metadata: dict[str, Any], **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
                "config": MODEL_CONFIG, "level": metadata["command"]["arguments"]["level"],
                "deck": metadata["command"]["arguments"]["deck"],
                "profile": "Adventure-II, six slots, no store items", "provenance": metadata, **fields}, path)


def episode_targets(steps: list[dict[str, Any]], index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    observation = steps[index]["observation"]
    lanes = torch.zeros(1, 6, device=device)
    for zombie in observation["zombies"]:
        lanes[0, zombie["row"]] = 1
    next_wave = 0.0
    if index + 1 < len(steps):
        next_wave = float(steps[index + 1]["events"].get("waves_started", 0) > 0)
    return lanes, torch.tensor([next_wave], dtype=torch.float32, device=device)


def train(model: GameplayModelV1, episodes: list[dict[str, Any]], epochs: int, device: torch.device) -> tuple[list[float], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    plant_steps = sum(step["action"]["type"] == "plant" for episode in episodes for step in episode["steps"])
    other_steps = sum(len(episode["steps"]) for episode in episodes) - plant_steps
    plant_weight = other_steps / max(plant_steps, 1)
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        order = list(range(len(episodes)))
        random.shuffle(order)
        total_loss = 0.0
        total_steps = 0
        for episode_index in order:
            episode = episodes[episode_index]
            steps = episode["steps"]
            for start in range(0, len(steps), 64):
                burn_start = max(0, start - 32)
                hidden = None
                with torch.no_grad():
                    for index in range(burn_start, start):
                        previous = steps[index].get("previous_action", steps[index - 1]["action"] if index else None)
                        warm = model.step(steps[index]["observation"], hidden, previous,
                                          steps[index]["delta_ticks"], steps[index]["events"])
                        hidden = warm["hidden"]
                if hidden is not None:
                    hidden = hidden.detach()
                losses = []
                for index in range(start, min(start + 64, len(steps))):
                    step = steps[index]
                    previous = step.get("previous_action", steps[index - 1]["action"] if index else None)
                    output = model.step(step["observation"], hidden, previous, step["delta_ticks"], step["events"])
                    hidden = output["hidden"]
                    loss = behavior_cloning_loss(
                        model, output, step["observation"], step["action"], plant_weight,
                        step.get("candidate_actions"), step.get("search_policy"),
                    )
                    lanes, next_wave = episode_targets(steps, index, device)
                    outcome = float(episode.get("won", True))
                    value_target = float(step.get("search_value") if step.get("search_value") is not None else outcome)
                    loss = loss + 0.05 * F.mse_loss(output["value"], torch.full_like(output["value"], value_target))
                    loss = loss + 0.05 * F.cross_entropy(output["aux_outcome"], torch.full((1,), int(outcome), dtype=torch.long, device=device))
                    loss = loss + 0.05 * F.binary_cross_entropy_with_logits(output["aux_lane_threat"], lanes)
                    loss = loss + 0.05 * F.binary_cross_entropy_with_logits(output["aux_next_spawn"].view(1), next_wave)
                    losses.append(loss)
                optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(torch.stack([loss.detach() for loss in losses]).sum().item())
                total_steps += len(losses)
        epoch_loss = total_loss / max(total_steps, 1)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}", flush=True)
    return history, plant_weight


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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts" / "adventure2_level7")
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=parse_seeds, default=list(DECK))
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--train-seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dagger-seeds", default=None)
    parser.add_argument("--eval-seeds", default="30000,30001,30002,30003")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--search-width", type=int, default=4)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=4)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.dagger_rounds < 1:
        parser.error("--dagger-rounds must be positive")
    if min(args.search_width, args.search_depth, args.search_candidates) < 1:
        parser.error("search width, depth, and candidate count must be positive")

    torch.manual_seed(17)
    random.seed(17)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.output_dir / "replays"
    data_path = args.output_dir / "teacher_trajectories.json.gz"
    train_seeds = parse_seeds(args.train_seeds)
    dagger_seeds = parse_seeds(args.dagger_seeds) if args.dagger_seeds else train_seeds

    if not args.train_only:
        episodes = []
        with PvZEnv(resource_dir=args.resource_dir) as env:
            for seed in train_seeds:
                episode = collect_episode(env, seed, replay_dir, args.max_actions, args.level, tuple(args.deck),
                                          args.zombie_count_multiplier, args.search_width,
                                          args.search_depth, args.search_candidates)
                episodes.append(episode)
                print(f"teacher seed={seed} won={episode['won']} wave={episode['wave']}/{episode['wave_count']} "
                      f"tick={episode['tick']} steps={len(episode['steps'])}", flush=True)
        write_episodes(data_path, episodes)
        print(f"saved {sum(len(e['steps']) for e in episodes)} demonstrations to {data_path}", flush=True)
        teacher_metadata = provenance(args, {"teacher": data_path}, train_seeds, [], device)
        (args.output_dir / "teacher_trajectory_metadata.json").write_text(
            json.dumps({"provenance": teacher_metadata}, indent=2) + "\n", encoding="utf-8")
        if args.collect_only:
            return
    else:
        episodes = read_episodes(data_path)
        if any(episode.get("observation_version") != 2 or episode.get("task_version") != 2
               or episode.get("teacher_label_version") != 1 for episode in episodes):
            raise ValueError("training trajectories predate the current environment or search teacher; recollect them without --train-only")

    model = GameplayModelV1().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"training on {device}; parameters={parameter_count:,}; architecture={MODEL_CONFIG}", flush=True)
    losses, plant_weight = train(model, episodes, args.epochs, device)
    teacher_data = {"teacher": data_path}
    bc_metadata = provenance(args, teacher_data, [episode["seed"] for episode in episodes], [], device)
    bc_checkpoint = args.output_dir / "gameplay_model_v1_bc.pt"
    save_checkpoint(bc_checkpoint, model, bc_metadata, plant_action_weight=plant_weight,
                    training_seeds=[episode["seed"] for episode in episodes],
                    parameter_count=parameter_count, epochs=args.epochs, losses=losses)
    (args.output_dir / "gameplay_model_v1_bc.json").write_text(
        json.dumps({"checkpoint": bc_checkpoint.name, "provenance": bc_metadata,
                    "training_seeds": [episode["seed"] for episode in episodes], "losses": losses},
                   indent=2) + "\n", encoding="utf-8")
    dagger_path = args.output_dir / "dagger_trajectories.json.gz"
    dagger_episodes = []
    with PvZEnv(resource_dir=args.resource_dir) as env:
        for dagger_round in range(1, args.dagger_rounds + 1):
            new_episodes = [collect_dagger_episode(
                model, env, seed, device, replay_dir, args.max_actions, args.level, tuple(args.deck),
                args.zombie_count_multiplier, args.search_width, args.search_depth, args.search_candidates,
            ) for seed in dagger_seeds]
            dagger_episodes.extend(new_episodes)
            episodes.extend(new_episodes)
            write_episodes(dagger_path, dagger_episodes)
            for episode in new_episodes:
                print(f"dagger round={dagger_round} seed={episode['seed']} won={episode['won']} "
                      f"wave={episode['wave']}/{episode['wave_count']} tick={episode['tick']} "
                      f"steps={len(episode['steps'])}", flush=True)
            losses, plant_weight = train(model, episodes, args.epochs, device)
        results = evaluate(model, env, parse_seeds(args.eval_seeds), device, args.max_actions,
                           args.level, tuple(args.deck), args.zombie_count_multiplier)
    checkpoint = args.output_dir / "gameplay_model_v1.pt"
    final_metadata = provenance(args, {"teacher": data_path, "dagger": dagger_path},
                                train_seeds, dagger_seeds, device)
    save_checkpoint(checkpoint, model, final_metadata, plant_action_weight=plant_weight,
                    training_seeds=[episode["seed"] for episode in episodes],
                    parameter_count=parameter_count, epochs=args.epochs, losses=losses,
                    evaluation=results)
    summary = {
        "model": checkpoint.name, "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "protocol_version": 1, "observation_version": 2, "task_version": 2,
        "level": args.level, "playthrough": 2, "deck": args.deck,
        "zombie_count_multiplier": args.zombie_count_multiplier, "device": str(device),
        "parameter_count": parameter_count, "training_episodes": len(episodes),
        "training_steps": sum(len(episode["steps"]) for episode in episodes),
        "training_seeds": [episode["seed"] for episode in episodes], "plant_action_weight": plant_weight,
        "losses": losses,
        "provenance": final_metadata,
        "evaluation": results, "wins": sum(record["won"] for record in results),
        "evaluation_count": len(results),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={checkpoint}", flush=True)


if __name__ == "__main__":
    main()
