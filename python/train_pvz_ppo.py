"""Train a recurrent PPO policy on Adventure-II from a DAgger checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F

from benchmark_pvz_agent import DEFAULT_SEEDS, read_seed_set
from pvz_agent import (GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG,
                       resolve_device, select_action)
from pvz_env import PlayerProfileContext, PvZEnv, TaskSpec


LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)
DISCOUNT_REFERENCE_TICKS = 300
ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout)
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def collect_episode(model: GameplayModelV1, env: PvZEnv, seed: int,
                    max_actions: int, replay_path: Path, level: int = LEVEL,
                    deck: tuple[int, ...] = DECK, zombie_count_multiplier: float = 1.0) -> dict[str, Any]:
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=PlayerProfileContext(),
                    zombie_count_multiplier=zombie_count_multiplier)
    observation, _ = env.reset(deck=deck, task=task)
    hidden = None
    previous_action = None
    elapsed_since_previous_observation = 0
    events: dict[str, Any] = {}
    transitions = []
    started = time.perf_counter()
    for decision_index in range(max_actions):
        hidden_before = None if hidden is None else hidden.detach().clone()
        with torch.no_grad():
            output = model.step(observation, hidden, previous_action,
                                elapsed_since_previous_observation, events)
            action, log_prob, _ = select_action(model, output, observation)
        transition = {
            "decision_index": decision_index,
            "observation": observation, "previous_action": previous_action,
            "elapsed_since_previous_observation": elapsed_since_previous_observation,
            "events": events, "action": action,
            "log_prob": float(log_prob.item()), "value": float(output["value"].item()),
            "hidden": hidden_before,
        }
        transitions.append(transition)
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"model selected an illegal action on seed {seed}: {action}")
        hidden = output["hidden"]
        previous_action = action
        transition["action_duration_ticks"] = info.get(
            "ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0
        )
        elapsed_since_previous_observation = transition["action_duration_ticks"]
        events = info["events"]
        if done:
            break
    if not observation["terminal"]:
        raise RuntimeError(f"PPO episode exceeded {max_actions} decisions on seed {seed}")
    won = observation["result"] == 1
    transitions[-1]["reward"] = 1.0 if won else -1.0
    for transition in transitions[:-1]:
        transition["reward"] = 0.0
    env.save_replay(replay_path)
    return {"seed": seed, "replay_id": replay_path.name, "won": won, "result": observation["result"],
            "wave": observation["wave"], "wave_count": observation["wave_count"],
            "tick": observation["tick"], "seconds": time.perf_counter() - started,
            "transitions": transitions}


def add_advantages(episodes: list[dict[str, Any]], gamma: float, gae_lambda: float) -> None:
    for episode in episodes:
        transitions = episode["transitions"]
        advantage = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            transition = transitions[index]
            next_value = transitions[index + 1]["value"] if index + 1 < len(transitions) else 0.0
            discount = gamma ** (transition["action_duration_ticks"] / DISCOUNT_REFERENCE_TICKS)
            delta = transition["reward"] + discount * next_value - transition["value"]
            advantage = delta + discount * gae_lambda * advantage
            transition["advantage"] = advantage
            transition["return"] = advantage + transition["value"]


def train_update(model: GameplayModelV1, episodes: list[dict[str, Any]], optimizer: torch.optim.Optimizer,
                 device: torch.device, ppo_epochs: int, sequence_length: int,
                 clip_epsilon: float, value_coefficient: float, entropy_coefficient: float) -> dict[str, float]:
    advantages = torch.tensor([transition["advantage"] for episode in episodes
                               for transition in episode["transitions"]], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    offset = 0
    for episode in episodes:
        for transition in episode["transitions"]:
            transition["normalized_advantage"] = advantages[offset]
            offset += 1

    chunks = [(episode, start, min(start + sequence_length, len(episode["transitions"])))
              for episode in episodes
              for start in range(0, len(episode["transitions"]), sequence_length)]
    policy_losses, value_losses, entropies = [], [], []
    model.train()
    for _ in range(ppo_epochs):
        random.shuffle(chunks)
        for episode, start, end in chunks:
            transitions = episode["transitions"]
            hidden = transitions[start]["hidden"]
            hidden = None if hidden is None else hidden.to(device).detach()
            log_probs, values, entropies_for_chunk = [], [], []
            for transition in transitions[start:end]:
                output = model.step(transition["observation"], hidden, transition["previous_action"],
                                    transition["elapsed_since_previous_observation"], transition["events"])
                _, log_prob, entropy = select_action(model, output, transition["observation"],
                                                     action=transition["action"])
                hidden = output["hidden"]
                log_probs.append(log_prob)
                values.append(output["value"].squeeze())
                entropies_for_chunk.append(entropy)
            new_log_prob = torch.stack(log_probs)
            old_log_prob = torch.tensor([transition["log_prob"] for transition in transitions[start:end]],
                                        dtype=torch.float32, device=device)
            advantage = torch.stack([transition["normalized_advantage"]
                                     for transition in transitions[start:end]])
            ratio = torch.exp(new_log_prob - old_log_prob)
            policy_loss = -torch.minimum(ratio * advantage,
                                         ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage).mean()
            returns = torch.tensor([transition["return"] for transition in transitions[start:end]],
                                   dtype=torch.float32, device=device)
            value_loss = F.mse_loss(torch.stack(values), returns)
            entropy = torch.stack(entropies_for_chunk).mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            entropies.append(float(entropy.detach().item()))
    model.eval()
    return {"policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": sum(entropies) / len(entropies)}


def episode_hash(episode: dict[str, Any]) -> str:
    steps = [{key: transition[key] for key in
              ("observation", "previous_action", "elapsed_since_previous_observation", "action_duration_ticks",
               "events", "action", "log_prob", "value", "reward")}
             for transition in episode["transitions"]]
    payload = json.dumps({"seed": episode["seed"], "steps": steps,
                          "result": episode["result"]}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--init-checkpoint", type=Path,
                        default=ROOT / "artifacts" / "adventure2_level7" / "gameplay_model_v1.pt")
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "adventure2_level7")
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=lambda value: tuple(map(int, value.split(","))), default=DECK)
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--train-seed-end", type=int, default=10000)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if args.updates < 1 or args.rollout_episodes < 1 or args.ppo_epochs < 1:
        parser.error("updates, rollout episodes, and PPO epochs must be positive")
    if args.train_seed_start >= args.train_seed_end:
        parser.error("training seed range must be nonempty")
    if not 1.0 <= args.zombie_count_multiplier <= 10.0:
        parser.error("--zombie-count-multiplier must be from 1 to 10")
    evaluation_seeds = set(read_seed_set(args.seeds, args.level))
    if any(args.train_seed_start <= seed < args.train_seed_end for seed in evaluation_seeds):
        parser.error("training seed range overlaps the frozen evaluation seeds")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    initial_checkpoint_sha = sha256_file(args.init_checkpoint)
    trajectory_dir = args.init_checkpoint.expanduser().resolve().parent
    initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    provenance = initial["provenance"]
    if (initial["model_architecture_version"] != MODEL_ARCHITECTURE_VERSION
            or provenance["observation_version"] != 2 or provenance["task_version"] != 2):
        raise ValueError("initial checkpoint versions do not match the current model, observation, and task")
    model = GameplayModelV1().to(device)
    model.load_state_dict(initial["state_dict"])
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gameplay_model_v1_ppo.pt"
    summary_path = args.output_dir / "ppo_training_summary.json"
    revision, dirty = git_metadata()
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    resource_hashes = {"main.pak": sha256_file(resource_dir / "main.pak"),
                       "properties/partner.xml": sha256_file(resource_dir / "properties" / "partner.xml")}
    trajectory_hashes = {}
    history = []
    config = {key: value for key, value in vars(args).items()}
    config.update({"resource_dir": str(resource_dir), "init_checkpoint": str(args.init_checkpoint),
                   "seeds": str(args.seeds), "output_dir": str(args.output_dir),
                   "discount_reference_ticks": DISCOUNT_REFERENCE_TICKS,
                   "resolved_device": str(device)})
    train_seeds = list(range(args.train_seed_start, args.train_seed_end))
    with PvZEnv(resource_dir=resource_dir) as env:
        for update in range(1, args.updates + 1):
            seeds = random.sample(train_seeds, args.rollout_episodes)
            episodes = [collect_episode(model, env, seed, args.max_actions,
                                        args.output_dir / "replays" / f"ppo_update_{update}_seed_{seed}.jsonl.gz",
                                        args.level, args.deck, args.zombie_count_multiplier)
                        for seed in seeds]
            add_advantages(episodes, args.gamma, args.gae_lambda)
            losses = train_update(model, episodes, optimizer, device, args.ppo_epochs,
                                  args.sequence_length, args.clip_epsilon,
                                  args.value_coefficient, args.entropy_coefficient)
            hashes = {f"{update}:{episode['seed']}": episode_hash(episode) for episode in episodes}
            trajectory_hashes.update(hashes)
            row = {"update": update, "seeds": seeds,
                   "wins": sum(episode["won"] for episode in episodes),
                   "episodes": [{"seed": episode["seed"], "replay_id": episode["replay_id"], "won": episode["won"],
                                 "wave": episode["wave"], "actions": len(episode["transitions"]),
                                 "seconds": round(episode["seconds"], 3),
                                 "sha256": hashes[f"{update}:{episode['seed']}"]}
                                for episode in episodes],
                   "losses": losses}
            history.append(row)
            if not env.episode:
                raise RuntimeError("environment did not record resource provenance")
            provenance = {
                "git_sha": revision, "git_dirty": dirty,
                "command": {"argv": sys.argv, "arguments": config},
                "trajectory_sha256": {
                    "teacher": sha256_file(trajectory_dir / "teacher_trajectories.json.gz"),
                    "dagger": sha256_file(trajectory_dir / "dagger_trajectories.json.gz"),
                    "ppo_rollouts_by_seed": trajectory_hashes,
                },
                "resource_sha256": resource_hashes,
                "frozen_evaluation_seed_sha256": sha256_file(args.seeds),
                "initial_checkpoint_sha256": initial_checkpoint_sha,
                "random_seeds": {"python_torch": args.seed, "training_seed_range": [args.train_seed_start,
                                                                                      args.train_seed_end - 1]},
                "model_config": MODEL_CONFIG, "observation_version": 2, "task_version": 2,
            }
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
                        "config": MODEL_CONFIG, "level": args.level, "deck": args.deck,
                        "profile": "Adventure-II, six slots, no store items",
                        "update": update, "ppo_config": config, "losses": losses,
                        "provenance": provenance}, checkpoint_path)
            summary = {"checkpoint": checkpoint_path.name, "provenance": provenance,
                       "initial_checkpoint": str(args.init_checkpoint), "updates": history}
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"update={update}/{args.updates} wins={row['wins']}/{len(episodes)} "
                  f"policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f} "
                  f"entropy={losses['entropy']:.3f}", flush=True)
    print(f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
