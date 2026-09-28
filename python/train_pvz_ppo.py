"""Train a recurrent PPO policy on Adventure-II from a DAgger checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F

from pvz_agent_model import (GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG,
                             resolve_device, select_action)
from pvz_common import (
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    TASK_VERSION,
    canonical_digest,
    git_metadata,
    sha256_file,
)
from pvz_env import PvZEnv, training_task
from pvz_seed_sets import DEFAULT_DEV_SEEDS, DEFAULT_TEST_SEEDS, read_seed_set
from pvz_value import DISCOUNT_REFERENCE_TICKS, SEARCH_LABEL_VERSION, VALUE_GAMMA, VALUE_SEMANTICS

LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)
ROOT = Path(__file__).resolve().parent.parent


def collect_episode(model: GameplayModelV1, env: PvZEnv, seed: int,
                    max_actions: int, replay_path: Path, level: int = LEVEL,
                    deck: tuple[int, ...] = DECK, zombie_count_multiplier: float = 1.0) -> dict[str, Any]:
    task = training_task(seed, level, zombie_count_multiplier)
    observation, _ = env.reset(deck=deck, task=task)
    hidden = None
    previous_action = None
    elapsed_since_previous_observation = 0
    events: dict[str, Any] = {}
    transitions = []
    started = time.perf_counter()
    for decision_index in range(max_actions):
        with torch.no_grad():
            output = model.step(observation, hidden, previous_action,
                                elapsed_since_previous_observation, events)
            action, log_prob, _ = select_action(model, output, observation)
        transition = {
            "decision_index": decision_index,
            "observation": observation,
            "previous_action": previous_action,
            "elapsed_since_previous_observation": elapsed_since_previous_observation,
            "events": events,
            "action": action,
            "log_prob": float(log_prob.item()),
            "value": float(output["value"].item()),
        }
        transitions.append(transition)
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"model selected an illegal action on seed {seed}: {action}")
        hidden = output["hidden"]
        previous_action = action
        transition["action_duration_ticks"] = info["ticks_advanced"]
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
    return {
        "seed": seed,
        "replay_id": replay_path.name,
        "won": won,
        "result": observation["result"],
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "tick": observation["tick"],
        "seconds": time.perf_counter() - started,
        "transitions": transitions,
    }


def add_advantages(episodes: list[dict[str, Any]], gae_lambda: float) -> None:
    for episode in episodes:
        transitions = episode["transitions"]
        advantage = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            transition = transitions[index]
            duration_ratio = transition["action_duration_ticks"] / DISCOUNT_REFERENCE_TICKS
            discount = VALUE_GAMMA ** duration_ratio
            trace_discount = discount * (gae_lambda ** duration_ratio)
            next_value = transitions[index + 1]["value"] if index + 1 < len(transitions) else 0.0
            end_of_step_reward = discount * transition["reward"]
            delta = end_of_step_reward + discount * next_value - transition["value"]
            advantage = delta + trace_discount * advantage
            transition["advantage"] = advantage
            transition["return"] = advantage + transition["value"]


def _rebuild_hidden(model: GameplayModelV1, transitions: list[dict[str, Any]], start: int) -> torch.Tensor | None:
    hidden = None
    if start <= 0:
        return hidden
    with torch.no_grad():
        for transition in transitions[:start]:
            output = model.step(transition["observation"], hidden, transition["previous_action"],
                                transition["elapsed_since_previous_observation"], transition["events"])
            hidden = output["hidden"]
    return None if hidden is None else hidden.detach()


def train_update(model: GameplayModelV1, episodes: list[dict[str, Any]], optimizer: torch.optim.Optimizer,
                 device: torch.device, ppo_epochs: int, sequence_length: int,
                 clip_epsilon: float, value_coefficient: float, entropy_coefficient: float) -> dict[str, float]:
    advantages = torch.tensor([
        transition["advantage"] for episode in episodes for transition in episode["transitions"]
    ], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    offset = 0
    for episode in episodes:
        for transition in episode["transitions"]:
            transition["normalized_advantage"] = advantages[offset]
            offset += 1

    chunks = [
        (episode, start, min(start + sequence_length, len(episode["transitions"])))
        for episode in episodes
        for start in range(0, len(episode["transitions"]), sequence_length)
    ]
    policy_losses, value_losses, entropies = [], [], []
    model.train()
    for _ in range(ppo_epochs):
        random.shuffle(chunks)
        for episode, start, end in chunks:
            transitions = episode["transitions"]
            hidden = _rebuild_hidden(model, transitions, start)
            log_probs, values, entropies_for_chunk = [], [], []
            for transition in transitions[start:end]:
                output = model.step(transition["observation"], hidden, transition["previous_action"],
                                    transition["elapsed_since_previous_observation"], transition["events"])
                _, log_prob, entropy = select_action(
                    model, output, transition["observation"], action=transition["action"]
                )
                hidden = output["hidden"]
                log_probs.append(log_prob)
                values.append(output["value"].squeeze())
                entropies_for_chunk.append(entropy)
            new_log_prob = torch.stack(log_probs)
            old_log_prob = torch.tensor(
                [transition["log_prob"] for transition in transitions[start:end]],
                dtype=torch.float32,
                device=device,
            )
            advantage = torch.stack([
                transition["normalized_advantage"] for transition in transitions[start:end]
            ])
            ratio = torch.exp(new_log_prob - old_log_prob)
            policy_loss = -torch.minimum(
                ratio * advantage,
                ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage,
            ).mean()
            returns = torch.tensor(
                [transition["return"] for transition in transitions[start:end]],
                dtype=torch.float32,
                device=device,
            )
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
    return {
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "entropy": sum(entropies) / len(entropies),
    }


def episode_hash(episode: dict[str, Any]) -> str:
    steps = [{
        key: transition[key]
        for key in (
            "observation", "previous_action", "elapsed_since_previous_observation",
            "action_duration_ticks", "events", "action", "log_prob", "value", "reward",
        )
    } for transition in episode["transitions"]]
    return canonical_digest({"seed": episode["seed"], "steps": steps, "result": episode["result"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--init-checkpoint", type=Path,
                        default=ROOT / "artifacts" / "adventure2_level7" / "gameplay_model_v1.pt")
    parser.add_argument("--dev-seeds", type=Path, default=DEFAULT_DEV_SEEDS)
    parser.add_argument("--test-seeds", type=Path, default=DEFAULT_TEST_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "adventure2_level7")
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=lambda value: tuple(map(int, value.split(","))), default=DECK)
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
        help="auto = cuda if available else cpu. MPS is never chosen automatically: every "
             "model call here is a batch-of-1 forward pass, where MPS measured 1.71x slower "
             "end to end. Pass --device mps explicitly to opt in.",
    )
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--train-seed-end", type=int, default=10000)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if args.updates < 1 or args.rollout_episodes < 1 or args.ppo_epochs < 1 or args.sequence_length < 1:
        parser.error("updates, rollout episodes, PPO epochs, and sequence length must be positive")
    if not 0.0 < args.gae_lambda <= 1.0:
        parser.error("--gae-lambda must be in (0, 1]")
    if args.train_seed_start >= args.train_seed_end:
        parser.error("training seed range must be nonempty")
    if not 1.0 <= args.zombie_count_multiplier <= 10.0:
        parser.error("--zombie-count-multiplier must be from 1 to 10")

    development_seeds = set(read_seed_set(args.dev_seeds, args.level, "development"))
    final_test_seeds = set(read_seed_set(args.test_seeds, args.level, "final_test"))
    frozen_seeds = development_seeds | final_test_seeds
    if development_seeds & final_test_seeds:
        parser.error("development and final-test seed sets overlap")
    if any(args.train_seed_start <= seed < args.train_seed_end for seed in frozen_seeds):
        parser.error("training seed range overlaps a frozen development/final-test seed")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    initial_checkpoint_sha = sha256_file(args.init_checkpoint)
    trajectory_dir = args.init_checkpoint.expanduser().resolve().parent
    initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    provenance = initial["provenance"]
    if (initial["model_architecture_version"] != MODEL_ARCHITECTURE_VERSION
            or initial.get("value_semantics") != VALUE_SEMANTICS
            or provenance.get("protocol_version") != ENV_PROTOCOL_VERSION
            or provenance.get("search_label_version") != SEARCH_LABEL_VERSION
            or provenance["observation_version"] != OBSERVATION_VERSION
            or provenance["task_version"] != TASK_VERSION):
        raise ValueError("initial checkpoint does not match the current model/search semantics")
    model = GameplayModelV1().to(device)
    model.load_state_dict(initial["state_dict"])
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gameplay_model_v1_ppo.pt"
    summary_path = args.output_dir / "ppo_training_summary.json"
    revision, dirty = git_metadata(ROOT)
    resource_dir = Path(args.resource_dir).expanduser().resolve()
    resource_hashes = {
        "main.pak": sha256_file(resource_dir / "main.pak"),
        "properties/partner.xml": sha256_file(resource_dir / "properties" / "partner.xml"),
    }
    trajectory_hashes: dict[str, str] = {}
    history = []
    config = {key: value for key, value in vars(args).items()}
    config.update({
        "resource_dir": str(resource_dir),
        "init_checkpoint": str(args.init_checkpoint),
        "dev_seeds": str(args.dev_seeds),
        "test_seeds": str(args.test_seeds),
        "output_dir": str(args.output_dir),
        "discount_reference_ticks": DISCOUNT_REFERENCE_TICKS,
        "value_gamma": VALUE_GAMMA,
        "value_semantics": VALUE_SEMANTICS,
        "resolved_device": str(device),
    })
    train_seeds = list(range(args.train_seed_start, args.train_seed_end))
    with PvZEnv(resource_dir=resource_dir) as env:
        for update in range(1, args.updates + 1):
            seeds = random.sample(train_seeds, args.rollout_episodes)
            episodes = [
                collect_episode(
                    model,
                    env,
                    seed,
                    args.max_actions,
                    args.output_dir / "replays" / f"ppo_update_{update}_seed_{seed}.jsonl.gz",
                    args.level,
                    args.deck,
                    args.zombie_count_multiplier,
                )
                for seed in seeds
            ]
            add_advantages(episodes, args.gae_lambda)
            losses = train_update(
                model, episodes, optimizer, device, args.ppo_epochs,
                args.sequence_length, args.clip_epsilon,
                args.value_coefficient, args.entropy_coefficient,
            )
            hashes = {f"{update}:{episode['seed']}": episode_hash(episode) for episode in episodes}
            trajectory_hashes.update(hashes)
            row = {
                "update": update,
                "seeds": seeds,
                "wins": sum(episode["won"] for episode in episodes),
                "episodes": [{
                    "seed": episode["seed"],
                    "replay_id": episode["replay_id"],
                    "won": episode["won"],
                    "wave": episode["wave"],
                    "actions": len(episode["transitions"]),
                    "seconds": round(episode["seconds"], 3),
                    "sha256": hashes[f"{update}:{episode['seed']}"],
                } for episode in episodes],
                "losses": losses,
            }
            history.append(row)
            if not env.episode:
                raise RuntimeError("environment did not record resource provenance")
            provenance = {
                "git_sha": revision,
                "git_dirty": dirty,
                "protocol_version": ENV_PROTOCOL_VERSION,
                "command": {"argv": sys.argv, "arguments": config},
                "trajectory_sha256": {
                    "search": sha256_file(trajectory_dir / "search_trajectories.json.gz"),
                    "dagger_search": sha256_file(trajectory_dir / "dagger_search_trajectories.json.gz"),
                    "ppo_rollouts_by_seed": trajectory_hashes,
                },
                "resource_sha256": resource_hashes,
                "frozen_development_seed_sha256": sha256_file(args.dev_seeds),
                "frozen_final_test_seed_sha256": sha256_file(args.test_seeds),
                "initial_checkpoint_sha256": initial_checkpoint_sha,
                "random_seeds": {
                    "python_torch": args.seed,
                    "training_seed_range": [args.train_seed_start, args.train_seed_end - 1],
                },
                "model_config": MODEL_CONFIG,
                "observation_version": OBSERVATION_VERSION,
                "task_version": TASK_VERSION,
                "search_label_version": SEARCH_LABEL_VERSION,
                "value_semantics": VALUE_SEMANTICS,
            }
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
                "value_semantics": VALUE_SEMANTICS,
                "config": MODEL_CONFIG,
                "level": args.level,
                "deck": args.deck,
                "profile": "Adventure-II, six slots, no store items",
                "update": update,
                "ppo_config": config,
                "losses": losses,
                "provenance": provenance,
            }, checkpoint_path)
            summary = {
                "checkpoint": checkpoint_path.name,
                "provenance": provenance,
                "initial_checkpoint": str(args.init_checkpoint),
                "updates": history,
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(
                f"update={update}/{args.updates} wins={row['wins']}/{len(episodes)} "
                f"policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f} "
                f"entropy={losses['entropy']:.3f}",
                flush=True,
            )
    print(f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
