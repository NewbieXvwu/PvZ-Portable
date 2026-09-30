"""Train a recurrent PPO policy on Adventure-II from a DAgger checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pvz_agent_model import (FLEX_ATTENTION_AVAILABLE, GameplayModelV1, MODEL_ARCHITECTURE_VERSION, MODEL_CONFIG,
                             configure_torch_threads, legal_summary, observation_tokens,
                             pack_tokens, replay_log_probs, resolve_device, select_action)
from pvz_common import (
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    TASK_VERSION,
    canonical_digest,
    git_metadata,
    sha256_file,
)
from pvz_env import PvZEnv, TaskSpec, training_task
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


def _task_spec(task: dict[str, Any], seed: int) -> TaskSpec:
    return TaskSpec(
        level=task["level"], seed=seed, playthrough=task["playthrough"],
        zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
        preplanted=tuple(tuple(plant) for plant in task["preplanted"]),
    )


def potential(observation: dict[str, Any]) -> float:
    if observation["terminal"]:
        return 0.0
    sun = min(1.0, max(0.0, observation["sun"] / 1000.0))
    progress = min(1.0, max(0.0, observation["wave"] / max(1, observation["wave_count"])))
    health = [
        min(1.0, max(0.0, plant.get("health", 0) / max(1, plant.get("max_health", 1))))
        for plant in observation["plants"]
        if not plant.get("squished") and plant.get("health", 0) > 0
    ]
    plant_health = sum(health) / len(health) if health else 0.0
    return (sun + progress + plant_health) / 3.0


def collect_task_episode(model: GameplayModelV1, env: PvZEnv, task: dict[str, Any],
                         environment_seed: int, job_id: int, max_actions: int) -> dict[str, Any]:
    started = time.perf_counter()
    observation, _ = env.reset(deck=task["deck"], task=_task_spec(task, environment_seed))
    reset_seconds = time.perf_counter() - started
    hidden = None
    previous_action = None
    elapsed_since_previous_observation = 0
    events: dict[str, Any] = {}
    transitions = []
    model_seconds = 0.0
    environment_seconds = reset_seconds
    critic_inputs_seconds = 0.0
    tokenization_seconds = 0.0
    for decision_index in range(max_actions):
        wave = observation["wave"]
        critic_started = time.perf_counter()
        critic_inputs = env.critic_inputs(wave)
        critic_inputs_seconds += time.perf_counter() - critic_started
        tokenize_started = time.perf_counter()
        tensors, metadata = observation_tokens(observation)
        packed = pack_tokens(tensors, metadata)
        legal = legal_summary(observation["legal_actions"])
        tokenization_seconds += time.perf_counter() - tokenize_started
        model_started = time.perf_counter()
        with torch.no_grad():
            output = model.step_tokens(tensors, metadata, wave, hidden, previous_action,
                                       elapsed_since_previous_observation, events)
            action, log_prob, _ = select_action(model, output, legal)
            critic_extra = model.privileged_extra_from_inputs(
                critic_inputs["wave_timer"], critic_inputs["wave_zombies"])
            value = model.privileged_value_from_extra(output, critic_extra)
        model_seconds += time.perf_counter() - model_started
        current_potential = potential(observation)
        transition = {
            "decision_index": decision_index,
            "tokens": packed,
            "wave": wave,
            "legal": legal,
            "previous_action": previous_action,
            "elapsed_since_previous_observation": elapsed_since_previous_observation,
            "events": events,
            "critic_extra": critic_extra,
            "action": action,
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "potential": current_potential,
        }
        transitions.append(transition)
        environment_started = time.perf_counter()
        observation, _, done, _, info = env.step(action)
        environment_seconds += time.perf_counter() - environment_started
        if not info.get("ok"):
            raise RuntimeError(f"model selected an illegal action on seed {environment_seed}: {action}")
        hidden = output["hidden"]
        previous_action = action
        transition["action_duration_ticks"] = info["ticks_advanced"]
        elapsed_since_previous_observation = transition["action_duration_ticks"]
        events = info["events"]
        duration_ratio = transition["action_duration_ticks"] / DISCOUNT_REFERENCE_TICKS
        discount = VALUE_GAMMA ** duration_ratio
        next_potential = potential(observation)
        shaping_reward = discount * next_potential - current_potential
        transition["shaping_reward"] = shaping_reward
        transition["reward"] = shaping_reward
        transition["terminal_outcome"] = 0.0
        if done:
            transition["terminal_outcome"] = 1.0 if observation["result"] == 1 else -1.0
            transition["reward"] += transition["terminal_outcome"]
            break
    if not observation["terminal"]:
        raise RuntimeError(f"PPO episode exceeded {max_actions} decisions on seed {environment_seed}")
    elapsed = time.perf_counter() - started
    return {
        "seed": job_id,
        "task_seed": environment_seed,
        "task_id": task["task_id"],
        "won": observation["result"] == 1,
        "result": observation["result"],
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "tick": observation["tick"],
        "seconds": elapsed,
        "profile_seconds": {
            "model": model_seconds,
            "environment": environment_seconds,
            "critic_inputs": critic_inputs_seconds,
            "tokenization": tokenization_seconds,
        },
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
            delta = transition["reward"] + discount * next_value - transition["value"]
            advantage = delta + trace_discount * advantage
            transition["advantage"] = advantage
            transition["return"] = advantage + transition["value"]



def train_update(model: GameplayModelV1, episodes: list[dict[str, Any]], optimizer: torch.optim.Optimizer,
                 device: torch.device, ppo_epochs: int, sequence_length: int,
                 clip_epsilon: float, value_coefficient: float, entropy_coefficient: float,
                 minibatch_chunks: int = 1,
                 attention_backend: str = "auto") -> dict[str, float]:
    """Layered-batch PPO update.

    Semantics CHANGE vs the previous per-chunk loop: chunks at the same position
    across episodes form a layer, a layer runs as ONE batched forward (hidden
    flows along layers, replacing per-chunk prefix rebuilds), and the optimizer
    steps once per minibatch of chunks instead of once per chunk.  Effective
    batch grows by ~minibatch_chunks, so the learning rate must be retuned
    (default raised accordingly); the stage-0 learning-signal gate is the
    arbiter of whether the new configuration actually learns.
    """
    advantages = torch.tensor([
        transition["advantage"] for episode in episodes for transition in episode["transitions"]
    ], dtype=torch.float32, device=device)
    advantage_mean = float(advantages.mean().item())
    advantage_std = float(advantages.std(unbiased=False).item())
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    offset = 0
    for episode in episodes:
        for transition in episode["transitions"]:
            transition["normalized_advantage"] = advantages[offset]
            offset += 1

    if attention_backend not in ("auto", "dense", "flex"):
        raise ValueError(f"unsupported attention backend: {attention_backend}")
    if attention_backend == "flex" and (device.type != "cuda" or not FLEX_ATTENTION_AVAILABLE):
        raise ValueError("FlexAttention updates require CUDA and a supported PyTorch build")
    use_flex = (device.type == "cuda" and attention_backend != "dense"
                and FLEX_ATTENTION_AVAILABLE)
    for layer in model.encoder:
        layer.attention.use_flex_attention = use_flex

    # cut chunks and group them by position across episodes; each chunk knows
    # the key of its predecessor (same episode, previous position) so hidden
    # can flow layer by layer instead of rebuilding prefixes per chunk
    layers: list[list[tuple[int, dict[str, Any], int, int]]] = []
    previous_key: dict[int, int | None] = {}
    key_counter = 0
    for episode in episodes:
        transitions = episode["transitions"]
        prev_key = None
        for position, start in enumerate(range(0, len(transitions), sequence_length)):
            end = min(start + sequence_length, len(transitions))
            while len(layers) <= position:
                layers.append([])
            key_counter += 1
            key = key_counter
            previous_key[key] = prev_key
            layers[position].append((key, episode, start, end))
            prev_key = key

    policy_losses, value_losses, entropies = [], [], []
    model.train()
    for _ in range(ppo_epochs):
        # hidden chains are recomputed every epoch: parameters may have moved
        hidden_by_key: dict[int, torch.Tensor | None] = {
            key: None for key in previous_key
        }
        for layer in layers:
            random.shuffle(layer)
            for batch_start in range(0, len(layer), minibatch_chunks):
                batch = layer[batch_start:batch_start + minibatch_chunks]
                sequences = [episode["transitions"][start:end] for _, episode, start, end in batch]
                hiddens = [hidden_by_key[key] for key, _, _, _ in batch]
                outputs, hidden_out = model.forward_sequences(sequences, hiddens)
                flat_transitions = [transition for sequence in sequences for transition in sequence]
                log_probs, entropies_for_chunk = replay_log_probs(model, outputs, flat_transitions)
                belief = torch.cat([output["belief"] for output in outputs], dim=0)
                extras = torch.tensor([transition["critic_extra"] for transition in flat_transitions],
                                      dtype=torch.float32, device=device)
                values = model.privileged_value_batch(belief, extras).squeeze(-1)
                new_log_prob = log_probs
                old_log_prob = torch.tensor(
                    [transition["log_prob"] for transition in flat_transitions],
                    dtype=torch.float32,
                    device=device,
                )
                advantage = torch.stack([
                    transition["normalized_advantage"] for transition in flat_transitions
                ])
                ratio = torch.exp(new_log_prob - old_log_prob)
                policy_loss = -torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage,
                ).mean()
                returns = torch.tensor(
                    [transition["return"] for transition in flat_transitions],
                    dtype=torch.float32,
                    device=device,
                )
                value_loss = F.mse_loss(values, returns)
                entropy = entropies_for_chunk.mean()
                loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("PPO gradient norm became non-finite")
                optimizer.step()
                policy_losses.append(float(policy_loss.detach().item()))
                value_losses.append(float(value_loss.detach().item()))
                entropies.append(float(entropy.detach().item()))
                for b, (key, _, _, _) in enumerate(batch):
                    hidden_by_key[key] = hidden_out[:, b, :].detach()
    model.eval()
    return {
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "entropy": sum(entropies) / len(entropies),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "gradient_norm": float(gradient_norm.detach().item()),
    }


def _jsonable(value: Any) -> Any:
    """numpy arrays (packed tokens) are not JSON-serializable; lists are."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def episode_hash(episode: dict[str, Any]) -> str:
    """Legacy JSON-path digest; see :func:`episode_digest` for the packed-bytes one.

    Kept because ``artifacts/adventure2_level7/training_summary.json`` records the
    digests this function produced, and re-deriving them must stay possible.
    ``episode_digest`` is what the T5 trainer uses.  Measured on this machine it
    costs 13.9 ms/episode against 1.75 ms/episode, an 8.0x saving (5.3x on a
    shorter episode sample); an earlier note claiming 45x was never reproducible.
    """
    steps = [_jsonable({
        key: transition[key]
        for key in (
            "tokens", "wave", "legal", "previous_action", "elapsed_since_previous_observation",
            "action_duration_ticks", "events", "critic_extra", "action", "log_prob",
            "value", "potential", "shaping_reward", "terminal_outcome", "reward",
        ) if key in transition
    }) for transition in episode["transitions"]]
    return canonical_digest({
        "seed": episode["seed"], "task_seed": episode.get("task_seed"),
        "task_id": episode.get("task_id"), "steps": steps, "result": episode["result"],
    })


# The fields that identify an episode's trajectory, in a fixed order.  ``episode_hash``
# uses the same tuple; keep the two in step.
EPISODE_DIGEST_FIELDS = (
    "tokens", "wave", "legal", "previous_action", "elapsed_since_previous_observation",
    "action_duration_ticks", "events", "critic_extra", "action", "log_prob",
    "value", "potential", "shaping_reward", "terminal_outcome", "reward",
)

# Recorded alongside every digest so a reader can tell which algorithm produced the
# trajectory hashes without guessing from the digest length.
EPISODE_DIGEST_ALGORITHM = "blake2b-128 over tagged packed transition bytes (episode_digest)"


def _digest_into(hasher: "hashlib._Hash", value: Any) -> None:
    """Feed *value* into *hasher* as an unambiguous, canonical byte stream.

    Arrays go in through their raw buffer, so a packed ``int8``/``float16`` token
    block is never expanded into Python scalars -- that expansion was the entire cost
    of the JSON path (220,809 scalars per episode, 4.42e8 per 2,000-episode update).
    Every branch writes a type tag and, where the payload is variable length, a
    length, so distinct structures cannot collide.
    """
    if isinstance(value, np.ndarray):
        hasher.update(b"A" + value.dtype.str.encode("ascii") + b"\x00")
        hasher.update(",".join(str(size) for size in value.shape).encode("ascii") + b"\x00")
        hasher.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, np.generic):
        # ``tobytes`` keeps the width, so float32 1.0 and float64 1.0 stay distinct.
        hasher.update(b"G" + value.dtype.str.encode("ascii") + b"\x00" + value.tobytes())
        return
    if isinstance(value, dict):
        hasher.update(b"D" + str(len(value)).encode("ascii") + b"\x00")
        for key in sorted(value, key=str):
            _digest_into(hasher, str(key))
            _digest_into(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        # Lists and tuples digest alike, matching the JSON path where both became arrays.
        hasher.update(b"L" + str(len(value)).encode("ascii") + b"\x00")
        for item in value:
            _digest_into(hasher, item)
        return
    if isinstance(value, bool):
        hasher.update(b"T" if value else b"F")
        return
    if isinstance(value, int):
        hasher.update(b"I" + str(value).encode("ascii") + b"\x00")
        return
    if isinstance(value, float):
        hasher.update(b"R" + struct.pack("<d", value))
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(b"S" + str(len(encoded)).encode("ascii") + b"\x00" + encoded)
        return
    if value is None:
        hasher.update(b"N")
        return
    raise TypeError(f"episode digest cannot encode {type(value).__name__}")


def episode_digest(episode: dict[str, Any], *, digest_size: int = 16) -> str:
    """Hex blake2b digest of an episode's packed transition bytes.

    Equivalent in purpose to :func:`episode_hash` but cheaper, because packed
    observation tokens stay as raw bytes instead of being rebuilt as Python objects
    and JSON-encoded.  Measured 1.75 ms/episode against 13.9 ms/episode, an 8.0x
    saving, so ~3.5 s instead of ~27.9 s for a 2000-episode rollout batch.
    The digest values differ from ``episode_hash``, so
    ``trajectory_sha256`` recorded by one is not comparable with the other; the T5
    trainer records which function it used.
    """
    hasher = hashlib.blake2b(digest_size=digest_size)
    hasher.update(b"pvz-episode-digest-v1\x00")
    _digest_into(hasher, episode["seed"])
    _digest_into(hasher, episode.get("task_seed"))
    _digest_into(hasher, episode.get("task_id"))
    _digest_into(hasher, episode["result"])
    for transition in episode["transitions"]:
        hasher.update(b"|")
        for key in EPISODE_DIGEST_FIELDS:
            if key in transition:
                _digest_into(hasher, key)
                _digest_into(hasher, transition[key])
    return hasher.hexdigest()


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
    parser.add_argument(
        "--threads", type=int, default=0,
        help="CPU thread count for torch; 0 selects the measured default (4). This is a "
             "reproducibility knob, not a correctness one: thread count perturbs the low "
             "order bits (<=6e-7 relative) but the decision error budget is 1e-5..1e-4. "
             "Pass --threads 1 only to reproduce artifacts from an older run.",
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
    configure_torch_threads(args.threads)
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
