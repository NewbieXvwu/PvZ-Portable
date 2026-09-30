"""The PPO update and the task-family episode collector.

This module is the library half of the T5 trainer: ``train_pvz_ppo_task_family``
imports :func:`collect_task_episode` and :func:`train_update` from here.  It used to
also carry its own single-level Adventure-II entry point, and a ``collect_episode``
that fed it; both went away with the rest of the T4 search-teacher pipeline.
"""

from __future__ import annotations

import hashlib
import random
import struct
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from pvz_agent_model import (FLEX_ATTENTION_AVAILABLE, GameplayModelV1, legal_summary,
                             observation_tokens, pack_tokens, replay_log_probs, select_action)
from pvz_common import canonical_digest
from pvz_env import PvZEnv, TaskSpec
from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA


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
    # ``CRITIC_INPUTS`` returns the current wave's zombie roster, which is fixed for
    # the whole wave (measured: 272 calls across a 5-wave level produced 4 distinct
    # payloads, i.e. 1.5%).  The other field it returns, ``wave_timer``, is the same
    # ``mZombieCountDown`` the public observation already carries -- verified equal on
    # 120/120 decisions -- so it comes from ``observation`` instead of a second round
    # trip.  One request per wave instead of one per decision.
    wave_rosters: dict[int, list[int]] = {}
    for decision_index in range(max_actions):
        wave = observation["wave"]
        critic_started = time.perf_counter()
        roster = wave_rosters.get(wave)
        if roster is None:
            roster = env.critic_inputs(wave)["wave_zombies"]
            wave_rosters[wave] = roster
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
                observation["wave_timer"], roster)
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
                 attention_backend: str = "auto",
                 label: str | None = None) -> dict[str, float]:
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
    # ``auto`` resolves to the dense path, including on CUDA.  This used to pick
    # FlexAttention whenever CUDA was available, on the strength of a recorded
    # sweep that measured one attention layer at 71.69 ms dense against 18.92 ms
    # flex.  Re-measured on the same RTX 5080 (PPO_UPDATE_ANATOMY.md §10): that
    # dense number described the relation-bias assembly *before* it was compiled,
    # and the current dense path is 8.25 ms against flex's 22.40 ms.  A 512-episode
    # update agrees: 81.7 ms per optimizer step dense against 125.8 ms flex, 1.54x.
    #
    # The mechanism is in the backward pass, not the forward: FlexAttention has to
    # reduce 14.2M score-element gradients into 12 / 72 / 108 / 600 table entries,
    # and the dense path lets ``torch.compile`` do that in one kernel instead
    # (906 ms eager against 5.1 ms compiled).  FlexAttention stays available as an
    # explicit ``--attention-backend flex`` so the measurement can be redone.
    use_flex = (device.type == "cuda" and attention_backend == "flex"
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
    for epoch in range(ppo_epochs):
        # A 2,000-episode update is minutes of GPU time with no output at all.  One
        # line per epoch turns a silent block into a countable one; ``label`` is
        # optional so library callers (and tests) stay quiet.
        if label is not None:
            print(f"{label} ppo epoch {epoch + 1}/{ppo_epochs}", flush=True)
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

    ``episode_digest`` is what the T5 trainer uses.  This one survives only as the
    baseline that ``scripts/training_hotspot_profile.py`` measures it against:
    13.9 ms/episode versus 1.75 ms/episode, an 8.0x saving (5.3x on a shorter
    episode sample); an earlier note claiming 45x was never reproducible.
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


