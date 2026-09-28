"""Behavior cloning and auxiliary training for simulator-search trajectories."""

from __future__ import annotations

from typing import Any

import random
import torch
from torch.nn import functional as F

from pvz_agent_model import GameplayModelV1, hard_behavior_cloning_loss
from pvz_value import discounted_terminal_value

AUXILIARY_LOSS_WEIGHT = 0.05
LANE_COUNT = 6


def episode_targets(steps: list[dict[str, Any]], index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    observation = steps[index]["observation"]
    lanes = torch.zeros(1, LANE_COUNT, device=device)
    for zombie in observation["zombies"]:
        row = int(zombie["row"])
        if 0 <= row < LANE_COUNT:
            lanes[0, row] = 1
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
            terminal_tick = int(episode["tick"])
            hidden = None
            for start in range(0, len(steps), 64):
                losses = []
                for index in range(start, min(start + 64, len(steps))):
                    step = steps[index]
                    previous = step.get("previous_action", steps[index - 1]["action"] if index else None)
                    output = model.step(step["observation"], hidden, previous, step["delta_ticks"], step["events"])
                    hidden = output["hidden"]
                    loss = hard_behavior_cloning_loss(model, output, step["observation"], step["action"], plant_weight)
                    lanes, next_wave = episode_targets(steps, index, device)
                    won = bool(episode["won"])
                    remaining_ticks = max(0, terminal_tick - int(step["observation"]["tick"]))
                    value_target = discounted_terminal_value(won, remaining_ticks)
                    outcome_class = 1 if won else 0
                    loss = loss + AUXILIARY_LOSS_WEIGHT * F.mse_loss(
                        output["value"], torch.full_like(output["value"], value_target))
                    loss = loss + AUXILIARY_LOSS_WEIGHT * F.cross_entropy(
                        output["aux_outcome"], torch.full((1,), outcome_class, dtype=torch.long, device=device))
                    loss = loss + AUXILIARY_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(
                        output["aux_lane_threat"], lanes)
                    loss = loss + AUXILIARY_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(
                        output["aux_next_spawn"].view(1), next_wave)
                    losses.append(loss)
                optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if hidden is not None:
                    hidden = hidden.detach()
                total_loss += float(torch.stack([loss.detach() for loss in losses]).sum().item())
                total_steps += len(losses)
        epoch_loss = total_loss / max(total_steps, 1)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}", flush=True)
    return history, plant_weight
