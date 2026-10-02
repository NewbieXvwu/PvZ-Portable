"""Finite decision history with vectorized feedforward prediction.

Stored history contains projected public inputs, never an implicit recurrent
state. PPO re-encodes the necessary leading frames with current parameters.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


SHORT_FIELDS = {"history_decisions", "projection_width", "mlp_width", "mlp_layers"}


def validate_short_config(config: dict[str, Any]) -> None:
    settings = config["short_history"]
    if (not isinstance(settings, dict) or set(settings) != SHORT_FIELDS
            or any(type(value) is not int or value < 1 for value in settings.values())):
        raise ValueError("short_history requires explicit positive integer dimensions")
    if "slow_memory" in config:
        raise ValueError("first short-history candidate cannot also contain slow_memory")
    if config.get("wait_mask") != "progress_v1" or config.get("input_flags") != 7:
        raise ValueError("short_history requires repaired input7/progress_v1")


class ShortHistoryMemory(nn.Module):
    def __init__(self, input_width: int, output_width: int, settings: dict[str, int]):
        super().__init__()
        self.history_decisions = settings["history_decisions"]
        self.projection_width = settings["projection_width"]
        self.history_projection = nn.Sequential(nn.Linear(input_width, self.projection_width), nn.SiLU())
        layers: list[nn.Module] = []
        width = self.history_decisions * self.projection_width
        for _ in range(settings["mlp_layers"]):
            layers.extend((nn.Linear(width, settings["mlp_width"]), nn.SiLU()))
            width = settings["mlp_width"]
        layers.extend((nn.Linear(width, output_width), nn.Tanh()))
        self.prediction = nn.Sequential(*layers)

    def forward_sequences(self, inputs: Tensor, lengths: list[int],
                          histories: list[Tensor | None]) -> tuple[Tensor, Tensor]:
        projected = self.history_projection(inputs)
        beliefs, next_histories = [], []
        offset, retained = 0, self.history_decisions - 1
        for length, history in zip(lengths, histories, strict=True):
            if length < 1:
                raise ValueError("short history requires nonempty sequences")
            if history is None:
                history = projected.new_zeros(retained, self.projection_width)
            if history.shape != (retained, self.projection_width):
                raise ValueError("short-history cache shape differs from saved configuration")
            joined = torch.cat((history, projected[offset:offset + length]))
            windows = joined.unfold(0, self.history_decisions, 1).transpose(1, 2)
            beliefs.append(self.prediction(windows.reshape(length, -1)))
            next_histories.append((joined[-retained:] if retained else joined[:0]).unsqueeze(1))
            offset += length
        if offset != inputs.shape[0]:
            raise ValueError("short-history sequence lengths do not cover the input")
        return torch.cat(beliefs), torch.cat(next_histories, dim=1)
