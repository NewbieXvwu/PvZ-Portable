"""Independent value model used only by simulator search."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pvz_common import ENV_PROTOCOL_VERSION
from pvz_search_candidates import lane_pressure
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS, discounted_terminal_value

SEARCH_VALUE_VERSION = 2
SEARCH_VALUE_FEATURE_VERSION = 2

PLANT_TYPE_COUNT = 49
ZOMBIE_TYPE_COUNT = 100
MAX_SEED_SLOTS = 10

# One extra slot per block so an unmodelled type is *visible* to the value model
# instead of being silently indistinguishable from "nothing here".
UNKNOWN_PLANT_SLOT = PLANT_TYPE_COUNT
UNKNOWN_ZOMBIE_SLOT = ZOMBIE_TYPE_COUNT
PLANT_SLOT_COUNT = PLANT_TYPE_COUNT + 1
ZOMBIE_SLOT_COUNT = ZOMBIE_TYPE_COUNT + 1

ROW_COUNT = 6
COL_COUNT = 9
COL_GROUPS = 3

GLOBAL_FEATURES = 16 + ROW_COUNT * 8
PLANT_BLOCK_FEATURES = ROW_COUNT * COL_GROUPS * PLANT_SLOT_COUNT * 2
ZOMBIE_BLOCK_FEATURES = ROW_COUNT * ZOMBIE_SLOT_COUNT * 2
PACKET_WIDTH = PLANT_SLOT_COUNT * 2 + 4
PACKET_BLOCK_FEATURES = MAX_SEED_SLOTS * PACKET_WIDTH
SEARCH_VALUE_FEATURES = (
    GLOBAL_FEATURES + PLANT_BLOCK_FEATURES + ZOMBIE_BLOCK_FEATURES + PACKET_BLOCK_FEATURES
)

# The two halves of each spatial block: occupancy counts first, then intensity.
PLANT_BLOCK_HALF = PLANT_BLOCK_FEATURES // 2
ZOMBIE_BLOCK_HALF = ZOMBIE_BLOCK_FEATURES // 2


def _slot(raw: Any, count: int, unknown: int) -> int:
    """Map a raw type id to a feature slot, routing unmodelled values to *unknown*."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return unknown
    return value if 0 <= value < count else unknown


def search_value_features(observation: dict[str, Any]) -> torch.Tensor:
    """Build the 4116-wide value-model input vector for *observation*.

    The accumulator is a ``float64`` numpy array rather than a Python list of
    floats.  Two properties make that substitution free of any semantic drift:

    * Python ``float`` *is* IEEE-754 binary64, so every value written into the
      accumulator is represented exactly as it was before, and numpy's
      ``arr[i] += x`` performs the same round-to-nearest-even binary64 addition
      that ``list[i] += x`` performed.
    * The conversion to ``float32`` still happens exactly once, at the end, on
      the same binary64 inputs.

    ``torch.tensor(list_of_floats, dtype=float32)`` and
    ``torch.from_numpy(arr.astype(float32))`` were verified to agree bit-for-bit
    on 2.57M adversarial doubles (denormals, float32 tie-to-even boundaries,
    magnitudes at both ends of the range); see
    ``test_search_value_feature_equivalence.py``.
    """
    plants = [plant for plant in observation["plants"]
              if not plant.get("squished") and plant.get("health", 0) > 0]
    zombies = [zombie for zombie in observation["zombies"]
               if zombie.get("body_health", 0) > 0 or zombie.get("helm_health", 0) > 0
               or zombie.get("shield_health", 0) > 0]
    projectiles = observation.get("projectiles", [])
    packets = observation.get("packets", [])
    pressure = lane_pressure(observation)
    rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"] > 0})

    # Preallocate the exact width so the vector can never come out short.
    features = np.zeros(SEARCH_VALUE_FEATURES, dtype=np.float64)
    features[0:16] = [
        float(observation["sun"]) / 1000.0,
        float(observation["wave"]) / max(1.0, float(observation["wave_count"])),
        float(observation["tick"]) / 60000.0,
        float(observation.get("zombie_count_multiplier", 1.0)) / 10.0,
        float(bool(observation.get("night"))),
        float(bool(observation.get("pool"))),
        float(bool(observation.get("fog"))),
        float(bool(observation.get("roof"))),
        len(plants) / 40.0,
        len(zombies) / 40.0,
        len(projectiles) / 50.0,
        sum(bool(packet.get("active")) for packet in packets) / 10.0,
        sum(float(packet.get("cooldown", 0)) for packet in packets) / max(1.0, 3000.0 * len(packets)),
        sum(float(packet.get("cost", 0)) for packet in packets) / max(1.0, 500.0 * len(packets)),
        sum(defense.get("state") == 1 for defense in observation.get("defenses", [])) / max(1.0, float(len(rows))),
        1.0,
    ]

    cursor = 16
    for row in range(ROW_COUNT):
        row_zombies = [zombie for zombie in zombies if zombie.get("row") == row]
        row_plants = [plant for plant in plants if plant.get("row") == row]
        nearest = min((float(zombie.get("x", 900.0)) for zombie in row_zombies), default=900.0)
        zombie_hp = sum(
            max(0.0, float(zombie.get("body_health", 0)))
            + max(0.0, float(zombie.get("helm_health", 0)))
            + max(0.0, float(zombie.get("shield_health", 0)))
            for zombie in row_zombies
        )
        plant_health = sum(
            max(0.0, min(1.0, float(plant.get("health", 0)) / max(1.0, float(plant.get("max_health", 1)))))
            for plant in row_plants
        )
        features[cursor:cursor + 8] = [
            len(row_zombies) / 10.0,
            min(2.0, zombie_hp / 4000.0),
            min(2.0, pressure.get(row, 0.0) / 8.0),
            nearest / 900.0,
            len(row_plants) / 10.0,
            plant_health / 10.0,
            min((float(zombie.get("x", 900.0)) for zombie in row_zombies if zombie.get("is_eating")),
                default=900.0) / 900.0,
            sum(bool(zombie.get("is_eating")) for zombie in row_zombies) / 5.0,
        ]
        cursor += 8

    # Basic slicing yields *views*, so the sparse accumulations below land
    # directly in ``features`` without a copy-back step.
    plant_spatial = features[cursor:cursor + PLANT_BLOCK_HALF]
    plant_health_by_slot = features[cursor + PLANT_BLOCK_HALF:cursor + PLANT_BLOCK_FEATURES]
    cursor += PLANT_BLOCK_FEATURES
    for plant in plants:
        row = int(plant.get("row", -1))
        col = int(plant.get("col", -1))
        if not (0 <= row < ROW_COUNT and 0 <= col < COL_COUNT):
            continue
        index = (row * COL_GROUPS + col // COL_GROUPS) * PLANT_SLOT_COUNT + _slot(
            plant.get("type"), PLANT_TYPE_COUNT, UNKNOWN_PLANT_SLOT
        )
        plant_spatial[index] += 1.0 / COL_GROUPS
        plant_health_by_slot[index] += max(
            0.0, min(1.0, float(plant.get("health", 0)) / max(1.0, float(plant.get("max_health", 1))))
        ) / COL_GROUPS

    zombie_types = features[cursor:cursor + ZOMBIE_BLOCK_HALF]
    zombie_threat = features[cursor + ZOMBIE_BLOCK_HALF:cursor + ZOMBIE_BLOCK_FEATURES]
    cursor += ZOMBIE_BLOCK_FEATURES
    for zombie in zombies:
        row = int(zombie.get("row", -1))
        if not (0 <= row < ROW_COUNT):
            continue
        index = row * ZOMBIE_SLOT_COUNT + _slot(zombie.get("type"), ZOMBIE_TYPE_COUNT, UNKNOWN_ZOMBIE_SLOT)
        zombie_types[index] += 0.2
        urgency = max(0.0, min(1.5, (760.0 - float(zombie.get("x", 900.0))) / 520.0))
        zombie_threat[index] += urgency / 3.0

    packet_features = features[cursor:cursor + PACKET_BLOCK_FEATURES]
    cursor += PACKET_BLOCK_FEATURES
    for packet in packets:
        index = int(packet.get("index", -1))
        if not (0 <= index < MAX_SEED_SLOTS):
            continue
        base = index * PACKET_WIDTH
        packet_features[base + _slot(packet.get("type"), PLANT_TYPE_COUNT, UNKNOWN_PLANT_SLOT)] = 1.0
        imitater_type = packet.get("imitater_type")
        # -1 / None mean "this card is not an imitater"; anything >= 0 is a target type.
        if imitater_type is not None and int(imitater_type) >= 0:
            imitater_slot = _slot(imitater_type, PLANT_TYPE_COUNT, UNKNOWN_PLANT_SLOT)
            packet_features[base + PLANT_SLOT_COUNT + imitater_slot] = 1.0
        meta = base + PLANT_SLOT_COUNT * 2
        packet_features[meta] = float(bool(packet.get("active")))
        refresh = max(1.0, float(packet.get("refresh_time", 3000)))
        packet_features[meta + 1] = max(0.0, min(1.0, float(packet.get("cooldown", 0)) / refresh))
        packet_features[meta + 2] = min(2.0, refresh / 3000.0)
        packet_features[meta + 3] = min(2.0, float(packet.get("cost", 0)) / 500.0)

    if cursor != SEARCH_VALUE_FEATURES:
        raise RuntimeError(
            f"search value feature layout is inconsistent: wrote {cursor}, expected {SEARCH_VALUE_FEATURES}"
        )
    return torch.from_numpy(features.astype(np.float32))


class SearchValueModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(SEARCH_VALUE_FEATURES, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        # ``predict`` runs once per search leaf (~200 times per decision), and
        # ``next(self.parameters()).device`` costs ~7us because it builds a
        # generator plus a dict of named parameters.  A non-persistent buffer is
        # moved by ``Module.to`` exactly like the parameters are, so it tracks the
        # device for free -- and staying out of ``state_dict`` keeps existing
        # checkpoints loadable.
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)

    def device(self) -> torch.device:
        return self._device_anchor.device

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    @torch.inference_mode()
    def predict(self, observation: dict[str, Any]) -> float:
        value = self(search_value_features(observation).to(self.device()).unsqueeze(0))
        return float(value.item())


def _dataset(episodes: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[torch.Tensor] = []
    targets: list[float] = []
    for episode in episodes:
        terminal_tick = int(episode["tick"])
        won = bool(episode["won"])
        for step in episode["steps"]:
            observation = step["observation"]
            rows.append(search_value_features(observation))
            remaining = max(0, terminal_tick - int(observation["tick"]))
            targets.append(discounted_terminal_value(won, remaining))
    if not rows:
        raise ValueError("search-value training requires at least one state")
    return torch.stack(rows), torch.tensor(targets, dtype=torch.float32)


def train_search_value(
    model: SearchValueModel,
    episodes: list[dict[str, Any]],
    epochs: int,
    device: torch.device,
    batch_size: int = 256,
) -> list[float]:
    features, targets = _dataset(episodes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    history: list[float] = []
    indices = list(range(len(targets)))
    for _ in range(epochs):
        random.shuffle(indices)
        total = 0.0
        count = 0
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            prediction = model(features[batch].to(device))
            loss = F.mse_loss(prediction, targets[batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().item()) * len(batch)
            count += len(batch)
        history.append(total / max(count, 1))
    model.eval()
    return history


def save_search_value(path: Path, model: SearchValueModel, task_signature: dict[str, Any], **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "search_value_version": SEARCH_VALUE_VERSION,
        "feature_version": SEARCH_VALUE_FEATURE_VERSION,
        "protocol_version": ENV_PROTOCOL_VERSION,
        "search_label_version": SEARCH_LABEL_VERSION,
        "value_semantics": VALUE_SEMANTICS,
        "task_signature": task_signature,
        **metadata,
    }, path)


def load_search_value(path: Path, device: torch.device,
                      expected_task_signature: dict[str, Any]) -> tuple[SearchValueModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("search_value_version") != SEARCH_VALUE_VERSION:
        raise ValueError("search-value checkpoint version mismatch")
    if checkpoint.get("feature_version") != SEARCH_VALUE_FEATURE_VERSION:
        raise ValueError("search-value feature version mismatch")
    if checkpoint.get("protocol_version") != ENV_PROTOCOL_VERSION:
        raise ValueError("search-value protocol version mismatch")
    if checkpoint.get("search_label_version") != SEARCH_LABEL_VERSION:
        raise ValueError("search-value label version mismatch")
    if checkpoint.get("value_semantics") != VALUE_SEMANTICS:
        raise ValueError("search-value semantics mismatch")
    if checkpoint.get("task_signature") != expected_task_signature:
        raise ValueError("search-value task signature mismatch")
    model = SearchValueModel().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
