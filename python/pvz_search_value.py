"""Independent value model used only by simulator search."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from pvz_env import ENV_PROTOCOL_VERSION
from pvz_search_candidates import lane_pressure
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS, discounted_terminal_value

SEARCH_VALUE_VERSION = 1
SEARCH_VALUE_FEATURE_VERSION = 1
SEARCH_VALUE_FEATURES = 64


def search_value_features(observation: dict[str, Any]) -> torch.Tensor:
    plants = [plant for plant in observation["plants"]
              if not plant.get("squished") and plant.get("health", 0) > 0]
    zombies = [zombie for zombie in observation["zombies"]
               if zombie.get("body_health", 0) > 0 or zombie.get("helm_health", 0) > 0
               or zombie.get("shield_health", 0) > 0]
    projectiles = observation.get("projectiles", [])
    packets = observation.get("packets", [])
    pressure = lane_pressure(observation)
    rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"] > 0})
    features: list[float] = [
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
    for row in range(6):
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
        features.extend([
            len(row_zombies) / 10.0,
            min(2.0, zombie_hp / 4000.0),
            min(2.0, pressure.get(row, 0.0) / 8.0),
            nearest / 900.0,
            len(row_plants) / 10.0,
            plant_health / 10.0,
            min((float(zombie.get("x", 900.0)) for zombie in row_zombies if zombie.get("is_eating")),
                default=900.0) / 900.0,
            sum(bool(zombie.get("is_eating")) for zombie in row_zombies) / 5.0,
        ])
    if len(features) > SEARCH_VALUE_FEATURES:
        raise RuntimeError("search value feature vector exceeds configured width")
    features.extend([0.0] * (SEARCH_VALUE_FEATURES - len(features)))
    return torch.tensor(features, dtype=torch.float32)


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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    @torch.inference_mode()
    def predict(self, observation: dict[str, Any]) -> float:
        device = next(self.parameters()).device
        value = self(search_value_features(observation).to(device).unsqueeze(0))
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


def save_search_value(path: Path, model: SearchValueModel, **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "search_value_version": SEARCH_VALUE_VERSION,
        "feature_version": SEARCH_VALUE_FEATURE_VERSION,
        "protocol_version": ENV_PROTOCOL_VERSION,
        "search_label_version": SEARCH_LABEL_VERSION,
        "value_semantics": VALUE_SEMANTICS,
        **metadata,
    }, path)


def load_search_value(path: Path, device: torch.device) -> tuple[SearchValueModel, dict[str, Any]]:
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
    model = SearchValueModel().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
