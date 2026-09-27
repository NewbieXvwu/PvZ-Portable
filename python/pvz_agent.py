"""Structured recurrent GameplayModel-v0 policy and behavior-cloning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TOKEN_KINDS = {
    "global": 0,
    "profile": 1,
    "cell": 2,
    "plant": 3,
    "zombie": 4,
    "projectile": 5,
    "defense": 6,
    "grid_item": 7,
    "seed_packet": 8,
    "zombie_roster": 9,
}
WAIT_TICKS = (150, 300, 600, 1200, 2400)
MODEL_CONFIG = {"layers": 4, "width": 192, "heads": 6, "ff_width": 768, "gru_layers": 2, "gru_width": 256}
FEATURE_COUNT = 32


def _ratio(value: float, scale: float) -> float:
    return max(-2.0, min(2.0, float(value) / scale))


def observation_tokens(observation: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    kinds: list[int] = []
    categories: list[int] = []
    variants: list[int] = []
    features: list[list[float]] = []
    rows: list[int] = []
    cols: list[int] = []
    packet_tokens: dict[int, int] = {}
    cell_tokens: dict[int, int] = {}

    def add(kind: str, category: int = -1, variant: int = -1, values: tuple[float, ...] = (),
            row: int = -1, col: int = -1) -> int:
        index = len(kinds)
        kinds.append(TOKEN_KINDS[kind])
        categories.append(max(0, min(127, int(category) + 1)))
        variants.append(max(0, min(127, int(variant) + 1)))
        feature_row = [0.0] * FEATURE_COUNT
        feature_row[:min(FEATURE_COUNT, len(values))] = values[:FEATURE_COUNT]
        features.append(feature_row)
        rows.append(row)
        cols.append(col)
        return index

    plants = observation["plants"]
    zombies = observation["zombies"]
    projectiles = observation["projectiles"]
    add("global", values=(
        _ratio(observation["sun"], 1000), _ratio(observation["wave"], max(observation["wave_count"], 1)),
        _ratio(observation["tick"], 60000), _ratio(observation["wave_count"], 30),
        float(observation["night"]), float(observation["pool"]), float(observation["fog"]), float(observation["roof"]),
        _ratio(len(plants), 40), _ratio(len(zombies), 40), _ratio(len(projectiles), 50),
        float(observation["terminal"]), _ratio(observation["result"], 2),
    ))
    profile = observation["player_profile"]
    add("profile", values=(
        _ratio(profile["playthrough"], 2), _ratio(profile["seed_slot_count"], 10),
        _ratio(len(profile["owned_upgrade_plants"]), 8), float(profile["imitater_owned"]),
        float(profile["first_aid_owned"]), float(profile["pool_cleaner_owned"]),
        float(profile["roof_cleaner_owned"]), _ratio(profile["rake_charges_remaining"], 3),
    ))

    for cell in observation["cells"]:
        row, col = cell["row"], cell["col"]
        item_types = cell["grid_item_types"]
        cell_tokens[row * 9 + col] = add("cell", cell["terrain"], cell["row_type"], (
            _ratio(row, 5), _ratio(col, 8), _ratio(cell["terrain"], 8), _ratio(cell["row_type"], 3),
            _ratio(len(cell["plant_types"]), 5), float(11 in item_types), float(1 in item_types),
            float(2 in item_types),
        ), row=row, col=col)

    for plant in plants:
        row, col = plant["row"], plant["col"]
        add("plant", plant["type"], plant["imitater_type"], (
            _ratio(row, 5), _ratio(col, 8), _ratio(plant["health"], max(plant["max_health"], 1)),
            _ratio(plant["health"], 3000), _ratio(plant["max_health"], 3000), _ratio(plant["state"], 100),
            _ratio(plant["state_countdown"], 1200), _ratio(plant["launch_counter"], 1200),
            _ratio(plant["launch_rate"], 1200), _ratio(plant["shooting_counter"], 1200),
            _ratio(plant["wake_up_counter"], 1200), float(plant["asleep"]), float(plant["squished"]),
            _ratio(plant["bungee_state"], 3), float(plant["target_zombie_id"] > 0),
        ), row=row, col=col)

    for zombie in zombies:
        row = zombie["row"]
        col = max(0, min(8, int((zombie["x"] - 40) / 80)))
        add("zombie", zombie["type"], values=(
            _ratio(row, 5), _ratio(col, 8), _ratio(zombie["x"], 900), _ratio(zombie["y"], 700),
            _ratio(zombie["body_health"], max(zombie["body_max_health"], 1)), _ratio(zombie["body_health"], 2000),
            _ratio(zombie["helm_health"], max(zombie["helm_max_health"], 1)),
            _ratio(zombie["shield_health"], max(zombie["shield_max_health"], 1)),
            _ratio(zombie["phase"], 32), _ratio(zombie["phase_counter"], 1200),
            _ratio(zombie["velocity_x"], 5), _ratio(zombie["chilled"], 1200),
            _ratio(zombie["buttered"], 1200), _ratio(zombie["ice_trap"], 1200),
            float(zombie["has_head"]), float(zombie["has_arm"]), float(zombie["has_object"]),
            float(zombie["is_eating"]), _ratio(zombie["target_col"], 8), _ratio(zombie["target_row"], 5),
        ), row=row, col=col)

    for projectile in projectiles:
        row = projectile["row"]
        col = max(0, min(8, int((projectile["x"] - 40) / 80)))
        add("projectile", projectile["type"], projectile["motion"], (
            _ratio(row, 5), _ratio(col, 8), _ratio(projectile["x"], 900), _ratio(projectile["y"], 700),
            _ratio(projectile["z"], 500), _ratio(projectile["vx"], 20), _ratio(projectile["vy"], 20),
            _ratio(projectile["vz"], 20), _ratio(projectile["motion"], 8), _ratio(projectile["damage"], 500),
            _ratio(projectile["age"], 1200), float(projectile["target_zombie_id"] > 0),
        ), row=row, col=col)

    for defense in observation["defenses"]:
        row = defense["row"]
        col = max(0, min(8, int((defense["x"] - 40) / 80)))
        add("defense", defense["type"], values=(
            _ratio(row, 5), _ratio(defense["state"], 5), _ratio(defense["x"], 900), _ratio(defense["y"], 700),
        ), row=row, col=col)

    for item in observation["grid_items"]:
        row, col = item["row"], item["col"]
        add("grid_item", item["type"], item["state"], (
            _ratio(row, 5), _ratio(col, 8), _ratio(item["counter"], 1200),
            _ratio(item["x"], 900), _ratio(item["y"], 700), _ratio(item["zombie_type"], 40),
            _ratio(item["seed_type"], 50),
        ), row, col)

    for packet in observation["packets"]:
        packet_tokens[packet["index"]] = add("seed_packet", packet["type"], packet["imitater_type"], (
            _ratio(packet["index"], 10), _ratio(packet["cost"], 500), _ratio(packet["cooldown"], 3000),
            _ratio(packet["refresh_time"], 3000), float(packet["active"]),
        ))

    for zombie_type in observation["loadout_context"]["zombie_roster"]:
        add("zombie_roster", zombie_type, values=(1.0,))

    device = torch.device("cpu")
    tensors = {
        "kinds": torch.tensor(kinds, dtype=torch.long, device=device),
        "categories": torch.tensor(categories, dtype=torch.long, device=device),
        "variants": torch.tensor(variants, dtype=torch.long, device=device),
        "features": torch.tensor(features, dtype=torch.float32, device=device),
        "rows": torch.tensor(rows, dtype=torch.long, device=device),
        "cols": torch.tensor(cols, dtype=torch.long, device=device),
    }
    return tensors, {"packet_tokens": packet_tokens, "cell_tokens": cell_tokens}


class RelationAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.relation_bias_enabled = True
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.kind_pair_bias = nn.Parameter(torch.zeros(heads, len(TOKEN_KINDS), len(TOKEN_KINDS)))
        self.row_bias = nn.Embedding(12, heads)
        self.col_bias = nn.Embedding(18, heads)
        self.same_cell_bias = nn.Embedding(2, heads)

    def forward(self, x: Tensor, kinds: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        batch, count, width = x.shape
        qkv = self.qkv(x).view(batch, count, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scale = self.head_width ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if self.relation_bias_enabled:
            kind_pair = self.kind_pair_bias[:, kinds[:, None], kinds[None, :]]
            row_known = (rows[:, None] >= 0) & (rows[None, :] >= 0)
            col_known = (cols[:, None] >= 0) & (cols[None, :] >= 0)
            row_delta = (rows[:, None] - rows[None, :]).clamp(-5, 5) + 5
            col_delta = (cols[:, None] - cols[None, :]).clamp(-8, 8) + 8
            row_bucket = torch.where(row_known, row_delta, 11)
            col_bucket = torch.where(col_known, col_delta, 17)
            same_cell = (row_known & col_known & (rows[:, None] == rows[None, :]) & (cols[:, None] == cols[None, :])).long()
            relation = kind_pair + self.row_bias(row_bucket).permute(2, 0, 1)
            relation = relation + self.col_bias(col_bucket).permute(2, 0, 1) + self.same_cell_bias(same_cell).permute(2, 0, 1)
            scores = scores + relation.unsqueeze(0)
        attended = torch.softmax(scores, dim=-1)
        value = torch.matmul(attended, value).transpose(1, 2).contiguous().view(batch, count, width)
        return self.projection(value)


class RelationLayer(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = RelationAttention(width, heads)
        self.ff_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, ff_width)
        self.value = nn.Linear(width, ff_width)
        self.down = nn.Linear(ff_width, width)

    def forward(self, x: Tensor, kinds: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x), kinds, rows, cols)
        x = x + self.down(F.silu(self.gate(self.ff_norm(x))) * self.value(self.ff_norm(x)))
        return x


class GameplayModelV0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = MODEL_CONFIG["width"]
        self.kind_embedding = nn.Embedding(len(TOKEN_KINDS), width)
        self.category_embedding = nn.Embedding(128, width)
        self.variant_embedding = nn.Embedding(128, width)
        self.feature_projection = nn.Sequential(nn.Linear(FEATURE_COUNT, width), nn.SiLU(), nn.Linear(width, width))
        self.row_embedding = nn.Embedding(8, width)
        self.col_embedding = nn.Embedding(11, width)
        self.encoder = nn.ModuleList([
            RelationLayer(width, MODEL_CONFIG["heads"], MODEL_CONFIG["ff_width"])
            for _ in range(MODEL_CONFIG["layers"])
        ])
        self.encoder_norm = nn.LayerNorm(width)

        self.action_embedding = nn.Embedding(3, 64)
        self.previous_packet_embedding = nn.Embedding(12, 64)
        self.previous_cell_embedding = nn.Embedding(55, 64)
        self.previous_wait_embedding = nn.Embedding(len(WAIT_TICKS) + 1, 64)
        self.previous_action_projection = nn.Sequential(nn.Linear(256, 64), nn.SiLU())
        self.delta_embedding = nn.Embedding(32, 32)
        self.event_projection = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.belief = nn.GRU(
            width + 128, MODEL_CONFIG["gru_width"], MODEL_CONFIG["gru_layers"], batch_first=True
        )
        hidden = MODEL_CONFIG["gru_width"]
        self.action_type = nn.Linear(hidden, 3)
        self.packet_query = nn.Linear(hidden, width)
        self.packet_key = nn.Linear(width, width)
        self.cell_key = nn.Linear(width, width)
        self.plant_cell_query = nn.Linear(hidden + width, width)
        self.shovel_cell_query = nn.Linear(hidden, width)
        self.wait_duration = nn.Linear(hidden, len(WAIT_TICKS))
        self.value = nn.Linear(hidden, 1)
        self.aux_next_spawn = nn.Linear(hidden, 1)
        self.aux_lane_threat = nn.Linear(hidden, 6)
        self.aux_outcome = nn.Linear(hidden, 2)
        self.privileged_features = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.privileged_critic = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def _previous_action(self, action: dict[str, Any] | None, device: torch.device) -> Tensor:
        if action is None:
            return torch.zeros(1, 64, device=device)
        kind = {"plant": 0, "shovel": 1, "wait": 2, "wait_decision": 2}[action["type"]]
        packet = max(0, min(10, int(action.get("packet", -1)) + 1))
        cell = 0
        if "row" in action and "col" in action:
            cell = max(0, min(54, int(action["row"]) * 9 + int(action["col"]) + 1))
        ticks = action.get("ticks", 0)
        duration = min(range(len(WAIT_TICKS)), key=lambda i: abs(WAIT_TICKS[i] - ticks)) + 1 if ticks else 0
        parts = torch.cat((
            self.action_embedding(torch.tensor([kind], device=device)),
            self.previous_packet_embedding(torch.tensor([packet], device=device)),
            self.previous_cell_embedding(torch.tensor([cell], device=device)),
            self.previous_wait_embedding(torch.tensor([duration], device=device)),
        ), dim=-1)
        return self.previous_action_projection(parts)

    @staticmethod
    def _event_features(events: dict[str, Any] | None, device: torch.device) -> Tensor:
        events = events or {}
        values = [
            _ratio(events.get("zombies_killed", 0), 10), _ratio(events.get("plants_eaten", 0), 5),
            _ratio(events.get("sun_produced", 0), 250), _ratio(events.get("sun_spent", 0), 300),
            float(events.get("mower_triggered", 0) > 0), _ratio(events.get("waves_started", 0), 3),
            float(events.get("level_won", False)), float(events.get("level_lost", False)),
        ]
        return torch.tensor([values], dtype=torch.float32, device=device)

    def step(
        self,
        observation: dict[str, Any],
        hidden: Tensor | None = None,
        previous_action: dict[str, Any] | None = None,
        delta_ticks: int = 0,
        events: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        device = next(self.parameters()).device
        tensors, metadata = observation_tokens(observation)
        kinds = tensors["kinds"].to(device)
        rows = tensors["rows"].to(device)
        cols = tensors["cols"].to(device)
        x = self.kind_embedding(kinds) + self.category_embedding(tensors["categories"].to(device))
        x = x + self.variant_embedding(tensors["variants"].to(device))
        x = x + self.feature_projection(tensors["features"].to(device))
        x = x + self.row_embedding((rows + 1).clamp(0, 7)) + self.col_embedding((cols + 1).clamp(0, 10))
        x = x.unsqueeze(0)
        for layer in self.encoder:
            x = layer(x, kinds, rows, cols)
        x = self.encoder_norm(x)

        action_vector = self._previous_action(previous_action, device)
        delta_index = min(31, max(0, int(math.log2(max(0, delta_ticks) + 1))))
        delta_vector = self.delta_embedding(torch.tensor([delta_index], device=device))
        event_vector = self.event_projection(self._event_features(events, device))
        recurrent_input = torch.cat((x[:, 0, :], action_vector, delta_vector, event_vector), dim=-1).unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(MODEL_CONFIG["gru_layers"], 1, MODEL_CONFIG["gru_width"], device=device)
        belief, hidden = self.belief(recurrent_input, hidden)
        belief = belief[:, 0, :]

        packet_ids = sorted(metadata["packet_tokens"])
        cell_ids = list(range(54))
        packet_tokens = torch.stack([x[0, metadata["packet_tokens"][i]] for i in packet_ids]) if packet_ids else x.new_zeros((0, MODEL_CONFIG["width"]))
        cell_tokens = torch.stack([x[0, metadata["cell_tokens"][i]] for i in cell_ids])
        packet_logits = (self.packet_key(packet_tokens) * self.packet_query(belief)).sum(-1) / math.sqrt(MODEL_CONFIG["width"])
        return {
            "hidden": hidden,
            "belief": belief,
            "type_logits": self.action_type(belief)[0],
            "wait_logits": self.wait_duration(belief)[0],
            "packet_logits": packet_logits,
            "packet_ids": packet_ids,
            "packet_tokens": packet_tokens,
            "cell_tokens": cell_tokens,
            "cell_keys": self.cell_key(cell_tokens),
            "value": self.value(belief).squeeze(-1),
            "aux_next_spawn": self.aux_next_spawn(belief),
            "aux_lane_threat": self.aux_lane_threat(belief),
            "aux_outcome": self.aux_outcome(belief),
            "wave_index": observation["wave"],
        }

    def plant_cell_scores(self, output: dict[str, Any], packet: int) -> Tensor:
        packet_index = output["packet_ids"].index(packet)
        query = self.plant_cell_query(torch.cat((output["belief"], output["packet_tokens"][packet_index].unsqueeze(0)), dim=-1))
        return (output["cell_keys"] * query).sum(-1) / math.sqrt(MODEL_CONFIG["width"])

    def shovel_cell_scores(self, output: dict[str, Any]) -> Tensor:
        query = self.shovel_cell_query(output["belief"])
        return (output["cell_keys"] * query).sum(-1) / math.sqrt(MODEL_CONFIG["width"])

    def privileged_value(self, output: dict[str, Any], privileged_state: dict[str, Any] | None) -> Tensor:
        values = [0.0] * 16
        if privileged_state:
            hidden = privileged_state.get("hidden", {})
            values[0] = _ratio(hidden.get("wave_timer", 0), 6000)
            waves = hidden.get("zombies_in_wave", [])
            current = min(max(0, output["wave_index"]), max(0, len(waves) - 1))
            for zombie_type in waves[current][:15] if waves else []:
                if 0 <= zombie_type < 15:
                    values[1 + zombie_type] += 0.1
        device = output["belief"].device
        extra = self.privileged_features(torch.tensor([values], dtype=torch.float32, device=device))
        return self.privileged_critic(torch.cat((output["belief"], extra), dim=-1))


def select_action(
    model: GameplayModelV0,
    output: dict[str, Any],
    observation: dict[str, Any],
    action: dict[str, Any] | None = None,
    deterministic: bool = False,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    device = output["type_logits"].device
    legal = observation["legal_actions"]
    valid_packets = sorted({item["packet"] for item in legal["plants"]})
    valid_shovels = {row * 9 + col for col, row in legal["shovels"]}
    type_logits = output["type_logits"].clone()
    type_logits[0] = type_logits[0] if valid_packets else -1e9
    type_logits[1] = type_logits[1] if valid_shovels else -1e9
    type_dist = torch.distributions.Categorical(logits=type_logits)
    action_types = {"plant": 0, "shovel": 1, "wait": 2, "wait_decision": 2}
    if action is not None and action.get("type") not in action_types:
        raise ValueError(f"unsupported action type: {action.get('type')}")
    type_index = (type_logits.argmax() if deterministic else type_dist.sample()) if action is None else torch.tensor(
        action_types[action["type"]], device=device
    )
    if action is not None and ((int(type_index.item()) == 0 and not valid_packets)
                               or (int(type_index.item()) == 1 and not valid_shovels)):
        raise ValueError(f"action is illegal in this observation: {action}")
    log_prob = type_dist.log_prob(type_index)
    entropy = type_dist.entropy()
    action_type = int(type_index.item())

    if action_type == 0:
        packet_logits = output["packet_logits"].clone()
        allowed = set(valid_packets)
        for index, packet_id in enumerate(output["packet_ids"]):
            if packet_id not in allowed:
                packet_logits[index] = -1e9
        packet_dist = torch.distributions.Categorical(logits=packet_logits)
        if action is not None and action["packet"] not in allowed:
            raise ValueError(f"illegal plant packet: {action}")
        packet_index = (packet_logits.argmax() if deterministic else packet_dist.sample()) if action is None else torch.tensor(
            output["packet_ids"].index(action["packet"]), device=device
        )
        packet = output["packet_ids"][int(packet_index.item())]
        log_prob = log_prob + packet_dist.log_prob(packet_index)
        entropy = entropy + packet_dist.entropy()
        cell_logits = model.plant_cell_scores(output, packet)
        valid_cells = {a["row"] * 9 + a["col"] for a in legal["plants"] if a["packet"] == packet}
        for cell in range(54):
            if cell not in valid_cells:
                cell_logits[cell] = -1e9
        cell_dist = torch.distributions.Categorical(logits=cell_logits)
        if action is not None and action["row"] * 9 + action["col"] not in valid_cells:
            raise ValueError(f"illegal plant cell: {action}")
        cell_index = (cell_logits.argmax() if deterministic else cell_dist.sample()) if action is None else torch.tensor(
            action["row"] * 9 + action["col"], device=device
        )
        cell = int(cell_index.item())
        log_prob = log_prob + cell_dist.log_prob(cell_index)
        entropy = entropy + cell_dist.entropy()
        selected = {"type": "plant", "packet": packet, "col": cell % 9, "row": cell // 9} if action is None else dict(action)
    elif action_type == 1:
        cell_logits = model.shovel_cell_scores(output)
        for cell in range(54):
            if cell not in valid_shovels:
                cell_logits[cell] = -1e9
        cell_dist = torch.distributions.Categorical(logits=cell_logits)
        if action is not None and action["row"] * 9 + action["col"] not in valid_shovels:
            raise ValueError(f"illegal shovel cell: {action}")
        cell_index = (cell_logits.argmax() if deterministic else cell_dist.sample()) if action is None else torch.tensor(
            action["row"] * 9 + action["col"], device=device
        )
        cell = int(cell_index.item())
        log_prob = log_prob + cell_dist.log_prob(cell_index)
        entropy = entropy + cell_dist.entropy()
        selected = {"type": "shovel", "col": cell % 9, "row": cell // 9} if action is None else dict(action)
    else:
        wait_dist = torch.distributions.Categorical(logits=output["wait_logits"])
        duration = (output["wait_logits"].argmax() if deterministic else wait_dist.sample()) if action is None else torch.tensor(
            min(range(len(WAIT_TICKS)), key=lambda i: abs(WAIT_TICKS[i] - action.get("ticks", 300))), device=device
        )
        log_prob = log_prob + wait_dist.log_prob(duration)
        entropy = entropy + wait_dist.entropy()
        selected = {"type": "wait", "ticks": WAIT_TICKS[int(duration.item())]} if action is None else dict(action)
    return selected, log_prob, entropy


@torch.no_grad()
def predict_action(
    model: GameplayModelV0,
    observation: dict[str, Any],
    hidden: Tensor | None,
    previous_action: dict[str, Any] | None,
    delta_ticks: int,
    events: dict[str, Any] | None,
) -> tuple[dict[str, Any], Tensor, dict[str, Any]]:
    output = model.step(observation, hidden, previous_action, delta_ticks, events)
    action, _, _ = select_action(model, output, observation, deterministic=True)
    return action, output["hidden"], output


def behavior_cloning_loss(model: GameplayModelV0, output: dict[str, Any], observation: dict[str, Any], action: dict[str, Any], plant_weight: float = 1.0) -> Tensor:
    legal = observation["legal_actions"]
    device = output["type_logits"].device
    valid_packets = sorted({item["packet"] for item in legal["plants"]})
    valid_shovels = [row * 9 + col for col, row in legal["shovels"]]
    type_logits = output["type_logits"].clone()
    if not valid_packets: type_logits[0] = -1e9
    if not valid_shovels: type_logits[1] = -1e9
    target_type = {"plant": 0, "shovel": 1, "wait": 2, "wait_decision": 2}[action["type"]]
    type_loss = F.cross_entropy(type_logits.unsqueeze(0), torch.tensor([target_type], device=device))
    losses = [type_loss * (plant_weight if target_type == 0 else 1.0)]
    if target_type == 0:
        packet_logits = output["packet_logits"].clone()
        for index, packet_id in enumerate(output["packet_ids"]):
            if packet_id not in valid_packets:
                packet_logits[index] = -1e9
        packet_index = output["packet_ids"].index(action["packet"])
        losses.append(F.cross_entropy(packet_logits.unsqueeze(0), torch.tensor([packet_index], device=device)))
        cell_logits = model.plant_cell_scores(output, action["packet"]).clone()
        valid_cells = {a["row"] * 9 + a["col"] for a in legal["plants"] if a["packet"] == action["packet"]}
        for cell in range(54):
            if cell not in valid_cells:
                cell_logits[cell] = -1e9
        target_cell = action["row"] * 9 + action["col"]
        losses.append(F.cross_entropy(cell_logits.unsqueeze(0), torch.tensor([target_cell], device=device)))
    elif target_type == 1:
        cell_logits = model.shovel_cell_scores(output).clone()
        for cell in range(54):
            if cell not in valid_shovels:
                cell_logits[cell] = -1e9
        target_cell = action["row"] * 9 + action["col"]
        losses.append(F.cross_entropy(cell_logits.unsqueeze(0), torch.tensor([target_cell], device=device)))
    else:
        target_ticks = action.get("ticks", 300)
        duration = min(range(len(WAIT_TICKS)), key=lambda i: abs(WAIT_TICKS[i] - target_ticks))
        losses.append(F.cross_entropy(output["wait_logits"].unsqueeze(0), torch.tensor([duration], device=device)))
    return torch.stack(losses).sum()


def teacher_v0_action(observation: dict[str, Any]) -> dict[str, Any]:
    rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"] > 0})
    plants, zombies, sun = observation["plants"], observation["zombies"], observation["sun"]
    packets = {packet["type"]: packet for packet in observation["packets"]}
    legal = observation["legal_actions"]["plants"]
    attackers = [plant for plant in plants if plant["type"] in (0, 5)]
    shooter_rows = {plant["row"] for plant in attackers}
    zombies_by_row = {row: [zombie for zombie in zombies if zombie["row"] == row] for row in rows}
    chosen = None

    if sun >= 150 and packets[2]["active"] and packets[2]["cooldown"] == 0:
        for row in rows:
            close = [zombie for zombie in zombies_by_row[row] if zombie["x"] < 320]
            if len(close) >= 2:
                choices = [a for a in legal if a["packet"] == packets[2]["index"] and a["row"] == row and 1 <= a["col"] <= 4]
                if choices:
                    chosen = min(choices, key=lambda a: abs(a["col"] - 2))
                    break

    if chosen is None and sun >= 50 and packets[3]["active"] and packets[3]["cooldown"] == 0:
        for row in rows:
            has_threat = any(zombie["x"] < 400 for zombie in zombies_by_row[row])
            has_wallnut = any(plant["type"] == 3 and plant["row"] == row and plant["col"] >= 3 for plant in plants)
            if has_threat and not has_wallnut:
                choices = [a for a in legal if a["packet"] == packets[3]["index"] and a["row"] == row and a["col"] == 4]
                if choices:
                    chosen = choices[0]
                    break

    target_rows = [row for row in rows if zombies_by_row[row] and row not in shooter_rows]
    if not target_rows:
        target_rows = [row for row in rows if row not in shooter_rows]
    if chosen is None and target_rows:
        row = min(target_rows, key=lambda r: (min((z["x"] for z in zombies_by_row[r]), default=9999), abs(r - 2)))
        for seed_type, cost in ((0, 100), (5, 175)):
            packet = packets[seed_type]
            if sun >= cost and packet["active"] and packet["cooldown"] == 0:
                choices = [a for a in legal if a["packet"] == packet["index"] and a["row"] == row and 2 <= a["col"] <= 4]
                if choices:
                    chosen = min(choices, key=lambda a: abs(a["col"] - 3))
                    break

    sunflowers = [plant for plant in plants if plant["type"] == 1]
    sunflower_target = 2 if len(shooter_rows) < len(rows) else 5
    if chosen is None and sun >= 50 and len(sunflowers) < sunflower_target and packets[1]["active"] and packets[1]["cooldown"] == 0:
        occupied = {(plant["row"], plant["col"]) for plant in plants}
        choices = [a for a in legal if a["packet"] == packets[1]["index"] and a["col"] in (0, 1) and (a["row"], a["col"]) not in occupied]
        if choices:
            chosen = min(choices, key=lambda a: (abs(a["row"] - 2), a["col"]))

    if chosen is None and sun >= 175 and packets[5]["active"] and packets[5]["cooldown"] == 0:
        choices = [a for a in legal if a["packet"] == packets[5]["index"] and a["row"] in shooter_rows and 2 <= a["col"] <= 4]
        urgent = [a for a in choices if zombies_by_row[a["row"]]]
        if urgent:
            chosen = min(urgent, key=lambda a: (min(z["x"] for z in zombies_by_row[a["row"]]), abs(a["col"] - 3)))

    if chosen is None and sun >= 100 and packets[0]["active"] and packets[0]["cooldown"] == 0:
        counts = {row: sum(plant["row"] == row and plant["type"] in (0, 5) for plant in plants) for row in rows}
        candidates = [row for row in rows if counts[row] < 2]
        if candidates:
            row = min(candidates, key=lambda r: (counts[r], abs(r - 2)))
            choices = [a for a in legal if a["packet"] == packets[0]["index"] and a["row"] == row and 2 <= a["col"] <= 4]
            if choices:
                chosen = min(choices, key=lambda a: abs(a["col"] - 3))

    if chosen is None and sun >= 50 and len(sunflowers) < 8 and packets[1]["active"] and packets[1]["cooldown"] == 0:
        occupied = {(plant["row"], plant["col"]) for plant in plants}
        choices = [a for a in legal if a["packet"] == packets[1]["index"] and a["col"] in (0, 1) and (a["row"], a["col"]) not in occupied]
        if choices:
            chosen = min(choices, key=lambda a: (abs(a["row"] - 2), a["col"]))

    if chosen is not None:
        return {"type": "plant", **chosen}
    return {"type": "wait", "ticks": 300}


@dataclass(frozen=True)
class TeacherAdvice:
    action: dict[str, Any]
    candidates: list[tuple[dict[str, Any], float]]


def teacher_advice(observation: dict[str, Any], sunflower_placements: int = 0) -> TeacherAdvice:
    rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"] > 0})
    zombies = observation["zombies"]
    plants = observation["plants"]
    zombie_multiplier = max(1.0, float(observation.get("zombie_count_multiplier", 1.0)))
    safe_shot_gap = 240.0 + 30.0 * (zombie_multiplier - 1.0)
    emergency_front = max(180.0, 450.0 - 50.0 * (zombie_multiplier - 1.0))
    packets = {packet["index"]: packet for packet in observation["packets"]}
    legal = observation["legal_actions"]
    plants_by_row = {row: [] for row in rows}
    zombies_by_row = {row: [zombie for zombie in zombies if zombie["row"] == row] for row in rows}

    def plant_type(plant: dict[str, Any]) -> int:
        return plant["imitater_type"] if plant["type"] == 48 and plant["imitater_type"] >= 0 else plant["type"]

    for plant in plants:
        if plant["row"] in plants_by_row:
            plants_by_row[plant["row"]].append(plant)

    def zombie_urgency(zombie: dict[str, Any]) -> float:
        return max(0.0, min(1.0, (650.0 - zombie["x"]) / 420.0))

    def zombie_hp(zombie: dict[str, Any]) -> float:
        return sum(max(0.0, zombie.get(key, 0.0)) for key in ("body_health", "helm_health", "shield_health"))

    def expected_shots(zombie: dict[str, Any], plant_x: float) -> float:
        speed = abs(zombie.get("velocity_x", 0.0))
        if speed <= 0.01:
            return 0.0
        firing_window = max(0.0, (zombie["x"] - plant_x - 80.0) / speed)
        projectile_travel = max(0.0, zombie["x"] - plant_x) / 5.0
        return max(0.0, firing_window - projectile_travel) / 150.0

    attackers = {
        row: [p for p in plants_by_row[row]
              if plant_type(p) in (0, 5) and not p["squished"]
              and p["health"] > 0.1 * max(1, p["max_health"])
              and min((z["x"] for z in zombies_by_row[row]), default=9999.0) > 160 + 80 * p["col"]]
        for row in rows
    }
    wallnuts = {row: [p for p in plants_by_row[row] if plant_type(p) == 3 and not p["squished"]]
                for row in rows}
    mine_rows = {row for row in rows if any(plant_type(p) == 4 for p in plants_by_row[row])}
    uncovered_rows = {row for row in rows if zombies_by_row[row] and not attackers[row]}
    ready_mowers = {defense["row"] for defense in observation["defenses"] if defense["state"] == 1}
    sunflowers = [p for p in plants if plant_type(p) == 1]
    active_shooter_rows = sum(bool(attackers[row]) for row in rows)
    shooter_rows_built = sum(any(plant_type(p) in (0, 5) and not p["squished"] and
                                 p["health"] > 0.1 * max(1, p["max_health"])
                                 for p in plants_by_row[row]) for row in rows)
    shooter_rows_before_economy = min(len(rows), max(2, math.ceil(zombie_multiplier)))
    sunflower_goal = min(8, 2 + math.ceil(zombie_multiplier - 1.0)) if observation["wave"] < 3 else min(8, 3 + round(zombie_multiplier - 1.0))
    if observation["wave"] >= 10:
        sunflower_goal = 8
    elif active_shooter_rows == len(rows):
        sunflower_goal = max(sunflower_goal, 5)
    mower_defense_front = 500.0 + 40.0 * (zombie_multiplier - 1.0)
    sunflower_safety_front = mower_defense_front
    rake_rows = {item["row"] for item in observation["grid_items"] if item["type"] == 11 and item["state"] == 26}
    reserve_target = round(100.0 + 25.0 * (zombie_multiplier - 1.0))
    coverage_bonus = 24.0 + 12.0 * (zombie_multiplier - 1.0)
    sun = observation["sun"]
    scores: list[tuple[dict[str, Any], float]] = []

    def add(action: dict[str, Any], score: float) -> None:
        scores.append((action, score))

    for placement in legal["plants"]:
        packet = packets[placement["packet"]]
        seed_type = packet["imitater_type"] if packet["type"] == 48 else packet["type"]
        row, col = placement["row"], placement["col"]
        lane_zombies = zombies_by_row[row]
        front = min((z["x"] for z in lane_zombies), default=9999.0)
        urgency = max((zombie_urgency(z) for z in lane_zombies), default=0.0)
        pressure = sum((1.0 + zombie_urgency(z)) * min(3.0, max(0.5, zombie_hp(z) / 200.0)) for z in lane_zombies)
        count_attackers = len(attackers[row])
        mower_risk = max(0.0, min(1.0, (450.0 - front) / 300.0)) if row in ready_mowers else 0.0
        plant_x = 80 + 80 * col
        action = {"type": "plant", **placement}
        cost = packet["cost"]
        if sun < cost:
            continue
        reserve_penalty = 100.0 if sun - cost < reserve_target else 0.0

        if seed_type in (0, 5):
            if front <= emergency_front or not attackers[row]:
                reserve_penalty = 0.0
            if seed_type == 0 and not attackers[row] and front > mower_defense_front and sun < 150:
                reserve_penalty = 200.0
            targets = [z for z in lane_zombies if z["x"] > plant_x + 80]
            if targets:
                useful_count = sum(1 for z in targets if z["x"] > plant_x + 180)
                value = (25.0 if seed_type == 5 else 22.0) + 10.0 * urgency
                value += min(30.0, pressure * 4.0) / (1 + count_attackers * 0.5)
                value += min(4.0, useful_count * 0.7) - count_attackers * 4.0
                gap = min(z["x"] - plant_x for z in targets)
                value += max(-5.0, 7.0 - abs(gap - safe_shot_gap) / 35.0) + 12.0 * mower_risk
                value += coverage_bonus if row in uncovered_rows else -coverage_bonus * len(uncovered_rows)
                needed_attackers = min(3, max(1, (len(targets) + 1) // 2))
                value += 20.0 * max(0, needed_attackers - count_attackers)
                extra_damage = 0.0
                for zombie in targets:
                    shots = expected_shots(zombie, plant_x) * (1.75 if seed_type == 5 else 1.0)
                    current = min(zombie_hp(zombie), count_attackers * shots * 20.0)
                    added = min(zombie_hp(zombie), (count_attackers + 1) * shots * 20.0)
                    extra_damage += added - current
                value += min(200.0, extra_damage * 0.5)
                if seed_type == 5:
                    value += 40.0 + min(8.0, pressure) + 8.0 * (zombie_multiplier - 1.0)
                    reserve_penalty = 0.0
                if row in ready_mowers and front < mower_defense_front:
                    value += 600.0
                    reserve_penalty = 0.0
                add(action, value - cost * 0.01 - reserve_penalty)
            elif not lane_zombies and count_attackers == 0:
                value = (8.0 if seed_type == 0 else 5.0) + (2.0 if col == 2 else 0.0)
                if seed_type == 0:
                    value += 12.0 * (zombie_multiplier - 1.0)
                add(action, value - cost * 0.01 - reserve_penalty)
            continue

        if seed_type == 3 and lane_zombies and not wallnuts[row] and plant_x < front - 35:
            # Put the blocker on the house side of the front zombie, as close as a safe legal tile allows.
            value = 6.0 + 11.0 * urgency + min(4.0, pressure) + 10.0 * mower_risk
            value += max(-5.0, 7.0 - abs((front - plant_x) - safe_shot_gap) / 35.0)
            value -= cost * 0.01
            if row in ready_mowers and front < emergency_front + 70 and (attackers[row] or front < 200):
                reserve_penalty = 0.0
            add(action, (value if attackers[row] else value - 5.0) - reserve_penalty)
        elif seed_type == 4 and lane_zombies and row not in mine_rows:
            viable = []
            for zombie in lane_zombies:
                speed = abs(zombie.get("velocity_x", 0.0))
                distance = zombie["x"] - plant_x - 25
                if speed > 0.01 and distance > 0:
                    arrival = distance / speed
                    if arrival >= 3200:
                        viable.append(arrival)
            if viable:
                arrival = min(viable)
                value = 3.0 + min(3.0, len(lane_zombies)) * 1.2 + 2.0 * urgency
                value -= abs(arrival - 2600.0) / 900.0 + min(8.0, count_attackers * 5.0)
                if not attackers[row]:
                    value += 16.0 + max(0.0, 8.0 - abs(arrival - 2600.0) / 350.0)
                    if observation["wave"] <= 1:
                        reserve_penalty = 0.0
                add(action, value - cost * 0.01 - reserve_penalty)
        elif seed_type == 2:
            targets = [z for z in zombies if abs(z["row"] - row) <= 1 and abs(z["x"] - plant_x) <= 115]
            if targets:
                value = 8.0 + sum(
                    15.0 + 20.0 * zombie_urgency(z) + min(8.0, zombie_hp(z) / 180.0)
                    + 10.0 * (zombie_multiplier - 1.0)
                    for z in targets
                )
                if len(targets) >= 2 and max(zombie_urgency(z) for z in targets) > 0.35:
                    reserve_penalty = 0.0
                if any(z["row"] in ready_mowers and z["x"] < emergency_front - 70 for z in targets):
                    value += 14.0
                    reserve_penalty = 0.0
                if any(z["row"] in ready_mowers and z["x"] < mower_defense_front for z in targets):
                    value += 50.0
                    reserve_penalty = 0.0
                uncovered_targets = sum(z["row"] in uncovered_rows for z in targets)
                value += coverage_bonus * uncovered_targets
                if not uncovered_targets and uncovered_rows:
                    value -= coverage_bonus * len(uncovered_rows)
                add(action, value - cost * 0.015 - reserve_penalty)
        elif seed_type == 1:
            opening_sunflowers = max(1, round(2.0 / zombie_multiplier))
            if (sunflower_placements >= opening_sunflowers and
                    shooter_rows_built < shooter_rows_before_economy):
                continue
            if len(sunflowers) >= sunflower_goal:
                continue
            row_front = min((z["x"] for z in lane_zombies), default=9999.0)
            if lane_zombies and row_front < sunflower_safety_front and not any(p["col"] > col for p in wallnuts[row]):
                continue
            reserve_penalty = 0.0
            safety = max(0.0, min(1.0, (row_front - 420.0) / 300.0))
            economy = max(-3.0, 16.0 - len(sunflowers) * 3.5)
            economy += max(0, sunflower_goal - max(len(sunflowers), sunflower_placements)) * 8.0
            add(action, economy + safety * 4.0 + (1.0 if col == 1 else 0.0) - cost * 0.01)

    for col, row in legal["shovels"]:
        plant = next((p for p in plants_by_row[row] if p["col"] == col), None)
        if plant is None:
            continue
        durability = plant["health"] / max(1, plant["max_health"])
        nearest = min((z["x"] for z in zombies_by_row[row]), default=9999.0)
        if plant.get("squished") or (durability < 0.12 and nearest > 620):
            add({"type": "shovel", "col": col, "row": row}, 4.0 - durability)

    if not zombies:
        wait_ticks, wait_score = 600, 3.0
    else:
        nearest = min(zombie["x"] for zombie in zombies)
        wait_ticks = 150 if nearest < 380 else 300 if nearest < 540 else 600
        mower_front = min((zombie["x"] for zombie in zombies if zombie["row"] in ready_mowers),
                          default=9999.0)
        if mower_front < mower_defense_front:
            wait_ticks = min(wait_ticks, 30)
        elif nearest < 540:
            wait_ticks = min(wait_ticks, 60)
        wait_score = 0.0 if nearest < 540 else 4.0
        if any(zombie["row"] in ready_mowers and zombie["x"] < emergency_front for zombie in zombies):
            wait_score -= 6.0
        if any(zombie["row"] in rake_rows and zombie["x"] > 600 for zombie in zombies):
            wait_score += 3.0
    add({"type": "wait", "ticks": wait_ticks}, wait_score)

    scores.sort(key=lambda item: item[1], reverse=True)
    candidates = scores[:8]
    return TeacherAdvice(action=candidates[0][0], candidates=candidates)


def teacher_action(observation: dict[str, Any]) -> dict[str, Any]:
    return teacher_advice(observation).action


class TeacherPolicy:
    def __init__(self) -> None:
        self.sunflower_placements = 0

    def advice(self, observation: dict[str, Any]) -> TeacherAdvice:
        return teacher_advice(observation, self.sunflower_placements)

    def record_action(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if action.get("type") != "plant":
            return
        packet = next((item for item in observation["packets"] if item["index"] == action["packet"]), None)
        if packet is None:
            return
        seed_type = packet["imitater_type"] if packet["type"] == 48 else packet["type"]
        if seed_type == 1:
            self.sunflower_placements += 1
