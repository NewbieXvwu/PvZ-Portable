"""Structured recurrent GameplayModel-v1 network and action distribution helpers."""

from __future__ import annotations

import math
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
WAIT_TICKS = (60, 150, 300)
MODEL_CONFIG = {"layers": 4, "width": 192, "heads": 6, "ff_width": 768, "gru_layers": 2, "gru_width": 256}
MODEL_ARCHITECTURE_VERSION = 4
FEATURE_COUNT = 32


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a ``--device`` choice, preferring CUDA and falling back to the CPU.

    ``auto`` deliberately does **not** consider MPS.  Every model invocation in
    this project is a batch-of-1 forward pass -- the search value model is
    evaluated once per search leaf (~250 times per decision), and
    ``GameplayModelV1.step`` carries a GRU hidden state so it is called one
    observation at a time.  Measured on an Apple M5 Pro, MPS is slower on
    exactly those shapes:

    * ``SearchValueModel.predict`` 68 us -> 451 us, a 6.6x regression;
    * ``advice()`` on the real simulator 162 ms -> 282 ms (leaf evaluation goes
      from 8.5% to 47% of the search);
    * a whole rollout episode 13.2 s -> 22.6 s.

    MPS only wins once tensors are batched (value forward batch=256: 1.47x;
    encoder batch=64: 2.44x), and no such path exists in the training loop.
    The one place MPS does help is the backward pass of the student network
    (1.29x), which is ~7.5 min of a ~50 min run and is dwarfed by the 30 min it
    costs in rollout collection.  Pass ``--device mps`` explicitly to opt in.
    """
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return torch.device(requested)


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
        _ratio(observation["zombie_count_multiplier"], 10),
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

    tensors = {
        "kinds": torch.tensor(kinds, dtype=torch.long),
        "categories": torch.tensor(categories, dtype=torch.long),
        "variants": torch.tensor(variants, dtype=torch.long),
        "features": torch.tensor(features, dtype=torch.float32),
        "rows": torch.tensor(rows, dtype=torch.long),
        "cols": torch.tensor(cols, dtype=torch.long),
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
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_width ** -0.5)
        if self.relation_bias_enabled:
            kind_pair = self.kind_pair_bias[:, kinds[:, None], kinds[None, :]]
            row_known = (rows[:, None] >= 0) & (rows[None, :] >= 0)
            col_known = (cols[:, None] >= 0) & (cols[None, :] >= 0)
            row_delta = (rows[:, None] - rows[None, :]).clamp(-5, 5) + 5
            col_delta = (cols[:, None] - cols[None, :]).clamp(-8, 8) + 8
            row_bucket = torch.where(row_known, row_delta, 11)
            col_bucket = torch.where(col_known, col_delta, 17)
            same_cell = (row_known & col_known & (rows[:, None] == rows[None, :])
                         & (cols[:, None] == cols[None, :])).long()
            relation = kind_pair + self.row_bias(row_bucket).permute(2, 0, 1)
            relation = relation + self.col_bias(col_bucket).permute(2, 0, 1)
            relation = relation + self.same_cell_bias(same_cell).permute(2, 0, 1)
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
        normalized = self.ff_norm(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.value(normalized))
        return x


class GameplayModelV1(nn.Module):
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
        self.belief = nn.GRU(width + 128, MODEL_CONFIG["gru_width"], MODEL_CONFIG["gru_layers"], batch_first=True)
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
        action_types = {"plant": 0, "shovel": 1, "wait": 2}
        if action.get("type") not in action_types:
            raise ValueError(f"unsupported action type: {action.get('type')}")
        kind = action_types[action["type"]]
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

    def step(self, observation: dict[str, Any], hidden: Tensor | None = None,
             previous_action: dict[str, Any] | None = None, delta_ticks: int = 0,
             events: dict[str, Any] | None = None) -> dict[str, Any]:
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
        packet_tokens = (torch.stack([x[0, metadata["packet_tokens"][i]] for i in packet_ids])
                         if packet_ids else x.new_zeros((0, MODEL_CONFIG["width"])))
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
            "value": torch.tanh(self.value(belief)).squeeze(-1),
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


def select_action(model: GameplayModelV1, output: dict[str, Any], observation: dict[str, Any],
                  action: dict[str, Any] | None = None, deterministic: bool = False) -> tuple[dict[str, Any], Tensor, Tensor]:
    device = output["type_logits"].device
    legal = observation["legal_actions"]
    valid_packets = sorted({item["packet"] for item in legal["plants"]})
    valid_shovels = {row * 9 + col for col, row in legal["shovels"]}
    type_logits = output["type_logits"].clone()
    if not valid_packets:
        type_logits[0] = -1e9
    if not valid_shovels:
        type_logits[1] = -1e9
    if not legal.get("wait", True):
        type_logits[2:] = -1e9
    type_dist = torch.distributions.Categorical(logits=type_logits)
    action_types = {"plant": 0, "shovel": 1, "wait": 2}
    if action is not None and action.get("type") not in action_types:
        raise ValueError(f"unsupported action type: {action.get('type')}")
    type_index = ((type_logits.argmax() if deterministic else type_dist.sample()) if action is None
                  else torch.tensor(action_types[action["type"]], device=device))
    if action is not None and ((int(type_index.item()) == 0 and not valid_packets)
                               or (int(type_index.item()) == 1 and not valid_shovels)
                               or (int(type_index.item()) >= 2 and not legal.get("wait", True))):
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
        packet_index = ((packet_logits.argmax() if deterministic else packet_dist.sample()) if action is None
                        else torch.tensor(output["packet_ids"].index(action["packet"]), device=device))
        packet = output["packet_ids"][int(packet_index.item())]
        log_prob = log_prob + packet_dist.log_prob(packet_index)
        entropy = entropy + packet_dist.entropy()
        cell_logits = model.plant_cell_scores(output, packet).clone()
        valid_cells = {a["row"] * 9 + a["col"] for a in legal["plants"] if a["packet"] == packet}
        for cell in range(54):
            if cell not in valid_cells:
                cell_logits[cell] = -1e9
        cell_dist = torch.distributions.Categorical(logits=cell_logits)
        if action is not None and action["row"] * 9 + action["col"] not in valid_cells:
            raise ValueError(f"illegal plant cell: {action}")
        cell_index = ((cell_logits.argmax() if deterministic else cell_dist.sample()) if action is None
                      else torch.tensor(action["row"] * 9 + action["col"], device=device))
        cell = int(cell_index.item())
        log_prob = log_prob + cell_dist.log_prob(cell_index)
        entropy = entropy + cell_dist.entropy()
        selected = {"type": "plant", "packet": packet, "col": cell % 9, "row": cell // 9} if action is None else dict(action)
    elif action_type == 1:
        cell_logits = model.shovel_cell_scores(output).clone()
        for cell in range(54):
            if cell not in valid_shovels:
                cell_logits[cell] = -1e9
        cell_dist = torch.distributions.Categorical(logits=cell_logits)
        if action is not None and action["row"] * 9 + action["col"] not in valid_shovels:
            raise ValueError(f"illegal shovel cell: {action}")
        cell_index = ((cell_logits.argmax() if deterministic else cell_dist.sample()) if action is None
                      else torch.tensor(action["row"] * 9 + action["col"], device=device))
        cell = int(cell_index.item())
        log_prob = log_prob + cell_dist.log_prob(cell_index)
        entropy = entropy + cell_dist.entropy()
        selected = {"type": "shovel", "col": cell % 9, "row": cell // 9} if action is None else dict(action)
    elif action_type == 2:
        wait_dist = torch.distributions.Categorical(logits=output["wait_logits"])
        if action is not None and action.get("ticks", 150) not in WAIT_TICKS:
            raise ValueError(f"wait ticks must be one of {WAIT_TICKS}: {action}")
        duration = ((output["wait_logits"].argmax() if deterministic else wait_dist.sample()) if action is None
                    else torch.tensor(min(range(len(WAIT_TICKS)),
                                          key=lambda i: abs(WAIT_TICKS[i] - action.get("ticks", 150))), device=device))
        log_prob = log_prob + wait_dist.log_prob(duration)
        entropy = entropy + wait_dist.entropy()
        selected = {"type": "wait", "ticks": WAIT_TICKS[int(duration.item())]} if action is None else dict(action)
    else:
        raise ValueError(f"unsupported action index: {action_type}")
    return selected, log_prob, entropy


@torch.no_grad()
def predict_action(model: GameplayModelV1, observation: dict[str, Any], hidden: Tensor | None,
                   previous_action: dict[str, Any] | None, delta_ticks: int,
                   events: dict[str, Any] | None) -> tuple[dict[str, Any], Tensor, dict[str, Any]]:
    output = model.step(observation, hidden, previous_action, delta_ticks, events)
    action, _, _ = select_action(model, output, observation, deterministic=True)
    return action, output["hidden"], output


def hard_behavior_cloning_loss(model: GameplayModelV1, output: dict[str, Any], observation: dict[str, Any],
                               action: dict[str, Any], plant_weight: float = 1.0) -> Tensor:
    legal = observation["legal_actions"]
    device = output["type_logits"].device
    valid_packets = sorted({item["packet"] for item in legal["plants"]})
    valid_shovels = [row * 9 + col for col, row in legal["shovels"]]
    type_logits = output["type_logits"].clone()
    if not valid_packets:
        type_logits[0] = -1e9
    if not valid_shovels:
        type_logits[1] = -1e9
    if not legal.get("wait", True):
        type_logits[2:] = -1e9
    action_types = {"plant": 0, "shovel": 1, "wait": 2}
    if action.get("type") not in action_types:
        raise ValueError(f"unsupported action type: {action.get('type')}")
    target_type = action_types[action["type"]]
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
    elif target_type == 2:
        target_ticks = action.get("ticks", 150)
        if target_ticks not in WAIT_TICKS:
            raise ValueError(f"wait ticks must be one of {WAIT_TICKS}: {action}")
        duration = min(range(len(WAIT_TICKS)), key=lambda i: abs(WAIT_TICKS[i] - target_ticks))
        losses.append(F.cross_entropy(output["wait_logits"].unsqueeze(0), torch.tensor([duration], device=device)))
    return torch.stack(losses).sum()
