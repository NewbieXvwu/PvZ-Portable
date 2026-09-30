"""Structured recurrent GameplayModel-v1 network and action distribution helpers."""

from __future__ import annotations

import math
import os
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:  # pragma: no cover - depends on the installed PyTorch build
    flex_attention = None


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
    "lane": 10,
}
WAIT_TICKS = (60, 150, 300)
MODEL_CONFIG = {"layers": 4, "width": 192, "heads": 6, "ff_width": 768, "gru_layers": 2, "gru_width": 256}
MODEL_ARCHITECTURE_VERSION = 5
FEATURE_COUNT = 32

# Fill value for the padded columns of the batched packet-logit table.  Those
# columns are always overwritten by ``masked_fill`` before the distribution is
# built, so this value is never read -- it exists only so the table is a plain
# finite tensor rather than one carrying -inf, which keeps it representable if a
# caller ever inspects it under fp16 autocast.
PACKET_LOGIT_FLOOR = -1e9


def env_flag(name: str, *, default: bool) -> bool:
    """Read a boolean environment flag, accepting the usual off spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


# Fusion of the relation-bias assembly, on by default.  The chain dispatches 117
# operators eagerly while doing microseconds of arithmetic, so fusing it is a measured
# 1.41x on the encoder and 1.30x on a full rollout step, bit-identically.
#
# It matters for the *eager* attention path, which is now both rollout (batch of 1)
# and the training update: ``train_update`` resolves ``--attention-backend auto`` to
# dense even on CUDA, because the fused assembly is 177x faster than the eager one
# there (906 ms -> 5.1 ms for one layer's backward) and beats FlexAttention's
# ``score_mod`` path 8.25 ms to 22.40 ms.  See ``PPO_UPDATE_ANATOMY.md`` §10.
#
# Each process pays a one-off compile cost, and the rollout pool pays it 18 times in
# parallel.  Set ``PVZ_RELATION_BIAS_FUSION=0`` (or call
# ``set_relation_bias_fusion(False)``) to go back to the eager chain -- but do not
# do that on CUDA: the eager chain's backward is a 14.2M-to-12-element reduction
# per table, and it is 110x slower end to end.
RELATION_BIAS_FUSION = env_flag("PVZ_RELATION_BIAS_FUSION", default=True)
FLEX_ATTENTION_AVAILABLE = flex_attention is not None and hasattr(torch, "compile")
_COMPILED_FLEX_ATTENTION = (
    torch.compile(flex_attention, dynamic=True) if FLEX_ATTENTION_AVAILABLE else None
)


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


# Four threads was the best measured setting on both machines tested (an Apple
# M5 Pro and an i7-12700F), and raising the thread count did not change any output
# bitwise.  The sweep that produced this lived in ``scripts/thread_effect.py``,
# which was deleted with the frozen search teacher it also measured.
DEFAULT_TORCH_THREADS = 4


def configure_torch_threads(requested: int = 0) -> int:
    """Pin the CPU thread count and return the value actually applied.

    The training entry points used to hard-code ``torch.set_num_threads(1)`` right
    next to ``random.seed`` / ``torch.manual_seed`` -- a reproducibility choice.
    That pin is free on Apple Silicon (Accelerate already saturates a single
    thread) but expensive on an x86 desktop, where four threads run the
    batch-of-1 search shape **2.0x** faster and the 64-step training window
    **1.7x** faster.

    Thread count does perturb the low-order bits, but far below anything that
    matters here.  Measured against ``threads=1``:

    ==================  ====================  =====================
    quantity            i7-12700F (rel.)      M5 Pro (rel.)
    ==================  ====================  =====================
    value forward       4.0e-07               0 (bit-identical)
    gameplay step       2.3e-07               0 (bit-identical)
    after 64-step win.  6.1e-07               7.0e-10
    ==================  ====================  =====================

    The decision error budget measured on the real simulator is 1e-5..1e-4, so
    these differences are two or more orders of magnitude too small to flip a
    search decision or change training behaviour.

    Reproducibility survives as long as the value is *pinned*: the same machine
    at the same thread count is deterministic either way.  Pass ``--threads 1``
    when you need bit-compatibility with artifacts from an older run; ``0`` (the
    default) means :data:`DEFAULT_TORCH_THREADS`.
    """
    resolved = DEFAULT_TORCH_THREADS if requested <= 0 else requested
    torch.set_num_threads(resolved)
    return resolved


def _ratio(value: float, scale: float) -> float:
    return max(-2.0, min(2.0, float(value) / scale))


ECONOMIC_PLANT_TYPES = frozenset({1, 9, 38, 41})
SHOOTER_PLANT_TYPES = frozenset({
    0, 5, 7, 8, 10, 13, 18, 24, 26, 28, 29, 32, 34, 39, 40, 42, 43, 44, 47, 52,
})


def derive_observation_features(observation: dict[str, Any]) -> dict[str, Any]:
    """Compute trainable row summaries and global economy/wave features from an observation."""
    row_threat = [0.0] * 6
    row_nearest = [1.0] * 6
    row_shooters = [0] * 6
    row_plant_health = [0.0] * 6
    row_zombie_seen = [False] * 6
    economic_count = 0
    shooter_count = 0

    for zombie in observation["zombies"]:
        row = int(zombie["row"])
        if not 0 <= row < 6:
            continue
        distance = max(0.0, min(1.0, (float(zombie["x"]) - 40.0) / 860.0))
        health = max(0.0, float(zombie["body_health"]) + float(zombie["helm_health"])
                     + float(zombie["shield_health"]))
        row_threat[row] += (1.0 - distance) * (1.0 + max(0.0, min(1.0, health / 2000.0)))
        row_nearest[row] = min(row_nearest[row], distance)
        row_zombie_seen[row] = True

    for plant in observation["plants"]:
        row = int(plant["row"])
        if not 0 <= row < 6:
            continue
        plant_type = int(plant["type"])
        if plant_type == 48 and int(plant.get("imitater_type", -1)) >= 0:
            plant_type = int(plant["imitater_type"])
        if plant_type in ECONOMIC_PLANT_TYPES:
            economic_count += 1
        if plant_type in SHOOTER_PLANT_TYPES:
            shooter_count += 1
            row_shooters[row] += 1
        row_plant_health[row] += max(0.0, float(plant["health"]))

    lane_features = [
        (_ratio(row_threat[row], 5.0), row_nearest[row] if row_zombie_seen[row] else 1.0,
         _ratio(row_shooters[row], 5.0), _ratio(row_plant_health[row], 3000.0))
        for row in range(6)
    ]
    wave_count = max(1, int(observation["wave_count"]))
    wave_progress = max(0.0, min(1.0, float(observation["wave"]) / wave_count))
    return {
        "lane_features": lane_features,
        "sun_income_rate": float(observation["sun_income_rate"]),
        "economic_fire_ratio": economic_count / max(1, shooter_count),
        "wave_progress": wave_progress,
        "next_wave_distance": max(0.0, min(1.0, float(observation["wave_timer"]) / 6000.0)),
    }


def observation_tokens(observation: dict[str, Any]) -> tuple[dict[str, Tensor], dict[str, Any]]:
    kinds: list[int] = []
    categories: list[int] = []
    variants: list[int] = []
    features: list[list[float]] = []
    rows: list[int] = []
    cols: list[int] = []
    packet_tokens: dict[int, int] = {}
    cell_tokens: dict[int, int] = {}
    lane_tokens: dict[int, int] = {}
    derived = derive_observation_features(observation)

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
        _ratio(derived["sun_income_rate"], 50.0),
        _ratio(derived["economic_fire_ratio"], 1.0),
        derived["wave_progress"], derived["next_wave_distance"],
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

    for row, values in enumerate(derived["lane_features"]):
        lane_tokens[row] = add("lane", row=row, values=values)

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
        # Rollouts store packed token features as float16. Round here as well so
        # inference and PPO replay consume the same values instead of allowing a
        # storage conversion to change a close action ranking.
        "features": torch.tensor(features, dtype=torch.float16).to(torch.float32),
        "rows": torch.tensor(rows, dtype=torch.long),
        "cols": torch.tensor(cols, dtype=torch.long),
    }
    return tensors, {"packet_tokens": packet_tokens, "cell_tokens": cell_tokens,
                     "lane_tokens": lane_tokens}


TOKEN_ID_FIELDS = ("kinds", "categories", "variants", "rows", "cols")


def pack_tokens(tensors: dict[str, Tensor], metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten a tokenization into contiguous numpy arrays for storage.

    A raw observation costs ~42 KiB as Python objects (measured: 4.0 KiB pickled,
    so the object representation inflates it 10.6x).  The packed form is
    5*8 + 32*4 = 168 bytes per token -- about 11 KiB for a typical 70-token
    board -- and pickles as two arrays instead of six tensors plus nested dicts.
    """
    ids = np.stack([tensors[field].numpy() for field in TOKEN_ID_FIELDS], axis=1).astype(np.int8)
    packet_ids = sorted(metadata["packet_tokens"])
    return {
        "ids": ids,
        "features": tensors["features"].numpy().astype(np.float16),
        "cell_index": np.array([metadata["cell_tokens"][cell] for cell in range(54)], dtype=np.uint16),
        "packet_ids": np.array(packet_ids, dtype=np.uint8),
        "packet_index": np.array([metadata["packet_tokens"][packet] for packet in packet_ids], dtype=np.uint16),
    }


def unpack_tokens(packed: dict[str, Any], device: torch.device) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Inverse of :func:`pack_tokens`.  Cheap enough to call per training step."""
    ids = torch.from_numpy(np.ascontiguousarray(packed["ids"])).to(device=device, dtype=torch.long)
    tensors = {field: ids[:, index].contiguous() for index, field in enumerate(TOKEN_ID_FIELDS)}
    tensors["features"] = torch.from_numpy(np.ascontiguousarray(packed["features"])).to(
        device=device, dtype=torch.float32)
    packet_ids = [int(value) for value in packed["packet_ids"]]
    packet_index = [int(value) for value in packed["packet_index"]]
    metadata = {
        "packet_tokens": dict(zip(packet_ids, packet_index)),
        "cell_tokens": {cell: int(packed["cell_index"][cell]) for cell in range(54)},
    }
    return tensors, metadata


def legal_summary(legal_actions: dict[str, Any]) -> dict[str, Any]:
    """Compact view of ``observation["legal_actions"]`` (9.3 KiB as objects).

    Keeps exactly the sets ``select_action`` reads: which packets are plantable,
    which cells each packet may go on, which cells are shovellable, and whether
    waiting is allowed.  Cells are stored as 54-bit masks.
    """
    by_packet: dict[int, int] = {}
    for item in legal_actions.get("plants", ()):
        packet = int(item["packet"])
        by_packet[packet] = by_packet.get(packet, 0) | (1 << (int(item["row"]) * 9 + int(item["col"])))
    shovel_mask = 0
    for col, row in legal_actions.get("shovels", ()):
        shovel_mask |= 1 << (int(row) * 9 + int(col))
    return {
        "packets": tuple(sorted(by_packet)),
        "plant_mask": tuple(by_packet[packet] for packet in sorted(by_packet)),
        "shovel_mask": shovel_mask,
        "wait": bool(legal_actions.get("wait", True)),
    }


def _legal_of(observation_or_summary: dict[str, Any]) -> dict[str, Any]:
    """Accept either a raw observation (has ``legal_actions``) or a summary."""
    if "legal_actions" in observation_or_summary:
        return legal_summary(observation_or_summary["legal_actions"])
    return observation_or_summary


def _mask_cells(mask: int) -> set[int]:
    return {cell for cell in range(54) if (mask >> cell) & 1}


class RelationBiasIndices(NamedTuple):
    """Integer indices into the per-layer relation-bias tables.

    These depend only on the discrete token layout (``kinds``/``rows``/``cols``),
    never on the learned features or on any layer's parameters, so the encoder
    computes them once and every layer reuses them.  This is the PvZ analogue of
    cross-layer index sharing: the indices cost more operator dispatches than the
    lookups they feed, and they are identical for all layers.
    """

    row_bucket: Tensor
    col_bucket: Tensor
    same_cell: Tensor


def relation_bias_indices(rows: Tensor, cols: Tensor) -> RelationBiasIndices:
    """Compute the layer-independent relation-bias indices.

    ``rows``/``cols`` are ``(batch, count)``; the result is three
    ``(batch, count, count)`` integer tensors.
    """
    row_known = (rows[:, :, None] >= 0) & (rows[:, None, :] >= 0)
    col_known = (cols[:, :, None] >= 0) & (cols[:, None, :] >= 0)
    row_delta = (rows[:, :, None] - rows[:, None, :]).clamp(-5, 5) + 5
    col_delta = (cols[:, :, None] - cols[:, None, :]).clamp(-8, 8) + 8
    row_bucket = torch.where(row_known, row_delta, 11)
    col_bucket = torch.where(col_known, col_delta, 17)
    same_cell = (row_known & col_known & (rows[:, :, None] == rows[:, None, :])
                 & (cols[:, :, None] == cols[:, None, :])).long()
    return RelationBiasIndices(row_bucket=row_bucket, col_bucket=col_bucket, same_cell=same_cell)


def relation_bias_from_indices(kinds: Tensor, row_bucket: Tensor, col_bucket: Tensor,
                               same_cell: Tensor, kind_pair_bias: Tensor,
                               row_bias_weight: Tensor, col_bias_weight: Tensor,
                               same_bias_weight: Tensor) -> Tensor:
    """Assemble ``(batch, heads, count, count)`` relation bias from precomputed indices.

    Takes every table as an explicit argument so the whole chain can be fused by
    ``torch.compile``.

    Every term is assembled in ``(batch, count, count, heads)`` order and the
    heads dimension is moved last only once, at the end.  ``permute`` is a view,
    so the cost is not the permutation itself but the strided read the following
    addition has to perform: summing in ``(batch, count, count, heads)`` keeps all
    four operands contiguous and pays one strided read instead of four.  Measured
    on an Apple M5 Pro, one CPU thread, over four encoder layers: 1.263 ms ->
    0.957 ms for a single decision (108 tokens) and 178.6 ms -> 131.0 ms for a
    256-transition minibatch (80 tokens).  The addition order within each element
    is unchanged, so the result is bit-identical.

    ``kind_pair_bias`` is ``(heads, K, K)``, so the head dimension is moved last
    first; that permutation touches ``heads * K * K`` elements, not
    ``batch * count * count * heads``.
    """
    relation = kind_pair_bias.permute(1, 2, 0)[kinds[:, :, None], kinds[:, None, :]]
    relation = relation + row_bias_weight[row_bucket]
    relation = relation + col_bias_weight[col_bucket]
    relation = relation + same_bias_weight[same_cell]
    return relation.permute(0, 3, 1, 2)


_FUSED_RELATION_BIAS = None
_FUSED_RELATION_BIAS_FAILED = False


def fused_relation_bias() -> Any | None:
    """Lazily compiled :func:`relation_bias_from_indices`, or ``None`` if unavailable.

    ``dynamic=True`` keeps one compiled artefact across the token counts a board
    produces (measured 74-116) instead of recompiling per shape.
    """
    global _FUSED_RELATION_BIAS, _FUSED_RELATION_BIAS_FAILED
    if _FUSED_RELATION_BIAS_FAILED:
        return None
    if _FUSED_RELATION_BIAS is None:
        try:
            _FUSED_RELATION_BIAS = torch.compile(relation_bias_from_indices, dynamic=True)
        except Exception:  # noqa: BLE001 - fall back to the eager path
            _FUSED_RELATION_BIAS_FAILED = True
            return None
    return _FUSED_RELATION_BIAS


def use_fused_relation_bias() -> bool:
    """Whether the fused relation-bias path is enabled and available.

    Only the eager attention path consults this; the CUDA ``FlexAttention`` path
    computes the same bias inside its kernel regardless.
    """
    return RELATION_BIAS_FUSION and fused_relation_bias() is not None


def set_relation_bias_fusion(enabled: bool) -> bool:
    """Enable or disable the fused relation-bias path.

    Returns the value actually in effect (``False`` if compilation is
    unavailable).  Both paths are bit-identical, so this only trades start-up
    compile time against steady-state speed.
    """
    global RELATION_BIAS_FUSION
    RELATION_BIAS_FUSION = bool(enabled)
    if RELATION_BIAS_FUSION and fused_relation_bias() is None:
        RELATION_BIAS_FUSION = False
    return RELATION_BIAS_FUSION


class RelationAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.relation_bias_enabled = True
        self.use_flex_attention = False
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.kind_pair_bias = nn.Parameter(torch.zeros(heads, len(TOKEN_KINDS), len(TOKEN_KINDS)))
        self.row_bias = nn.Embedding(12, heads)
        self.col_bias = nn.Embedding(18, heads)
        self.same_cell_bias = nn.Embedding(2, heads)

    def forward(self, x: Tensor, kinds: Tensor, rows: Tensor, cols: Tensor,
                key_mask: Tensor | None = None,
                indices: RelationBiasIndices | None = None) -> Tensor:
        batch, count, width = x.shape
        if kinds.dim() == 1:
            # Single-step call: kinds/rows/cols are (count,).  Lift to (batch, count)
            # so relation biases compute per sample; for batch=1 the arithmetic is
            # identical to the original 1-D form.
            kinds = kinds.unsqueeze(0)
            rows = rows.unsqueeze(0)
            cols = cols.unsqueeze(0)
        qkv = self.qkv(x).view(batch, count, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if (self.use_flex_attention and _COMPILED_FLEX_ATTENTION is not None
                and query.device.type == "cuda" and batch >= 32):
            kind_bias = self.kind_pair_bias
            row_bias = self.row_bias.weight
            col_bias = self.col_bias.weight
            same_bias = self.same_cell_bias.weight

            def relation_score(score: Tensor, batch_index: Tensor, head: Tensor,
                               query_index: Tensor, key_index: Tensor) -> Tensor:
                query_kind = kinds[batch_index, query_index]
                key_kind = kinds[batch_index, key_index]
                query_row = rows[batch_index, query_index]
                key_row = rows[batch_index, key_index]
                query_col = cols[batch_index, query_index]
                key_col = cols[batch_index, key_index]
                row_known = (query_row >= 0) & (key_row >= 0)
                col_known = (query_col >= 0) & (key_col >= 0)
                row_bucket = torch.where(
                    row_known, (query_row - key_row).clamp(-5, 5) + 5, 11)
                col_bucket = torch.where(
                    col_known, (query_col - key_col).clamp(-8, 8) + 8, 17)
                same_cell = (row_known & col_known & (query_row == key_row)
                             & (query_col == key_col)).long()
                relation = (kind_bias[head, query_kind, key_kind]
                            + row_bias[row_bucket, head]
                            + col_bias[col_bucket, head]
                            + same_bias[same_cell, head])
                score = score + relation
                if key_mask is not None:
                    score = score.masked_fill(
                        ~key_mask[batch_index, key_index], torch.finfo(score.dtype).min)
                return score

            attended = _COMPILED_FLEX_ATTENTION(
                query, key, value,
                score_mod=relation_score if self.relation_bias_enabled else None,
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, count, width)
            return self.projection(attended)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_width ** -0.5)
        if key_mask is not None:
            # key_mask: (batch, count), True on real tokens. Use the dtype's finite
            # minimum so a padded query row remains finite under FP16 autocast too.
            # Padded queries are never gathered.
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if self.relation_bias_enabled:
            if indices is None:
                indices = relation_bias_indices(rows, cols)
            if use_fused_relation_bias():
                relation = fused_relation_bias()(
                    kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
                    self.kind_pair_bias, self.row_bias.weight,
                    self.col_bias.weight, self.same_cell_bias.weight)
            else:
                relation = relation_bias_from_indices(
                    kinds, indices.row_bucket, indices.col_bucket, indices.same_cell,
                    self.kind_pair_bias, self.row_bias.weight,
                    self.col_bias.weight, self.same_cell_bias.weight)
            scores = scores + relation
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

    def forward(self, x: Tensor, kinds: Tensor, rows: Tensor, cols: Tensor,
                key_mask: Tensor | None = None,
                indices: RelationBiasIndices | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), kinds, rows, cols, key_mask, indices)
        normalized = self.ff_norm(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.value(normalized))
        return x

class GameplayModelV1(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = {**MODEL_CONFIG, **(config or {})}
        allowed = set(MODEL_CONFIG) | {"critic_width", "critic_layers"}
        if set(self.config) - allowed:
            raise ValueError(f"unknown model settings: {sorted(set(self.config) - allowed)}")
        if any(type(value) is not int or value < 1 for value in self.config.values()):
            raise ValueError("model dimensions must be positive integers")
        if self.config["width"] % self.config["heads"]:
            raise ValueError("encoder width must be divisible by attention heads")
        width = self.config["width"]
        # Preserve the seed-0 initialization stream for every pre-existing weight.
        # The new lane-kind row is initialized with a private generator, so adding
        # this vocabulary entry does not shift all later parameters' RNG draws.
        self.kind_embedding = nn.Embedding(TOKEN_KINDS["lane"], width)
        lane_generator = torch.Generator(device="cpu").manual_seed(50_210)
        lane_embedding = torch.empty(1, width).normal_(generator=lane_generator)
        self.kind_embedding.weight = nn.Parameter(torch.cat((self.kind_embedding.weight.detach(), lane_embedding)))
        self.category_embedding = nn.Embedding(128, width)
        self.variant_embedding = nn.Embedding(128, width)
        self.feature_projection = nn.Sequential(nn.Linear(FEATURE_COUNT, width), nn.SiLU(), nn.Linear(width, width))
        self.row_embedding = nn.Embedding(8, width)
        self.col_embedding = nn.Embedding(11, width)
        self.encoder = nn.ModuleList([
            RelationLayer(width, self.config["heads"], self.config["ff_width"])
            for _ in range(self.config["layers"])
        ])
        self.encoder_norm = nn.LayerNorm(width)

        self.action_embedding = nn.Embedding(3, 64)
        self.previous_packet_embedding = nn.Embedding(12, 64)
        self.previous_cell_embedding = nn.Embedding(55, 64)
        self.previous_wait_embedding = nn.Embedding(len(WAIT_TICKS) + 1, 64)
        self.previous_action_projection = nn.Sequential(nn.Linear(256, 64), nn.SiLU())
        self.delta_embedding = nn.Embedding(32, 32)
        self.event_projection = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.belief = nn.GRU(width + 128, self.config["gru_width"], self.config["gru_layers"], batch_first=True)
        hidden = self.config["gru_width"]
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
        critic_width = self.config.get("critic_width", hidden)
        critic_layers = self.config.get("critic_layers", 1)
        critic: list[nn.Module] = [nn.Linear(hidden + 64, critic_width), nn.Tanh()]
        for _ in range(critic_layers - 1):
            critic.extend((nn.Linear(critic_width, critic_width), nn.Tanh()))
        critic.append(nn.Linear(critic_width, 1))
        self.privileged_critic = nn.Sequential(*critic)

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
        tensors, metadata = observation_tokens(observation)
        return self.step_tokens(tensors, metadata, observation["wave"], hidden,
                                previous_action, delta_ticks, events)

    def step_tokens(self, tensors: dict[str, Tensor], metadata: dict[str, Any], wave: int,
                    hidden: Tensor | None = None, previous_action: dict[str, Any] | None = None,
                    delta_ticks: int = 0, events: dict[str, Any] | None = None) -> dict[str, Any]:
        """Same forward as :meth:`step`, but over pre-tokenized input.

        Rollouts tokenize once and store the packed tokens; training reuses them
        for every PPO epoch instead of re-reading the 42 KiB observation dict
        each time.  Tokenization is stateless, so results are identical.
        """
        device = next(self.parameters()).device
        kinds = tensors["kinds"].to(device)
        rows = tensors["rows"].to(device)
        cols = tensors["cols"].to(device)
        x = self.kind_embedding(kinds) + self.category_embedding(tensors["categories"].to(device))
        x = x + self.variant_embedding(tensors["variants"].to(device))
        x = x + self.feature_projection(tensors["features"].to(device))
        x = x + self.row_embedding((rows + 1).clamp(0, 7)) + self.col_embedding((cols + 1).clamp(0, 10))
        x = x.unsqueeze(0)
        # The relation-bias indices depend only on the token layout, so compute
        # them once for the whole encoder instead of once per layer.
        indices = relation_bias_indices(rows.unsqueeze(0), cols.unsqueeze(0))
        for layer in self.encoder:
            x = layer(x, kinds, rows, cols, indices=indices)
        x = self.encoder_norm(x)

        action_vector = self._previous_action(previous_action, device)
        delta_index = min(31, max(0, int(math.log2(max(0, delta_ticks) + 1))))
        delta_vector = self.delta_embedding(torch.tensor([delta_index], device=device))
        event_vector = self.event_projection(self._event_features(events, device))
        recurrent_input = torch.cat((x[:, 0, :], action_vector, delta_vector, event_vector), dim=-1).unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(self.config["gru_layers"], 1, self.config["gru_width"], device=device)
        belief, hidden = self.belief(recurrent_input, hidden)
        belief = belief[:, 0, :]

        packet_ids = sorted(metadata["packet_tokens"])
        cell_ids = list(range(54))
        packet_tokens = (torch.stack([x[0, metadata["packet_tokens"][i]] for i in packet_ids])
                         if packet_ids else x.new_zeros((0, self.config["width"])))
        cell_tokens = torch.stack([x[0, metadata["cell_tokens"][i]] for i in cell_ids])
        packet_logits = (self.packet_key(packet_tokens) * self.packet_query(belief)).sum(-1) / math.sqrt(self.config["width"])
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
            "wave_index": wave,
        }

    def _previous_action_batch(self, actions: list[dict[str, Any] | None], device: torch.device) -> Tensor:
        """Batch form of ``_previous_action``; None rows stay exactly zero."""
        action_types = {"plant": 0, "shovel": 1, "wait": 2}
        kinds, packets, cells, durations = [], [], [], []
        for action in actions:
            if action is None:
                kinds.append(0)
                packets.append(0)
                cells.append(0)
                durations.append(0)
                continue
            kind = action_types[action["type"]]
            packet = max(0, min(10, int(action.get("packet", -1)) + 1))
            cell = 0
            if "row" in action and "col" in action:
                cell = max(0, min(54, int(action["row"]) * 9 + int(action["col"]) + 1))
            ticks = action.get("ticks", 0)
            duration = (min(range(len(WAIT_TICKS)), key=lambda i: abs(WAIT_TICKS[i] - ticks)) + 1
                        if ticks else 0)
            kinds.append(kind)
            packets.append(packet)
            cells.append(cell)
            durations.append(duration)
        index = torch.tensor(list(zip(kinds, packets, cells, durations)), dtype=torch.long, device=device)
        parts = torch.cat((
            self.action_embedding(index[:, 0]),
            self.previous_packet_embedding(index[:, 1]),
            self.previous_cell_embedding(index[:, 2]),
            self.previous_wait_embedding(index[:, 3]),
        ), dim=-1)
        vector = self.previous_action_projection(parts)
        present = torch.tensor([action is not None for action in actions], device=device)
        return vector * present[:, None].to(vector.dtype)

    @staticmethod
    def _event_features_batch(events_list: list[dict[str, Any] | None], device: torch.device) -> Tensor:
        rows = []
        for events in events_list:
            events = events or {}
            rows.append([
                _ratio(events.get("zombies_killed", 0), 10), _ratio(events.get("plants_eaten", 0), 5),
                _ratio(events.get("sun_produced", 0), 250), _ratio(events.get("sun_spent", 0), 300),
                float(events.get("mower_triggered", 0) > 0), _ratio(events.get("waves_started", 0), 3),
                float(events.get("level_won", False)), float(events.get("level_lost", False)),
            ])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def forward_sequences(self, sequences: list[list[dict[str, Any]]],
                          hiddens: list[Tensor | None]) -> tuple[list[dict[str, Any]], Tensor]:
        """Run many independent sequences as one batched forward.

        This is the unit of work for layered PPO: every episode is cut into
        fixed-length chunks, chunks at the same position across episodes form a
        layer, and a layer is one forward_sequences call.  Variable sequence
        lengths are handled with pack_padded_sequence, so the returned hidden is
        each sequence's own true final hidden (pad steps never touch it).

        Equivalent to calling step_tokens once per transition (identical weight
        paths, padding keys masked out of attention); GEMM blocking differs, so
        results agree to ~1e-6 rather than bitwise.

        Returns (outputs flat in sequence order, hidden_out (gru_layers, B, H)).
        """
        device = next(self.parameters()).device
        lengths_seq = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
        t_max = int(lengths_seq.max())
        flat = [transition for sequence in sequences for transition in sequence]
        count = len(flat)
        offsets = np.concatenate(([0], np.cumsum(lengths_seq)))

        # token padding over all flattened transitions
        packed_list = [transition["tokens"] for transition in flat]
        token_lengths = np.array([packed["ids"].shape[0] for packed in packed_list], dtype=np.int64)
        l_max = int(token_lengths.max())
        ids = np.zeros((count, l_max, len(TOKEN_ID_FIELDS)), dtype=np.int64)
        features = np.zeros((count, l_max, FEATURE_COUNT), dtype=np.float16)
        key_mask = np.zeros((count, l_max), dtype=bool)
        for index, packed in enumerate(packed_list):
            real = packed["ids"].shape[0]
            ids[index, :real] = packed["ids"]
            features[index, :real] = packed["features"]
            key_mask[index, :real] = True
        p_max = max(max(1, packed["packet_ids"].shape[0]) for packed in packed_list)
        packet_index = np.zeros((count, p_max), dtype=np.int64)
        packet_counts = []
        for index, packed in enumerate(packed_list):
            many = packed["packet_ids"].shape[0]
            packet_counts.append(many)
            if many:
                packet_index[index, :many] = packed["packet_index"]
        cell_index = np.stack([packed["cell_index"] for packed in packed_list])

        ids_t = torch.from_numpy(ids).to(device)
        kinds = ids_t[:, :, 0]
        rows = ids_t[:, :, 3]
        cols = ids_t[:, :, 4]
        x = (self.kind_embedding(kinds)
             + self.category_embedding(ids_t[:, :, 1])
             + self.variant_embedding(ids_t[:, :, 2])
             + self.feature_projection(torch.from_numpy(features).to(device=device, dtype=torch.float32))
             + self.row_embedding((rows + 1).clamp(0, 7))
             + self.col_embedding((cols + 1).clamp(0, 10)))
        mask = torch.from_numpy(key_mask).to(device)
        indices = relation_bias_indices(rows, cols)
        for layer in self.encoder:
            x = layer(x, kinds, rows, cols, mask, indices)
        x = self.encoder_norm(x)

        action_vector = self._previous_action_batch(
            [transition["previous_action"] for transition in flat], device)
        delta_index = torch.tensor(
            [min(31, max(0, int(math.log2(max(0, int(transition["elapsed_since_previous_observation"])) + 1))))
             for transition in flat], dtype=torch.long, device=device)
        delta_vector = self.delta_embedding(delta_index)
        event_vector = self.event_projection(self._event_features_batch(
            [transition["events"] for transition in flat], device))
        recurrent_flat = torch.cat((x[:, 0, :], action_vector, delta_vector, event_vector), dim=-1)

        # regroup the flattened steps into (B, T_max, D) grid for the GRU;
        # grid[b, t] = flat index for real steps, -1 on padding
        batch = len(sequences)
        grid = np.full((batch, t_max), -1, dtype=np.int64)
        for b in range(batch):
            grid[b, :lengths_seq[b]] = np.arange(offsets[b], offsets[b + 1])
        grid_t = torch.from_numpy(grid).to(device)
        recurrent = recurrent_flat[grid_t.clamp(min=0)]
        seq_lengths = torch.from_numpy(lengths_seq).to(device)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            recurrent, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden_in = torch.zeros(self.config["gru_layers"], batch, self.config["gru_width"], device=device)
        for b, start_hidden in enumerate(hiddens):
            if start_hidden is not None:
                hidden_in[:, b, :] = start_hidden
        belief_seq, hidden_out = self.belief(packed_input, hidden_in)
        belief_padded, _ = nn.utils.rnn.pad_packed_sequence(belief_seq, batch_first=True, total_length=t_max)
        belief = belief_padded[grid_t >= 0]  # (N, H); row-major mask order == flat order

        type_logits = self.action_type(belief)
        wait_logits = self.wait_duration(belief)
        value_out = torch.tanh(self.value(belief))
        aux_next_spawn = self.aux_next_spawn(belief)
        aux_lane_threat = self.aux_lane_threat(belief)
        aux_outcome = self.aux_outcome(belief)
        packet_query_all = self.packet_query(belief)
        arange = torch.arange(count, device=device)
        cell_index_t = torch.from_numpy(cell_index).to(device=device, dtype=torch.long)
        packet_index_t = torch.from_numpy(packet_index).to(device=device, dtype=torch.long)
        cell_tokens = x[arange[:, None], cell_index_t]
        cell_keys = self.cell_key(cell_tokens)
        packet_tokens = x[arange[:, None], packet_index_t]
        packet_keys = self.packet_key(packet_tokens)
        packet_logits = (packet_keys * packet_query_all[:, None, :]).sum(-1) / math.sqrt(self.config["width"])

        outputs = []
        for index, transition in enumerate(flat):
            many = packet_counts[index]
            outputs.append({
                "hidden": None,
                "belief": belief[index:index + 1],
                "type_logits": type_logits[index],
                "wait_logits": wait_logits[index],
                "packet_logits": packet_logits[index, :many],
                "packet_ids": [int(value) for value in transition["tokens"]["packet_ids"]],
                "packet_tokens": packet_tokens[index, :many],
                "cell_tokens": cell_tokens[index],
                "cell_keys": cell_keys[index],
                "value": value_out[index],
                "aux_next_spawn": aux_next_spawn[index:index + 1],
                "aux_lane_threat": aux_lane_threat[index:index + 1],
                "aux_outcome": aux_outcome[index:index + 1],
                "wave_index": transition["wave"],
            })
        return outputs, hidden_out

    def plant_cell_scores(self, output: dict[str, Any], packet: int) -> Tensor:
        packet_index = output["packet_ids"].index(packet)
        query = self.plant_cell_query(torch.cat((output["belief"], output["packet_tokens"][packet_index].unsqueeze(0)), dim=-1))
        return (output["cell_keys"] * query).sum(-1) / math.sqrt(self.config["width"])

    def shovel_cell_scores(self, output: dict[str, Any]) -> Tensor:
        query = self.shovel_cell_query(output["belief"])
        return (output["cell_keys"] * query).sum(-1) / math.sqrt(self.config["width"])

    def privileged_extra(self, privileged_state: dict[str, Any] | None, wave_index: int) -> list[float]:
        """Reduce a privileged state to the 16 floats the critic actually reads.

        Measured: a full privileged_state costs 76.9 KiB as Python objects, but
        only ``hidden["wave_timer"]`` and the current wave's zombie roster feed
        the critic -- 99.7% of the stored bytes were never used.  Rollouts store
        these 16 floats (196 bytes) instead of the state dict.
        """
        if not privileged_state:
            return [0.0] * 16
        hidden = privileged_state.get("hidden", {})
        waves = hidden.get("zombies_in_wave", [])
        current = min(max(0, wave_index), max(0, len(waves) - 1))
        return self.privileged_extra_from_inputs(
            hidden.get("wave_timer", 0), waves[current] if waves else [])

    @staticmethod
    def privileged_extra_from_inputs(wave_timer: int, wave_zombies: list[int]) -> list[float]:
        values = [0.0] * 16
        values[0] = _ratio(wave_timer, 6000)
        for zombie_type in wave_zombies[:15]:
            if 0 <= zombie_type < 15:
                values[1 + zombie_type] += 0.1
        return values

    def privileged_value_from_extra(self, output: dict[str, Any], extra: list[float]) -> Tensor:
        device = output["belief"].device
        extra_features = self.privileged_features(torch.tensor([extra], dtype=torch.float32, device=device))
        return self.privileged_critic(torch.cat((output["belief"], extra_features), dim=-1))

    def privileged_value_batch(self, belief: Tensor, extras: Tensor) -> Tensor:
        """Batch form of ``privileged_value_from_extra``: belief (T,H), extras (T,16)."""
        extra_features = self.privileged_features(extras)
        return self.privileged_critic(torch.cat((belief, extra_features), dim=-1))


def select_action(model: GameplayModelV1, output: dict[str, Any], observation: dict[str, Any],
                  action: dict[str, Any] | None = None, deterministic: bool = False) -> tuple[dict[str, Any], Tensor, Tensor]:
    device = output["type_logits"].device
    legal = _legal_of(observation)
    plant_masks = dict(zip(legal["packets"], legal["plant_mask"]))
    valid_packets = list(legal["packets"])
    valid_shovels = _mask_cells(legal["shovel_mask"])
    wait_allowed = legal["wait"]
    type_logits = output["type_logits"].clone()
    if not valid_packets:
        type_logits[0] = torch.finfo(type_logits.dtype).min
    if not valid_shovels:
        type_logits[1] = torch.finfo(type_logits.dtype).min
    if not wait_allowed:
        type_logits[2:] = torch.finfo(type_logits.dtype).min
    type_dist = torch.distributions.Categorical(logits=type_logits)
    action_types = {"plant": 0, "shovel": 1, "wait": 2}
    if action is not None and action.get("type") not in action_types:
        raise ValueError(f"unsupported action type: {action.get('type')}")
    type_index = ((type_logits.argmax() if deterministic else type_dist.sample()) if action is None
                  else torch.tensor(action_types[action["type"]], device=device))
    if action is not None and ((int(type_index.item()) == 0 and not valid_packets)
                               or (int(type_index.item()) == 1 and not valid_shovels)
                               or (int(type_index.item()) >= 2 and not wait_allowed)):
        raise ValueError(f"action is illegal in this observation: {action}")
    log_prob = type_dist.log_prob(type_index)
    entropy = type_dist.entropy()
    action_type = int(type_index.item())

    if action_type == 0:
        packet_logits = output["packet_logits"].clone()
        allowed = set(valid_packets)
        for index, packet_id in enumerate(output["packet_ids"]):
            if packet_id not in allowed:
                packet_logits[index] = torch.finfo(packet_logits.dtype).min
        packet_dist = torch.distributions.Categorical(logits=packet_logits)
        if action is not None and action["packet"] not in allowed:
            raise ValueError(f"illegal plant packet: {action}")
        packet_index = ((packet_logits.argmax() if deterministic else packet_dist.sample()) if action is None
                        else torch.tensor(output["packet_ids"].index(action["packet"]), device=device))
        packet = output["packet_ids"][int(packet_index.item())]
        log_prob = log_prob + packet_dist.log_prob(packet_index)
        entropy = entropy + packet_dist.entropy()
        cell_logits = model.plant_cell_scores(output, packet).clone()
        valid_cells = _mask_cells(plant_masks[packet])
        for cell in range(54):
            if cell not in valid_cells:
                cell_logits[cell] = torch.finfo(cell_logits.dtype).min
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
                cell_logits[cell] = torch.finfo(cell_logits.dtype).min
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


def replay_log_probs(model: GameplayModelV1, outputs: list[dict[str, Any]],
                     transitions: list[dict[str, Any]]) -> tuple[Tensor, Tensor]:
    """Batched PPO replay: (log_prob, entropy) of the recorded actions.

    Semantically identical to calling ``select_action(model, output, legal,
    action=recorded)`` per transition and stacking the results, but evaluates
    every distribution in one tensor op per chunk instead of ~30 small ops per
    transition.  Only the sampling call sites differ (replay never samples);
    masking follows the same finite minimum convention, so probabilities match to
    GEMM tolerance (~1e-6).

    Every mask and index table is built as a Python object first and moved to the
    device in one copy.  Building them by writing into a device tensor row by row
    -- which is what this function used to do -- costs one kernel launch per row:
    measured on an Apple M5 Pro, the same arithmetic takes 2.9 ms on the CPU and
    51.3 ms on MPS for 256 transitions, a 17.5x penalty with no compute behind it.
    A row-at-a-time write is the whole difference.

    The recorded packet index is deliberately kept as a Python ``int``: reading a
    scalar out of a device tensor (``int(sel_pos[r])``) synchronises the device,
    so doing it once per plant row serialised the update 150 times per minibatch.
    """
    device = outputs[0]["type_logits"].device
    total = len(outputs)
    action_types = {"plant": 0, "shovel": 1, "wait": 2}

    type_logits = torch.stack([output["type_logits"] for output in outputs])
    wait_logits = torch.stack([output["wait_logits"] for output in outputs])
    belief = torch.cat([output["belief"] for output in outputs], dim=0)
    cell_keys = torch.stack([output["cell_keys"] for output in outputs])

    # ``packet_logits`` rows are padded to the widest row in the chunk.  The pad
    # columns are masked out below by ``packet_mask``, so the fill value is only a
    # placeholder; what matters is that the scatter puts each row's real logits in
    # that row's leading columns.  The scatter is built from shapes and from one
    # ``torch.cat``, never from ``float(tensor_element)``: reading a scalar off a
    # device tensor synchronises it, and doing that per packet row cost 1,536
    # synchronisations per call.
    p_max = max(1, max(output["packet_logits"].shape[0] for output in outputs))
    packet_ids_rows = [list(output["packet_ids"]) for output in outputs]
    counts = [output["packet_logits"].shape[0] for output in outputs]
    packet_logits = torch.full((total, p_max), PACKET_LOGIT_FLOOR, device=device)
    if any(counts):
        packet_logits = packet_logits.index_put(
            (torch.tensor([row for row, count in enumerate(counts) for _ in range(count)],
                          dtype=torch.long, device=device),
             torch.tensor([column for count in counts for column in range(count)],
                          dtype=torch.long, device=device)),
            torch.cat([output["packet_logits"] for output in outputs]))

    type_index = torch.tensor([action_types[tr["action"]["type"]] for tr in transitions],
                              dtype=torch.long, device=device)
    type_mask = torch.tensor(
        [[bool(tr["legal"]["packets"]), bool(tr["legal"]["shovel_mask"] != 0), bool(tr["legal"]["wait"])]
         for tr in transitions],
        dtype=torch.bool, device=device)
    type_dist = torch.distributions.Categorical(
        logits=type_logits.masked_fill(~type_mask, torch.finfo(type_logits.dtype).min))
    log_prob = type_dist.log_prob(type_index)
    entropy = type_dist.entropy()

    plant_rows = [i for i, tr in enumerate(transitions) if tr["action"]["type"] == "plant"]
    shovel_rows = [i for i, tr in enumerate(transitions) if tr["action"]["type"] == "shovel"]
    wait_rows = [i for i, tr in enumerate(transitions) if tr["action"]["type"] == "wait"]
    add_lp = torch.zeros(total, device=device)
    add_ent = torch.zeros(total, device=device)

    if plant_rows:
        # Python ints, so the packet gather below never reads a scalar off the device.
        selected_positions = [
            packet_ids_rows[i].index(transitions[i]["action"]["packet"]) for i in plant_rows]
        packet_mask = torch.tensor(
            [[pid in set(transitions[i]["legal"]["packets"]) for pid in packet_ids_rows[i]]
             + [False] * (p_max - len(packet_ids_rows[i]))
             for i in plant_rows],
            dtype=torch.bool, device=device)
        packet_dist = torch.distributions.Categorical(
            logits=packet_logits[plant_rows].masked_fill(
                ~packet_mask, torch.finfo(packet_logits.dtype).min))
        selected_tensor = torch.tensor(selected_positions, dtype=torch.long, device=device)
        lp_packet = packet_dist.log_prob(selected_tensor)
        ent_packet = packet_dist.entropy()

        sel_tokens = torch.stack([outputs[i]["packet_tokens"][position]
                                  for i, position in zip(plant_rows, selected_positions)])
        query = model.plant_cell_query(torch.cat((belief[plant_rows], sel_tokens), dim=-1))
        cell_logits = (torch.einsum("nw,ntw->nt", query, cell_keys[plant_rows])
                       / math.sqrt(model.config["width"]))
        bits = torch.tensor([dict(zip(transitions[i]["legal"]["packets"],
                                      transitions[i]["legal"]["plant_mask"]))[
                                  transitions[i]["action"]["packet"]] for i in plant_rows],
                            dtype=torch.long, device=device)
        cell_mask = ((bits[:, None] >> torch.arange(54, device=device)[None, :]) & 1) == 1
        cell_dist = torch.distributions.Categorical(
            logits=cell_logits.masked_fill(~cell_mask, torch.finfo(cell_logits.dtype).min))
        cell_index = torch.tensor([transitions[i]["action"]["row"] * 9 + transitions[i]["action"]["col"]
                                   for i in plant_rows], dtype=torch.long, device=device)
        lp_cell = cell_dist.log_prob(cell_index)
        ent_cell = cell_dist.entropy()

        rows_t = torch.tensor(plant_rows, dtype=torch.long, device=device)
        add_lp = add_lp.index_put((rows_t,), lp_packet + lp_cell)
        add_ent = add_ent.index_put((rows_t,), ent_packet + ent_cell)

    if shovel_rows:
        query = model.shovel_cell_query(belief[shovel_rows])
        cell_logits = (torch.einsum("nw,ntw->nt", query, cell_keys[shovel_rows])
                       / math.sqrt(model.config["width"]))
        bits = torch.tensor([transitions[i]["legal"]["shovel_mask"] for i in shovel_rows],
                            dtype=torch.long, device=device)
        cell_mask = ((bits[:, None] >> torch.arange(54, device=device)[None, :]) & 1) == 1
        cell_dist = torch.distributions.Categorical(
            logits=cell_logits.masked_fill(~cell_mask, torch.finfo(cell_logits.dtype).min))
        cell_index = torch.tensor([transitions[i]["action"]["row"] * 9 + transitions[i]["action"]["col"]
                                   for i in shovel_rows], dtype=torch.long, device=device)
        rows_t = torch.tensor(shovel_rows, dtype=torch.long, device=device)
        add_lp = add_lp.index_put((rows_t,), cell_dist.log_prob(cell_index))
        add_ent = add_ent.index_put((rows_t,), cell_dist.entropy())

    if wait_rows:
        duration = torch.tensor([
            min(range(len(WAIT_TICKS)),
                key=lambda k: abs(WAIT_TICKS[k] - transitions[i]["action"].get("ticks", 150)))
            for i in wait_rows], dtype=torch.long, device=device)
        wait_dist = torch.distributions.Categorical(logits=wait_logits[wait_rows])
        rows_t = torch.tensor(wait_rows, dtype=torch.long, device=device)
        add_lp = add_lp.index_put((rows_t,), wait_dist.log_prob(duration))
        add_ent = add_ent.index_put((rows_t,), wait_dist.entropy())

    return log_prob + add_lp, entropy + add_ent


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
        type_logits[0] = torch.finfo(type_logits.dtype).min
    if not valid_shovels:
        type_logits[1] = torch.finfo(type_logits.dtype).min
    if not legal.get("wait", True):
        type_logits[2:] = torch.finfo(type_logits.dtype).min
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
                packet_logits[index] = torch.finfo(packet_logits.dtype).min
        packet_index = output["packet_ids"].index(action["packet"])
        losses.append(F.cross_entropy(packet_logits.unsqueeze(0), torch.tensor([packet_index], device=device)))
        cell_logits = model.plant_cell_scores(output, action["packet"]).clone()
        valid_cells = {a["row"] * 9 + a["col"] for a in legal["plants"] if a["packet"] == action["packet"]}
        for cell in range(54):
            if cell not in valid_cells:
                cell_logits[cell] = torch.finfo(cell_logits.dtype).min
        target_cell = action["row"] * 9 + action["col"]
        losses.append(F.cross_entropy(cell_logits.unsqueeze(0), torch.tensor([target_cell], device=device)))
    elif target_type == 1:
        cell_logits = model.shovel_cell_scores(output).clone()
        for cell in range(54):
            if cell not in valid_shovels:
                cell_logits[cell] = torch.finfo(cell_logits.dtype).min
        target_cell = action["row"] * 9 + action["col"]
        losses.append(F.cross_entropy(cell_logits.unsqueeze(0), torch.tensor([target_cell], device=device)))
    elif target_type == 2:
        target_ticks = action.get("ticks", 150)
        if target_ticks not in WAIT_TICKS:
            raise ValueError(f"wait ticks must be one of {WAIT_TICKS}: {action}")
        duration = min(range(len(WAIT_TICKS)), key=lambda i: abs(WAIT_TICKS[i] - target_ticks))
        losses.append(F.cross_entropy(output["wait_logits"].unsqueeze(0), torch.tensor([duration], device=device)))
    return torch.stack(losses).sum()


def factored_action_log_prob(model: GameplayModelV1, output: dict[str, Any], observation: dict[str, Any],
                             action: dict[str, Any]) -> Tensor:
    """``log p(action)`` under the model's factored action distribution.

    ``select_action`` already computes exactly this, and it is the same
    factorisation ``hard_behavior_cloning_loss`` assumes when it sums the
    type/packet/cell cross-entropies: a plant is scored as
    ``p(type) * p(packet) * p(cell | packet)``, so the three factors can never be
    recombined into an action the teacher never proposed.

    Routing the soft-label term through ``select_action`` rather than re-deriving
    the masking is what makes ``-factored_action_log_prob`` provably equal to
    ``hard_behavior_cloning_loss`` for the same action, and the tests pin that.
    """
    _, log_prob, _ = select_action(model, output, observation, action=action)
    return log_prob


def soft_behavior_cloning_loss(model: GameplayModelV1, output: dict[str, Any], observation: dict[str, Any],
                               candidates: list[dict[str, Any]], policy: list[float],
                               plant_weight: float = 1.0) -> Tensor:
    """Expected negative log-likelihood of the search teacher's *whole* candidate set.

    The teacher emits a distribution over every root candidate
    (``SearchTeacher._policy_from_results`` concentrates it on the best outcome
    class and softmaxes the scores within it).  Training only on its argmax --
    which is what ``hard_behavior_cloning_loss`` does -- discards the margin
    between the runner-up lines, even though ``collect_search_episode`` already
    writes that distribution into every step.

    ``plant_weight`` is applied per candidate and the weights are then
    renormalised, so the result stays an expectation (weights sum to one) while
    plant candidates carry the same upweighting the hard loss gives its type
    factor.
    """
    if len(candidates) != len(policy):
        raise ValueError(f"{len(candidates)} candidates but {len(policy)} policy entries")
    weighted: list[tuple[dict[str, Any], float]] = []
    for action, probability in zip(candidates, policy):
        weight = max(0.0, float(probability))
        if weight <= 0.0:
            continue
        if plant_weight != 1.0 and action.get("type") == "plant":
            weight *= plant_weight
        weighted.append((action, weight))
    total = sum(weight for _, weight in weighted)
    if total <= 0.0:
        raise ValueError("search policy assigns no probability to any candidate")
    return torch.stack([
        -(weight / total) * factored_action_log_prob(model, output, observation, action)
        for action, weight in weighted
    ]).sum()
