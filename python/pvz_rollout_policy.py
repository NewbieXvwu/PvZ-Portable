"""Fast rollout policy: a small net that imitates pi, for long look-ahead in search (DESIGN.md §13.5).

AlphaGo's search used a rollout policy far weaker than its main policy but about a thousand
times faster. Here pi costs ~18 ms per step at batch size 1 (the 7.9M-parameter transformer),
while the simulator itself costs ~1.3 ms, so look-ahead is limited by pi. This net reads the same
public board encoding as the value network (``pvz_value_net.featurize``) and predicts pi's next
action. In practice pi only ever plays ``plant(packet, row, col)`` or ``wait 60 / timeout``, so
there are three heads: plant-or-wait, which packet, and which cell given the packet.

Imitating pi itself is learnable (pi is nearly deterministic); what stage 3 could not do was
imitate the search's counter-intuitive choices.
"""
from __future__ import annotations

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn

from pvz_value_net import BOARD_CHANNELS, COLS, CONTEXT, ROWS, SCALARS, featurize_numpy
from pvz_value_net import PLANT_TYPES as PlantTypes
from pvz_student_observation import OBS_EXTRA

PACKETS = 10            # seed-bank slots (SEEDBANK_MAX)
LAWN_COLS = 9
EXTRA = 2               # previous action was a plant; ticks since the previous decision / 300
WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}


# LEGACY(conveyor-start-key): only the L10 conveyor tasks carry the key.
def packet_features(frame: dict) -> torch.Tensor:
    """Public per-slot [plant type, active, cooldown progress, affordable]; -1 marks an empty slot.

    Kept separate from the legacy scalar encoding: duplicate cards can have different cooldowns.
    No slot position is a learned feature of the identity head.
    """
    result = np.zeros((PACKETS, 4), dtype=np.float32)
    result[:, 0] = -1
    # The engine reports the start mode it actually used. On the original conveyor start
    # SeedPacket::CanPickUp skips the sun check when the board has a conveyor bank, so the
    # nominal packet cost is not a budget gate there. This is the engine's own flag, not a
    # Python "conveyor is free" rule; tasks without it keep the nominal-cost comparison.
    free_cards = frame.get('conveyor_start') == 'original'
    for slot, kind, active, cooldown, refresh, cost in frame['packets']:
        if not 0 <= slot < PACKETS or kind < 0:
            continue
        result[slot] = (min(kind, PlantTypes - 1), bool(active),
                        min(1.0, 1.0 if active or not refresh else cooldown / refresh),
                        True if free_cards else frame['sun'] >= cost)
    return torch.from_numpy(result)


def identity_wave_scalar(frame: dict) -> float:
    """Canonical public wave-count feature; legacy featurize overwrites it for SEED_NONE (-1)."""
    waves = max(1, frame['scene'].get('wave_count') or 1)
    return math.log1p(waves) / math.log1p(50)


def identity_scalars(frame: dict, scalars: torch.Tensor) -> torch.Tensor:
    """Use with identity inputs, including direct encode/forward callers; slot features stay legacy."""
    result = scalars.clone()
    result[4] = identity_wave_scalar(frame)
    return result


def features(frame: dict, previous: dict | None, context: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Board and scalars; with ``context`` (``pvz_value_net.observation_context``: zombie-count multiplier and
    the level's zombie roster) appended for students trained with ``--context``."""
    board, scalars = featurize_numpy(frame)
    action = frame.get("action") if previous is not None else None
    extra = np.array([float(bool(action) and action.get("type") == "plant"),
                      ((frame["tick"] - previous["tick"]) / 300.0) if previous else 0.0], dtype=np.float32)
    parts = [scalars, extra] if context is None else [scalars, extra, np.asarray(context, dtype=np.float32)]
    return torch.from_numpy(board), torch.from_numpy(np.concatenate(parts))


class RolloutNet(nn.Module):
    def __init__(self, width: int = 64, pool: str = "mean", context: bool = False, obs: str = "v1",
                 row_pool: bool = False, aux: bool = False, packet_head: str = "slot",
                 plant_embedding_dim: int | None = None):
        super().__init__()
        if obs not in OBS_EXTRA:
            raise ValueError(f"unknown student observation version: {obs}")
        self.obs = obs
        if packet_head not in ('slot', 'identity'):
            raise ValueError(f'unknown packet head: {packet_head}')
        self.packet_head = packet_head
        self.plant_embedding_dim = width if plant_embedding_dim is None else plant_embedding_dim
        if self.plant_embedding_dim < 1 or (packet_head == 'slot' and self.plant_embedding_dim != width):
            raise ValueError('invalid plant embedding dimension')
        # With context the student also reads the task (zombie-count multiplier, zombie roster): early in a
        # 1.5x game the board looks like a 1.0x game, but the teacher already plays differently (2026-10-16:
        # v12 copies the teacher's early wall-nuts 13% of the time).
        self.context = context
        # "meanmax" adds a max over the board to the global context: the mean dilutes one lane's
        # threat over 54 cells, which a cherry-bomb or wall-nut decision is about (2026-10-16).
        self.pool = pool
        g = (3 if pool == "meanmax" else 2) * width
        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS + OBS_EXTRA[obs][0], width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, (1, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.scalar = nn.Sequential(nn.Linear(SCALARS + EXTRA + (CONTEXT if context else 0)
                                             + OBS_EXTRA[obs][1], width), nn.ReLU())
        self.plant = nn.Linear(g, 1)
        e = self.plant_embedding_dim
        self.packet = (nn.Linear(g, PACKETS) if packet_head == 'slot' else
                       nn.Sequential(nn.Linear(g + e + 3, width), nn.ReLU(), nn.Linear(width, 1)))
        self.packet_embedding = nn.Embedding(PACKETS if packet_head == 'slot' else PlantTypes, e)
        # Each cell sees its own features, the chosen packet and the global context.
        self.cell = nn.Sequential(nn.Conv2d(width + e + g, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1))
        self.row_pool = row_pool
        self.row_projection = nn.Conv2d(2 * width, width, 1) if row_pool else None
        self.aux = nn.Conv2d(width, 1, 1) if aux else None

    def forward(self, board: torch.Tensor, scalars: torch.Tensor, packet: torch.Tensor | None = None,
                auxiliary: bool = False, packet_info: torch.Tensor | None = None):
        h, g = self.encode(board, scalars, packet_info=packet_info)
        out = {"plant": self.plant(g).squeeze(1), "packet": self.packet_logits(g, packet_info)}
        if packet is not None:
            out["cell"] = self.cell_logits(h, g, packet, packet_info)
        if auxiliary:
            out["aux"] = self.aux(h).mean(dim=3).squeeze(1)
        return out

    def encode(self, board: torch.Tensor, scalars: torch.Tensor,
               sparse_input: bool = False, packet_info: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.packet_head == 'identity':
            self._check_packet_info(packet_info)
            # Symmetric per-type aggregate, including duplicate cards; legacy features use last slot.
            scalars = scalars.clone()
            scalars[:, 5:113] = 0
            kinds = packet_info[:, :, 0].long()
            valid = kinds >= 0
            ready = packet_info[:, :, 1] * packet_info[:, :, 3]
            for start, values in ((5, ready), (59, packet_info[:, :, 2])):
                scalars[:, start:start + PlantTypes].scatter_reduce_(
                    1, kinds.clamp(min=0), values.masked_fill(~valid, 0), reduce='amax', include_self=True)
        if sparse_input:
            h = self._trunk_sparse(board)
        else:
            h = self.trunk(board)                               # [B, W, 6, 10]
        if self.row_pool:
            rows = torch.cat([h.mean(dim=3, keepdim=True), h.amax(dim=3, keepdim=True)], dim=1)
            h = F.relu(h + self.row_projection(rows))
        s = self.scalar(scalars)                                # [B, W]
        if self.pool == "meanmax":
            return h, torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), s], dim=1)
        return h, torch.cat([h.mean(dim=(2, 3)), s], dim=1)

    def _trunk_sparse(self, board: torch.Tensor) -> torch.Tensor:
        """The trunk with the first convolution restricted to input channels that are non-zero
        somewhere in the batch. A board has ~7 of 117 channels in use (one per plant/zombie type on
        the lawn, plus lanes/mowers), and an all-zero channel adds exactly 0 to every output, so the
        result is the same convolution; scripts/perf_rollout_check.py checks it bitwise."""
        conv = self.trunk[0]
        active = board.flatten(2).any(2).any(0).nonzero().squeeze(1)
        if active.numel() == 0:
            return self.trunk(board)
        x = F.conv2d(board.index_select(1, active), conv.weight.index_select(1, active), conv.bias,
                     padding=conv.padding)
        return self.trunk[1:](x)

    @staticmethod
    def _check_packet_info(packet_info):
        if packet_info is None or packet_info.ndim != 3 or packet_info.shape[1:] != (PACKETS, 4):
            raise ValueError('identity packet head requires [batch,10,4] public packet_info')

    def packet_logits(self, g: torch.Tensor, packet_info: torch.Tensor | None = None) -> torch.Tensor:
        # LEGACY(slot-packet-head): v34 and earlier students pick a card slot, not a plant identity.
        if self.packet_head == 'slot':
            return self.packet(g)
        self._check_packet_info(packet_info)
        kind = packet_info[:, :, 0].long()
        embedding = self.packet_embedding(kind.clamp(min=0))
        global_context = g[:, None, :].expand(-1, PACKETS, -1)
        logits = self.packet(torch.cat([global_context, embedding, packet_info[:, :, 1:]], dim=2)).squeeze(2)
        # CanPickUp/CanPlantAt are authoritative at action selection: conveyor cards are free
        # even when their displayed price exceeds the current sun. Do not invent a budget gate.
        return logits.masked_fill(kind < 0, -1e9)

    def cell_logits(self, h: torch.Tensor, g: torch.Tensor, packet: torch.Tensor,
                    packet_info: torch.Tensor | None = None) -> torch.Tensor:
        if self.packet_head == 'identity':
            self._check_packet_info(packet_info)
            packet = packet_info[:, :, 0].long().gather(1, packet[:, None]).squeeze(1).clamp(min=0)
        e = self.packet_embedding(packet)[:, :, None, None].expand(-1, -1, ROWS, COLS)
        ctx = g[:, :, None, None].expand(-1, -1, ROWS, COLS)
        return self.cell(torch.cat([h, e, ctx], dim=1))[:, 0, :, :LAWN_COLS].flatten(1)  # [B, 54]


@torch.inference_mode()
def act(net: RolloutNet, frame: dict, previous: dict | None, legal_plants: list[dict],
        sparse_input: bool = True, context: np.ndarray | None = None, history=None) -> dict:
    """Greedy action restricted to the simulator's legal plants; otherwise wait like pi does."""
    if net.obs != "v1":
        if history is None:
            raise ValueError(f"{net.obs} requires per-episode StudentHistory, updated at every decision")
        if history.obs != net.obs:
            raise ValueError("student history observation version differs from network")
        board, scalars = history.features(frame, previous, context if net.context else None)
    if not legal_plants:
        return dict(WAIT)
    if net.context and context is None:
        raise ValueError("this student reads the task context: pass act(..., context=observation_context(obs))")
    if net.obs == "v1":
        board, scalars = features(frame, previous, context if net.context else None)
    # The trunk is computed once and shared by the three heads: the cell head used to rerun the
    # whole network on the same input, which gives the same numbers at twice the cost.
    info = packet_features(frame)[None] if net.packet_head == 'identity' else None
    if info is not None:
        scalars = identity_scalars(frame, scalars)
    h, g = net.encode(board[None], scalars[None], sparse_input=sparse_input, packet_info=info)
    if float(net.plant(g)[0, 0]) <= 0.0:
        return dict(WAIT)
    by_packet: dict[int, list[dict]] = {}
    for a in legal_plants:
        by_packet.setdefault(a["packet"], []).append(a)
    packet_logits = net.packet_logits(g, info)[0].tolist()
    packet = max(by_packet, key=(lambda p: (packet_logits[p], -int(info[0, p, 0])))
                 if info is not None else (lambda p: packet_logits[p] if p < PACKETS else -1e9))
    cells = net.cell_logits(h, g, torch.tensor([packet]), info)[0].tolist()
    best = max(by_packet[packet], key=lambda a: cells[a["row"] * LAWN_COLS + a["col"]])
    return {"type": "plant", "packet": packet, "col": best["col"], "row": best["row"]}


@torch.inference_mode()
def ranked_plants(net: RolloutNet, frame: dict, previous: dict | None, legal_plants: list[dict],
                  context: np.ndarray | None = None, history=None, sparse_input: bool = True) -> list[tuple[float, dict]]:
    """The student's own preference order over the *legal plant actions*, best first.

    Same heads as `act` (plant-or-wait, packet, cell), but every legal action gets a score instead
    of only the argmax: P(plant) x P(packet | legal packets) x P(cell | legal cells of that packet).
    `act` is the greedy corner of this distribution, so the first entry is exactly the action `act`
    returns whenever it plants -- scripts/search_play.py --student-candidates widens the teacher's
    candidate set with entries 2..k. Without that flag nothing calls this function and the teacher's
    candidate set is bit-identical to before.
    """
    if not legal_plants:
        return []
    if net.obs != "v1":
        if history is None:
            raise ValueError(f"{net.obs} requires per-episode StudentHistory, updated at every decision")
        if history.obs != net.obs:
            raise ValueError("student history observation version differs from network")
        board, scalars = history.features(frame, previous, context if net.context else None)
    else:
        if net.context and context is None:
            raise ValueError("this student reads the task context: pass context=observation_context(obs)")
        board, scalars = features(frame, previous, context if net.context else None)
    info = packet_features(frame)[None] if net.packet_head == 'identity' else None
    if info is not None:
        scalars = identity_scalars(frame, scalars)
    h, g = net.encode(board[None], scalars[None], sparse_input=sparse_input, packet_info=info)
    plant_p = float(torch.sigmoid(net.plant(g))[0, 0])
    by_packet: dict[int, list[dict]] = {}
    for a in legal_plants:
        by_packet.setdefault(a["packet"], []).append(a)
    packets = sorted(by_packet)
    packet_p = torch.softmax(net.packet_logits(g, info)[0][packets], dim=0).tolist()
    out: list[tuple[float, dict]] = []
    for packet, p_packet in zip(packets, packet_p):
        cells = net.cell_logits(h, g, torch.tensor([packet]), info)[0]
        index = torch.tensor([a["row"] * LAWN_COLS + a["col"] for a in by_packet[packet]])
        cell_p = torch.softmax(cells[index], dim=0).tolist()
        for a, p_cell in zip(by_packet[packet], cell_p):
            out.append((plant_p * p_packet * p_cell, a))
    out.sort(key=(lambda item: (-item[0], int(info[0, item[1]['packet'], 0]),
                               item[1]['row'], item[1]['col'], item[1]['packet']))
             if info is not None else (lambda item: -item[0]))
    return out


def warm_start_student(net: RolloutNet, initial: RolloutNet) -> None:
    """Copy a parent's function into a wider, semantically compatible input.

    Only prefix encodings may be widened: fog/size/lite inputs carry different
    features and cannot be reinterpreted as boss/bungee/sleep channels.
    """
    for name in ('pool', 'context', 'row_pool', 'packet_head', 'plant_embedding_dim'):
        if getattr(net, name) != getattr(initial, name):
            raise ValueError(f'warm-start architecture differs: {name}')
    upgrades = {('v1', 'v2'), ('v2', 'v2fog')}
    upgrades.update((obs, 'v2sleep') for obs in ('v1', 'v2', 'v2boss', 'v2bungee'))
    # v2timer 只是 v2sleep 后面再接 4 个通道，凡是它的前缀都能零权重加宽过来。
    upgrades.update((obs, 'v2timer') for obs in ('v1', 'v2', 'v2boss', 'v2bungee', 'v2sleep'))
    if net.obs != initial.obs and (initial.obs, net.obs) not in upgrades:
        raise ValueError(f'no function-preserving input upgrade: {initial.obs} -> {net.obs}')
    weights = dict(initial.state_dict())
    target = net.state_dict()
    for name in ('trunk.0.weight', 'scalar.0.weight'):
        old, new = weights[name], target[name]
        if old.shape[0] != new.shape[0] or old.shape[2:] != new.shape[2:] or old.shape[1] > new.shape[1]:
            raise ValueError(f'warm-start layer shape differs: {name}')
        expanded = old.new_zeros(new.shape)
        expanded[:, :old.shape[1]] = old
        weights[name] = expanded
    if net.aux is not None and initial.aux is None:
        weights.update({name: value for name, value in target.items() if name.startswith('aux.')})
    net.load_state_dict(weights)


def load_student(path) -> RolloutNet:
    """A student or fast_ppo checkpoint as a RolloutNet in eval mode (width read from the weights;
    fast_ppo's value head is dropped)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["state_dict"]
    net = RolloutNet(weights["trunk.0.weight"].shape[0], checkpoint.get("pool", "mean"),
                     bool(checkpoint.get("context", False)), checkpoint.get("obs", "v1"),
                     bool(checkpoint.get("row_pool", False)), packet_head=checkpoint.get('packet_head', 'slot'),
                     plant_embedding_dim=checkpoint.get('plant_embedding_dim'))
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith(("value.", "aux."))})
    return net.eval()


def export_student(net: RolloutNet, path) -> None:
    """v6 adds an identity packet MLP/shared plant embeddings; legacy exports remain byte-identical."""
    import struct
    names = ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
             "trunk.4.weight", "trunk.4.bias", "scalar.0.weight", "scalar.0.bias",
             "plant.weight", "plant.bias", "packet.weight", "packet.bias", "packet_embedding.weight",
             "cell.0.weight", "cell.0.bias", "cell.2.weight", "cell.2.bias")
    weights = net.state_dict()
    if net.packet_head == 'identity':
        names = tuple(n.replace('packet.weight', 'packet.0.weight').replace('packet.bias', 'packet.0.bias') for n in names)
    if net.row_pool:
        names += ("row_projection.weight", "row_projection.bias")
    if net.packet_head == 'identity':
        names += ('packet.2.weight', 'packet.2.bias')
    with open(path, "wb") as stream:
        identity = net.packet_head == 'identity'
        stream.write((b"PVZRN006" if identity else b"PVZRN005" if net.row_pool else b"PVZRN004" if net.obs != "v1" else b"PVZRN003" if net.context else b"PVZRN002")
                     + struct.pack("<II", net.trunk[0].out_channels, int(net.pool == "meanmax")))
        if identity or net.row_pool or net.context or net.obs != "v1":
            stream.write(struct.pack("<I", int(net.context)))
        if identity or net.row_pool or net.obs != "v1":
            stream.write(struct.pack("<I", 9 if net.obs == "v2timer" else 8 if net.obs == "v2sleep" else 7 if net.obs == "v2bungee" else 6 if net.obs == "v2boss" else 5 if net.obs == "v2size" else 4 if net.obs == "v2fog" else 3 if net.obs == "v2" else 2 if net.obs == "v2lite" else 1))
        if identity or net.row_pool:
            stream.write(struct.pack("<I", int(net.row_pool)))
        if identity:
            stream.write(struct.pack('<II', 1, net.plant_embedding_dim))
        for name in names:
            stream.write(weights[name].detach().cpu().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export student / fast_ppo weights for the C++ simulator")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    options = parser.parse_args()
    export_student(load_student(options.checkpoint), options.output)
