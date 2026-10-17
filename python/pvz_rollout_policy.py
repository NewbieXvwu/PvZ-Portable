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
import torch
import torch.nn.functional as F
from torch import nn

from pvz_value_net import BOARD_CHANNELS, COLS, CONTEXT, ROWS, SCALARS, featurize_numpy

PACKETS = 10            # seed-bank slots (SEEDBANK_MAX)
LAWN_COLS = 9
EXTRA = 2               # previous action was a plant; ticks since the previous decision / 300
WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}


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
    def __init__(self, width: int = 64, pool: str = "mean", context: bool = False):
        super().__init__()
        # With context the student also reads the task (zombie-count multiplier, zombie roster): early in a
        # 1.5x game the board looks like a 1.0x game, but the teacher already plays differently (2026-10-16:
        # v12 copies the teacher's early wall-nuts 13% of the time).
        self.context = context
        # "meanmax" adds a max over the board to the global context: the mean dilutes one lane's
        # threat over 54 cells, which a cherry-bomb or wall-nut decision is about (2026-10-16).
        self.pool = pool
        g = (3 if pool == "meanmax" else 2) * width
        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, (1, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.scalar = nn.Sequential(nn.Linear(SCALARS + EXTRA + (CONTEXT if context else 0), width), nn.ReLU())
        self.plant = nn.Linear(g, 1)
        self.packet = nn.Linear(g, PACKETS)
        self.packet_embedding = nn.Embedding(PACKETS, width)
        # Each cell sees its own features, the chosen packet and the global context.
        self.cell = nn.Sequential(nn.Conv2d(2 * width + g, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1))

    def forward(self, board: torch.Tensor, scalars: torch.Tensor, packet: torch.Tensor | None = None):
        h, g = self.encode(board, scalars)
        out = {"plant": self.plant(g).squeeze(1), "packet": self.packet(g)}
        if packet is not None:
            out["cell"] = self.cell_logits(h, g, packet)
        return out

    def encode(self, board: torch.Tensor, scalars: torch.Tensor,
               sparse_input: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if sparse_input:
            h = self._trunk_sparse(board)
        else:
            h = self.trunk(board)                               # [B, W, 6, 10]
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

    def cell_logits(self, h: torch.Tensor, g: torch.Tensor, packet: torch.Tensor) -> torch.Tensor:
        e = self.packet_embedding(packet)[:, :, None, None].expand(-1, -1, ROWS, COLS)
        ctx = g[:, :, None, None].expand(-1, -1, ROWS, COLS)
        return self.cell(torch.cat([h, e, ctx], dim=1))[:, 0, :, :LAWN_COLS].flatten(1)  # [B, 54]


@torch.inference_mode()
def act(net: RolloutNet, frame: dict, previous: dict | None, legal_plants: list[dict],
        sparse_input: bool = True, context: np.ndarray | None = None) -> dict:
    """Greedy action restricted to the simulator's legal plants; otherwise wait like pi does."""
    if not legal_plants:
        return dict(WAIT)
    if net.context and context is None:
        raise ValueError("this student reads the task context: pass act(..., context=observation_context(obs))")
    board, scalars = features(frame, previous, context if net.context else None)
    # The trunk is computed once and shared by the three heads: the cell head used to rerun the
    # whole network on the same input, which gives the same numbers at twice the cost.
    h, g = net.encode(board[None], scalars[None], sparse_input=sparse_input)
    if float(net.plant(g)[0, 0]) <= 0.0:
        return dict(WAIT)
    by_packet: dict[int, list[dict]] = {}
    for a in legal_plants:
        by_packet.setdefault(a["packet"], []).append(a)
    packet_logits = net.packet(g)[0].tolist()
    packet = max(by_packet, key=lambda p: packet_logits[p] if p < PACKETS else -1e9)
    cells = net.cell_logits(h, g, torch.tensor([packet]))[0].tolist()
    best = max(by_packet[packet], key=lambda a: cells[a["row"] * LAWN_COLS + a["col"]])
    return {"type": "plant", "packet": packet, "col": best["col"], "row": best["row"]}


def load_student(path) -> RolloutNet:
    """A student or fast_ppo checkpoint as a RolloutNet in eval mode (width read from the weights;
    fast_ppo's value head is dropped)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["state_dict"]
    net = RolloutNet(weights["trunk.0.weight"].shape[0], checkpoint.get("pool", "mean"), bool(checkpoint.get("context", False)))
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")})
    return net.eval()


def export_student(net: RolloutNet, path) -> None:
    """Write little-endian v2 weights, or v3 with an additional task-context flag."""
    import struct
    names = ("trunk.0.weight", "trunk.0.bias", "trunk.2.weight", "trunk.2.bias",
             "trunk.4.weight", "trunk.4.bias", "scalar.0.weight", "scalar.0.bias",
             "plant.weight", "plant.bias", "packet.weight", "packet.bias", "packet_embedding.weight",
             "cell.0.weight", "cell.0.bias", "cell.2.weight", "cell.2.bias")
    weights = net.state_dict()
    with open(path, "wb") as stream:
        stream.write((b"PVZRN003" if net.context else b"PVZRN002")
                     + struct.pack("<II", net.trunk[0].out_channels, int(net.pool == "meanmax")))
        if net.context:
            stream.write(struct.pack("<I", 1))
        for name in names:
            stream.write(weights[name].detach().cpu().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export student / fast_ppo weights for the C++ simulator")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    options = parser.parse_args()
    export_student(load_student(options.checkpoint), options.output)
