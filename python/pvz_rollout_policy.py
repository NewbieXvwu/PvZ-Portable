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
from torch import nn

from pvz_value_net import BOARD_CHANNELS, COLS, ROWS, SCALARS, featurize_numpy

PACKETS = 10            # seed-bank slots (SEEDBANK_MAX)
LAWN_COLS = 9
EXTRA = 2               # previous action was a plant; ticks since the previous decision / 300
WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}


def features(frame: dict, previous: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
    board, scalars = featurize_numpy(frame)
    action = frame.get("action") if previous is not None else None
    extra = np.array([float(bool(action) and action.get("type") == "plant"),
                      ((frame["tick"] - previous["tick"]) / 300.0) if previous else 0.0], dtype=np.float32)
    return torch.from_numpy(board), torch.from_numpy(np.concatenate([scalars, extra]))


class RolloutNet(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, (1, 5), padding=(0, 2)), nn.ReLU(),
        )
        self.scalar = nn.Sequential(nn.Linear(SCALARS + EXTRA, width), nn.ReLU())
        self.plant = nn.Linear(2 * width, 1)
        self.packet = nn.Linear(2 * width, PACKETS)
        self.packet_embedding = nn.Embedding(PACKETS, width)
        # Each cell sees its own features, the chosen packet and the global context.
        self.cell = nn.Sequential(nn.Conv2d(4 * width, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1))

    def forward(self, board: torch.Tensor, scalars: torch.Tensor, packet: torch.Tensor | None = None):
        h, g = self.encode(board, scalars)
        out = {"plant": self.plant(g).squeeze(1), "packet": self.packet(g)}
        if packet is not None:
            out["cell"] = self.cell_logits(h, g, packet)
        return out

    def encode(self, board: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(board)                                   # [B, W, 6, 10]
        s = self.scalar(scalars)                                # [B, W]
        return h, torch.cat([h.mean(dim=(2, 3)), s], dim=1)

    def cell_logits(self, h: torch.Tensor, g: torch.Tensor, packet: torch.Tensor) -> torch.Tensor:
        e = self.packet_embedding(packet)[:, :, None, None].expand(-1, -1, ROWS, COLS)
        ctx = g[:, :, None, None].expand(-1, -1, ROWS, COLS)
        return self.cell(torch.cat([h, e, ctx], dim=1))[:, 0, :, :LAWN_COLS].flatten(1)  # [B, 54]


@torch.inference_mode()
def act(net: RolloutNet, frame: dict, previous: dict | None, legal_plants: list[dict]) -> dict:
    """Greedy action restricted to the simulator's legal plants; otherwise wait like pi does."""
    if not legal_plants:
        return dict(WAIT)
    board, scalars = features(frame, previous)
    # The trunk is computed once and shared by the three heads: the cell head used to rerun the
    # whole network on the same input, which gives the same numbers at twice the cost.
    h, g = net.encode(board[None], scalars[None])
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
    weights = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    net = RolloutNet(weights["trunk.0.weight"].shape[0])
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")})
    return net.eval()
