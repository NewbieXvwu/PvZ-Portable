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

import torch
from torch import nn

from pvz_value_net import BOARD_CHANNELS, COLS, ROWS, SCALARS, featurize

PACKETS = 10            # seed-bank slots (SEEDBANK_MAX)
LAWN_COLS = 9
EXTRA = 2               # previous action was a plant; ticks since the previous decision / 300
WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}


def features(frame: dict, previous: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
    board, scalars = featurize(frame)
    action = frame.get("action") if previous is not None else None
    extra = torch.tensor([float(bool(action) and action.get("type") == "plant"),
                          ((frame["tick"] - previous["tick"]) / 300.0) if previous else 0.0])
    return board, torch.cat([scalars, extra])


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
        h = self.trunk(board)                                   # [B, W, 6, 10]
        s = self.scalar(scalars)                                # [B, W]
        g = torch.cat([h.mean(dim=(2, 3)), s], dim=1)
        out = {"plant": self.plant(g).squeeze(1), "packet": self.packet(g)}
        if packet is not None:
            e = self.packet_embedding(packet)[:, :, None, None].expand(-1, -1, ROWS, COLS)
            ctx = g[:, :, None, None].expand(-1, -1, ROWS, COLS)
            out["cell"] = self.cell(torch.cat([h, e, ctx], dim=1))[:, 0, :, :LAWN_COLS].flatten(1)  # [B, 54]
        return out


@torch.inference_mode()
def act(net: RolloutNet, frame: dict, previous: dict | None, legal_plants: list[dict]) -> dict:
    """Greedy action restricted to the simulator's legal plants; otherwise wait like pi does."""
    if not legal_plants:
        return dict(WAIT)
    board, scalars = features(frame, previous)
    out = net(board[None], scalars[None])
    if float(out["plant"][0]) <= 0.0:
        return dict(WAIT)
    by_packet: dict[int, list[dict]] = {}
    for a in legal_plants:
        by_packet.setdefault(a["packet"], []).append(a)
    packet = max(by_packet, key=lambda p: float(out["packet"][0, p]) if p < PACKETS else -1e9)
    cells = net(board[None], scalars[None], torch.tensor([packet]))["cell"][0]
    best = max(by_packet[packet], key=lambda a: float(cells[a["row"] * LAWN_COLS + a["col"]]))
    return {"type": "plant", "packet": packet, "col": best["col"], "row": best["row"]}
