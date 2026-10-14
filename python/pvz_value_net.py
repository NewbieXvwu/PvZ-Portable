"""Value network: estimate P(win) from a public board frame (DESIGN.md §13).

Input is the compact frame produced by ``scripts/render_episode._frame(obs)``, so stored
episodes and live environment observations go through exactly the same encoding.

Encoding: a 6-row x 10-column board tensor plus a small vector of global scalars.
Column 9 collects everything right of the lawn (zombies that have not entered yet), and
rows cover the 6-row pool layout so the same encoder works on every terrain.
Only public information is used -- nothing a player could not see on screen.
"""
from __future__ import annotations

import math

import torch
from torch import nn

ROWS = 6
COLS = 10                 # 9 lawn columns + 1 "beyond the lawn" column
LAWN_XMIN = 40
CELL_PX = 80
PLANT_TYPES = 54          # SeedType ids, clipped
ZOMBIE_TYPES = 34         # ZombieType ids, clipped

# Board channels.
CH_PLANT = 0                              # PLANT_TYPES channels: health fraction
CH_ZOMBIE = CH_PLANT + PLANT_TYPES        # ZOMBIE_TYPES channels: summed health / 200
CH_EATING = CH_ZOMBIE + ZOMBIE_TYPES      # zombies currently eating in this cell
CH_CHILLED = CH_EATING + 1                # chilled zombies in this cell
CH_MOWER = CH_CHILLED + 1                 # lawnmower ready (column 0 only)
CH_LANE = CH_MOWER + 1                    # row is a playable lane
BOARD_CHANNELS = CH_LANE + 1

# Global scalars: sun, wave progress, time, income, and per-plant-type packet state.
SCALAR_BASE = 5
SCALARS = SCALAR_BASE + 2 * PLANT_TYPES


def _col_of(x: float) -> int:
    return min(COLS - 1, max(0, int((x - LAWN_XMIN) // CELL_PX)))


def featurize(frame: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(board[C,6,10], scalars[S])`` for one frame."""
    board = torch.zeros(BOARD_CHANNELS, ROWS, COLS)
    for row, col, kind, health, max_health in frame["plants"]:
        if 0 <= row < ROWS and 0 <= col < COLS - 1:
            board[CH_PLANT + min(kind, PLANT_TYPES - 1), row, col] += health / max(1, max_health)
    for z in frame["zombies"]:
        row, kind, x, _vx, health, on_board, eating, chilled = z[:8]
        if not on_board or not 0 <= row < ROWS:
            continue
        col = _col_of(x)
        board[CH_ZOMBIE + min(kind, ZOMBIE_TYPES - 1), row, col] += health / 200.0
        board[CH_EATING, row, col] += float(eating)
        board[CH_CHILLED, row, col] += float(chilled > 0)
    for row, state in frame["mowers"]:
        if 0 <= row < ROWS:
            board[CH_MOWER, row, 0] = float(state == 1)
    for row in frame.get("lanes", []):
        if 0 <= row < ROWS:
            board[CH_LANE, row, :] = 1.0

    scalars = torch.zeros(SCALARS)
    wave_count = max(1, frame["scene"].get("wave_count") or 1)
    sun = frame["sun"]
    scalars[0] = min(sun, 2000) / 1000.0
    scalars[1] = frame["wave"] / wave_count
    scalars[2] = frame["tick"] / 60000.0
    scalars[3] = (frame.get("sun_income_rate") or 0.0) / 100.0
    scalars[4] = math.log1p(wave_count) / math.log1p(50)
    for _index, kind, active, cooldown, refresh, cost in frame["packets"]:
        kind = min(kind, PLANT_TYPES - 1)
        ready = bool(active) and sun >= cost
        scalars[SCALAR_BASE + kind] = float(ready)
        progress = 1.0 if active else (cooldown / refresh if refresh else 1.0)
        scalars[SCALAR_BASE + PLANT_TYPES + kind] = min(1.0, progress)
    return board, scalars


def potential_phi(frame: dict) -> float:
    """The shaping potential the PPO reward uses (notes/training.md), as a baseline score."""
    plants = frame["plants"]
    health = (sum(p[3] / max(1, p[4]) for p in plants) / len(plants)) if plants else 0.0
    wave_count = max(1, frame["scene"].get("wave_count") or 1)
    return (min(1.0, frame["sun"] / 1000) + min(1.0, frame["wave"] / wave_count) + health) / 3


class ValueNet(nn.Module):
    """Small conv net over the board, joined with the scalars, producing a win logit.

    Convolutions see neighbouring cells (a zombie two columns in front of a peashooter);
    a per-row max pool keeps "which lane is in trouble" before everything is averaged.
    """

    def __init__(self, width: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, (1, 3), padding=(0, 1)), nn.ReLU(),
        )
        self.scalar = nn.Sequential(nn.Linear(SCALARS, width), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(width * (2 + ROWS), width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, board: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        h = self.conv(board)                            # [B, W, 6, 10]
        rows = h.amax(dim=3).flatten(1)                 # [B, W*6] worst cell per lane
        pooled = h.mean(dim=(2, 3))                     # [B, W]
        x = torch.cat([pooled, rows, self.scalar(scalars)], dim=1)
        return self.head(x).squeeze(1)                  # logit of P(win)
