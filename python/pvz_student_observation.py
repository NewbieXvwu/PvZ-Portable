"""History-derived student inputs using only the archived public frames."""
import numpy as np
import torch

from pvz_value_net import CELL_PX, COLS, LAWN_XMIN, ROWS, _col_of

SPEED_SCALE = 1.0


class StudentHistory:
    def __init__(self):
        self.plants = {}
        self.wave = None
        self.wave_tick = 0

    def extra(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        tick = frame["tick"]
        if frame["wave"] != self.wave:
            self.wave, self.wave_tick = frame["wave"], tick
        current = {(row, col, kind): self.plants.get((row, col, kind), tick)
                   for row, col, kind, *_ in frame["plants"]
                   if 0 <= row < ROWS and 0 <= col < COLS - 1}
        self.plants = current
        board = np.zeros((4, ROWS, COLS), dtype=np.float32)
        ages = {}
        for (row, col, _kind), born in current.items():
            ages[row, col] = min(ages.get((row, col), 1.0), (tick - born) / 1500.0)
        for (row, col), age in ages.items():
            board[3, row, col] = age
        totals = np.zeros((3, ROWS, COLS), dtype=np.float64)
        for zombie in frame["zombies"]:
            row, _kind, position, velocity, health, on_board = zombie[:6]
            if not on_board or not 0 <= row < ROWS:
                continue
            col = _col_of(position)
            fraction = min(1.0, max(0.0, (position - LAWN_XMIN - col * CELL_PX) / CELL_PX))
            totals[:, row, col] += (health * fraction, health * velocity / SPEED_SCALE, health)
            board[2, row, col] = max(board[2, row, col], velocity / SPEED_SCALE)
        np.divide(totals[:2], totals[2:3], out=board[:2], where=totals[2:3] > 0)
        return board, np.array([min(1.0, (tick - self.wave_tick) / 3000.0)], dtype=np.float32)

    def features(self, frame: dict, previous: dict | None, context=None):
        from pvz_rollout_policy import features
        board, scalars = features(frame, previous, context)
        extra_board, extra_scalars = self.extra(frame)
        return (torch.cat([board, torch.from_numpy(extra_board)]),
                torch.cat([scalars, torch.from_numpy(extra_scalars)]))
