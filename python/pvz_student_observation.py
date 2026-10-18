"""History-derived student inputs using only the archived public frames."""
import numpy as np
import torch

from pvz_value_net import CELL_PX, COLS, LAWN_XMIN, ROWS, _col_of

SPEED_SCALE = 1.0
OBS_EXTRA = {"v1": (0, 0), "v2lite": (4, 1), "v2": (9, 7)}


def v2_extra(frame: dict, wave_tick: int) -> tuple[np.ndarray, np.ndarray]:
    board = np.zeros((9, ROWS, COLS), dtype=np.float32)
    totals = np.zeros((3, ROWS, COLS), dtype=np.float64)
    maximum = np.full((ROWS, COLS), -np.inf)
    raw = frame.get("v2")
    for index, zombie in enumerate(frame["zombies"]):
        row, kind, position, velocity, health, on_board = zombie[:6]
        if not on_board or not 0 <= row < ROWS:
            continue
        col = _col_of(position)
        fraction = min(1.0, max(0.0, (position - LAWN_XMIN - col * CELL_PX) / CELL_PX))
        totals[:, row, col] += (health * fraction, health * velocity / 0.7, health)
        maximum[row, col] = max(maximum[row, col], velocity / 0.7)
        if raw is not None:
            has_object, helm, shield = raw["zombies"][index]
            if kind == 3 and has_object:
                board[3, row, col] += health / 200.0
            board[4, row, col] += (helm + shield) / 370.0
            board[5, row, col] = max(float(board[5, row, col]), zombie[7] / 2000.0)
    np.divide(totals[:2], totals[2:3], out=board[:2], where=totals[2:3] > 0)
    board[2] = np.where(np.isfinite(maximum), maximum, 0.0)
    scalars = np.zeros(7, dtype=np.float32)
    scalars[5] = min(1.0, (frame["tick"] - wave_tick) / 3000.0)
    if raw is None:
        return board, scalars
    for row, col, kind, state, countdown, special, launch, rate in raw["plants"]:
        if not (0 <= row < ROWS and 0 <= col < COLS - 1):
            continue
        progress = 0.0
        if kind == 4 and state in (0, 15, 16):
            progress = 1.0 - countdown / 1500.0 if state == 0 else 1.0
        elif kind in (2, 20):
            progress = 1.0 - special / 100.0
        board[6, row, col] = max(float(board[6, row, col]), progress)
        if rate > 0:
            board[7, row, col] = max(float(board[7, row, col]), 1.0 - launch / rate)
    for row, position, damage in raw["projectiles"]:
        if 0 <= row < ROWS:
            board[8, row, _col_of(position)] += damage / 200.0
    scalars[:5] = (raw["wave_timer"] / 4500.0, float(raw["huge_wave_countdown"] > 0),
                   raw["wave_health"] / raw["wave_start_health"] if raw["wave_start_health"] else 0.0,
                   raw["sun_countdown"] / 1224.0, raw["sun_pending"] / 200.0)
    scalars[6] = 1.0
    return board, scalars


class StudentHistory:
    def __init__(self, obs: str = "v2lite"):
        self.obs = obs
        self.plants = {}
        self.wave = None
        self.wave_tick = 0

    def extra(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        tick = frame["tick"]
        if frame["wave"] != self.wave:
            self.wave, self.wave_tick = frame["wave"], tick
        if self.obs == "v2":
            return v2_extra(frame, self.wave_tick)
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
