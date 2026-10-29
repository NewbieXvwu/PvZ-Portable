"""History-derived student inputs using only the archived public frames."""
import numpy as np
import torch

from pvz_value_net import (CELL_PX, COLS, LAWN_XMIN, ROWS, _col_of, bungee_features, sleep_features,
                           timer_features)

SPEED_SCALE = 1.0
OBS_EXTRA = {"v1": (0, 0), "v2lite": (4, 1), "v2": (9, 7), "v2fog": (10, 7),
             "v2fogstack4": (112, 7), "v2size": (10, 7), "v2boss": (13, 9), "v2bungee": (19, 9), "v2sleep": (21, 9),
             # task145：v2sleep + 可见计时 4 通道（引爆/唤醒进度、状态进度、传送带卡序、卡是否就绪）
             "v2timer": (25, 9)}


def fog_channel(frame: dict) -> np.ndarray:
    """Engine fog opacity / 255, including the off-board rightmost encoder column."""
    board = np.zeros((1, ROWS, COLS), dtype=np.float32)
    if 'fog_amount' in frame:
        raw = np.asarray(frame['fog_amount'], dtype=np.float32)
        if raw.shape != (ROWS, COLS - 1):
            raise ValueError('fog_amount must be a 6 by 9 engine grid')
        board[0, :, :COLS - 1] = raw / 255.0
        board[0, :, -1] = raw[:, -1] / 255.0
    return board


def v2_extra(frame: dict, wave_tick: int) -> tuple[np.ndarray, np.ndarray]:
    board = np.zeros((13 if frame.get('_boss_features') else 9, ROWS, COLS), dtype=np.float32)
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
            has_object, helm, shield = raw["zombies"][index][:3]
            if kind == 3 and has_object:
                board[3, row, col] += health / 200.0
            board[4, row, col] += (helm + shield) / 370.0
            board[5, row, col] = max(float(board[5, row, col]), zombie[7] / 2000.0)
    np.divide(totals[:2], totals[2:3], out=board[:2], where=totals[2:3] > 0)
    board[2] = np.where(np.isfinite(maximum), maximum, 0.0)
    scalars = np.zeros(9 if frame.get('_boss_features') else 7, dtype=np.float32)
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
    if frame.get('_boss_features'):
        boss = frame.get('boss') or {}
        if boss.get('active'):
            board[9, 0, 0] = float(boss.get('head', False))
            projectile = boss.get('projectile')
            if projectile and 0 <= projectile.get('row', -1) < ROWS:
                board[10 if projectile.get('type') == 'fire' else 11, projectile['row'], _col_of(projectile.get('x', 0))] = 1.0
            if boss.get('action') in ('summon', 'stomp', 'bungee', 'rv'):
                row, col = boss.get('target_row', -1), boss.get('target_col', -1)
                if 0 <= row < ROWS and 0 <= col < COLS:
                    board[12, row, col] = 1.0
            scalars[7] = boss.get('stage', 0) / 2.0
            scalars[8] = boss.get('projectile', {}).get('row', 0) / 5.0 if boss.get('projectile') else 0.0
    return board, scalars


class StudentHistory:
    def __init__(self, obs: str = "v2lite"):
        self.obs = obs
        self.plants = {}
        self.wave = None
        self.wave_tick = 0
        self.zombie_frames = []

    def extra(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        if self.obs == 'v3':
            return np.zeros((0, ROWS, COLS), np.float32), np.zeros(0, np.float32)
        tick = frame["tick"]
        if frame["wave"] != self.wave:
            self.wave, self.wave_tick = frame["wave"], tick
        if self.obs in ("v2", "v2fog", "v2fogstack4", "v2size", "v2boss", "v2bungee", "v2sleep", "v2timer"):
            frame = {**frame, '_boss_features': self.obs in ('v2boss', 'v2bungee', 'v2sleep', 'v2timer')}
            board, scalars = v2_extra(frame, self.wave_tick)
            if self.obs in ('v2bungee', 'v2sleep', 'v2timer'):
                board = np.concatenate([board, bungee_features(frame)])
            if self.obs in ('v2sleep', 'v2timer'):
                board = np.concatenate([board, sleep_features(frame)])
            if self.obs == 'v2timer':
                board = np.concatenate([board, timer_features(frame)])
            if self.obs == 'v2size':
                size = np.zeros((1, ROWS, COLS), dtype=np.float32)
                for zombie in frame['zombies']:
                    row, _kind, position, _velocity, _health, on_board = zombie[:6]
                    if on_board and 0 <= row < ROWS:
                        size[0, row, _col_of(position)] += zombie[10] if len(zombie) > 10 else 1.0
                board = np.concatenate([board, size])
            if self.obs in ("v2fog", "v2fogstack4"):
                board = np.concatenate([board, fog_channel(frame)])
            if self.obs == 'v2fogstack4':
                from pvz_value_net import featurize_numpy
                # Generic public channel copies, sampled by elapsed game time. No tracking,
                # extrapolation, hidden IDs, or fog-specific interpretation of past zombies.
                current = featurize_numpy(frame)[0][54:88].copy()
                past = [values for time, values in reversed(self.zombie_frames) if time <= tick - 60][:3]
                missing = [np.zeros_like(current) for _ in range(3 - len(past))]
                board = np.concatenate([board, *past, *missing])
                if not self.zombie_frames or tick - self.zombie_frames[-1][0] >= 60:
                    self.zombie_frames.append((tick, current))
                    self.zombie_frames = self.zombie_frames[-4:]
            return board, scalars
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
        if self.obs == 'v3':
            return board, scalars  # v3 tokens have no per-episode history.
        extra_board, extra_scalars = self.extra(frame)
        return (torch.cat([board, torch.from_numpy(extra_board)]),
                torch.cat([scalars, torch.from_numpy(extra_scalars)]))
