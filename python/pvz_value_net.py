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

import numpy as np

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

# Optional task context, appended to the scalars by the caller: the zombie-count multiplier and the
# level's zombie roster. A player sees both before the level starts. Without them a net trained on
# mixed difficulties learns "which difficulty is this" (high AUC, useless for comparing two moves in
# one position: value_hard1, 2026-10-16). The roster, unlike a level number, means something on an
# unseen level.
CONTEXT = 1 + ZOMBIE_TYPES


def _col_of(x: float) -> int:
    return min(COLS - 1, max(0, int((x - LAWN_XMIN) // CELL_PX)))


def context_vector(multiplier: float, roster) -> np.ndarray:
    """``CONTEXT`` floats: multiplier / 3, then a multi-hot of the zombie types the level can send."""
    context = np.zeros(CONTEXT, dtype=np.float32)
    context[0] = multiplier / 3.0
    for kind in roster:
        context[1 + min(int(kind), ZOMBIE_TYPES - 1)] = 1.0
    return context


def observation_context(obs: dict) -> np.ndarray:
    """The context of a live game, from a full (not OBS_LEAN) observation."""
    return context_vector(obs.get("zombie_count_multiplier", 1.0),
                          (obs.get("loadout_context") or {}).get("zombie_roster") or [])


def featurize_numpy(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """:func:`featurize` as float32 numpy arrays.

    Same arithmetic as :func:`_featurize_reference` (float32 cells, the same additions in the
    same order), but written into numpy arrays: one ``tensor[i, j, k] += x`` costs a full torch
    dispatch, which made this function ~9% of a search game's wall time. The outputs are bitwise
    equal (``scripts/perf_rollout_check.py`` checks this on real frames).
    """
    board = np.zeros((BOARD_CHANNELS, ROWS, COLS), dtype=np.float32)
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

    scalars = np.zeros(SCALARS, dtype=np.float32)
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


# v1 above is frozen: student archive signatures include its exact source text.
ENCODER_V1 = "pvz_value_net.featurize v1"
ENCODER_V2 = "pvz_value_net.featurize v2"
ENCODER_V3 = "pvz_value_net.featurize v3"
ENCODER_V4 = "pvz_value_net.featurize v4"
ENCODER_V5 = "pvz_value_net.featurize v5"
scene_scalar_count = 6  # night, pool, fog, roof, original conveyor, public grave count / 45


def bungee_features(frame: dict) -> np.ndarray:
    board = np.zeros((6, ROWS, COLS), dtype=np.float32)
    raw = frame.get('bungee') or {}
    for row, col, target, phase, altitude in raw.get('zombies', []):
        if not (0 <= row < ROWS and 0 <= col < COLS - 1):
            continue
        board[0, row, col] = max(float(board[0, row, col]), float(bool(target)))
        if 0 <= phase <= 2:
            board[1, row, col] = max(float(board[1, row, col]), min(600, max(0, altitude)) / 600.0)
            board[2 + phase, row, col] = 1.0
    for row, col in raw.get('plants', []):
        if 0 <= row < ROWS and 0 <= col < COLS - 1:
            board[5, row, col] = 1.0
    return board


def sleep_features(frame: dict) -> np.ndarray:
    board = np.zeros((2, ROWS, COLS), dtype=np.float32)
    for row, col, asleep, wake_up_counter in frame.get('sleep', []):
        if 0 <= row < ROWS and 0 <= col < COLS - 1:
            board[0, row, col] = max(float(board[0, row, col]), float(asleep > 0))
            board[1, row, col] = max(float(board[1, row, col]), float(wake_up_counter > 0))
    return board


# task145: 屏幕上看得见的动作时长（tick）。数值取自引擎源码 src/Lawn/Plant.cpp ——
# 每个 (植物, 状态) 的满值就是它刚进入该状态时赋给 mStateCountdown 的数。
# 名字一律小写：encoder_signature 会把模块里的大写 int/float 常量算进指纹，
# 加一个大写常量就会把 v1/v2 这些冻结版本一起改签名。
_timer_state = {
    (6, 10): 70,      # 大嘴花：咬下去（UpdateChomper）
    (6, 13): 4000,    # 大嘴花：咀嚼（吃完才能再咬）
    (17, 3): 80,      # 窝瓜：盯住目标（UpdateSquash）
    (17, 4): 45,      # 窝瓜：起跳前
    (17, 5): 50,      # 窝瓜：上升
    (17, 6): 10,      # 窝瓜：下落
    (17, 7): 100,     # 窝瓜：落地后的停顿
    (47, 35): 500,    # 玉米加农炮：架设装填（PlantInitialize）
}
# 引爆/唤醒倒计时：樱桃、辣椒、毁灭菇、寒冰菇、咖啡豆都是 100 tick
# （mDoSpecialCountdown，见 Plant.cpp 的 PlantInitialize 与 UpdateDoomShroom）。
_timer_special = 100


def timer_features(frame: dict) -> np.ndarray:
    """可见动作倒计时与传送带卡序（task145，v2timer 新增的 4 个通道）。

    通道 0 引爆/唤醒进度：1 - do_special_countdown / 100。与基础通道 6 的
            樱桃/辣椒重复是有意的 —— 这个通道自成一套，任意 special > 0 的
            植物都在里面（毁灭菇、寒冰菇、咖啡豆正是通道 6 漏掉的）。
    通道 1 状态进度：1 - state_countdown / 该状态的满值，只覆盖大嘴花、窝瓜、
            玉米加农炮（土豆雷的武装在通道 6 里）。
    通道 2 传送带卡序：**行 0，列 = 卡槽位置**，(type + 1) / 48；0 表示这个位置是空的。
            卡槽顺序就是屏幕上从左到右的排列，用掉一张后引擎会整体前移
            （SeedBank::RemoveSeed），所以同一张卡的位置会变。
    通道 3 传送带就绪：该位置这张卡现在能不能种（active 且冷却为 0）。

    只用屏幕上看得见的量。雾里被遮住的植物在 v2 段里 state / countdown 是 -1，
    这里保持 0（与通道 6 的"未知"处理一致），不猜。
    """
    board = np.zeros((4, ROWS, COLS), dtype=np.float32)
    for row, col, kind, state, countdown, special, _launch, _rate in (frame.get('v2') or {}).get('plants', []):
        if not (0 <= row < ROWS and 0 <= col < COLS - 1):
            continue
        if special > 0:
            board[0, row, col] = max(float(board[0, row, col]),
                                     min(1.0, max(0.0, 1.0 - special / _timer_special)))
        full = _timer_state.get((kind, state))
        if full and countdown >= 0:
            board[1, row, col] = max(float(board[1, row, col]),
                                     min(1.0, max(0.0, 1.0 - countdown / full)))
    for index, kind, active, cooldown, _refresh, _cost in frame.get('packets', []):
        if kind < 0 or not 0 <= index < COLS:
            continue
        board[2, 0, index] = (kind + 1) / 48.0
        board[3, 0, index] = float(bool(active) and cooldown == 0)
    return board


class ValueNet(nn.Module):
    """Small conv net over the board, joined with the scalars, producing a win logit.

    Convolutions see neighbouring cells (a zombie two columns in front of a peashooter);
    a per-row max pool keeps "which lane is in trouble" before everything is averaged.
    """

    def __init__(self, width: int = 64, context: bool = False, encoder_version: int = 1):
        super().__init__()
        if encoder_version not in (1, 2, 3, 4, 5):
            raise ValueError(f"unknown value encoder version: {encoder_version}")
        self.encoder_version = encoder_version
        self.context = context        # scalars then carry SCALARS + CONTEXT values
        self.conv = nn.Sequential(
            # v3 = +蹦极 6；v4 = +睡眠 2；v5 = +可见计时与卡序 4。
            nn.Conv2d(BOARD_CHANNELS + (12 if encoder_version == 5 else 8 if encoder_version == 4 else 6 if encoder_version == 3 else 0), width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, (1, 3), padding=(0, 1)), nn.ReLU(),
        )
        self.scalar = nn.Sequential(nn.Linear(SCALARS + (scene_scalar_count if encoder_version >= 2 else 0) + (CONTEXT if context else 0), width), nn.ReLU())
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


def load_value_net(path) -> ValueNet:
    """A value checkpoint in eval mode; ``net.context`` says whether it expects the task context."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = checkpoint.get("encoder", ENCODER_V1)
    versions = {ENCODER_V1: 1, ENCODER_V2: 2, ENCODER_V3: 3, ENCODER_V4: 4, ENCODER_V5: 5}
    if encoder not in versions:
        raise ValueError(f"unsupported value encoder: {encoder!r}")
    net = ValueNet(width=int(checkpoint.get("width", 64)), context=bool(checkpoint.get("context", False)),
                   encoder_version=versions[encoder])
    net.load_state_dict(checkpoint["state_dict"])
    return net.eval()
