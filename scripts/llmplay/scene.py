"""局面渲染：把引擎观测变成一屏文字。

渲染本身不在这里发明 —— 场景解释复用 episode_query._fmt_scene（地形/能不能种），
棋盘复用 render_episode._board_ascii（大小写 = 满血/受伤的约定就写在那边的注释里）。
这里只负责把它们排成模型一屏能读完的样子。
"""
from __future__ import annotations

from pvz_constants import PLANT_NAME, ZOMBIE_NAME
from render_episode import _board_ascii, _frame, _lane_rows
from episode_query import _fmt_scene

MOWER_STATE = {0: "已用", 1: "待命", 2: "启动", 3: "行驶"}


def _plant_name(seed_type: int) -> str:
    return PLANT_NAME.get(int(seed_type), "type%s" % seed_type)


def _zombie_name(zombie_type: int) -> str:
    return ZOMBIE_NAME.get(int(zombie_type), "type%s" % zombie_type)


def _packet_line(frame: dict) -> str:
    parts = []
    for index, seed_type, active, cooldown, refresh, cost in frame.get("packets") or []:
        ready = bool(active) and int(cooldown) == 0
        state = "就绪" if ready else "冷却%d" % int(cooldown)
        parts.append("[%d]%s %d☀ %s" % (index, _plant_name(seed_type), int(cost), state))
    return "卡槽：" + " | ".join(parts)


def _lane_line(frame: dict, game) -> str:
    parts = []
    for row in _lane_rows(frame):
        front = "—" if row["front_col"] is None else "c%d" % row["front_col"]
        mower = row["mower"] if game.saw_mowers else "本关无"
        parts.append("r%d 火力%d/僵尸%d 最前%s 割草机%s"
                     % (row["row"], row["shooters"], row["zombies"], front, mower))
    return "路况：" + "｜".join(parts)


def _header(frame: dict, game) -> str:
    obs = game.obs or {}
    bits = ["tick %s" % frame["tick"],
            "波 %s/%s" % (obs.get("wave"), obs.get("wave_count")),
            "阳光 %s" % frame["sun"],
            "植物 %d" % len(frame["plants"]),
            "僵尸 %d" % sum(1 for z in frame["zombies"] if z[5])]
    if obs.get("terminal"):
        result = obs.get("result")
        bits.append("**终局：%s**" % ("通关" if result == 1 else "失败"))
    else:
        bits.append("进行中")
    return " ".join(bits)


def render(game, level: str = "brief") -> str:
    """brief：场景 + 棋盘 + 卡槽 + 路况；full：再加僵尸密度、逐路明细、全量僵尸表。"""
    frame = _frame(game.obs)
    lines = [_header(frame, game), "", _fmt_scene(frame), "",
             "棋盘（植物；大写=满血，小写=受伤；~ = 水路）：",
             _board_ascii(frame, "plant"), "",
             _packet_line(frame), _lane_line(frame, game)]
    if level == "full":
        lines += ["", "棋盘（僵尸密度；数字=该格僵尸数，9 表示 9 只以上）：",
                  _board_ascii(frame, "zombie")]
        if frame.get("graves"):
            lines.append("占位物（墓碑/花瓶等，这些格种不了）：%s"
                         % ", ".join("(%d,%d)类型%s" % (g[0], g[1], g[2]) for g in frame["graves"]))
        lines += ["", "僵尸明细（行 类型 x坐标 血量 在场上 正在啃食 冰冻）："]
        for z in frame["zombies"]:
            if not z[5]:
                continue
            lines.append("  r%d %s x=%.1f 血%d%s%s"
                         % (z[0], _zombie_name(z[1]), z[2], z[4],
                            " 啃食中" if z[6] else "",
                            " 冰冻%d" % z[8] if z[8] else ""))
        totals = frame.get("totals") or {}
        lines += ["", "全盘计数：生产者 %s · 射手 %s · 睡莲 %s · 香蒲 %s"
                  % (totals.get("producers"), totals.get("shooters"),
                     totals.get("lilypads"), totals.get("cattails"))]
    return "\n".join(lines)


def legal(game) -> dict:
    """引擎给的合法动作。plants 按卡槽归组；shovels 是当前有植物、可以铲的格。"""
    obs = game.obs or {}
    actions = (obs.get("legal_actions") or {}).get("plants") or []
    grouped: dict[int, list] = {}
    for action in actions:
        grouped.setdefault(int(action["packet"]), []).append([int(action["row"]), int(action["col"])])
    frame = _frame(obs)
    packets = {int(p[0]): int(p[1]) for p in frame["packets"]}
    plants = [{"packet": packet, "type": _plant_name(packets.get(packet, -1)),
               "cells": cells}
              for packet, cells in sorted(grouped.items())]
    shovels = sorted({(int(p[0]), int(p[1])) for p in frame["plants"]})
    return {"plants": plants, "shovels": [list(cell) for cell in shovels]}


def summary(game) -> dict:
    """当前波 / 最远存活波 / 剩余小推车 / 阳光 / 各行威胁。"""
    obs = game.obs or {}
    frame = _frame(obs)
    defenses = obs.get("defenses") or []
    mowers_left = sum(1 for d in defenses if int(d[1]) == 1)
    lanes = []
    for row in _lane_rows(frame):
        lanes.append(dict(row=row["row"], shooters=row["shooters"], zombies=row["zombies"],
                          front_col=row["front_col"],
                          mower=row["mower"] if game.saw_mowers else "本关无",
                          zombie_hp=row["zombie_hp"]))
    result = obs.get("result")
    return dict(tick=frame["tick"], level=obs.get("level"),
                wave=obs.get("wave"), wave_count=obs.get("wave_count"),
                max_wave=game.max_wave, sun=frame["sun"],
                mowers_left=mowers_left, mowers_total=len(defenses),
                plants=len(frame["plants"]),
                zombies=sum(1 for z in frame["zombies"] if z[5]),
                terminal=bool(obs.get("terminal")),
                won=result == 1, result=result,
                lanes=lanes)
