"""把一局模拟器对局渲染成"LLM 能读懂的战报"。

为什么需要它
------------
原始观测是一堵数字墙：54 个格子、40 张种子卡、每只僵尸的 `phase_counter`。
从这堆数字里看不出"这把输在哪"——是某条草坪没有火力？是阳光一直没用出去？
还是纯粹被推平？三件事能把它变得可读：

1. **参照系**。单说"第 17 波 / 共 30 波"没有意义；和同一 seed 的空操作地板、
   以及已知最好的一局放在一起才有意义。
2. **图**。棋盘是 5×9 的空间结构，一格一个字符的 ASCII 网格一眼就能读，
   而 54 项的 JSON 数组不能。
3. **换算成游戏概念的派生量**。不说"僵尸 x=710"，而说"最前面的僵尸离割草机还有
   8.4 格、按当前速度约 3200 tick 到位"。

输出两份：给人看的文本报告，给提示词用的 JSON。

用法
----
    python3 scripts/render_episode.py --resource-dir DIR --seed 30001 --level 7 \
        --policy scripted [--json OUT] [--compare-seed 30000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from scripted_baseline import DECK as SCRIPTED_DECK  # noqa: E402
from scripted_baseline import choose as scripted_choose  # noqa: E402

DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"

MAX_ACTIONS = 4000
LAWN_XMIN = 40
CELL_PX = 80
GRID_ROWS = 5
GRID_COLS = 9

# ---------------------------------------------------------------- 类型名称表
# 编号来自 src/ConstEnums.h 的 SeedType / ZombieType，不要凭记忆改。

PLANT_GLYPH = {
    0: "P", 1: "S", 2: "C", 3: "W", 4: "T", 5: "N", 6: "H", 7: "R",
    8: "p", 9: "s", 10: "f", 11: "g", 12: "y", 13: "c", 14: "i", 15: "d",
    16: "L", 17: "Q", 18: "3", 19: "k", 20: "J", 21: "^", 22: "t", 23: "A",
    24: "e", 25: "l", 26: "x", 27: "b", 28: "2", 29: "*", 30: "O", 31: "m",
    32: "u", 33: "o", 34: "K", 35: ",", 36: "G", 37: "U", 38: "M", 39: "%",
    40: "4", 41: "$", 42: "@", 43: "!", 44: "&", 45: "+", 46: "#", 47: "0",
    48: "I",
}

PLANT_NAME = {
    0: "豌豆射手", 1: "向日葵", 2: "樱桃炸弹", 3: "坚果墙", 4: "土豆雷", 5: "寒冰射手",
    6: "大嘴花", 7: "双发射手", 8: "小喷菇", 9: "阳光菇", 10: "大喷菇", 11: "墓碑吞噬者",
    12: "魅惑菇", 13: "胆小菇", 14: "寒冰菇", 15: "毁灭菇", 16: "睡莲", 17: "窝瓜",
    18: "三线射手", 19: "缠绕水草", 20: "火爆辣椒", 21: "地刺", 22: "火炬树桩", 23: "高坚果",
    24: "海蘑菇", 25: "路灯花", 26: "仙人掌", 27: "三叶草", 28: "裂荚射手", 29: "杨桃",
    30: "南瓜头", 31: "磁力菇", 32: "卷心菜投手", 33: "花盆", 34: "玉米投手", 35: "咖啡豆",
    36: "大蒜", 37: "叶子保护伞", 38: "金盏花", 39: "西瓜投手", 40: "机枪射手", 41: "双子向日葵",
    42: "忧郁菇", 43: "香蒲", 44: "冰西瓜", 45: "吸金磁", 46: "地刺王", 47: "玉米加农炮",
    48: "模仿者",
}

ZOMBIE_NAME = {
    0: "普通僵尸", 1: "旗帜僵尸", 2: "路障僵尸", 3: "撑杆僵尸", 4: "铁桶僵尸",
    5: "报纸僵尸", 6: "铁门僵尸", 7: "橄榄球僵尸", 8: "舞王僵尸", 9: "伴舞僵尸",
    10: "鸭子救生圈僵尸", 11: "潜水僵尸", 12: "冰车僵尸", 13: "雪橇僵尸",
    14: "海豚骑士僵尸", 15: "小丑僵尸", 16: "气球僵尸", 17: "矿工僵尸",
    18: "跳跳僵尸", 19: "雪人僵尸", 20: "蹦极僵尸", 21: "梯子僵尸",
    22: "投石车僵尸", 23: "巨人僵尸", 24: "小鬼僵尸", 25: "僵王博士",
    26: "豌豆头僵尸", 27: "坚果头僵尸", 28: "辣椒头僵尸", 29: "机枪头僵尸",
    30: "窝瓜头僵尸", 31: "高坚果头僵尸", 32: "红眼巨人",
}

ZOMBIE_GLYPH = {
    0: "z", 1: "F", 2: "n", 3: "v", 4: "b", 5: "w", 6: "D", 7: "f", 8: "d",
    9: "d", 10: "u", 11: "s", 12: "Z", 13: "B", 14: "p", 15: "j", 16: "a",
    17: "g", 18: "o", 19: "Y", 20: "e", 21: "L", 22: "c", 23: "G", 24: "i",
    25: "X", 26: "z", 27: "z", 28: "z", 29: "z", 30: "z", 31: "z", 32: "R",
}

# 会打伤害的植物 / 产阳光的植物 / 挡路的植物。
# 这几张表决定了"这一路有没有火力"的判定，改动前先想清楚。
SHOOTER_TYPES = frozenset({
    0, 5, 7, 10, 13, 18, 22, 26, 28, 29, 32, 34, 39, 40, 42, 43, 44, 47,
})
ECONOMY_TYPES = frozenset({1, 9, 38, 41})
WALL_TYPES = frozenset({3, 23, 30, 36})

MOWER_STATE = {0: "入场中", 1: "待命", 2: "已触发", 3: "被压扁"}


# ---------------------------------------------------------------- 采集


def _col_of(x: float) -> int:
    """像素横坐标 → 草坪列号。Board::GridToPixelX 是 col*80+40。"""
    return int((x - LAWN_XMIN) // CELL_PX)


def _frame(obs: dict) -> dict:
    """把一帧观测压成紧凑记录。只留渲染要用的字段。"""
    plants = [
        (p["row"], p["col"], p["type"], p["health"], p["max_health"])
        for p in obs.get("plants") or []
        if not p.get("squished")
    ]
    zombies = [
        (
            z["row"], z["type"], z["x"], z["velocity_x"],
            z["body_health"] + z["helm_health"] + z["shield_health"],
            bool(z.get("on_board")), bool(z.get("is_eating")),
            int(z.get("chilled") or 0), int(z.get("ice_trap") or 0),
        )
        for z in obs.get("zombies") or []
    ]
    mowers = {}
    for d in obs.get("defenses") or []:
        mowers[d["row"]] = d["state"]
    return {
        "tick": obs["tick"],
        "wave": obs["wave"],
        "sun": obs["sun"],
        "sun_income_rate": obs.get("sun_income_rate"),
        "plants": plants,
        "zombies": zombies,
        "mowers": mowers,
        "result": obs.get("result"),
        "terminal": bool(obs.get("terminal")),
        "enemy_on_screen": bool(obs.get("enemy_zombies_on_screen")),
    }


def collect(resource_dir: str, seed: int, level: int, policy: str,
            max_actions: int = MAX_ACTIONS, deck=None) -> dict:
    """跑一局，记录每一帧的紧凑状态与事件增量。"""
    env = PvZEnv(resource_dir, headless=True)
    deck = deck or SCRIPTED_DECK
    task = TaskSpec(level=level, seed=seed, playthrough=2)
    obs, _ = env.reset(deck=deck, task=task)

    frames = [_frame(obs)]
    events_total = {"zombies_killed": 0, "plants_eaten": 0, "sun_produced": 0,
                    "mower_triggered": 0, "waves_started": 0, "level_lost": 0}
    log = []  # (tick, wave, kind, detail)
    actions = 0
    rejected = 0
    kills_since_wave = 0

    while not obs["terminal"] and actions < max_actions:
        action = scripted_choose(obs) if policy == "scripted" else {"type": "wait", "ticks": 60}
        obs, _, done, _, info = env.step(action)
        if not info.get("ok"):
            rejected += 1
            obs, _, done, _, info = env.step({"type": "wait", "ticks": 60})
        ev = info.get("events") or {}
        for key in events_total:
            events_total[key] += int(ev.get(key) or 0)
        actions += 1

        prev = frames[-1]
        cur = _frame(obs)

        # ---- 事件：从事件计数器和帧间差里还原"发生了什么"。
        # 逐只僵尸的死亡日志太吵，改成并进"开波"那一行（上一波清掉几只）。
        kills_since_wave += int(ev.get("zombies_killed") or 0)
        if ev.get("waves_started"):
            tail = f"（上一波清掉 {kills_since_wave} 只）" if kills_since_wave else ""
            log.append((cur["tick"], cur["wave"], "开波", f"第 {cur['wave']} 波开始{tail}"))
            kills_since_wave = 0
        for r in _mower_fired(prev, cur):
            log.append((cur["tick"], cur["wave"], "割草机启动", f"{r} 号草坪后手被用掉"))
        if ev.get("plants_eaten"):
            gone = _plant_diff(prev["plants"], cur["plants"])
            log.append((cur["tick"], cur["wave"], "植物被吃",
                        "、".join(f"({r},{c}) {PLANT_NAME.get(t, t)}" for r, c, t, _ in gone)
                        or f"{ev['plants_eaten']} 株"))
        placed = _plant_diff(cur["plants"], prev["plants"])
        if placed:
            log.append((cur["tick"], cur["wave"], "种植",
                        "、".join(f"({r},{c}) {PLANT_NAME.get(t, t)}" for r, c, t, _ in placed)))
        if ev.get("level_lost"):
            log.append((cur["tick"], cur["wave"], "防线失守", "僵尸进屋"))

        frames.append(cur)
        if done:
            break

    env.close()
    final = frames[-1]
    result = int(final["result"] or 0)
    return {
        "schema": "episode_report_v1",
        "task": {"level": level, "seed": seed, "playthrough": 2,
                 "deck": list(deck), "policy": policy},
        "outcome": {
            "result": result,
            "won": result == 1,
            "reason": ("通关" if result == 1 else
                       "僵尸进屋" if events_total["level_lost"] else
                       "其他终局" if final["terminal"] else "动作上限截断"),
            "final_wave": final["wave"],
            "wave_count": obs["wave_count"],
            "final_tick": final["tick"],
            "actions": actions,
            "rejected_actions": rejected,
        },
        "totals": events_total,
        "frames": frames,
        "log": log,
    }


def _plant_diff(a: list, b: list) -> list:
    """a 有、b 没有的植物（按 (row,col,type) 计数差）。"""
    from collections import Counter
    ca = Counter((r, c, t) for r, c, t, _, _ in a)
    cb = Counter((r, c, t) for r, c, t, _, _ in b)
    out = []
    for key, n in (ca - cb).items():
        hp = next((h for r, c, t, h, _ in a if (r, c, t) == key), 0)
        out.extend([(key[0], key[1], key[2], hp)] * n)
    return sorted(out)


def _mower_fired(prev: dict, cur: dict) -> list:
    """这一帧有哪些草坪的割草机被用掉了。

    两种情况都要算：状态变成 TRIGGERED(=2)，或者整条从 defenses 列表里消失
    （割草机跑完一趟就被回收，一次 60 tick 的等待足以跳过 state=2 那一帧）。
    """
    fired = [r for r, s in cur["mowers"].items()
             if s == 2 and prev["mowers"].get(r) != 2]
    fired += [r for r in prev["mowers"] if r not in cur["mowers"] and r not in fired]
    return sorted(fired)


# ---------------------------------------------------------------- 渲染


def _lane_rows(frame: dict) -> list:
    """每一路一行派生指标。这是"一眼看出哪路要炸"的核心。"""
    rows = []
    for r in range(GRID_ROWS):
        plants = [p for p in frame["plants"] if p[0] == r]
        zs = [z for z in frame["zombies"] if z[0] == r and z[5]]
        off = [z for z in frame["zombies"] if z[0] == r and not z[5]]

        shooters = [p for p in plants if p[2] in SHOOTER_TYPES]
        econ = [p for p in plants if p[2] in ECONOMY_TYPES]
        walls = [p for p in plants if p[2] in WALL_TYPES]

        mower_state = frame["mowers"].get(r)
        mower_txt = MOWER_STATE.get(mower_state, "无") if mower_state is not None else "已消耗"

        zhp = sum(z[4] for z in zs)
        front_col = eta = eta_label = None
        if zs:
            front = min(zs, key=lambda z: z[2])
            front_col = _col_of(front[2])
            # 到割草机还有多少 tick：用最前面那只僵尸自己的速度。
            # 停住啃植物的僵尸速度是 0，直接算会得到一个天文数字，必须单独标。
            speed, eating = front[3], front[6]
            if speed < 0.02:
                eta_label = "停住啃食" if eating else "停住"
            else:
                eta = int(max(0.0, (front[2] - LAWN_XMIN) / speed))
                eta_label = f"{eta}t"

        # 判定是启发式，不是真理。规则就写在下面这几行，方便被推翻。
        if mower_state is None and front_col is not None and front_col <= 0:
            verdict = "已破"          # 后手没了，僵尸已经到最左列
        elif mower_state is None:
            verdict = "无后手"        # 割草机用掉了，但还没被压到最左
        elif not shooters and zs:
            verdict = "裸路承压"      # 有僵尸、这一路一株火力都没有
        elif front_col is not None and front_col <= 1:
            verdict = "紧急"          # 最前僵尸进 c0/c1
        elif eta is not None and eta < 900:
            verdict = "紧急"
        elif zs:
            verdict = "交火中"
        else:
            verdict = "平静"

        rows.append({
            "row": r,
            "shooters": len(shooters),
            "economy": len(econ),
            "walls": len(walls),
            "plants": len(plants),
            "zombies": len(zs),
            "off_board": len(off),
            "zombie_hp": zhp,
            "front_col": front_col,
            "ticks_to_mower": eta,
            "eta_label": eta_label,
            "mower": mower_txt,
            "verdict": verdict,
        })
    return rows


def _board_ascii(frame: dict, mode: str) -> str:
    """mode='plant' 画植物，'zombie' 画僵尸密度。"""
    grid = [["." for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    if mode == "plant":
        for r, c, t, hp, mx in frame["plants"]:
            if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                g = PLANT_GLYPH.get(t, "?")
                grid[r][c] = g if hp >= mx else g.lower()
    else:
        for r, t, x, *_ in frame["zombies"]:
            if not (0 <= r < GRID_ROWS):
                continue
            c = _col_of(x)
            if not (0 <= c < GRID_COLS):
                continue
            cur = grid[r][c]
            n = 1 if cur == "." else int(cur) + 1
            grid[r][c] = str(min(n, 9))

    header = "        " + " ".join(f"c{c}" for c in range(GRID_COLS))
    lines = [header]
    for r in range(GRID_ROWS):
        lines.append(f"  r{r}     " + "  ".join(grid[r]))
    return "\n".join(lines)


def _disp(s) -> int:
    """终端显示宽度：中日韩字符算两格。表格对齐靠它。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))


def _pad(s, width: int) -> str:
    s = str(s)
    return " " * max(0, width - _disp(s)) + s


def _lane_table(frame: dict) -> str:
    cols = [("路", 2), ("火力", 4), ("阳光", 4), ("挡路", 4), ("僵尸", 4),
            ("僵尸总血", 8), ("最前僵尸", 8), ("距割草机", 10), ("割草机", 8), ("判定", 0)]

    def fmt(cells) -> str:
        return "  " + "  ".join(
            _pad(v, w) if w else str(v) for (_, w), v in zip(cols, cells))

    head = fmt([n for n, _ in cols])
    lines = [head, "  " + "-" * (_disp(head) - 2)]
    for row in _lane_rows(frame):
        lines.append(fmt([
            row["row"], row["shooters"], row["economy"], row["walls"],
            row["zombies"], row["zombie_hp"],
            "—" if row["front_col"] is None else f"c{row['front_col']}",
            row["eta_label"] or "—",
            row["mower"], row["verdict"],
        ]))
    return "\n".join(lines)


def _peak_frames(rec: dict) -> dict:
    """每一波"僵尸最多"的那一帧。决定胜负的是峰值压力，不是开波/收波时的平静。"""
    peak: dict[int, dict] = {}
    for f in rec["frames"]:
        cur = peak.get(f["wave"])
        if cur is None or len(f["zombies"]) > len(cur["zombies"]):
            peak[f["wave"]] = f
    return peak


def _mower_fired_waves(rec: dict) -> set:
    """{(wave, row)}：哪一波用掉了哪一路的割草机。

    每条路只有一台割草机，所以按 row 去重：触发时会被看到一次（state→2），
    跑完被回收时又会被看到一次（从列表消失），那还是同一台。
    """
    frames = rec["frames"]
    out, seen = set(), set()
    for i in range(1, len(frames)):
        for r in _mower_fired(frames[i - 1], frames[i]):
            if r not in seen:
                seen.add(r)
                out.add((frames[i]["wave"], r))
    return out


def _wave_matrix(rec: dict) -> str:
    """逐波态势矩阵：整局压成一张表。格子写成 火力/僵尸数。"""
    peak = _peak_frames(rec)
    fired = _mower_fired_waves(rec)

    cols = [("波", 4)] + [(f"{r}号路", 8) for r in range(GRID_ROWS)] + \
           [("阳光", 6), ("割草机已用", 10), ("备注", 0)]

    def fmt(cells) -> str:
        return "  " + "  ".join(
            _pad(v, w) if w else str(v) for (_, w), v in zip(cols, cells))

    head = fmt([n for n, _ in cols])
    lines = [head, "  " + "-" * (_disp(head) - 2)]

    for w in sorted(peak):
        f = peak[w]
        lanes = _lane_rows(f)
        cells = [w] + [f"{lanes[r]['shooters']}/{lanes[r]['zombies']}"
                       for r in range(GRID_ROWS)]
        notes = []
        hit = [r for r in range(GRID_ROWS) if (w, r) in fired]
        if hit:
            notes.append("割草机 " + "、".join(str(r) for r in hit))
        bare = [r for r in range(GRID_ROWS)
                if lanes[r]["shooters"] == 0 and lanes[r]["zombies"]]
        if bare:
            notes.append("裸路 " + "、".join(str(r) for r in bare))
        cells += [f["sun"], GRID_ROWS - len(f["mowers"]), "；".join(notes)]
        lines.append(fmt(cells))
    lines.append("  格内写法：火力/僵尸数。割草机已用 = 已消耗台数（满额 5）。")
    return "\n".join(lines)


def _diagnose(rec: dict) -> list:
    """用机械信号写几句人话诊断。

    规则全部可核对，不做因果推断：只说"哪条路的火力上限被锁死"、
    "哪条路在后半程压力一直超过火力"、"钱和火力有没有对上"。
    真正的因果解释交给读这份报告的人（或 LLM）。
    """
    peak = _peak_frames(rec)
    frames = rec["frames"]
    waves = sorted(peak)
    out = []

    # 1. 火力上限：整局每条路最多同时有几个火力植物
    ceiling = {r: 0 for r in range(GRID_ROWS)}
    for w in waves:
        for lane in _lane_rows(peak[w]):
            ceiling[lane["row"]] = max(ceiling[lane["row"]], lane["shooters"])
    if waves:
        worst = min(ceiling, key=lambda r: ceiling[r])
        best = max(ceiling.values())
        if best - ceiling[worst] >= 2:
            out.append(f"{worst} 号路全程火力上限只有 {ceiling[worst]}，"
                       f"其它路最高到 {best}——五条路里相对最弱的一条。")

    # 2. 后半程"僵尸数 > 火力数"的波次：输出跟不上压力
    half = waves[len(waves) // 2:] if waves else []
    over = {r: [] for r in range(GRID_ROWS)}
    for w in half:
        for lane in _lane_rows(peak[w]):
            if lane["zombies"] > lane["shooters"]:
                over[lane["row"]].append(w)
    if over:
        r = max(over, key=lambda k: len(over[k]))
        if len(over[r]) >= 2:
            out.append(f"{r} 号路在后半程（{half[0]}–{half[-1]} 波）有 {len(over[r])} 个波次"
                       f"僵尸数超过火力数，是全盘输出最跟不上压力的一路。")

    # 3. 植物净损失最多的路（反复补种的绞肉机）
    losses = {r: 0 for r in range(GRID_ROWS)}
    for i in range(1, len(frames)):
        for r, _c, _t, _h in _plant_diff(frames[i - 1]["plants"], frames[i]["plants"]):
            losses[r] += 1
    if any(losses.values()):
        worst = max(losses, key=lambda r: losses[r])
        out.append(f"{worst} 号路植物净损失 {losses[worst]} 株（全路合计 "
                   f"{sum(losses.values())} 株），是补种最频繁的一路。")

    # 4. 割草机在哪几波被用掉
    fired = _mower_fired_waves(rec)
    if fired:
        by_wave: dict[int, list] = {}
        for w, r in sorted(fired):
            by_wave.setdefault(w, []).append(r)
        detail = "；".join(f"第 {w} 波 {rs} 号路" for w, rs in by_wave.items())
        out.append(f"割草机共消耗 {len(fired)} 台：{detail}。")

    # 5. 终局时哪几路已经没后手
    last = _lane_rows(frames[-1])
    naked = [l["row"] for l in last if l["mower"] in ("已消耗", "已触发")]
    if naked:
        out.append(f"终局时 {naked} 号草坪已无割草机，这几路再被突破就是直接进屋。")

    # 6. 钱和火力没对上：阳光够买好几株了，却有路零火力还站着僵尸
    idle = [(f["sun"], f["wave"]) for f in frames
            if f["sun"] >= 300
            and any(l["shooters"] == 0 and l["zombies"] for l in _lane_rows(f))]
    if idle:
        s, w = max(idle)
        out.append(f"第 {w} 波时手里有 {s} 阳光，场上却仍有「有僵尸、零火力」的路"
                   f"——钱没换成火力。")

    if not out:
        out.append("没有触发任何机械告警信号；这一局可能是被整体压制而非单点失误。")
    return out


def _turning_points(rec: dict) -> list:
    """自动找"这把是从哪里开始输的"。每条都是可核对的机制信号。"""
    frames, out = rec["frames"], []

    # 1. 割草机第一次被用掉
    for i in range(1, len(frames)):
        fired = _mower_fired(frames[i - 1], frames[i])
        if fired:
            out.append({
                "kind": "割草机启动", "tick": frames[i]["tick"], "wave": frames[i]["wave"],
                "detail": f"第 {frames[i]['wave']} 波，{fired} 号草坪的后手被用掉",
                "index": i,
            })
            break

    # 2. 每条路第一次被打到 1 格以内
    for r in range(GRID_ROWS):
        for i, f in enumerate(frames):
            front = [z for z in f["zombies"] if z[0] == r and z[5]]
            if front and min(_col_of(z[2]) for z in front) <= 1:
                out.append({
                    "kind": f"{r} 号草坪被压到 1 格", "tick": f["tick"], "wave": f["wave"],
                    "detail": f"最前僵尸进入 c{min(_col_of(z[2]) for z in front)}",
                    "index": i,
                })
                break

    # 3. 波数停滞：最后一波推进发生在哪，之后卡了多久
    last_bump = 0
    for i in range(1, len(frames)):
        if frames[i]["wave"] > frames[i - 1]["wave"]:
            last_bump = i
    if last_bump < len(frames) - 1:
        stall = frames[-1]["tick"] - frames[last_bump]["tick"]
        out.append({
            "kind": "波数停滞", "tick": frames[last_bump]["tick"], "wave": frames[last_bump]["wave"],
            "detail": f"最后一次推进到第 {frames[last_bump]['wave']} 波后，"
                      f"又过了 {stall} tick 没能再推进（终局停在 {frames[-1]['wave']}）",
            "index": last_bump,
        })

    # 4. 阳光闲置：攒着钱、有路没火力、又很久没种东西
    last_plant_tick = 0
    for i in range(1, len(frames)):
        f = frames[i]
        planted = _plant_diff(f["plants"], frames[i - 1]["plants"])
        if planted:
            last_plant_tick = f["tick"]
        bare = [r for r in range(GRID_ROWS)
                if not any(p[2] in SHOOTER_TYPES for p in f["plants"] if p[0] == r)
                and any(z[0] == r and z[5] for z in f["zombies"])]
        if bare and f["sun"] >= 300 and f["tick"] - last_plant_tick >= 1500:
            out.append({
                "kind": "阳光闲置且有裸路", "tick": f["tick"], "wave": f["wave"],
                "detail": f"阳光 {f['sun']} 攒着未用，{bare} 号草坪有僵尸但无火力，"
                          f"已 {f['tick'] - last_plant_tick} tick 未种植",
                "index": i,
            })
            break

    # 5. 终局
    out.append({
        "kind": "终局", "tick": frames[-1]["tick"], "wave": frames[-1]["wave"],
        "detail": f"{rec['outcome']['reason']}，第 {frames[-1]['wave']}/{rec['outcome']['wave_count']} 波，"
                  f"场上还剩 {sum(1 for z in frames[-1]['zombies'] if z[5])} 只僵尸",
        "index": len(frames) - 1,
    })
    out.sort(key=lambda d: d["index"])
    return out


def render_text(rec: dict, reference: dict | None = None) -> str:
    o, t = rec["outcome"], rec["task"]
    L = []
    L.append("=" * 74)
    L.append(f" 战局报告   第 {t['level']} 关 / seed {t['seed']} / 策略 {t['policy']} / "
             f"{'通关' if o['won'] else '失败'}")
    L.append("=" * 74)

    # ---- 参照系
    L.append("")
    L.append("【这把什么水平】")
    L.append(f"  结果：{o['reason']}，推进到第 {o['final_wave']}/{o['wave_count']} 波，"
             f"共 {o['actions']} 次决策、{o['final_tick']} tick")
    if reference:
        ro = reference["outcome"]
        L.append(f"  同一 seed 空操作：第 {ro['final_wave']}/{ro['wave_count']} 波"
                 f"（{'通关' if ro['won'] else '失败'}）")
        if ro["won"]:
            L.append("  → 这一局比什么都不做还差" if not o["won"] else "  → 与空操作同为通关")
        else:
            delta = o["final_wave"] - ro["final_wave"]
            L.append(f"  → 比什么都不做{'多' if delta >= 0 else '少'}推进 {abs(delta)} 波")
    L.append(f"  事件合计：杀死 {rec['totals']['zombies_killed']} 只僵尸，"
             f"损失 {rec['totals']['plants_eaten']} 株植物，"
             f"割草机消耗 {rec['totals']['mower_triggered']} 台，"
             f"产阳光 {rec['totals']['sun_produced']}")

    # ---- 自动诊断：机械信号直接回答"输在哪"
    L.append("")
    L.append("【自动诊断】")
    for line in _diagnose(rec):
        L.append(f"  · {line}")

    # ---- 转折点（放前面，这是"输在哪"的答案）
    L.append("")
    L.append("【转折点】")
    for tp in _turning_points(rec):
        L.append(f"  tick {tp['tick']:>7}  第 {tp['wave']:>2} 波  {tp['kind']:<16} {tp['detail']}")

    # ---- 逐波态势（整局压成一张表）
    L.append("")
    L.append("【逐波态势】")
    L.append(_wave_matrix(rec))

    # ---- 关键帧棋盘
    key_idx = [tp["index"] for tp in _turning_points(rec)]
    seen, picks = set(), []
    for i in key_idx + [len(rec["frames"]) - 1]:
        if i not in seen:
            seen.add(i)
            picks.append(i)
    picks = picks[:6]
    L.append("")
    L.append("【关键帧棋盘】")
    for i in picks:
        f = rec["frames"][i]
        L.append("")
        L.append(f"  ── tick {f['tick']} / 第 {f['wave']}/{o['wave_count']} 波 / "
                 f"阳光 {f['sun']} ──")
        L.append("  植物（大写=满血，小写=受伤，. 空）：")
        L.append("  " + _board_ascii(f, "plant").replace("\n", "\n  "))
        L.append("  僵尸分布（数字=该格僵尸数）：")
        L.append("  " + _board_ascii(f, "zombie").replace("\n", "\n  "))
        L.append("  " + _lane_table(f).replace("\n", "\n  "))

    # ---- 终局棋盘
    L.append("")
    L.append("【终局棋盘】")
    f = rec["frames"][-1]
    L.append("  植物：")
    L.append("  " + _board_ascii(f, "plant").replace("\n", "\n  "))
    L.append("  僵尸：")
    L.append("  " + _board_ascii(f, "zombie").replace("\n", "\n  "))
    L.append("  " + _lane_table(f).replace("\n", "\n  "))

    # ---- 事件流水
    L.append("")
    L.append(f"【事件流水】共 {len(rec['log'])} 条")
    for tick, wave, kind, detail in rec["log"]:
        L.append(f"  tick {tick:>7}  第 {wave:>2} 波  {kind:<10} {detail}")

    L.append("")
    L.append("【图例】植物 P豌豆 S向日葵 N寒冰 C樱桃 W坚果 T土豆雷 H大嘴花 R双发 "
             "3三线 A高坚果 Q窝瓜 J辣椒 ^地刺 $双子葵")
    L.append("        僵尸 z普通 n路障 b铁桶 F旗帜 v撑杆 f橄榄球 D铁门 a气球 "
             "G巨人 i小鬼 L梯子 c投石车")
    L.append("")
    return "\n".join(L)


def to_payload(rec: dict, reference: dict | None = None, compact: bool = False) -> dict:
    """给提示词用的结构化版本。帧只保留关键帧，避免塞爆上下文。

    compact=True 时事件流水只留骨架（开波 / 割草机启动 / 防线失守 / 终局），
    去掉逐次种植与补种的流水；"这把输在哪"靠转折点+逐波矩阵已经能回答。
    """
    key_idx = sorted({tp["index"] for tp in _turning_points(rec)} | {len(rec["frames"]) - 1})
    peak, fired = _peak_frames(rec), _mower_fired_waves(rec)
    skeleton = {"开波", "割草机启动", "防线失守", "终局"}

    wave_matrix = []
    for w in sorted(peak):
        lanes = _lane_rows(peak[w])
        wave_matrix.append({
            "wave": w,
            "lanes": [{"row": r, "shooters": lanes[r]["shooters"],
                       "zombies": lanes[r]["zombies"]} for r in range(GRID_ROWS)],
            "sun": peak[w]["sun"],
            "mowers_used": GRID_ROWS - len(peak[w]["mowers"]),
            "mowers_fired_rows": [r for r in range(GRID_ROWS) if (w, r) in fired],
        })

    return {
        "schema": rec["schema"],
        "task": rec["task"],
        "outcome": rec["outcome"],
        "totals": rec["totals"],
        "reference": None if reference is None else {
            "policy": reference["task"]["policy"],
            "outcome": reference["outcome"],
        },
        "turning_points": [{k: v for k, v in tp.items() if k != "index"}
                           for tp in _turning_points(rec)],
        "diagnosis": _diagnose(rec),
        "wave_matrix": wave_matrix,
        "key_frames": [
            {
                "tick": rec["frames"][i]["tick"],
                "wave": rec["frames"][i]["wave"],
                "sun": rec["frames"][i]["sun"],
                "plants_ascii": _board_ascii(rec["frames"][i], "plant"),
                "zombies_ascii": _board_ascii(rec["frames"][i], "zombie"),
                "lanes": _lane_rows(rec["frames"][i]),
            }
            for i in key_idx
        ],
        "event_log": [
            {"tick": tick, "wave": wave, "kind": kind, "detail": detail}
            for tick, wave, kind, detail in rec["log"]
            if not compact or kind in skeleton
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--policy", choices=["scripted", "donothing"], default="scripted")
    ap.add_argument("--reference-policy", choices=["scripted", "donothing", "none"],
                    default="donothing")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    ap.add_argument("--deck", default=None,
                    help="逗号分隔的种子卡编号，默认用脚本教师的白天卡组 (0,1,2,3,4,5)。"
                         "夜晚关是 8,9,10,12,14,15；泳池关 0,1,2,3,4,16；"
                         "浓雾关 8,9,10,14,15,16；屋顶关 0,1,2,3,4,33。")
    ap.add_argument("--json", default=None, help="把结构化版本写到这个路径")
    ap.add_argument("--compact", action="store_true",
                    help="JSON 里的事件流水只留骨架（开波/割草机/失守/终局）")
    args = ap.parse_args()

    deck = ([int(v) for v in args.deck.split(",")] if args.deck else None)
    rec = collect(args.resource_dir, args.seed, args.level, args.policy,
                  args.max_actions, deck)
    ref = None
    if args.reference_policy != "none":
        ref = collect(args.resource_dir, args.seed, args.level,
                      args.reference_policy, args.max_actions, deck)
    print(render_text(rec, ref))
    if args.json:
        Path(args.json).write_text(
            json.dumps(to_payload(rec, ref, compact=args.compact),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[已写出结构化版本] {args.json}")


if __name__ == "__main__":
    main()
