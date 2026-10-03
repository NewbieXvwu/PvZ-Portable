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
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from pvz_constants import PLANT_NAME, grid_square_types  # noqa: E402
from scripted_baseline import choose as scripted_choose  # noqa: E402
from scripted_baseline import deck_for_level, POLICY_REVISION, profile_for_deck  # noqa: E402

# 真实路的地形集合：草地或水路。DIRT/NONE/HIGH_GROUND 是填充或特殊格。
_LANE_TERRAIN = frozenset(
    v for v, name in grid_square_types().items()
    if name in ("GRIDSQUARE_GRASS", "GRIDSQUARE_POOL"))

DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"

MAX_ACTIONS = 4000
LAWN_XMIN = 40
CELL_PX = 80
GRID_ROWS = 5    # ⚠ 只是"大多数关"的路数，**不是棋盘行数**。棋盘 grid 恒为 6×9，
GRID_COLS = 9    # 泳池/雾关是 6 条路（第 5 行是真实的草地路，割草机 6 台），
                 # 草地/屋顶关第 5 行是 DIRT 填充（地形 2，5 台割草机）。
                 # 2026-10-03 S5 实验中模型抓到这个不一致后修正：
                 # 一切"几条路"的判断都走 lane_indices()，别再用这个常数。


def lane_indices(frame: dict) -> list:
    """这一帧的**真实路号**（0 起）。

    新存档有 `lanes` 字段（_frame 按 GridSquareType ∈ {GRASS, POOL} 现算）；
    旧存档没有 → 回退为 row_terrain 的全部行（旧渲染只存了真实路，
    但泳池旧存档会缺第 5 路 —— 那是修复前的缺陷，别拿它当事实）。
    """
    lanes = frame.get("lanes")
    if lanes:
        return list(lanes)
    return list(range(len(frame.get("row_terrain") or [])))

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

# ---------------------------------------------------------------- 因果数值
# 全部来自 src/Lawn/Plant.cpp 的 mLaunchRate / mSeedCost 与 Projectile 的 mDamage，
# 不是估的。豌豆射手：每 150 tick 一发、每发 20 伤害 → 0.1333 伤害/tick。
# 有出处才能拿它做"够不够打"的判断；凭感觉编一个 DPS 会让整个因果链不可信。
PEA_DAMAGE = 20
PEA_INTERVAL = 150
PEA_DPS = PEA_DAMAGE / PEA_INTERVAL

SHOOTER_DPS = {
    0: PEA_DPS,          # 豌豆射手
    5: PEA_DPS,          # 寒冰射手（同伤害，附加减速）
    7: 2 * PEA_DPS,      # 双发射手（一次两发）
    40: 4 * PEA_DPS,     # 机枪射手
}
DEFAULT_SHOOTER_DPS = PEA_DPS  # 其余射手按豌豆射手估算，输出里会标注

PLANT_COST = {0: 100, 1: 50, 2: 150, 3: 50, 4: 25, 5: 175}

# 判定"火力够不够"用的余量倍数：到割草机之前能打出的伤害 / 最前那只僵尸的血。
# 2 倍以上算从容；1~2 倍算勉强；不到 1 倍就是打不死（僵尸会先到）。
LANE_COMFORT_RATIO = 2.0


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
            int(z["id"]),  # 索引 9：追踪单只僵尸用。加在末尾，前面的索引不变。
        )
        for z in obs.get("zombies") or []
    ]
    # 存成 [[row, state], ...] 而不是 {row: state}：JSON 的对象键永远是字符串，
    # 整数键的字典存盘再读回来会变成 "0"/"1"，后面按行号取就会炸。
    mowers = [[d["row"], d["state"]] for d in obs.get("defenses") or []]
    # 卡槽（seed bank）。原生观测**本来就发**这几个字段（见 LawnApp.cpp 的
    # `"packets":[...]`：index/type/imitater_type/active/cooldown/refresh_time/cost），
    # 是这个压缩函数原先把它们丢了。丢掉的代价是可实测的 —— S3 那轮模型写道：
    #     "What cards are in the deck? Let me check a frame's seed bank.
    #      The frame output didn't show seed bank. Let me check vocabulary—no.
    #      Maybe the frame tool has a way to show? Not in this output."
    # 它连试三条路都拿不到"手里有什么牌、哪张冷却好了"，最后只能从 `actions`
    # 里出现过的植物去反推卡组。数据一直都在，只是没往外送。
    packets = [
        [
            p["index"], p["type"], bool(p["active"]),
            int(p["cooldown"]), int(p["refresh_time"]), int(p["cost"]),
        ]
        for p in obs.get("packets") or []
    ]
    # 场景（关卡背景 + 每格地形）。原生观测一直发 `terrain` / `night` / `pool` /
    # `fog` / `roof` / `grid`（`pvz_agent_model.py` 就在读它们），是这个压缩函数
    # 原先把它们丢了。
    #
    # 丢掉的代价（2026-10-03 实测量化）：**水路能不能种**取决于这一格是不是
    # `GRIDSQUARE_POOL`，而卡组里没有睡莲时整条路就是死的。模型看不到地形，
    # 只能从"策略一直没往第 2、3 路种东西"去反推为什么 —— 而那个反推是错的，
    # 它会以为"策略选错了路"，实际是"那两路物理上种不了"。
    #
    # 这里只存**事实**（编号与布尔），不把它们翻译成人话 —— 名字与"能不能种"
    # 的推导交给 `episode_query._fmt_scene`，那边能拿到植物名表与卡组。
    grid = obs.get("grid") or []
    row_terrain = []
    for row in grid:
        vals = {int(v) for v in row}
        # 一行内地形不一致时记 -1（"混合"）。实测 25 关里没出现过，
        # 但**不能**假定一致 —— 假定错了会把水路说成旱地，那比不说更坏。
        row_terrain.append(vals.pop() if len(vals) == 1 else -1)
    # 真实路 = 地形是草地或水路的行。非泳池关的第 6 行（下标 5）是 DIRT 填充，
    # 不是路 —— 泳池/雾关它却是真实的草地路（实测 L21：row 5 有合法种植位、
    # 割草机 6 台）。之前按"前 5 行"硬截，泳池关整整丢了一条路，
    # S5 实验里的模型抓到了这个不一致（它发现策略在 (5,0) 种了向日葵
    # 而棋盘渲染里根本没有 r5）。
    lane_terrain = _LANE_TERRAIN
    lanes = [r for r, t in enumerate(row_terrain) if t in lane_terrain]
    # 全局计数：脚本策略的关键规则前置条件（③生产者<10、⑤射手<14、
    # ④香蒲<2）都是**全盘**量，而不是某一条路的。模型曾被迫从棋盘 ASCII
    # 里逐格数向日葵来判断"经济规则会不会触发"——数了三遍、每遍都标注
    # 不确定。这里直接给出可核对的事实，别让它做易错的数数工作。
    totals = {
        "producers": sum(1 for p in plants if p[2] in ECONOMY_TYPES),
        "shooters": sum(1 for p in plants if p[2] in SHOOTER_TYPES),
        "lilypads": sum(1 for p in plants if p[2] == 16),
        "cattails": sum(1 for p in plants if p[2] == 43),
    }
    return {
        "tick": obs["tick"],
        "wave": obs["wave"],
        "sun": obs["sun"],
        "sun_income_rate": obs.get("sun_income_rate"),
        "plants": plants,
        "zombies": zombies,
        "mowers": mowers,
        "packets": packets,
        "scene": {
            "level": obs.get("level"),
            "terrain": obs.get("terrain"),      # BackgroundType 的编号
            "night": bool(obs.get("night")),
            "pool": bool(obs.get("pool")),
            "fog": bool(obs.get("fog")),
            "roof": bool(obs.get("roof")),
            "wave_count": obs.get("wave_count"),
        },
        "row_terrain": row_terrain,             # grid 全部行，GridSquareType 编号
        "lanes": lanes,                         # 真实路号（泳池/雾 6 条，其余 5 条）
        "totals": totals,                       # 全盘生产者/射手/睡莲/香蒲计数
        # 坟墓/花瓶等占位物（夜间草地关与 ScaryPotter 关）：这些格子**种不了**，
        # 模型看不到它们会把"策略没往那格种"误读成决策错误。只存位置，
        # 翻译成人话在 episode_query._fmt_scene。大多数关没有 → 存 []。
        "graves": [[g["row"], g["col"], g["type"]] for g in (obs.get("grid_items") or [])],
        "result": obs.get("result"),
        "terminal": bool(obs.get("terminal")),
        "enemy_on_screen": bool(obs.get("enemy_zombies_on_screen")),
    }


def collect(resource_dir: str, seed: int, level: int, policy: str,
            max_actions: int = MAX_ACTIONS, deck=None, override: dict | None = None,
            capture_legal_at: int | None = None,
            prefix_actions: list | None = None) -> dict:
    """跑一局，记录每一帧的紧凑状态、**这一步的决策**、以及事件增量。

    frames[i] = 做完第 i 次决策之后的局面（frames[0] 是开局）。
    frames[i]["action"] = 第 i 次决策实际做了什么（frames[0] 是 None）。

    注意这里的编号：第 N 次决策是在循环里 actions == N-1 的那一轮做出的，
    所以 override / capture_legal_at 都按**决策序号（1 起）**收，内部减一。
    早期版本按 actions 直接匹配，整体错开一步，枚举到的是"决策已经做完之后"的
    状态（阳光已经花掉、没有合法种植位）——踩过。

    override = {决策序号: 动作}：到那一步时不用策略的默认选择，改用给定动作。
    环境是确定性的（同 task+seed+动作序列 → 逐位相同的结果），所以这是精确的
    反事实重放，不是近似。

    prefix_actions = 前缀动作表（第 1..N 步）：提供时这些步**照抄给定动作**、
    不问策略。whatif 重放旧存档用：把存档里第 1..N-1 步的真实动作作为前缀，
    第 N 步 override，之后让策略接管 —— 这样重放对**任何历史版本的策略**
    都忠实到分叉点为止，策略行为修订（见 scripted_baseline.POLICY_REVISION）
    不会让旧存档的 whatif 静默分叉。
    """
    env = PvZEnv(resource_dir, headless=True)
    # 卡组按关卡地形定（AGENTS.md「任务与卡组」）；显式传 deck 的调用方
    # （whatif 重放旧存档）用传入值，profile 也从这份卡组推导 —— 槽位数、
    # 升级植物所有权都和卡组一致，旧存档（6 卡、无升级）重放时行为不变。
    deck = tuple(deck) if deck else deck_for_level(level)
    task = TaskSpec(level=level, seed=seed, playthrough=2, profile=profile_for_deck(deck))
    obs, _ = env.reset(deck=deck, task=task)

    frames = [_frame(obs)]
    frames[0]["action"] = None
    events_total = {"zombies_killed": 0, "plants_eaten": 0, "sun_produced": 0,
                    "mower_triggered": 0, "waves_started": 0, "level_lost": 0}
    log = []  # (tick, wave, kind, detail)
    actions = 0
    rejected = 0
    kills_since_wave = 0
    legal_at = None

    while not obs["terminal"] and actions < max_actions:
        if capture_legal_at is not None and actions == capture_legal_at - 1:
            legal_at = obs.get("legal_actions")
        if prefix_actions is not None and actions < len(prefix_actions):
            action = prefix_actions[actions]
        else:
            action = scripted_choose(obs) if policy == "scripted" else {"type": "wait", "ticks": 60}
            if override and (actions + 1) in override:
                action = override[actions + 1]
        obs, _, done, _, info = env.step(action)
        was_rejected = not info.get("ok")
        if was_rejected:
            rejected += 1
            obs, _, done, _, info = env.step({"type": "wait", "ticks": 60})
        ev = info.get("events") or {}
        for key in events_total:
            events_total[key] += int(ev.get(key) or 0)
        actions += 1

        prev = frames[-1]
        cur = _frame(obs)
        cur["action"] = action
        cur["action_rejected"] = was_rejected

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
                 "deck": list(deck), "policy": policy,
                 "policy_revision": POLICY_REVISION},
        "legal_at": legal_at,
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


def _mower_map(frame: dict) -> dict:
    """这一帧哪几路还有割草机、什么状态。

    兼容两种存法：内存里的 {row: state}，以及存盘再读回来的 [[row, state], ...]。
    存档一律用后者，因为 JSON 对象键只能是字符串。
    """
    m = frame["mowers"]
    if isinstance(m, dict):
        return {int(k): v for k, v in m.items()}
    return {int(r): s for r, s in m}


def _mower_fired(prev: dict, cur: dict) -> list:
    """这一帧有哪些草坪的割草机被用掉了。

    一台割草机在 defenses 里走完两步：`state 1 → 2`（被触发、正在跑），
    然后整条从列表里消失（跑完一趟被回收）。**这是同一台机器的两个阶段，
    不是两台机器** —— 所以只在**第一次**能看见它出事的那一帧报一次。

    两种都要算，但第二种是有条件的：状态变成 TRIGGERED(=2)，或者整条从
    defenses 列表里消失**且上一帧还不是 2**（一次 60 tick 的等待足以跳过
    state=2 那一帧，那时只能靠"消失"来发现）。

    为什么加 `s != 2` 这个条件（2026-10-03 实测）：不加的话，同一台割草机会
    在触发帧和回收帧各报一次，文本一模一样。模型读到两条一模一样的
    "割草机被用掉"，只能猜"是不是有两台？"，然后写：
        "33600 割草机被用掉 and 33960 割草机被用掉 — maybe two mowers? Odd but whatever."
    它放过了这个疑点，但**它在那上面花掉的注意力是真的**。
    """
    a, b = _mower_map(prev), _mower_map(cur)
    fired = [r for r, s in b.items() if s == 2 and a.get(r) != 2]
    fired += [r for r, s in a.items() if r not in b and s != 2 and r not in fired]
    return sorted(fired)


# ---------------------------------------------------------------- 渲染


def _reachable_dps(frame: dict, row: int, front_x: float | None) -> tuple:
    """这一路**真正打得到最前面那只僵尸**的火力。返回 (可打到 dps, 可打到株数, 总株数)。

    植物朝右打，僵尸从右往左走。僵尸一旦走到某株植物的**左边**，那株就再也
    打不到它了 —— 所以"这一路有几个豌豆射手"不等于"有几个正在输出"。

    2026-10-03 修：原先直接数这一路全部射手，会把僵尸身后的植物也算进火力，
    凭空多算输出、把必输的路判成守得住。种子 30001 第 266 步那株种在 c2 的
    豌豆射手，僵尸在它左边 c1 —— 它一炮都打不出去，旧公式却把它算成 0.13 dps。
    """
    front_col = None if front_x is None else _col_of(front_x)
    dps = 0.0
    n_reach = n_all = 0
    for p in frame["plants"]:
        if p[0] != row or p[2] not in SHOOTER_TYPES:
            continue
        n_all += 1
        # p[1] 是植物所在列。同列（僵尸正啃它）也算打不到。
        if front_col is not None and p[1] >= front_col:
            continue
        n_reach += 1
        dps += SHOOTER_DPS.get(p[2], DEFAULT_SHOOTER_DPS)
    return dps, n_reach, n_all


def _capacity(*, dps: float, front_x: float, speed: float, front_hp: int,
              n_reach: int, n_all: int, n_zombies: int, hp_total: int) -> tuple:
    """纯函数：只吃数字，判断"这一路的火力够不够"。返回 (符号, 短语, 展开句)。

    判据是**物理量**，不是经验阈值：

        到割草机还要多少 tick   = (僵尸 x − 割草机 x) / 僵尸速度
        这段时间能打出的总伤害  = **打得到这只僵尸的**火力 dps 之和 × 上面的 tick 数

    伤害率来自 src/Lawn/Plant.cpp（豌豆 20 伤害 / 每 150 tick 一发），不是估的。
    所以"打不死"是推出来的结论，读者可以自己验算，也可以直接反驳这个判据。

    `dps` / `n_reach` 必须是**打得到这只僵尸**的火力（见 `_reachable_dps`），
    不是这一路的火力总和；`n_all` 是这一路射手总株数，只用来区分两种"零火力"：
    真的一株都没有（裸路）vs 有射手但全在僵尸右边打不到（被绕过）。
    这两种情况的应对完全相反 —— 前者要种，后者要种在**更右边**。

    做成纯函数的原因：它同时被 `_lane_rows`（出表格里的短语）和 `_lane_causal`
    （出完整因果句）调用，而 `_lane_causal` 又要读 `_lane_rows` 的输出 ——
    写成读 frame 的形式会互相递归。数字进、结论出，谁都能调。

    短语里带上余量倍数：单帧的 ✗/⚠ 会在 1.0 附近来回跳（差 6 tick 就能翻），
    只给符号读起来像噪声；给出数字，读的人才知道"这是在临界线上"。
    """
    if n_zombies == 0:
        return "✓", "平静", None
    if dps <= 0:
        if n_all > 0:
            return "✗", "火力被绕过", (
                f"这一路有 {n_all} 株射手，但**全都打不到**最前那只僵尸 —— "
                f"它已经走到射手左边（或正在啃那一株），射手朝右打，够不着它。"
                f"这种情况下再往同一条路上补种，种在同样的位置等于白种。")
        return "✗", "裸路承压", (f"没有火力能打到它。{n_zombies} 只僵尸共 {hp_total} 血"
                                 f"没人挡，最前那只已经走到 c{_col_of(front_x)}")
    if speed < 0.02:
        return "⚠", "啃食僵局", (f"{n_reach} 个火力打死最前那只（{front_hp} 血）要 "
                                 f"{front_hp / dps:,.0f} tick。僵尸此刻正停住啃植物，"
                                 f"所以它还没走到割草机，但也没被打死 —— 是个僵局")

    arrive = (front_x - LAWN_XMIN) / speed
    cap = dps * arrive               # 到割草机之前这一路能打出的总伤害
    ratio = cap / front_hp if front_hp > 0 else 0.0
    rest = hp_total - front_hp
    queue = f"，后面还排着 {n_zombies - 1} 只共 {rest} 血" if rest else ""

    if ratio < 1:
        return "✗", f"打不过({ratio:.1f}x)", (
            f"{n_reach} 个火力。最前那只 {arrive:,.0f} tick 后到割草机，"
            f"这段时间最多打出 {cap:,.0f} 伤害，而它有 {front_hp} 血"
            f"—— 打不死{queue}")
    if rest > 0 or ratio < LANE_COMFORT_RATIO:
        return "⚠", f"余量不足({ratio:.1f}x)", (
            f"{n_reach} 个火力能在僵尸到达前打死最前那只"
            f"（{front_hp} 血），余量 {ratio:.1f} 倍{queue}")
    return "✓", f"压得住({ratio:.1f}x)", None


def _lane_rows(frame: dict) -> list:
    """每一路一行派生指标。这是"一眼看出哪路要炸"的核心。"""
    rows = []
    for r in lane_indices(frame):
        plants = [p for p in frame["plants"] if p[0] == r]
        zs = [z for z in frame["zombies"] if z[0] == r and z[5]]
        off = [z for z in frame["zombies"] if z[0] == r and not z[5]]

        shooters = [p for p in plants if p[2] in SHOOTER_TYPES]
        econ = [p for p in plants if p[2] in ECONOMY_TYPES]
        walls = [p for p in plants if p[2] in WALL_TYPES]

        mower_state = _mower_map(frame).get(r)
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

        # 判定 = 物理结论（_capacity 算的）＋ 一条加注。**不是**优先级阶梯。
        #
        # 2026-10-03 踩过的坑：原先的阶梯把"割草机已用"排在第一位，于是
        # 一条被 2 个豌豆射手稳稳压住 20 多波的路，全程被标成"无后手" ——
        # 那是假警报，会主动把读的人引到错的地方去。
        # 后手没了是**风险加注**，不是"这条路要炸"；表格里本来就有独立的
        # 「割草机」列在显示它，判定列再喊一遍就是噪音。
        # 火力只算**打得到最前面那只**的：植物朝右打，僵尸走到植物左边之后
        # 那株就废了。数全部射手会把输出算多（见 _reachable_dps 的注释）。
        if zs:
            dps, n_reach, n_all = _reachable_dps(frame, r, front[2])
            sym, short, _ = _capacity(
                dps=dps, front_x=front[2], speed=front[3], front_hp=front[4],
                n_reach=n_reach, n_all=n_all, n_zombies=len(zs), hp_total=zhp)
        else:
            sym, short, _ = _capacity(
                dps=0.0, front_x=0.0, speed=1.0, front_hp=0,
                n_reach=0, n_all=0, n_zombies=0, hp_total=0)

        if mower_state is None and front_col is not None and front_col <= 0:
            verdict = "已破"                      # 后手没了 + 僵尸到最左列 = 真·即将进屋
        elif mower_state is None:
            verdict = f"{sym} {short}（无后手）"
        else:
            verdict = f"{sym} {short}"

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


def _sun_phrase(sun: int) -> str:
    """把阳光数换成一株植物的量。读者不需要自己记价格表。"""
    n = sun // PLANT_COST[0]
    return (f"{sun}（够买 {n} 个豌豆射手）" if n >= 1
            else f"{sun}（不够买一个豌豆射手，要 {PLANT_COST[0]}）")


def _lane_causal(frame: dict, row: int) -> tuple:
    """一条路的火力够不够 —— 返回 (符号, 展开句)，句子只在"值得说"时才有。

    判据全在 `_capacity` 里（物理量），这里只负责取数和处理"没有僵尸"的情况。
    """
    lane = _lane_rows(frame)[row]
    zs = [z for z in frame["zombies"] if z[0] == row and z[5]]

    if not zs:
        # 空路 + 没后手：只有在**同时没有火力**时才算风险。有火力守着的空路
        # 不需要报警 —— 它马上就会把下一只打死。
        n_all = _reachable_dps(frame, row, None)[2]
        if n_all == 0 and lane["mower"] in ("已消耗", "已触发"):
            return "⚠", ("这一路没有火力也没有割草机 —— 下一只僵尸进来就是直接进屋，"
                         "中间没有任何东西能挡")
        return "✓", None

    front = min(zs, key=lambda z: z[2])
    hp_total = sum(z[4] for z in zs)
    dps, n_reach, n_all = _reachable_dps(frame, row, front[2])
    sym, _, why = _capacity(
        dps=dps, front_x=front[2], speed=front[3], front_hp=front[4],
        n_reach=n_reach, n_all=n_all, n_zombies=len(zs), hp_total=hp_total)
    if why and dps > 0 and n_all > n_reach:
        # 这是最容易被看漏的一类败因：植物是种下去了，但种在僵尸后面。
        # dps <= 0 时 _capacity 自己已经说清楚了，不重复。
        why += (f"（这一路一共 {n_all} 株射手，其中 {n_all - n_reach} 株在最前那只"
                f"僵尸的右边/同格，**打不到它**，所以没算进火力）")
    return sym, why


def _lane_line(frame: dict, causal: bool = True) -> str:
    """一行态势。稳的路只给一个符号，不稳的路才展开成句子。"""
    if not causal:
        return "｜".join(
            f"{l['row']}路 {l['shooters']}火力/{l['zombies']}僵尸"
            + ("❗" if (l["shooters"] == 0 and l["zombies"]) else "")
            for l in _lane_rows(frame)) + f"｜阳光 {frame['sun']}"
    parts = []
    for r in lane_indices(frame):
        sym, _ = _lane_causal(frame, r)
        parts.append(f"{r}路{sym}")
    return "｜".join(parts) + f"｜阳光 {_sun_phrase(frame['sun'])}"


def _lane_warnings(frame: dict) -> list:
    """只把不稳的几条路展开成因果句子。"""
    out = []
    for r in lane_indices(frame):
        sym, why = _lane_causal(frame, r)
        if why:
            out.append((r, sym, why))
    return out


def _board_ascii(frame: dict, mode: str) -> str:
    """mode='plant' 画植物，'zombie' 画僵尸密度。"""
    lanes = lane_indices(frame)
    nrows = (max(lanes) + 1) if lanes else GRID_ROWS
    grid = [["." for _ in range(GRID_COLS)] for _ in range(nrows)]
    # 地形底纹：水路画成 `~`。让"这两路种不了"在棋盘上**看得见** ——
    # 只看文字说明，模型仍会去算"该在第 2 路种什么"，而那两路物理上种不下。
    #
    # 只画水路（GridSquareType 3）。**屋顶不画**：屋顶关每格的 terrain 仍是
    # 普通地面（实测第 41 关），花盆要求来自 `roof` 标志位；画了会让人以为
    # "只有这几格需要花盆"，那是错的。见 pvz_constants._fmt_terrain。
    row_terrain = frame.get("row_terrain") or []
    for r, t in enumerate(row_terrain):
        if r < nrows and t == 3:  # GRIDSQUARE_POOL
            grid[r] = ["~" for _ in range(GRID_COLS)]
    if mode == "plant":
        for r, c, t, hp, mx in frame["plants"]:
            if 0 <= r < nrows and 0 <= c < GRID_COLS:
                g = PLANT_GLYPH.get(t, "?")
                grid[r][c] = g if hp >= mx else g.lower()
    else:
        for r, t, x, *_ in frame["zombies"]:
            if not (0 <= r < nrows):
                continue
            c = _col_of(x)
            if not (0 <= c < GRID_COLS):
                continue
            cur = grid[r][c]
            # `~`（水路）也当空格处理 —— 僵尸是会游过水路的，
            # 而且 `int("~")` 会抛异常，那不是"没有僵尸"，是程序炸了。
            n = 1 if cur in (".", "~") else int(cur) + 1
            grid[r][c] = str(min(n, 9))

    header = "        " + " ".join(f"c{c}" for c in range(GRID_COLS))
    lines = [header]
    for r in range(nrows):
        lines.append(f"  r{r}     " + "  ".join(grid[r]))
    return "\n".join(lines)


def _disp(s) -> int:
    """终端显示宽度：中日韩字符算两格。表格对齐靠它。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))


def _pad(s, width: int) -> str:
    """右对齐补齐（数字列用）。保留是因为别处还在按这个名字调用。"""
    s = str(s)
    return " " * max(0, width - _disp(s)) + s


def _cell(s, width: int, align: str = ">") -> str:
    """按显示宽度补齐一个单元格。align='>' 右对齐（数字），'<' 左对齐（文字）。

    为什么必须分对齐：中文字符在终端占两格，靠空格硬拼列一定歪；
    而**文字列右对齐**会在列首留出一大片空白，读起来像"两列之间隔了十几个空格"
    —— 那不是列宽，那是没对齐。数字列才需要右对齐（位数对齐才能比大小）。
    """
    if not width:
        return str(s)
    gap = " " * max(0, width - _disp(s))
    return gap + str(s) if align == ">" else str(s) + gap


def _render_row(cols, cells, indent: str = "  ") -> str:
    """按列宽渲染一行。列描述是 (表头, 宽度[, 对齐])；宽度 0 = 末列不补齐。

    全仓库的表都走这里，别各自再写一份 fmt 闭包 —— 对齐规则只该有一份。
    """
    out = []
    for col, v in zip(cols, cells):
        width = col[1]
        align = col[2] if len(col) > 2 else ">"
        out.append(_cell(v, width, align) if width else str(v))
    return (indent + "  ".join(out)).rstrip()


def _lane_table(frame: dict) -> str:
    cols = [("路", 2), ("火力", 4), ("阳光", 4), ("挡路", 4), ("僵尸", 4),
            ("僵尸总血", 8), ("最前僵尸", 8), ("距割草机", 10, "<"), ("割草机", 8),
            ("判定", 0)]

    def fmt(cells) -> str:
        return _render_row(cols, cells)

    head = fmt([n for n, *_ in cols])
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
    # 路号从数据来（泳池/雾 6 条，其余 5 条），别用 GRID_ROWS 硬编码。
    lane_ids = lane_indices(peak[sorted(peak)[-1]]) if peak else []

    cols = [("波", 4)] + [(f"{r}号路", 8) for r in lane_ids] + \
           [("阳光", 6), ("割草机已用", 10), ("备注", 0)]

    def fmt(cells) -> str:
        return _render_row(cols, cells)

    head = fmt([n for n, *_ in cols])
    lines = [head, "  " + "-" * (_disp(head) - 2)]

    for w in sorted(peak):
        f = peak[w]
        # 按**路号**索引，别按下标 —— 泳池/雾关 lanes=[0,1,4,5]，
        # 列表下标与路号错一位（6 路棋盘的实测坑）。
        lanes = {l["row"]: l for l in _lane_rows(f)}
        lane_ids = sorted(lanes)
        cells = [w] + [f"{lanes[r]['shooters']}/{lanes[r]['zombies']}"
                       for r in lane_ids]
        notes = []
        hit = [r for r in lane_ids if (w, r) in fired]
        if hit:
            notes.append("割草机 " + "、".join(str(r) for r in hit))
        bare = [r for r in lane_ids
                if lanes[r]["shooters"] == 0 and lanes[r]["zombies"]]
        if bare:
            notes.append("裸路 " + "、".join(str(r) for r in bare))
        cells += [f["sun"], len(lane_ids) - len(f["mowers"]), "；".join(notes)]
        lines.append(fmt(cells))
    lines.append(f"  格内写法：火力/僵尸数。割草机已用 = 已消耗台数（满额 {len(lane_ids)}）。")
    return "\n".join(lines)


def _plant_survivals(frames: list, row: int, t0=None, t1=None) -> list:
    """某条路上每株植物从种下到消失（被吃或被铲）活了多少 tick。

    按格子占用区间算，不靠植物身份——同一格反复补种同一种植物也能分开计。
    """
    occ, out = {}, []
    for f in frames:
        cur = {p[1]: p[2] for p in f["plants"] if p[0] == row}
        for c, (st, _ty) in list(occ.items()):
            if c not in cur:
                out.append((st, f["tick"] - st))
                del occ[c]
        for c, ty in cur.items():
            if c not in occ:
                occ[c] = (f["tick"], ty)
    for c, (st, _ty) in occ.items():
        out.append((st, frames[-1]["tick"] - st))
    return [x for x in out if (t0 is None or x[0] >= t0) and (t1 is None or x[0] < t1)]


def _lane_collapse(rec: dict, min_samples: int = 3) -> list:
    """每条路"丢割草机之前 vs 之后"植物存活时长的对比。

    它说明的是**这一路是不是绞肉机**（投进去的植物活不久），
    **不是**"这把要输"的预言。两次实测：

    - 第一轮（脚本教师，第 7 关，seed 30000–30015 共 16 局）：崩塌比 ≥4× 的 7 局
      全部失败，≤3× 的 9 局里 8 局通关 —— 看起来像个预言，但样本只有 16。
    - 2026-10-03 反事实立刻给出反例：seed 30001 把第 266 步那株豌豆射手从 c2 挪到
      c6，**通关 30/30**，而这一局的 1 号路崩塌比是 **28.3×**（远高于 4× 的"阈值"）。
      1 号路确实成了绞肉机，但赢是靠别的路赢的。

    所以：**高崩塌 ≠ 会输**。它是"别再往这条路投钱"的线索，不是败因判定。
    任何把它当阈值规则用的代码都是在过度解读。
    """
    frames = rec["frames"]
    out = []
    for r in lane_indices(frames[-1]):
        mt = next((frames[i]["tick"] for i in range(1, len(frames))
                   if r in _mower_fired(frames[i - 1], frames[i])), None)
        if mt is None:
            continue
        before = [d for _s, d in _plant_survivals(frames, r, None, mt)]
        after = [d for _s, d in _plant_survivals(frames, r, mt, None)]
        if len(before) < min_samples or len(after) < min_samples:
            continue
        mb, ma = statistics.median(before), statistics.median(after)
        if ma <= 0:
            continue
        out.append({"row": r, "tick": mt, "before": mb, "after": ma,
                    "ratio": mb / ma, "n_before": len(before), "n_after": len(after)})
    return out


def _signals(rec: dict) -> list:
    """机械信号：一组**可核对的候选线索**，不是结论。

    这是唯一的规则实现处。`render_episode` 的文本报告和 `episode_query` 的目录
    都从这里取，避免两份实现漂移（这个仓库为重复实现付过代价）。

    每条返回 {rule, text, hint?}：rule 是规则名（可以被逐条反驳），
    hint 是"接下来该去查什么"（只有能接上 episode_query 的规则才给）。

    **这些规则盖不全失败方式，而且不能区分胜负。** 2026-10-03 实测把同一组规则
    跑在一败（seed 30001 基线，第 17/30 波）一胜（同 seed 的反事实，30/30 通关）
    两局上：火力上限、压力>火力、净损失、无后手、阳光闲置、存活崩塌 —— **六条
    在胜局里同样全部触发**，只有"哪一路"不同。所以这里给的是**去看哪里的索引**，
    不是败因。真正的因果要靠 `episode_query whatif`（确定性重放）去测。
    """
    peak = _peak_frames(rec)
    frames = rec["frames"]
    waves = sorted(peak)
    out = []
    # 路号从数据来（泳池/雾 6 条，其余 5 条），别用 GRID_ROWS 硬编码。
    lane_ids = lane_indices(frames[-1])

    # 1. 火力上限：整局每条路最多同时有几个火力植物
    ceiling = {r: 0 for r in lane_ids}
    for w in waves:
        for lane in _lane_rows(peak[w]):
            ceiling[lane["row"]] = max(ceiling[lane["row"]], lane["shooters"])
    if waves:
        worst = min(ceiling, key=lambda r: ceiling[r])
        best = max(ceiling.values())
        if best - ceiling[worst] >= 2:
            out.append({
                "rule": "火力上限",
                "text": f"{worst} 号路全程火力上限只有 {ceiling[worst]}，"
                        f"其它路最高到 {best}——{len(lane_ids)} 条路里相对最弱的一条。",
                "hint": f"lane --row {worst}",
            })

    # 2. 后半程"僵尸数 > 火力数"的波次：输出跟不上压力
    half = waves[len(waves) // 2:] if waves else []
    over = {r: [] for r in lane_ids}
    for w in half:
        for lane in _lane_rows(peak[w]):
            if lane["zombies"] > lane["shooters"]:
                over[lane["row"]].append(w)
    if over:
        r = max(over, key=lambda k: len(over[k]))
        if len(over[r]) >= 2:
            out.append({
                "rule": "压力>火力",
                "text": f"{r} 号路在后半程（{half[0]}–{half[-1]} 波）有 {len(over[r])} 个波次"
                        f"僵尸数超过火力数，是全盘输出最跟不上压力的一路。",
                "hint": f"between --wave {half[0]} --to-wave {half[-1]}",
            })

    # 3. 植物净损失最多的路（反复补种的绞肉机）
    losses = {r: 0 for r in lane_ids}
    for i in range(1, len(frames)):
        for r, _c, _t, _h in _plant_diff(frames[i - 1]["plants"], frames[i]["plants"]):
            losses[r] += 1
    if any(losses.values()):
        worst = max(losses, key=lambda r: losses[r])
        out.append({
            "rule": "净损失",
            "text": f"{worst} 号路植物净损失 {losses[worst]} 株（全路合计 "
                    f"{sum(losses.values())} 株），是补种最频繁的一路。",
            "hint": f"lane --row {worst}",
        })

    # 4. 割草机在哪几波被用掉
    fired = _mower_fired_waves(rec)
    if fired:
        by_wave: dict[int, list] = {}
        for w, r in sorted(fired):
            by_wave.setdefault(w, []).append(r)
        detail = "；".join(f"第 {w} 波 {rs} 号路" for w, rs in by_wave.items())
        out.append({
            "rule": "割草机",
            "text": f"割草机共消耗 {len(fired)} 台：{detail}。",
            "hint": None,
        })

    # 5. 终局时哪几路已经没后手
    last = _lane_rows(frames[-1])
    naked = [l for l in last if l["mower"] in ("已消耗", "已触发")]
    if naked:
        detail = "、".join(f"{l['row']} 号路（{l['shooters']} 个火力）" for l in naked)
        out.append({
            "rule": "无后手",
            "text": f"终局时这几路已经没有割草机：{detail}。"
                    f"没有割草机 = 再被突破就直接进屋，没有缓冲；"
                    f"但火力够的路照样守得住 —— 这是风险敞口，不是失败原因。",
            "hint": None,
        })

    # 6. 钱和火力没对上：阳光够买好几株了，却有路零火力还站着僵尸
    idle = [(f["sun"], f["wave"], f["tick"]) for f in frames
            if f["sun"] >= 300
            and any(l["shooters"] == 0 and l["zombies"] for l in _lane_rows(f))]
    if idle:
        s, w, tk = max(idle)
        out.append({
            "rule": "阳光闲置",
            "text": f"第 {w} 波时手里有 {_sun_phrase(s)}，场上却仍有"
                    f"「有僵尸、零火力」的路——钱没换成火力。",
            "hint": f"frame --tick {tk}",
        })

    # 7. 存活崩塌：某条路越过不可恢复点。
    #    判据是"丢割草机之后新种的植物还能活多久"，不是"现在压力大不大"。
    #    实测 16 个种子：>=4x 的 7 局全败，<=3x 的 9 局里 8 局通关。
    coll = _lane_collapse(rec)
    if coll:
        worst = max(coll, key=lambda d: d["ratio"])
        if worst["ratio"] >= 4:
            out.append({
                "rule": "存活崩塌",
                "text": f"{worst['row']} 号路在 tick {worst['tick']} 丢掉割草机后，"
                        f"新种下去的植物中位只能活 {worst['after']:,.0f} tick"
                        f"（之前是 {worst['before']:,.0f} tick，差 {worst['ratio']:.1f} 倍）"
                        f"—— 这一路已经是绞肉机，往这里投的植物基本等于扔钱。"
                        f"**注意：这不代表会输**，反事实胜局里出现过 28.3 倍的崩塌"
                        f"仍然通关（赢是靠别的路赢的）。",
                "hint": f"lane --row {worst['row']}",
            })

    return out


def _diagnose(rec: dict) -> list:
    """只取文本，给不需要 hint 的地方用。"""
    sigs = _signals(rec)
    if not sigs:
        return ["没有触发任何机械告警信号；这一局可能是被整体压制而非单点失误。"]
    return [s["text"] for s in sigs]


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
    for r in lane_indices(frames[-1]):
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
        bare = [r for r in lane_indices(f)
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
        lanes = {l["row"]: l for l in _lane_rows(peak[w])}
        lane_ids = sorted(lanes)
        wave_matrix.append({
            "wave": w,
            "lanes": [{"row": r, "shooters": lanes[r]["shooters"],
                       "zombies": lanes[r]["zombies"]} for r in lane_ids],
            "sun": peak[w]["sun"],
            "mowers_used": len(lane_ids) - len(peak[w]["mowers"]),
            "mowers_fired_rows": [r for r in lane_ids if (w, r) in fired],
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
