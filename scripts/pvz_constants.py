#!/usr/bin/env python3
"""游戏常量表 —— **从 C++ 源码现场解析**，不手抄。

为什么要有这个
--------------
实测模型的推理轨迹里，有一类反复出现的消耗：它在**猜游戏常量**。

    "I don't know the exact arm time. The emulator might model potato mine
     arm time as 15s = 900 ticks."

它先猜 900，又反推到 ~1600，正确答案是 **1500 tick 的倒计时 + 一段升起动画**。
错的地方不是"15 秒"这个游戏常识，而是 **tick 与秒的换算率**：它假设 60 tick/s，
这个模拟器是 100。一个常数猜错，它写下的"可迁移结论"整条就跟着错 ——
而那条结论是要沉淀成 skill 的。

所以：**常量必须由能回答它的东西给出来。** 这里能回答它的，是模拟器自己的源码。

设计立场
--------
1. **单一事实源 = 真正跑起来的那份 C++。** 手抄的常量表会漂移：源码改了、抄本没改，
   模型拿到的就是错的，而且**错得没有症状**（不会报错，只会给出一个像模像样的错答案）。
   这里全部现解析。
2. **解析失败要炸，不要静默返回空表。** 空表比没有更坏：模型会转而相信自己的 PvZ 常识，
   而我们正是为了纠正那个才做这个工具。所以任何一节解析不出来都直接抛 RuntimeError，
   并指明是哪个文件、哪个符号。
3. **只给事实，不给结论。** 这里不写"所以土豆雷该种在 c3" —— 那是模型的活。

用法
----
    python3 scripts/pvz_constants.py                  # 全部
    python3 scripts/pvz_constants.py --list           # 列出有哪些节
    python3 scripts/pvz_constants.py --section time   # 只取一节（省 token）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

# 植物 / 僵尸的 SeedType/ZombieType 编号 -> 中文名。
# 以前放在 render_episode 里，导致 pvz_constants → render_episode →
# scripted_baseline → pvz_constants 的循环导入（2026-10-03），搬到这里归位：
# 名字表本来就是常量。
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

PLANT_CPP = ROOT / "src" / "Lawn" / "Plant.cpp"
ZOMBIE_CPP = ROOT / "src" / "Lawn" / "Zombie.cpp"
SEXY_CPP = ROOT / "src" / "SexyAppFramework" / "SexyAppBase.cpp"
CONST_ENUMS = ROOT / "src" / "ConstEnums.h"
CHALLENGE_CPP = ROOT / "src" / "Lawn" / "Challenge.cpp"
BOARD_CPP = ROOT / "src" / "Lawn" / "Board.cpp"


# ---------------------------------------------------------------- 解析原语


def _read(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(
            f"常量来源文件不存在：{path}\n"
            f"常量表是从 C++ 源码现场解析的，源码挪了位置就得改这里的路径常量，"
            f"不能改成手抄表。"
        )
    return path.read_text(encoding="utf-8", errors="replace")


_FUNC_DEF = re.compile(r"(?m)^[A-Za-z_][^\n;{}]*?\b([A-Za-z_]\w*)::([A-Za-z_]\w*)\s*\(")


def _function_span(text: str, needle: str) -> tuple[int, int]:
    """返回包含 `needle` 首次出现位置的**顶层函数**的字符区间。

    为什么按函数切：同一个 `case SeedType::X:` 在文件里出现多次（初始化、伤害判定、
    渲染各一份），整文件扫会把它们混成一锅。按函数切才拿得到"初始化那一份"。
    """
    at = text.find(needle)
    if at < 0:
        raise RuntimeError(f"源码里找不到 {needle!r} —— 解析器的锚点失效了，去看一眼源码。")
    starts = [m.start() for m in _FUNC_DEF.finditer(text)]
    before = [s for s in starts if s <= at]
    if not before:
        raise RuntimeError(f"{needle!r} 不在任何 `Type::Func(` 形式的函数里。")
    after = [s for s in starts if s > at]
    return before[-1], (after[0] if after else len(text))


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _strip_comments(text: str) -> str:
    return _COMMENT.sub("", text)


_CASE_RE = re.compile(r"case\s+(\w+)::(\w+)\s*:")


def _case_blocks(body: str, enum: str, var_names: tuple[str, ...]):
    """把 switch 体拆成 `([标签...], {变量: 值})`。

    关键点：**共用 body 的连续 case 标签要归成一组**。
    `case A: case B: mPlantHealth = 4000;` 里 4000 是 A 和 B 共有的 ——
    按标签逐个切会把 B 漏掉，而漏掉的表现是"查不到这个植物的血量"，
    模型只好去猜。所以先累积标签，遇到有语句的段才收成一组。
    """
    marks = [(m.group(2), m.start(), m.end())
             for m in _CASE_RE.finditer(body) if m.group(1) == enum]
    if not marks:
        raise RuntimeError(f"给定函数体里没有 `case {enum}::` 标签 —— 锚点失效。")
    var_re = re.compile(r"\b(" + "|".join(var_names) + r")\s*=\s*(\d+)\s*;")
    groups: list[tuple[list[str], dict[str, int]]] = []
    pending: list[str] = []
    for i, (name, _start, end) in enumerate(marks):
        stop = marks[i + 1][1] if i + 1 < len(marks) else len(body)
        seg = _strip_comments(body[end:stop])
        pending.append(name)
        if seg.strip():
            groups.append((pending, {k: int(v) for k, v in var_re.findall(seg)}))
            pending = []
    if pending:
        groups.append((pending, {}))
    return groups


def _parse_enum(text: str, name: str) -> dict[str, int]:
    """解析 `enum <name> : <底层类型> { ... }`，支持显式赋值和隐式递增。

    为什么要解析枚举而不是"按表里的出现顺序编号"：动作参数要的是 **SeedType 的数值**，
    而表里的顺序只是"恰好"和它一致。用枚举是直接拿事实；用顺序是拿一个巧合。
    巧合会漂移（有人在中间插一个 SEED_*，表顺序就和枚举脱钩了），
    所以下面还有一道**交叉校验**：两者不一致就报错。
    """
    at = text.find(f"enum {name}")
    if at < 0:
        raise RuntimeError(f"{CONST_ENUMS} 里找不到 enum {name}。")
    open_brace = text.find("{", at)
    close = text.find("}", open_brace)
    if open_brace < 0 or close < 0:
        raise RuntimeError(f"enum {name} 的花括号没配平。")
    body = _strip_comments(text[open_brace + 1:close])
    out: dict[str, int] = {}
    prev = -1
    for token in body.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\w+)\s*=\s*(-?\d+)", token)
        if m:
            out[m.group(1)] = int(m.group(2))
            prev = int(m.group(2))
            continue
        m = re.fullmatch(r"(\w+)", token)
        if m:
            prev += 1
            out[m.group(1)] = prev
            continue
        raise RuntimeError(f"enum {name} 里有解析不了的条目：{token!r}")
    if not out:
        raise RuntimeError(f"enum {name} 解析出 0 项。")
    return out


def _array_body(text: str, symbol: str) -> str:
    at = text.find(symbol)
    if at < 0:
        raise RuntimeError(f"源码里找不到 {symbol} —— 表被改名了？")
    open_brace = text.find("{", at)
    close = text.find("};", open_brace)
    if open_brace < 0 or close < 0:
        raise RuntimeError(f"{symbol} 的数组体没找到（`{{` .. `}};`）。")
    return text[open_brace:close]


def _parse_rows(body: str, line_re: re.Pattern, symbol: str, marker: str) -> list[tuple]:
    """逐行解析表体。`marker` 是"这一行本该是一个条目"的特征串。

    行匹配不上有两种情况，必须分开：
    - 不是条目行（表体里的注释、空行）→ 跳过；
    - **是**条目行但解析不了 → 立刻报错。
    混在一起处理的话，字段改名会退化成"少解析出几行"，而少的那几行
    在模型眼里就是"这个植物没有血量数据"，它会去猜。
    """
    rows = []
    for line in body.splitlines():
        m = line_re.search(line)
        if m:
            rows.append(m.groups())
        elif marker in line:
            raise RuntimeError(
                f"{symbol} 里这一行有条目但解析不了：\n  {line.strip()[:160]}\n"
                f"字段顺序/名字变了？解析器要跟着改 —— 不要改成手抄表。"
            )
    if not rows:
        raise RuntimeError(f"{symbol} 解析出 0 条 —— 这比没有更坏，直接报错。")
    return rows


# ---------------------------------------------------------------- 各节数据


_PLANT_LINE = re.compile(
    r"\.mSeedType\s*=\s*SeedType::(\w+)\s*,"
    r".*?\.mPacketIndex\s*=\s*(\d+)\s*,"
    r".*?\.mSeedCost\s*=\s*(\d+)\s*,"
    r".*?\.mRefreshTime\s*=\s*(\d+)\s*,"
    r".*?\.mLaunchRate\s*=\s*(\d+)\s*,"
    r".*?\.mPlantName\s*=\s*\"([^\"]+)\""
)

_ZOMBIE_LINE = re.compile(
    r"\.mZombieType\s*=\s*(\w+)\s*,"
    r".*?\.mZombieValue\s*=\s*(\d+)\s*,"
    r".*?\.mStartingLevel\s*=\s*(\d+)\s*,"
    r".*?\.mFirstAllowedWave\s*=\s*(\d+)\s*,"
    r".*?\.mPickWeight\s*=\s*(\d+)\s*,"
    r".*?\.mZombieName\s*=\s*\"([^\"]+)\""
)

_PLANT_VARS = ("mPlantHealth", "mStateCountdown", "mDoSpecialCountdown", "mBlinkCountdown")
_ZOMBIE_VARS = ("mBodyHealth", "mHelmHealth", "mShieldHealth", "mFlyingHealth", "mPhaseCounter")

_CONSTEXPR = re.compile(r"(?m)^\s*constexpr\s+const\s+(\w+)\s+(\w+)\s*=\s*([^;]+);")


def tick_rate() -> int:
    """游戏逻辑帧率（每秒多少 tick）。来源：SexyAppBase 的 mSyncRefreshRate 默认值。

    这是整张表里**最要命的一个数**：它错了，所有 tick↔秒 的换算都错，
    而模型不会察觉。所以单独一个函数，单独报错。
    """
    text = _read(SEXY_CPP)
    m = re.search(r"mSyncRefreshRate\s*=\s*(\d+)\s*;", text)
    if not m:
        raise RuntimeError(f"{SEXY_CPP} 里找不到 mSyncRefreshRate 的默认赋值。")
    return int(m.group(1))


def plant_defs() -> list[dict]:
    body = _array_body(_read(PLANT_CPP), "gPlantDefs")
    ids = _parse_enum(_read(CONST_ENUMS), "SeedType")
    out = []
    for index, (seed, packet, cost, refresh, launch, name) in enumerate(
            _parse_rows(body, _PLANT_LINE, "gPlantDefs", ".mSeedType")):
        if ids.get(seed) != index:
            raise RuntimeError(
                f"gPlantDefs 第 {index} 行是 {seed}，但 enum SeedType 里它是 "
                f"{ids.get(seed)}。表顺序和枚举脱钩了 —— 动作参数用的是枚举值，"
                f"所以这里必须以枚举为准，别按顺序编号。"
            )
        out.append({
            "type": index,
            "seed": seed.removeprefix("SEED_"),
            "packet_index": int(packet),
            "cost": int(cost),
            "recharge": int(refresh),
            "launch": int(launch),
            "name": name,
        })
    return out


def plant_specials() -> dict[str, dict[str, int]]:
    """植物初始化分支里的血量与倒计时（`Plant::PlantInitialize` 那个 switch）。"""
    text = _read(PLANT_CPP)
    start, end = _function_span(text, "switch (theSeedType)")
    body = _strip_comments(text[start:end])
    default_health = None
    head = body[: body.find("switch (theSeedType)")]
    m = re.search(r"mPlantHealth\s*=\s*(\d+)\s*;", head)
    if m:
        default_health = int(m.group(1))
    out: dict[str, dict[str, int]] = {}
    for labels, values in _case_blocks(body, "SeedType", _PLANT_VARS):
        for label in labels:
            key = label.removeprefix("SEED_")
            out.setdefault(key, {}).update(values)
    if default_health is None:
        raise RuntimeError("没解析到植物血量的默认值（switch 之前的 mPlantHealth = N）。")
    # switch 里**没有**分支的植物用默认血量（向日葵、大喷菇、窝瓜……都是这一类）。
    # 不补这一手，它们在表里就会显示成 "?"，而模型看到 "?" 只会去猜。
    for d in plant_defs():
        entry = out.setdefault(d["seed"], {})
        entry.setdefault("mPlantHealth", default_health)
    return out


def zombie_defs() -> list[dict]:
    body = _array_body(_read(ZOMBIE_CPP), "gZombieDefs")
    ids = _parse_enum(_read(CONST_ENUMS), "ZombieType")
    out = []
    for index, (ztype, value, level, wave, weight, name) in enumerate(
            _parse_rows(body, _ZOMBIE_LINE, "gZombieDefs", ".mZombieType")):
        if ids.get(ztype) != index:
            raise RuntimeError(
                f"gZombieDefs 第 {index} 行是 {ztype}，但 enum ZombieType 里它是 "
                f"{ids.get(ztype)} —— 表顺序和枚举脱钩了。"
            )
        out.append({
            "type": ztype.removeprefix("ZOMBIE_"),
            "enum": index,
            "value": int(value),
            "level": int(level),
            "wave": int(wave),
            "weight": int(weight),
            "name": name,
        })
    return out


def zombie_specials() -> dict[str, dict[str, int]]:
    text = _read(ZOMBIE_CPP)
    start, end = _function_span(text, "switch (theType)")
    body = _strip_comments(text[start:end])
    default_health = None
    head = body[: body.find("switch (theType)")]
    m = re.search(r"mBodyHealth\s*=\s*(\d+)\s*;", head)
    if m:
        default_health = int(m.group(1))
    out: dict[str, dict[str, int]] = {}
    for labels, values in _case_blocks(body, "ZombieType", _ZOMBIE_VARS):
        for label in labels:
            key = label.removeprefix("ZOMBIE_")
            out.setdefault(key, {}).update(values)
    if default_health is None:
        raise RuntimeError("没解析到僵尸血量的默认值。")
    for d in zombie_defs():
        entry = out.setdefault(d["type"], {})
        entry.setdefault("mBodyHealth", default_health)
    return out


def zombie_wave_table() -> list[int]:
    """`gZombieWaves[NUM_LEVELS]`：每关的**基础**波数（下标 = 关卡号 - 1）。"""
    body = _array_body(_read(CHALLENGE_CPP), "gZombieWaves")
    rows = [int(v) for v in re.findall(r"-?\d+", body)]
    if len(rows) < 50:
        raise RuntimeError(f"gZombieWaves 解析出 {len(rows)} 项，少于 50 —— 表被改了？")
    return rows


def zombie_allowed_levels() -> dict[str, list[int]]:
    """`gZombieAllowedLevels`：每种僵尸在 1..50 关的允许表（0/1）。"""
    body = _array_body(_read(CHALLENGE_CPP), "gZombieAllowedLevels")
    out: dict[str, list[int]] = {}
    for ztype, inner in re.findall(r"\{\s*(\w+)\s*,\s*\{([^}]*)\}\s*\}", body):
        vals = [int(v) for v in re.findall(r"\d+", inner)]
        # C++ 聚合初始化允许写短：`{ ZOMBIE_DUCKY_TUBE, { 0 } }` 表示 50 项全是 0。
        # 所以短了要**补零**（这是 C++ 的语义，不是容错），长了才是真出问题。
        if len(vals) > 50:
            raise RuntimeError(f"gZombieAllowedLevels[{ztype}] 有 {len(vals)} 项，超过 50。")
        vals = vals + [0] * (50 - len(vals))
        out[ztype.removeprefix("ZOMBIE_")] = vals
    if not out:
        raise RuntimeError("gZombieAllowedLevels 解析出 0 项。")
    return out


def background_types() -> dict[int, str]:
    """`enum BackgroundType`：编号 -> 枚举名（关卡背景，决定这一关是白天/夜间/泳池/雾/屋顶）。"""
    return {v: k for k, v in _parse_enum(_read(CONST_ENUMS), "BackgroundType").items()}


def grid_square_types() -> dict[int, str]:
    """`enum GridSquareType`：编号 -> 枚举名（**每一格**的地形）。

    注意它和 `BackgroundType` 是两套东西：前者是"这一格能不能种"，
    后者是"这一关长什么样"。混淆它们会得出错误结论 —— 见下面的 `_fmt_terrain`。
    """
    return {v: k for k, v in _parse_enum(_read(CONST_ENUMS), "GridSquareType").items()}


# 枚举名 -> 给人/模型看的中文名。编号是**事实**（从 ConstEnums.h 解析），
# 中文名是**标签**。认不出就原样回英文，绝不猜 —— 猜错的标签比没有标签更坏。
_BACKGROUND_LABEL = {
    "BACKGROUND_1_DAY": "白天",
    "BACKGROUND_2_NIGHT": "夜间",
    "BACKGROUND_3_POOL": "白天泳池",
    "BACKGROUND_4_FOG": "夜间泳池（有雾）",
    "BACKGROUND_5_ROOF": "屋顶",
    "BACKGROUND_6_BOSS": "Boss 关",
}

_GRIDSQUARE_LABEL = {
    "GRIDSQUARE_NONE": "无（种不了）",
    "GRIDSQUARE_GRASS": "普通地面",
    "GRIDSQUARE_DIRT": "硬地（种不了）",
    "GRIDSQUARE_POOL": "水路",
    "GRIDSQUARE_HIGH_GROUND": "高地/屋顶",
}

# 水路 / 屋顶 各自需要的"底座"植物（SeedType 编号）。
#   睡莲：Board.cpp:2789 起 —— 非水生植物在水上必须先有睡莲。
#   花盆：Board.cpp:2822 起 —— `StageHasRoof() && !aHasFlowerPot` → 需要花盆。
#   香蒲：**睡莲的升级植物**，只能种在已有睡莲的那格上（Board.cpp:2849 起
#     `aUnderPlant->IsUpgradableTo(CATTAIL)` → OK）。空水格直接种会被
#     `Plant::IsUpgrade` 拦成 PLANTING_NEEDS_UPGRADE（Board.cpp:2866）。
#     曾把 2818 行"水上无睡莲时放行香蒲"误读成"香蒲可免睡莲直接种"——
#     那只是让香蒲**通过**水路检查，后面还有升级检查等着它（2026-10-03 实测踩过）。
#
# ⚠ 但**别拿源码规则当实测结论用**。屋顶那条就是这样栽的（2026-10-03）：
#   按源码推"卡组里没花盆 → 一棵都种不下"，实测第 41 关**开局就预置了 25 个花盆**
#   （c0–c4 × 5 路），直接种就行。所以 `episode_query._fmt_scene` 里屋顶那段是
#   **从这一帧的植物里读花盆在哪几列**，不照抄这条规则。
LILYPAD = 16
FLOWER_POT = 33
CATTAIL = 43
GRAVE_BUSTER = 11


def background_label(value) -> str:
    """BackgroundType 的编号 -> 中文名。认不出就回英文枚举名（不猜、不返回空）。"""
    name = background_types().get(int(value)) if value is not None else None
    if name is None:
        return f"(未知背景 {value})"
    return _BACKGROUND_LABEL.get(name, name)


def gridsquare_label(value) -> str:
    """GridSquareType 的编号 -> 中文名。同上：认不出就回英文枚举名。"""
    if value == -1:
        return "混合地形"
    name = grid_square_types().get(int(value)) if value is not None else None
    if name is None:
        return f"(未知地形 {value})"
    return _GRIDSQUARE_LABEL.get(name, name)


def upgrade_plants() -> frozenset[int]:
    """升级植物集合（SeedType 编号），从 `Plant::IsUpgrade`（Plant.cpp）解析。

    为什么要单独一个函数：升级植物进卡组前必须写进 profile 的
    `owned_upgrade_plants`，否则 reset 被拒；而且它们**种不下**——
    香蒲必须有睡莲垫底、双子向日葵必须有向日葵垫底（Board.cpp:2866
    `Plant::IsUpgrade → PLANTING_NEEDS_UPGRADE`，只有"种在底座上"的
    分支能提前放行）。模型不知道这条会得出"香蒲怎么种都不合法"的困惑。
    """
    start, end = _function_span(_read(PLANT_CPP), "bool Plant::IsUpgrade")
    body = _read(PLANT_CPP)[start:end]
    names = re.findall(r"SeedType::(SEED_[A-Z_]+)", body)
    if not names:
        raise RuntimeError("Plant::IsUpgrade 里没解析到任何 SEED_*。函数挪位置了？")
    ids = {seed: i for i, seed in
           ((k, v) for v, k in _parse_enum(_read(CONST_ENUMS), "SeedType").items())}
    missing = [n for n in names if n not in ids]
    if missing:
        raise RuntimeError(f"IsUpgrade 里的这些枚举在 SeedType 里找不到：{missing}")
    return frozenset(ids[n] for n in names)


def background_for_level(level: int) -> str:
    """关卡号 -> BackgroundType 枚举名。规则照抄 `Board::PickBackground`（Board.cpp）。

    特例：第 35 关是 ScaryPotter 挑战（LawnApp.cpp:2749
    `IsAdventureMode() && mBoard->mLevel == 35`），在 31..40 的雾区里
    ** override 成夜间草地**——只按"每 10 关一个区"推会推错这一关。
    实测印证（2026-10-03）：L35 terrain=1、night=1、无雾无水、15 个墓/壶格。
    """
    if not 1 <= level <= 50:
        raise RuntimeError(f"关卡号要在 1..50，收到 {level}。")
    if level == 35:  # IsScaryPotterLevel：优先于雾区判定（Board.cpp:867 起）
        return "BACKGROUND_2_NIGHT"
    text = _read(BOARD_CPP)
    start, end = _function_span(text, "void Board::PickBackground")
    body_bg = text[start:end]
    adventure = body_bg[body_bg.index("GAMEMODE_ADVENTURE"):body_bg.index("case GameMode::GAMEMODE_SURVIVAL")]
    bg = background_types()
    known = set(bg.values())
    # mLevel <= N * LEVELS_PER_AREA → BACKGROUND_X，按出现顺序取第一个命中的。
    thresholds = re.findall(
        r"mLevel <= (\d+) \* LEVELS_PER_AREA\)\s*\{\s*mBackground = BackgroundType::(BACKGROUND_\w+);",
        adventure)
    for n_str, bg_name in thresholds:
        if level <= int(n_str) * 10:
            if bg_name not in known:
                raise RuntimeError(f"PickBackground 里的 {bg_name} 不在 BackgroundType 枚举里。")
            return bg_name
    # 40 < level < FINAL_LEVEL → ROOF；== FINAL_LEVEL → BOSS（GameConstants.h: FINAL_LEVEL = 50）。
    if level < 50:
        return "BACKGROUND_5_ROOF"
    return "BACKGROUND_6_BOSS"


def level_report(level: int) -> list[str]:
    """按源码规则现算这一关的静态参数 —— **规则也一并给出**，不只给结论。

    为什么把规则写出来：这几个数都是**推导**出来的，不是表里的字面量。
    推导过程不写，模型就无法判断"这个数在什么条件下不成立"
    （比如首次通关和重复通关的波数不一样）。给结论不给条件，
    等于换一种方式让它猜。
    """
    if not 1 <= level <= 50:
        raise RuntimeError(f"关卡号要在 1..50，收到 {level}。")
    waves = zombie_wave_table()
    base = waves[level - 1]
    repeat = 20 if base < 10 else base + 10
    allowed = zombie_allowed_levels()
    defs = {d["type"]: d for d in zombie_defs()}
    specials = zombie_specials()

    L = [f"── level {level} 的静态参数（按源码规则现算）──", ""]
    L.append(f"  场景：{background_for_level(level)}"
             f"（规则 Board::PickBackground；第 35 关是 ScaryPotter 特例）")
    L.append("")
    L.append(f"  基础波数 gZombieWaves[{level - 1}] = {base}"
             f"（来源 src/Lawn/Challenge.cpp）")
    L.append("  实际波数规则（Board::PickZombieWaves）：")
    L.append(f"    · 首次通关：{base}")
    L.append(f"    · 非首次通关：< 10 取 20，否则 +10 → **{repeat}**")
    L.append("    （用存档里的 wave_count 核对一下是哪个分支，别默认。）")
    L.append("")
    L.append("  可能出现的僵尸（Board::CanZombieSpawnOnLevel，三个条件同时成立）：")
    L.append(f"    ① level >= mStartingLevel   ② mPickWeight != 0   "
             f"③ gZombieAllowedLevels[type].mAllowedOnLevel[{level - 1}] != 0")
    L.append("")
    hits = 0
    labels = {"mHelmHealth": "头盔", "mShieldHealth": "护盾", "mFlyingHealth": "气球"}
    for ztype, d in sorted(defs.items(), key=lambda kv: kv[1]["enum"]):
        if level < d["level"] or d["weight"] == 0:
            continue
        if not allowed.get(ztype, [0] * 50)[level - 1]:
            continue
        hits += 1
        sp = specials.get(ztype, {})
        hp = sp.get("mBodyHealth", "?")
        extra = {k: v for k, v in sp.items() if k in labels and v}
        extra_s = "".join(f" + {labels[k]} {v}" for k, v in sorted(extra.items()))
        cn = ZOMBIE_NAME.get(d["enum"], d["name"])
        L.append(f"    {cn:<10} {ztype:<16} 本体 {hp}{extra_s}")
    if hits == 0:
        L.append("    （一只都没有 —— 这三个条件至少有一个我解析错了，去看源码。）")
    return L


def numeric_constants() -> dict[str, list[tuple[str, str, str]]]:
    """源码里的 `constexpr const <type> NAME = expr;`，按文件分组。

    排除 `RENDER_GROUP_*`：那是渲染层编号，对"这一局怎么打"没有信息量。
    这是**唯一**一条人工过滤，且只做减法、不改值。
    """
    out: dict[str, list[tuple[str, str, str]]] = {}
    for path in (PLANT_CPP, ZOMBIE_CPP):
        rows = []
        for type_, name, expr in _CONSTEXPR.findall(_read(path)):
            if name.startswith("RENDER_GROUP_"):
                continue
            rows.append((type_, name, " ".join(expr.split())))
        out[path.name] = rows
    return out


# ---------------------------------------------------------------- 渲染

SECTIONS = ("time", "combat", "plants", "zombies", "level", "terrain", "literals")


def _fmt_terrain() -> list[str]:
    """场景与地形 —— 这一节存在的理由不是"把枚举列出来"，是**防止两套编号被搞混**。

    实测踩到的两个坑（2026-10-03），两个都是"照着源码推断"推错的：

      · **屋顶的花盆要求不能照抄源码。** 源码 Board.cpp:2822 写的是
        `StageHasRoof() && !aHasFlowerPot` → 需要花盆。按这条推出
        "卡组里没花盆 → 一棵都种不下"，**实测是错的**：本环境在屋顶关
        **开局就预置好了花盆**（第 41 关是 c0–c4 × 5 路，共 25 个），直接种就行。
        所以"屋顶哪几列能种"要看这一帧里花盆在哪 —— `episode_query._fmt_scene`
        是**从这一帧的植物里读出来的**，不照抄这条规则。
        另外屋顶关每格的 terrain 仍是 1（普通地面）：这个约束不体现在地形值上。

      · **水路是 terrain=3，不是 2。** `PlantRowType` 里水路才是 2；
        两套枚举编号不同，混用会把水路说成旱地。
    """
    L = ["── 场景与地形 ──", ""]
    L.append("  关卡背景（enum BackgroundType）= **这一关长什么样**：")
    for value, name in sorted(background_types().items()):
        L.append(f"    {value} = {name:<24} {_BACKGROUND_LABEL.get(name, name)}")
    L.append("")
    L.append("  每格地形（enum GridSquareType）= **这一格能不能种**：")
    note = {
        "GRIDSQUARE_GRASS": "直接种",
        "GRIDSQUARE_POOL": f"常规植物要先种睡莲({LILYPAD})；香蒲({CATTAIL})是睡莲的升级，只能种在已有睡莲的格上",
        # DIRT / NONE 的说明已经写在标签里了，这里不再重复一遍。
        "GRIDSQUARE_HIGH_GROUND": f"常规植物要先种花盆({FLOWER_POT})",
    }
    for value, name in sorted(grid_square_types().items()):
        tail = note.get(name, "")
        L.append(f"    {value} = {name:<24} {_GRIDSQUARE_LABEL.get(name, name)}"
                 + (f"　{tail}" if tail else ""))
    L.append("")
    L.append("  ⚠ **两套编号不是一回事**：背景说的是「这一关」，地形说的是「这一格」。")
    L.append("    另外 `PlantRowType` 是第三套（水路 = 2，而地形里水路 = 3），别拿它当地形用。")
    L.append("")
    L.append("  ⚠ **屋顶：别照源码推，看花盆在哪。** 源码 Board.cpp:2822 要求每格先有")
    L.append(f"    花盆({FLOWER_POT})，但本环境**开局就预置好了花盆**（实测第 41 关：")
    L.append("    c0–c4 × 5 路共 25 个），所以直接种就行、不用自己种花盆；")
    L.append("    而没有花盆的那几列（实测 c5–c8）**种不下**。")
    L.append("    另外屋顶关每格的 terrain 仍是 1（普通地面）—— 这个约束")
    L.append("    **不体现在地形值上**，看 terrain 判断不出屋顶。")
    L.append("")
    L.append("  ⚠ **夜间不掉自然阳光。** 实测对照（同一套卡组、什么都不种）：")
    L.append("    白天关 60 步后阳光 50 → 200；夜间关 60 步后仍是 50。")
    L.append("    所以夜间关阳光只能来自向日葵，开局经济节奏和白天完全不是一回事。")
    L.append("")
    L.append("  **哪一关是什么场景**（Board::PickBackground，每 10 关一个区；")
    L.append("    第 35 关是 ScaryPotter 挑战，override 成夜间草地）：")
    L.append("    1–10 白天草地 ｜ 11–20 夜间草地（有墓碑格，种不了）｜ 21–30 白天泳池")
    L.append("    ｜ 31–40 夜间泳池有雾（35 除外）｜ 41–49 屋顶 ｜ 50 Boss。")
    L.append("")
    L.append("  ⚠ **升级植物种不下底座以外的地方。** 从 Plant::IsUpgrade 解析出的升级植物：")
    L.append(f"    {', '.join(str(i) for i in sorted(upgrade_plants()))}（含香蒲 {CATTAIL}）。")
    L.append("    它们必须种在各自的底座上（香蒲 → 睡莲上；双子向日葵 → 向日葵上），")
    L.append("    且进卡组前必须在 profile 里声明 owned_upgrade_plants，否则 reset 直接被拒。")
    L.append("")
    L.append("  在工具输出里看这些：`frame` 的「场景：…」那一行；棋盘上 `~` = 水路。")
    return L


def _fmt_time(rate: int) -> list[str]:
    L = ["── 时间基准 ──", ""]
    L.append(f"  1 秒 = {rate} tick（游戏逻辑帧率 mSyncRefreshRate = {rate}，"
             f"来源 src/SexyAppFramework/SexyAppBase.cpp）")
    L.append(f"  tick → 秒：秒 = tick / {rate}")
    L.append(f"  秒 → tick：tick = 秒 × {rate}")
    L.append("")
    L.append("  ⚠ **不要假设 60 tick/s。** 这个模拟器是 100。实测模型的推理轨迹里")
    L.append("    出现过 \"15s = 900 ticks\"（按 60 算）—— 错在这里，不在游戏常识。")
    L.append("")
    L.append("  三处互相独立的印证（都不是推论，是源码里的数）：")
    L.append(f"    ① mSyncRefreshRate = {rate}")
    L.append("    ② 僵尸每 4 tick 咬一口、每口 4 点伤害 → 1 点/tick = 100 点/秒，")
    L.append("       与\"僵尸啃食 100 dps\"吻合")
    L.append("    ③ 土豆雷倒计时 1500 tick → 15 秒，与\"土豆雷 15 秒引爆\"吻合")
    return L


def _fmt_combat(consts: dict[str, list[tuple[str, str, str]]]) -> list[str]:
    L = ["── 战斗常量（源码 constexpr）──", ""]
    for fname, rows in consts.items():
        if not rows:
            continue
        L.append(f"  [{fname}]")
        for type_, name, expr in rows:
            L.append(f"    {name} = {expr}    ({type_})")
        L.append("")
    L.append("  啃食节奏（Zombie.cpp）：僵尸每 TICKS_BETWEEN_EATS=4 个 tick 咬一口，")
    L.append("  每口 DAMAGE_PER_EAT=4 点伤害 → 合计 1 点/tick = 100 点/秒。")
    L.append("  所以 300 血的豌豆射手被一只僵尸啃完要 3 秒（300 tick）。")
    return L


def _fmt_plants(defs: list[dict], specials: dict[str, dict[str, int]], rate: int) -> list[str]:
    L = ["── 植物表（gPlantDefs，src/Lawn/Plant.cpp）──", ""]
    L.append("  type  cost  冷却  攻击间隔  血量  中文名 / 英文名")
    L.append("  （type = SeedType 数值；cost = 阳光花费；冷却 = 卡片冷却 tick；")
    L.append("    攻击间隔 = 射击间隔 tick，0 = 不攻击；血量 = mPlantHealth）")
    L.append("")
    for d in defs:
        sp = specials.get(d["seed"], {})
        hp = sp.get("mPlantHealth", "?")
        cn = PLANT_NAME.get(d["type"]) or "(无中文名)"
        L.append(f"  {d['type']:>4}  {d['cost']:>4}  {d['recharge']:>5}  "
                 f"{d['launch']:>8}  {hp:>4}  {cn} / {d['name']}")
    L.append("")
    L.append("  type 就是动作 `plant:<n>:<row>:<col>` 里的 n（卡槽序号）**在默认卡组下**")
    L.append("  的取值：默认 7 关卡组是 0..5 这六张，槽位 i 装的正好是 type i。")
    L.append("  换成自定义卡组后 n 是「deck 里的第几个」，不再是植物种类 —— 见 vocabulary。")
    L.append("  （另有一个 mPacketIndex 卡面序号，是原始游戏的卡牌图编号，与动作参数无关，")
    L.append("    这里不列，免得和上面的 type 混淆。）")
    L.append("")
    L.append(f"── 植物特殊倒计时（Plant::PlantInitialize，单位 tick，1 s = {rate} tick）──")
    L.append("")
    named = [
        ("mStateCountdown", "状态倒计时"),
        ("mDoSpecialCountdown", "特殊动作倒计时"),
    ]
    for seed, sp in sorted(specials.items()):
        for var, label in named:
            val = sp.get(var)
            if val is None:
                continue
            cn = PLANT_NAME.get(_seed_index(seed), seed)
            L.append(f"  {cn:<8} {label} {var} = {val} tick = {val / rate:.1f} s")
    L.append("")
    L.append("  ⚠ 土豆雷：mStateCountdown = 1500 tick 只是**倒计时**，归零后还有一段升起")
    L.append("    动画（STATE_NOTREADY → STATE_POTATO_RISING → STATE_POTATO_ARMED）。")
    L.append("    所以「种下到能炸」的总时间 **大于** 15 秒。想知道确切总时长，")
    L.append("    用模拟器实测反推（实测约 1620 tick），别把 1500 当答案。")
    return L


_SEED_INDEX: dict[str, int] | None = None


def _seed_index(seed: str) -> int:
    """SEED_XXX 名字 → SeedType 数值（= PLANT_NAME 的键）。"""
    global _SEED_INDEX
    if _SEED_INDEX is None:
        _SEED_INDEX = {d["seed"]: d["type"] for d in plant_defs()}
    return _SEED_INDEX.get(seed, -1)


def _fmt_zombies(defs: list[dict], specials: dict[str, dict[str, int]]) -> list[str]:
    L = ["── 僵尸表（gZombieDefs，src/Lawn/Zombie.cpp）──", ""]
    L.append("  type  血量  威胁值  首次关卡  首次波  抽样权重  中文名 / 英文名")
    L.append("  （威胁值 = mZombieValue；首次关卡/首次波 = 这只僵尸最早能在哪出现；")
    L.append("    抽样权重 = mPickWeight，0 表示不靠抽样出现）")
    L.append("")
    for d in defs:
        hp = specials.get(d["type"], {}).get("mBodyHealth", "?")
        cn = ZOMBIE_NAME.get(d["enum"], "")
        L.append(f"  {d['type']:<18} {hp:>4}  {d['value']:>6}  {d['level']:>8} "
                 f"{d['wave']:>7}  {d['weight']:>8}  {cn or d['name']}")
    L.append("")
    L.append("  注：上面的血量只算**本体**。戴帽/持盾的僵尸在本体之外还有")
    L.append("  mHelmHealth（头盔）/ mShieldHealth（护盾）/ mFlyingHealth（气球）。")
    L.append("  所以「铁桶僵尸」的实际血量 = 本体 + 铁桶，不是表里那一个数。")
    L.append("  只列非零的：")
    L.append("")
    for ztype, sp in sorted(specials.items()):
        extra = {k: v for k, v in sp.items()
                 if k in ("mHelmHealth", "mShieldHealth", "mFlyingHealth") and v}
        if extra:
            parts = "  ".join(f"{k}={v}" for k, v in sorted(extra.items()))
            L.append(f"    {ztype:<22} {parts}")
    return L


def _fmt_literals(consts: dict[str, list[tuple[str, str, str]]]) -> list[str]:
    L = ["── 源码常量清单 ──", ""]
    L.append("  只列 Plant.cpp / Zombie.cpp 里 `constexpr const <类型> NAME = 值;` 的声明，")
    L.append("  原样给出表达式（不代算），排除渲染层编号 RENDER_GROUP_*。")
    L.append("")
    for fname, rows in consts.items():
        if not rows:
            continue
        L.append(f"  [{fname}]")
        for type_, name, expr in rows:
            L.append(f"    {type_:<8} {name} = {expr}")
        L.append("")
    return L


def render(sections: tuple[str, ...] = SECTIONS, level: int | None = None) -> str:
    rate = tick_rate()
    consts = numeric_constants()
    want = set(sections)
    L: list[str] = []
    if "time" in want:
        L += _fmt_time(rate)
        L.append("")
    if "combat" in want:
        L += _fmt_combat(consts)
        L.append("")
    if "plants" in want:
        L += _fmt_plants(plant_defs(), plant_specials(), rate)
        L.append("")
    if "zombies" in want:
        L += _fmt_zombies(zombie_defs(), zombie_specials())
        L.append("")
    if "level" in want:
        if level is None:
            L.append("── level 一节需要 --level N 才输出 ──")
            L.append("")
        else:
            L += level_report(level)
            L.append("")
    if "terrain" in want:
        L += _fmt_terrain()
        L.append("")
    if "literals" in want:
        L += _fmt_literals(consts)
    L.append("")
    L.append("  来源：src/Lawn/Plant.cpp、src/Lawn/Zombie.cpp、src/Lawn/Challenge.cpp、"
             "src/SexyAppFramework/SexyAppBase.cpp、src/ConstEnums.h —— 每次现解析，不是抄本。")
    return "\n".join(L).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", action="append", choices=SECTIONS, default=None,
                    help="只输出这一节，可重复。默认全部（level 一节需要 --level）。")
    ap.add_argument("--level", type=int, default=None,
                    help="关卡号 1..50。给了就输出该关的静态参数（波数、可能出现的僵尸）。")
    ap.add_argument("--list", action="store_true", help="列出可用的节名")
    args = ap.parse_args()
    if args.list:
        print("可用节名：" + "、".join(SECTIONS))
        return
    print(render(tuple(args.section) if args.section else SECTIONS, level=args.level),
          end="")


if __name__ == "__main__":
    main()
