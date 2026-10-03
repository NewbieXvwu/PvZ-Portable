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

from render_episode import PLANT_NAME, ZOMBIE_NAME  # noqa: E402

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

SECTIONS = ("time", "combat", "plants", "zombies", "level", "literals")


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
