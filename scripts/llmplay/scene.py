"""局面渲染：把引擎观测变成模型能一屏读完的文字局面（task81 §B）。

格式由主控 2026-10-25 定，顺序固定：
1. 状态行（预算/墙钟，来自 budget.Session）
2. 局面头：关卡、第几波、tick、距下一波、阳光、每行小推车、本关出怪类型
3. 空间网格：行 × 9 列，每格固定宽度，植物短代码 + 血量百分比，僵尸短代码 + 只数
4. 逐行明细：每行的植物（列/类型/血量/状态）与僵尸（类型/列号一位小数/本体+护具血量/
   减速冰冻/特殊状态），以及最前面那只多久到下一个植物列、多久到小推车
5. 卡片：费用、冷却、买不买得起、能种在哪些格
6. 事件：自上次 observe 以来发生了什么（种下/被吃/被偷/被砸/击杀/新波/小推车）
7. timeline()：每波一行的历史

这里不发明任何打法规则：植物/僵尸名字与短代码是常量表，能不能种走引擎的
legal_actions（Board::CanPlantAt），到小推车的 tick 数是 距离 / 当前速度。
"""
from __future__ import annotations

from pvz_constants import PLANT_NAME, ZOMBIE_NAME
from render_episode import (_frame, _col_of, lane_indices, LAWN_XMIN, CELL_PX, GRID_COLS)
from episode_query import _fmt_scene

CELL_WIDTH = 17          # 网格里每格固定宽度：植物 6 格 + 空格 + 僵尸 9 格 + 分隔空格
PLANT_PART = 6
ZOMBIE_PART = 9
MOWER_STATE = {0: "已用", 1: "待命", 2: "启动", 3: "行驶"}

# 短代码：3 个字母 + 血量百分比（Mel85 = 西瓜剩 85% 血）。
PLANT_CODE = {
    0: "Pea", 1: "Sun", 2: "Chr", 3: "Wal", 4: "Pto", 5: "Sno", 6: "Chm", 7: "Rep",
    8: "Puf", 9: "ShS", 10: "Fum", 11: "GrB", 12: "Hyp", 13: "Scd", 14: "IcS",
    15: "Doo", 16: "Lil", 17: "Squ", 18: "Thr", 19: "Tng", 20: "Jal", 21: "Spk",
    22: "Tor", 23: "Tal", 24: "Sea", 25: "Pln", 26: "Cac", 27: "Blo", 28: "Spl",
    29: "Sta", 30: "Pmp", 31: "Mag", 32: "Cbg", 33: "Flp", 34: "Krn", 35: "Cof",
    36: "Gar", 37: "Umb", 38: "Mar", 39: "Mel", 40: "Gat", 41: "Twn", 42: "Glm",
    43: "Cat", 44: "Wml", 45: "Gld", 46: "Spr", 47: "Cob", 48: "Imi",
}

# 僵尸短代码：3 个字母 + 只数（Gar1 = 1 只巨人，Imp2 = 2 只小鬼）。
# 顺序照 ZOMBIE_NAME 的类型编号，未列出的类型退化成 z<编号>。
ZOMBIE_CODE = {
    0: "Zom", 1: "Flg", 2: "Con", 3: "Pol", 4: "Buc", 5: "Nws", 6: "Scr", 7: "Ftb",
    8: "Dan", 9: "Bkp", 10: "Duk", 11: "Div", 12: "Zam", 13: "Sld", 14: "Dol",
    15: "Cln", 16: "Bal", 17: "Min", 18: "Pog", 19: "Yet", 20: "Bun", 21: "Lad",
    22: "Cat", 23: "Gar", 24: "Imp", 25: "Bss", 26: "Pea", 27: "Wal", 28: "Jal",
    29: "Gat", 30: "Squ", 31: "Tln", 32: "Red",
}

# PlantState 全表，顺序照 src/Lawn/Plant.h:49 的 enum（值就是下标）。
PLANT_STATE = {
    0: "待装填", 1: "就绪", 2: "特殊动作", 3: "窝瓜看目标", 4: "窝瓜起跳前",
    5: "窝瓜上升", 6: "窝瓜下落", 7: "窝瓜落地", 8: "墓碑吞噬者落地",
    9: "墓碑吞噬者啃食", 10: "大嘴花咬", 11: "大嘴花咬中", 12: "大嘴花咬空",
    13: "大嘴花消化", 14: "大嘴花吞咽", 15: "土豆雷出土", 16: "土豆雷已就绪",
    17: "土豆雷被压扁", 18: "地刺攻击", 19: "地刺攻击2", 20: "胆小菇缩下",
    21: "胆小菇躲藏", 22: "胆小菇抬起", 23: "阳光菇幼体", 24: "阳光菇生长",
    25: "阳光菇成年", 26: "磁力菇吸铁", 27: "磁力菇充能", 28: "保龄球上升",
    29: "保龄球下落", 30: "仙人掌低", 31: "仙人掌抬起", 32: "仙人掌高",
    33: "仙人掌落下", 34: "缠绕水草抓取", 35: "玉米炮装填", 36: "玉米炮装弹",
    37: "玉米炮就绪", 38: "玉米炮发射", 39: "玉米投手黄油", 40: "保护伞触发",
    41: "保护伞反弹", 42: "模仿者变形", 43: "禅境浇水", 44: "禅境有需求",
    45: "禅境满意", 46: "金盏花结束", 47: "花盆（无敌）", 48: "睡莲（无敌）",
}

ZOMBIE_LADDER = 21
ZOMBIE_BUNGEE = 20
ZOMBIE_CATAPULT = 22
ZOMBIE_GARGANTUAR = (23, 32)


def _pct(health, max_health) -> int:
    try:
        health, max_health = float(health), float(max_health)
    except (TypeError, ValueError):
        return 0
    if max_health <= 0:
        return 0
    return max(0, min(100, int(round(100.0 * health / max_health))))


def _plant_code(seed_type) -> str:
    return PLANT_CODE.get(int(seed_type), "t%02d" % int(seed_type))


def _plant_name(seed_type) -> str:
    return PLANT_NAME.get(int(seed_type), "type%s" % seed_type)


def _zombie_name(zombie_type) -> str:
    return ZOMBIE_NAME.get(int(zombie_type), "type%s" % zombie_type)


def _zombie_code(zombie_type) -> str:
    return ZOMBIE_CODE.get(int(zombie_type), "z%02d" % int(zombie_type))


def _zombie_label(zombie_type) -> str:
    """图例里的短名：去掉中文名末尾的"僵尸"，省空间（巨人僵尸 → 巨人）。"""
    name = _zombie_name(zombie_type)
    return name[:-2] if name.endswith("僵尸") else name


def _plants(obs) -> list:
    return [p for p in (obs.get("plants") or []) if not p.get("squished")]


def _zombies(obs) -> list:
    return list(obs.get("zombies") or [])


def _lane_list(obs) -> list:
    return lane_indices(_frame(obs))


def _fog_note(obs) -> str:
    if not obs.get("fog"):
        return ""
    return ("雾中遮住的僵尸看不见，僵尸数与最前位置仅计可见者；未见不代表没有。"
            "雾中植物的位置和种类保留，血量与状态看不见；合法种植和铲除仍由引擎判定。")


def _masked_plant(plant) -> bool:
    return plant.get("health") == -1 or plant.get("state") == -1


# ---------------------------------------------------------------- 2. 局面头

def header(game) -> str:
    obs = game.obs or {}
    frame = _frame(obs)
    wave, wave_count = obs.get("wave"), obs.get("wave_count")
    timer = obs.get("wave_timer")
    bits = ["L%s" % obs.get("level"), "第 %s/%s 波" % (wave, wave_count),
            "tick %s" % obs.get("tick")]
    if timer is not None:
        bits.append("距下一波约 %s tick" % int(timer))
    bits.append("阳光 %s" % obs.get("sun"))
    mowers = {int(d["row"]): int(d["state"]) for d in (obs.get("defenses") or [])}
    lanes = _lane_list(obs)
    if mowers:
        bits.append("小推车 " + " ".join(
            "r%d:%s" % (r, MOWER_STATE.get(mowers.get(r), "无")) for r in lanes))
    else:
        bits.append("本关无小推车")
    roster = (obs.get("loadout_context") or {}).get("zombie_roster") or []
    if roster:
        names = [_zombie_name(z) for z in roster]
        bits.append("本关出怪 " + "、".join(names))
    result = obs.get("result")
    if obs.get("terminal"):
        bits.append("**终局：%s**" % ("通关" if result == 1 else "失败"))
    else:
        bits.append("进行中")
    return " ".join(bits)


# ---------------------------------------------------------------- 3. 空间网格

def _zombie_tags(zombies) -> str:
    """格内僵尸：类型短代码 + 只数，最靠左（威胁最大）的排前面。

    放不下就丢掉末尾的类型，改写成 "+n"（n = 没显示出来的僵尸只数）。
    """
    counts, order = {}, []
    for zombie in sorted(zombies, key=lambda z: float(z.get("x") or 0)):
        code = _zombie_code(zombie["type"])
        if code not in counts:
            counts[code] = 0
            order.append(code)
        counts[code] += 1
    tokens = ["%s%d" % (code, counts[code]) for code in order]
    text = " ".join(tokens)
    hidden = 0
    while tokens and len(text) > ZOMBIE_PART:
        hidden += counts[order[len(tokens) - 1]]
        tokens.pop()
        text = " ".join(tokens)
        text = (text + " +%d" % hidden) if text else "+%d" % hidden
    return text[:ZOMBIE_PART]


def _cell(plant, zombies, pool: bool = False) -> str:
    if plant is None:
        left = ("~~" if pool else "..").ljust(PLANT_PART)
    elif _masked_plant(plant):
        left = ("%s??" % _plant_code(plant["type"])).ljust(PLANT_PART)
    else:
        left = ("%-3s%-2d" % (_plant_code(plant["type"]), _pct(plant.get("health"),
                                                             plant.get("max_health"))))
        left = left[:PLANT_PART].ljust(PLANT_PART)
    tags = (" " + _zombie_tags(zombies)) if zombies else ""
    # 末尾留一格空格，免得"Con1 Zom1"和右边格子的 ".." 粘在一起。
    return (left + tags.ljust(ZOMBIE_PART + 1) + " ")[:CELL_WIDTH]


def grid(game) -> str:
    obs = game.obs or {}
    lanes = _lane_list(obs)
    grid_rows = obs.get("grid") or []
    plants = _plants(obs)
    zombies = [z for z in _zombies(obs) if z.get("on_board")]
    by_cell = {}
    for plant in plants:
        by_cell.setdefault((int(plant["row"]), int(plant["col"])), []).append(plant)
    zcell = {}
    for zombie in zombies:
        cell = (int(zombie["row"]), _col_of(float(zombie["x"])))
        if 0 <= cell[1] < GRID_COLS:
            zcell.setdefault(cell, []).append(zombie)
    lines = ["        " + "".join(("c%d" % c).ljust(CELL_WIDTH) for c in range(GRID_COLS))]
    for row in lanes:
        cells = []
        for col in range(GRID_COLS):
            here = by_cell.get((row, col))
            plant = here[0] if here else None
            if plant is None and len(here or []) > 1:
                plant = here[0]
            pool = False
            if row < len(grid_rows) and col < len(grid_rows[row]):
                pool = int(grid_rows[row][col]) == 3
            cells.append(_cell(plant, zcell.get((row, col)), pool=pool))
        lines.append("  r%-2d   " % row + "".join(cells))
    outside = [z for z in _zombies(obs) if not z.get("on_board")]
    if outside:
        per_row = {}
        for zombie in outside:
            per_row.setdefault(int(zombie["row"]), []).append(zombie)
        parts = []
        for row in sorted(per_row):
            names = {}
            for zombie in per_row[row]:
                names[_zombie_name(zombie["type"])] = names.get(_zombie_name(zombie["type"]), 0) + 1
            parts.append("r%d %s" % (row, "、".join("%sx%d" % (k, v) for k, v in names.items())))
        lines.append("  场外来袭（还没走到棋盘上）：" + "；".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------- 4. 逐行明细

# 只有**投手**（有 mLaunchRate 的）用 STATE_READY/STATE_NOTREADY 表示"装填好了/还没"。
# 射手的状态位基本一直停在 STATE_NOTREADY（换弹由 shooting_counter 管），
# 花盆/睡莲/坚果更是永远停在 0 —— 把它们翻译成"待装填"会误导。
RELOAD_TYPES = {32, 34, 39, 44, 47}


# 屏幕上看得到倒计时的植物（task144）。数值来自引擎源码，见包里的 PLANTS.md；
# 这里只决定"哪个字段翻成哪句话"。引擎字段：mDoSpecialCountdown = 爆炸/唤醒倒计时，
# mStateCountdown = 该状态剩余 tick，mWakeUpCounter = 睡眠剩余 tick。
SPECIAL_LABEL = {2: "引爆倒计时", 20: "引爆倒计时", 15: "引爆倒计时", 14: "冰冻倒计时",
                 35: "唤醒倒计时"}
POTATO_MINE, CHOMPER, SQUASH, COB_CANNON, SUN_SHROOM = 4, 6, 17, 47, 9


def _plant_timers(obs) -> dict:
    """(行, 列) -> 完整观测 v2 段里的 mDoSpecialCountdown。

    完整观测的 plants 字典没有这个字段（引擎把它放在 v2 段里），所以按格子补回来。
    雾里被遮住的植物 v2 给 -1，这里原样保留"未知"。
    """
    out = {}
    for entry in ((obs or {}).get("v2") or {}).get("plants") or []:
        if len(entry) >= 6:
            out[(int(entry[0]), int(entry[1]))] = int(entry[5])
    return out


def _plant_state(plant, special=None) -> str:
    if _masked_plant(plant):
        return "状态未知（雾中看不见）"
    if plant.get("asleep"):
        left = int(plant.get("wake_up_counter") or 0)
        return "睡着（约 %d tick 后醒）" % left if left > 0 else "睡着"
    if int(plant.get("bungee_state") or 0):
        return "被蹦极吊起"
    kind = int(plant.get("type") or -1)
    state = int(plant.get("state") or 0)
    left = int(plant.get("state_countdown") or 0)
    if special is None:
        special = -1
    special = int(special)
    # 爆炸类与咖啡豆：动画一开始就在播，倒计时归零才生效。
    if special > 0 and kind in SPECIAL_LABEL:
        return "%s 剩 %d tick" % (SPECIAL_LABEL[kind], special)
    if left < 0 or state < 0:
        return "状态未知（雾里）"
    if kind == POTATO_MINE:
        if state == 0:
            return "武装中 剩 %d tick" % left
        if state == 15:
            return "出土中"
        if state == 16:
            return "已武装"
    elif kind == CHOMPER:
        if state == 10:
            return "咬 剩 %d tick" % left
        if state == 13:
            return "咀嚼 剩 %d tick" % left
        if state == 14:
            return "吞咽"
    elif kind == SQUASH:
        if state == 3:
            return "看目标 剩 %d tick" % left
        if state == 4:
            return "起跳前 剩 %d tick" % left
    elif kind == COB_CANNON and state == 35:
        return "装填中 剩 %d tick" % left
    elif kind == SUN_SHROOM and state == 23:
        return "幼体 剩 %d tick 长大" % left
    if state == 0 and kind not in RELOAD_TYPES:
        return "正常"
    return PLANT_STATE.get(state, "状态%d" % state)


def _zombie_line(zombie) -> str:
    kind = int(zombie["type"])
    bits = [_zombie_name(kind), "c%.1f" % (_col_of(float(zombie["x"])) + 0.5)]
    body = "%d/%d" % (int(zombie.get("body_health") or 0), int(zombie.get("body_max_health") or 0))
    text = "本体%s" % body
    helm_max = int(zombie.get("helm_max_health") or 0)
    if helm_max:
        text += " 护具%d/%d" % (int(zombie.get("helm_health") or 0), helm_max)
    shield_max = int(zombie.get("shield_max_health") or 0)
    if shield_max:
        text += " 盾%d/%d" % (int(zombie.get("shield_health") or 0), shield_max)
    bits.append(text)
    extra = []
    if kind == ZOMBIE_LADDER:
        extra.append("梯子已搭" if not zombie.get("has_object") else "梯子未搭")
    if kind == ZOMBIE_BUNGEE:
        extra.append("目标 r%sc%s" % (zombie.get("target_row"), zombie.get("target_col")))
    if kind == ZOMBIE_CATAPULT:
        extra.append("弹药 %s" % zombie.get("summon_counter"))
    if kind in ZOMBIE_GARGANTUAR:
        extra.append("已扔小鬼" if not zombie.get("has_object") else "小鬼还在手上")
    if int(zombie.get("chilled") or 0):
        extra.append("减速")
    if int(zombie.get("ice_trap") or 0):
        extra.append("冰冻")
    if int(zombie.get("buttered") or 0):
        extra.append("黄油定住")
    if zombie.get("is_eating"):
        extra.append("啃食中")
    if not zombie.get("has_head"):
        extra.append("无头")
    if extra:
        bits.append(" ".join(extra))
    return " ".join(bits)


def _eta(zombie, plants) -> str:
    speed = float(zombie.get("velocity_x") or 0)
    x = float(zombie.get("x") or 0)
    row = int(zombie["row"])
    col = _col_of(x)
    ahead = sorted({int(p["col"]) for p in plants
                    if int(p["row"]) == row and int(p["col"]) < col}, reverse=True)
    pieces = []
    if speed < 0.02:
        return "停住" + ("（正在啃食）" if zombie.get("is_eating") else "")
    if ahead:
        target = ahead[0]
        ticks = max(0, int((x - (LAWN_XMIN + target * CELL_PX + CELL_PX * 0.5)) / speed))
        pieces.append("约 %d tick 到第 %d 列" % (ticks, target))
    else:
        pieces.append("这一路前面没有植物")
    ticks = max(0, int((x - LAWN_XMIN) / speed))
    pieces.append("约 %d tick 到小推车" % ticks)
    return "，".join(pieces)


def rows(game) -> str:
    obs = game.obs or {}
    lanes = _lane_list(obs)
    plants = _plants(obs)
    zombies = _zombies(obs)
    timers = _plant_timers(obs)
    lines = []
    for row in lanes:
        row_plants = sorted([p for p in plants if int(p["row"]) == row], key=lambda p: int(p["col"]))
        row_zombies = [z for z in zombies if int(z["row"]) == row]
        # 本体血量 ≤ 0 的是正在倒下的尸体，不参与"最前"与到达时间。
        on_board = [z for z in row_zombies
                    if z.get("on_board") and int(z.get("body_health") or 0) > 0]
        if row_plants:
            text = " · ".join(
                "c%d %s %s %s" % (int(p["col"]), _plant_name(p["type"]),
                                     "血量看不见" if _masked_plant(p) else "%d/%d" % (
                                         int(p.get("health") or 0), int(p.get("max_health") or 0)),
                                     _plant_state(p, timers.get((int(p["row"]), int(p["col"])))))
                for p in row_plants)
        else:
            text = "（没有植物）"
        lines.append("  r%d 植物: %s" % (row, text))
        if on_board:
            lines.append("     僵尸: " + " | ".join(_zombie_line(z) for z in sorted(
                on_board, key=lambda z: float(z["x"]))))
            front = min(on_board, key=lambda z: float(z["x"]))
            lines.append("     最前: %s %s" % (_zombie_name(front["type"]), _eta(front, plants)))
        else:
            lines.append("     僵尸: " + ("未见（雾中看不见）" if obs.get("fog") else "无"))
    return "\n".join(lines)


# ---------------------------------------------------------------- 5. 卡片

def _packet_cells(obs, packet_index) -> list:
    cells = []
    for action in (obs.get("legal_actions") or {}).get("plants") or []:
        if int(action["packet"]) == int(packet_index):
            cells.append((int(action["row"]), int(action["col"])))
    return sorted(set(cells))


def cards(game) -> str:
    obs = game.obs or {}
    from . import cards as card_names
    sun = int(obs.get("sun") or 0)
    conveyor = obs.get("conveyor_start") == "original"
    lines = []
    if conveyor:
        lines.append("  （传送带：左→右排列，用掉一张后后面的卡整体前移，槽位号会变；"
                     "plant(card=\"名字\") 按名字种，不受影响）")
    for packet in obs.get("packets") or []:
        index = int(packet["index"])
        seed = int(packet["type"]) if packet.get("type") is not None else -1
        if seed < 0:
            lines.append("  [%d] （空槽）" % index)
            continue
        cost = int(packet.get("cost") or 0)
        cooldown = int(packet.get("cooldown") or 0)
        active = bool(packet.get("active"))
        ready = active and cooldown == 0
        cells = _packet_cells(obs, index)
        # 传送带关的卡不花阳光（Board::PlantSeed 对 HasConveyorBeltSeedBank 直接放行），
        # 卡片上的价钱只是这株植物的市价，说"阳光不够"会把模型劝退。
        if conveyor:
            afford = "免费（传送带）"
        else:
            afford = "买得起" if sun >= cost else "阳光不够（差 %d）" % (cost - sun)
        if cells:
            sample = "、".join("r%dc%d" % cell for cell in cells[:4])
            where = "可种 %d 格（例 %s）" % (len(cells), sample)
        else:
            where = "此刻没有可种的格"
        if ready:
            state = "就绪"
        elif cooldown > 0:
            state = "冷却剩 %d tick" % cooldown
        else:
            state = "未激活"
        lines.append("  [%d] %s(%s) %d☀ %s %s %s"
                     % (index, _plant_name(seed), card_names.slug(seed), cost,
                        state, afford, where))
    return "\n".join(lines) if lines else "  （没有卡槽）"


# ---------------------------------------------------------------- 6. 事件

EVENT_ORDER = ("wave", "plant", "lost", "kill", "mower", "sun")


def events(game) -> str:
    log = getattr(game, "events", None) or []
    if not log:
        return "  （自上次 observe 以来没有值得记的事件）"
    planted, lost, kills, waves, mowers = [], [], {}, [], []
    for item in log:
        kind = item.get("kind")
        if kind == "plant":
            planted.append("r%dc%d %s" % (item["row"], item["col"], _plant_name(item["plant"])))
        elif kind == "lost":
            lost.append("r%dc%d %s（%s）" % (item["row"], item["col"],
                                            _plant_name(item["plant"]), item.get("reason", "消失")))
        elif kind == "kill":
            name = _zombie_name(item["zombie"])
            kills[name] = kills.get(name, 0) + 1
        elif kind == "wave":
            waves.append(str(item["wave"]))
        elif kind == "mower":
            mowers.append("r%d" % item["row"])
    lines = []
    if planted:
        lines.append("  种下: " + "、".join(planted))
    if lost:
        lines.append("  失去: " + "、".join(lost))
    if kills:
        lines.append("  击杀: " + "、".join("%sx%d" % (k, v) for k, v in kills.items()))
    if waves:
        lines.append("  新波: 第 %s 波开始" % "、".join(waves))
    if mowers:
        lines.append("  小推车触发: " + "、".join(mowers))
    return "\n".join(lines)


# ---------------------------------------------------------------- 组装

def _code_table(per_line: int = 7) -> str:
    entries = ["%s=%s" % (ZOMBIE_CODE[z], _zombie_label(z)) for z in sorted(ZOMBIE_CODE)]
    rows = ["  " + " ".join(entries[i:i + per_line]) for i in range(0, len(entries), per_line)]
    return "\n".join(rows)


def legend() -> str:
    return ("图例：网格每格 = 植物短代码+血量百分比（Mel85 = 西瓜剩 85% 血）与僵尸短代码+只数。\n"
            "Gar1 = 1 只巨人，Gar1 Imp2 = 1 只巨人 + 2 只小鬼；格内按从左到右的威胁顺序，\n"
            "放不下时截断并标 +n（n = 还有几只没显示）。\n"
            ".. = 空地，~~ = 水路；r0–r5 = 行，c0–c8 = 列（c0 最左，僵尸从右往左走）。\n"
            "血量都是当前/上限，位置列号保留一位小数。\n"
            "僵尸短代码：\n" + _code_table())


def render(game, level: str = "full") -> str:
    from . import budget
    session = budget.session()
    first = session.claim_legend()
    lines = [session.status_line(game.budget_key), header(game), ""]
    if _fog_note(game.obs or {}):
        lines += [_fog_note(game.obs), "网格植物 ?? = 血量看不见；空格不代表雾中没有僵尸。", ""]
    if first:
        lines += [legend(), ""]
    lines += ["空间网格（植物 + 该格僵尸）：", grid(game), "",
              "逐行明细：", rows(game), "",
              "卡片：", cards(game), "",
              "事件（自上次 observe 以来）：", events(game), "",
              "逐波历史：", timeline(game)]
    return "\n".join(lines)


def legal(game) -> dict:
    obs = game.obs or {}
    actions = (obs.get("legal_actions") or {}).get("plants") or []
    packets = {int(p["index"]): int(p["type"]) for p in (obs.get("packets") or [])}
    grouped = {}
    for action in actions:
        grouped.setdefault(int(action["packet"]), []).append([int(action["row"]), int(action["col"])])
    plants = [{"packet": packet, "type": _plant_name(packets.get(packet, -1)),
               "cells": sorted(cells)}
              for packet, cells in sorted(grouped.items())]
    shovels = sorted({(int(p["row"]), int(p["col"])) for p in _plants(obs)})
    out = {"plants": plants, "shovels": [list(cell) for cell in shovels]}
    if _fog_note(obs):
        out["visibility"] = _fog_note(obs)
    return out


def summary(game) -> dict:
    obs = game.obs or {}
    defenses = obs.get("defenses") or []
    lanes = []
    for row in _lane_list(obs):
        row_zombies = [z for z in _zombies(obs) if int(z["row"]) == row and z.get("on_board")]
        front = min(row_zombies, key=lambda z: float(z["x"])) if row_zombies else None
        lanes.append(dict(row=row,
                          plants=sum(1 for p in _plants(obs) if int(p["row"]) == row),
                          zombies=len(row_zombies),
                          front_col=None if front is None else round(_col_of(float(front["x"])) + 0.5, 1),
                          front_type=None if front is None else _zombie_name(front["type"])))
    result = obs.get("result")
    out = dict(tick=obs.get("tick"), level=obs.get("level"), wave=obs.get("wave"),
                wave_count=obs.get("wave_count"), max_wave=game.max_wave,
                sun=obs.get("sun"),
                mowers_left=sum(1 for d in defenses if int(d["state"]) == 1),
                mowers_total=len(defenses),
                plants=len(_plants(obs)),
                zombies=sum(1 for z in _zombies(obs) if z.get("on_board")),
                terminal=bool(obs.get("terminal")), won=result == 1, result=result,
                lanes=lanes)
    if _fog_note(obs):
        out["visibility"] = _fog_note(obs)
    return out


def timeline(game) -> str:
    history = getattr(game, "wave_history", None) or []
    if not history:
        return "  （还没有跨过任何一波）"
    lines = ["  波  tick   阳光  植物  击杀  失去  小推车"]
    for row in history:
        lines.append("  %-3s %-6s %-5s %-5s %-5s %-5s %s"
                     % (row.get("wave"), row.get("tick"), row.get("sun"), row.get("plants"),
                        row.get("kills"), row.get("lost"), row.get("mowers")))
    return "\n".join(lines)
