"""A deliberately simple hand-written policy, used as a difficulty probe.

The point is not to write a good bot; it is to answer one question the win rate
cannot: is Adventure-II level 7 (30 waves, 6 base cards) winnable at all, or is
the 0/1024 result explained by the task?  If a 40-line rule bot gets much further
than the search teacher, the teacher is the problem.

    python scripts/scripted_baseline.py --resource-dir <PvZ 1.2.0.1073 dir> --seeds 30000 20000
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pvz_env import PvZEnv, TaskSpec, PlayerProfileContext  # noqa: E402
import pvz_constants  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
LEVEL = 7
PEASHOOTER, SUNFLOWER, CHERRY, WALLNUT, POTATO, SNOWPEA = range(6)
SUN_SHROOM = 9      # 阳光菇 25☀，夜间经济（夜间天不掉阳光，实测见 pvz_constants terrain 节）
LILY_PAD = 16       # 睡莲 25☀，水路底座
CATTAIL = 43        # 香蒲 225☀，睡莲的升级植物（只能种在睡莲上）
FLOWER_POT = 33     # 花盆 25☀，屋顶底座（本环境屋顶关开局已预置 c0–c4）


def deck_for_level(level: int) -> tuple[int, ...]:
    """按关卡地形排卡组。规则：**卡组必须覆盖这一关的每一格地形**。

    这是 2026-10-03 定下的任务约束（AGENTS.md「任务与卡组」节）：
    之前全关卡共用 0..5 六张牌，泳池关（21–30）两条水路没有任何植物能种，
    属于结构性输局 —— 诊断它会得出"卡组×地形不匹配"这种一次性的结论，
    而不是可教的政策缺陷。场景判定用 `pvz_constants.background_for_level`
    （从 Board::PickBackground 解析，含第 35 关 ScaryPotter 特例）。
    """
    bg = pvz_constants.background_for_level(level)
    if bg == "BACKGROUND_3_POOL":          # 21–30 白天泳池：水路两张
        return DECK + (LILY_PAD, CATTAIL)
    if bg == "BACKGROUND_4_FOG":           # 31–40 夜间泳池有雾：经济换阳光菇
        return (0, 9, 2, 3, 4, 5, LILY_PAD, CATTAIL)
    if bg == "BACKGROUND_2_NIGHT":         # 11–20（含 35）夜间草地：加阳光菇
        return DECK[:2] + (SUN_SHROOM,) + DECK[2:]
    if bg in ("BACKGROUND_5_ROOF", "BACKGROUND_6_BOSS"):   # 41–50：加花盆备用
        return DECK + (FLOWER_POT,)
    return DECK                            # 1–10 白天草地：原六卡


def profile_for_deck(deck: tuple[int, ...]) -> PlayerProfileContext:
    """卡组 -> PlayerProfileContext。seed 槽位数 = 卡组长度；
    卡组里出现的升级植物必须声明所有权，否则模拟器直接拒绝 reset
    （LawnApp.cpp:1329 校验；升级植物集合从 Plant::IsUpgrade 解析）。"""
    upgrades = tuple(sorted(set(deck) & set(pvz_constants.upgrade_plants())))
    return PlayerProfileContext(seed_slot_count=len(deck),
                                owned_upgrade_plants=upgrades)


def rows_with(plants: list[dict], kind: int) -> Counter:
    return Counter(plant["row"] for plant in plants if plant["type"] == kind)


# 决策规则的人话描述。给模型读的，所以**不给源码**、只给规则。
#
# 为什么放在这个文件里：它描述的就是下面那个 `choose()`。放在一起，
# 改代码的人抬头就能看见"还有一份描述要跟着改"。放到别处（README、
# MCP server、文档）必然会漂移，而漂移后的描述比没有描述更坏 ——
# 模型会照着一条**已经不存在**的规则去推理。
#
# 为什么用 ①②③ 而不是 1. 2. 3.：`decision_rules_text()` 会校验描述里
# 出现的每个数字都能在源码里找到。用阿拉伯数字做序号，序号本身会被
# 当成"常量"去校验，校验就变成了噪音。
DECISION_RULES = """\
── 脚本策略的决策规则（scripted policy）──

一句话：**这是一条固定顺序的 if-else 链，每个决策点只做一件事，没有全局规划。**
它不搜索、不预测、不记历史。当前状态不满足任何一条，就走 wait 60 tick。

卡组按关卡地形定（这一关是 {level}；场景判定见 constants 的 terrain 节）：
· 白天草地 1–10：0,1,2,3,4,5
· 夜间草地 11–20（含 35）：0,1,9,2,3,4,5（9 = 阳光菇 25☀，夜里天不掉阳光）
· 白天泳池 21–30：0,1,2,3,4,5,16,43（16 = 睡莲；43 = 香蒲 225☀，睡莲的升级植物）
· 夜间泳池有雾 31–40：0,9,2,3,4,5,16,43
· 屋顶/Boss 41–50：0,1,2,3,4,5,33（33 = 花盆；本环境屋顶关开局已预置 c0–c4 的花盆）

按下面的顺序判断，命中即返回：

  ① 紧急坚果墙
     某一路「最前僵尸的 x < 260」，且该路 col >= 5 处没有坚果墙
     → 在该路 col >= 5 处种坚果墙（取最靠右的合法列）。

  ② 樱桃炸弹清场
     x < 320 的僵尸 >= 4 只
     → 在僵尸最多的那一路种樱桃炸弹（取最靠左的合法列）。

  ③ 经济
     生产者 = 阳光菇(9)（夜间优先，或卡组里没有向日葵时）；否则向日葵(1)。
     生产者 < 10 株 → 在 col <= 2 种（优先最靠左的列；同列里优先生产者最少的那一路）。

  ④ 水路防御（卡组里有睡莲(16)才会触发；草地关没有这张牌）
     ④a. 某条水路上还没有睡莲/香蒲 → 在该路 col 2..5 种睡莲
         （优先最危险的水路，取最靠右的列）。
     ④b. 场上香蒲(43) < 2 株，且香蒲有合法位（必须种在睡莲上，225☀）
         → 种香蒲（优先最危险的那一路，取最靠右的列）。

  ⑤ 火力（射手 = 豌豆射手 + 寒冰射手，合计 < 14 株时）
     ⑤a. 寒冰射手 < 4 株且阳光 >= 275
         → 在 col 2..5 种寒冰射手（优先最危险的那一路）。
     ⑤b. 否则
         → 在 col 2..5 种豌豆射手（优先最危险的那一路；同一路优先最靠右的列）。

  ⑥ 早期地雷
     土豆雷 < 4 株且 wave <= 4
     → 在 col >= 6 种土豆雷（优先最危险的那一路）。

  ⑦ 以上都不满足 → wait 60 tick。

「最危险的那一路」= 该路「最前僵尸的 x」最小；该路没有僵尸时视为 9999。

三条容易误判的行为：
· 选中的落点若被引擎拒绝（例如格子已被占、那一格是墓碑），这一步**退化成 wait 60 tick**，
  策略不会换个落点重试 —— 所以某一步"没种下东西"不代表它没做决策。
· 所有阈值（260 / 320 / 4 / 10 / 14 / 275 / 6 / 2）都是写死的常数，
  不随波数、阳光余额或僵尸密度变化。
· 香蒲(43)是睡莲(16)的升级植物：空水格上**没有**香蒲的合法位，
  必须先有睡莲。legal_actions 里看不到香蒲的位置通常就是这两个原因之一
  （没睡莲垫底，或阳光不足 225——付不起的卡整个不出合法位）。
"""


def decision_rules_text() -> str:
    """把 `DECISION_RULES` 填上卡组/关卡，并**校验描述没有和代码脱钩**。

    校验方式：描述里出现的每一个数字，都必须能在本文件的源码里找到。
    为什么要这一步：这份描述是人写的（不像 `pvz_constants.py` 那样现解析），
    人写的就会过期。改 `choose()` 里的阈值却忘了改描述，模型会照着一份
    错误的规则推理，而且**它没有任何办法察觉**。
    所以宁可在这里硬失败，也不放一份可能过期的描述出去。
    """
    text = DECISION_RULES.format(level=LEVEL)
    source = Path(__file__).read_text(encoding="utf-8")
    # 只校验 DECISION_RULES 之后的部分，避免"数字出现在描述自己的定义里"这种自证。
    body = text
    missing = sorted({int(n) for n in re.findall(r"\d+", body)
                      if not re.search(rf"(?<!\d){n}(?!\d)", source)})
    if missing:
        raise RuntimeError(
            f"决策规则描述里的这些数字在 {Path(__file__).name} 里找不到：{missing}。"
            f"要么描述过期了（改 choose() 时漏改这里），要么阈值被挪去了别处。"
            f"先核对再放行 —— 一份和代码对不上的规则描述，比没有更坏。"
        )
    return body


def choose(observation: dict) -> dict:
    legal = observation["legal_actions"]["plants"]
    if not legal:
        return {"type": "wait", "ticks": 60}
    packets = observation["packets"]
    by_type: dict[int, list[dict]] = {}
    for placement in legal:
        kind = packets[placement["packet"]]["type"]
        by_type.setdefault(kind, []).append(placement)
    plants = observation["plants"]
    zombies = observation["zombies"]
    count = Counter(plant["type"] for plant in plants)

    def pick(kind, filter_fn, key_fn):
        options = [item for item in by_type.get(kind, []) if filter_fn(item)]
        if not options:
            return None
        return {"type": "plant", **max(options, key=key_fn)}

    def row_threat(row: int) -> float:
        xs = [z["x"] for z in zombies if z["row"] == row]
        return min(xs) if xs else 9999.0

    # 1. emergency wall-nut in front of anything close to the house
    for row in range(6):
        if row_threat(row) < 260 and not any(
                p["type"] == WALLNUT and p["row"] == row and p["col"] >= 5 for p in plants):
            action = pick(WALLNUT, lambda item: item["row"] == row and item["col"] >= 5,
                          lambda item: item["col"])
            if action:
                return action

    # 2. cherry bomb on a clustered breach
    cluster = [z for z in zombies if z["x"] < 320]
    if len(cluster) >= 4:
        row = Counter(z["row"] for z in cluster).most_common(1)[0][0]
        action = pick(CHERRY, lambda item: item["row"] == row, lambda item: -item["col"])
        if action:
            return action

    # 3. economy first: two columns of sun producers at the back
    #    夜间（天不掉阳光）优先阳光菇(9)，白天优先向日葵(1)；卡组里没有的退而求其次。
    producer = (SUN_SHROOM if by_type.get(SUN_SHROOM) else SUNFLOWER) \
        if observation.get("night") else \
        (SUNFLOWER if by_type.get(SUNFLOWER) else SUN_SHROOM)
    if producer and count[producer] < 10:
        action = pick(producer, lambda item: item["col"] <= 2,
                      lambda item: (-item["col"], rows_with(plants, producer)[item["row"]]))
        if action:
            return action

    # 4. water lanes: lily pads first, then cattails on them.
    #    只有卡组里有睡莲(16)才会触发 —— 草地关没有这张牌，到这里行为和原来完全一致。
    lily_options = by_type.get(LILY_PAD)
    if lily_options:
        water_rows = {item["row"] for item in lily_options}
        covered = rows_with(plants, LILY_PAD) + rows_with(plants, CATTAIL)
        uncovered = sorted((r for r in water_rows if covered[r] == 0), key=row_threat)
        if uncovered:
            target = uncovered[0]
            action = pick(LILY_PAD,
                          lambda item: item["row"] == target and 2 <= item["col"] <= 5,
                          lambda item: -item["col"])
            if action:
                return action
        cattail_options = by_type.get(CATTAIL)
        if cattail_options and count[CATTAIL] < 2:
            action = pick(CATTAIL, lambda item: 2 <= item["col"] <= 5,
                          lambda item: (-row_threat(item["row"]), -item["col"]))
            if action:
                return action

    # 5. offence: peashooters (with a few snow peas) in the middle columns
    shooters = count[PEASHOOTER] + count[SNOWPEA]
    if shooters < 14:
        if count[SNOWPEA] < 4 and observation["sun"] >= 275:
            action = pick(SNOWPEA, lambda item: 2 <= item["col"] <= 5,
                          lambda item: -row_threat(item["row"]))
            if action:
                return action
        action = pick(PEASHOOTER, lambda item: 2 <= item["col"] <= 5,
                      lambda item: (-row_threat(item["row"]), -item["col"]))
        if action:
            return action

    # 6. cheap early mines on the lane the first zombies walk down
    if count[POTATO] < 4 and observation["wave"] <= 4:
        action = pick(POTATO, lambda item: item["col"] >= 6, lambda item: -row_threat(item["row"]))
        if action:
            return action

    return {"type": "wait", "ticks": 60}


def run(env: PvZEnv, seed: int, level: int = LEVEL, max_actions: int = 4000,
        task: TaskSpec | None = None, deck: tuple[int, ...] | None = None) -> dict:
    deck = tuple(deck) if deck else deck_for_level(level)
    task = task or TaskSpec(level=level, seed=seed, playthrough=2,
                            profile=profile_for_deck(deck))
    observation, _ = env.reset(deck=deck, task=task)
    initial_plants = observation["plants"]
    initial_sun = observation["sun"]
    initial_off_board_zombies = sum(not zombie["on_board"] for zombie in observation["zombies"])
    if initial_off_board_zombies == 0 or observation["enemy_zombies_on_screen"]:
        raise AssertionError("preview zombies must be observed but excluded from enemy presence")
    actions = 0
    mower_triggered = 0
    while not observation["terminal"] and actions < max_actions:
        action = choose(observation)
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            # wait whenever the chosen placement was rejected
            observation, _, done, _, info = env.step({"type": "wait", "ticks": 60})
        mower_triggered += info["events"].get("mower_triggered", 0)
        actions += 1
        if done:
            break
    result = int(observation["result"])
    level_lost = bool(info.get("events", {}).get("level_lost"))
    if result == 1:
        terminal_reason = "won"
    elif result == 2 and level_lost:
        terminal_reason = "zombie_breach"
    elif result == 2:
        terminal_reason = "lost_other"
    elif observation["terminal"]:
        terminal_reason = "other_terminal"
    else:
        terminal_reason = "action_limit"
    if result == 1 and observation["enemy_zombies_on_screen"]:
        raise AssertionError("a winning terminal state must have no enemy zombies on screen")
    return {
        "seed": seed,
        "level": level,
        "won": result == 1,
        "result": result,
        "terminal_reason": terminal_reason,
        "terminal": bool(observation["terminal"]),
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "tick": observation["tick"],
        "actions": actions,
        "initial_plants": initial_plants,
        "initial_sun": initial_sun,
        "initial_off_board_zombies": initial_off_board_zombies,
        "initial_enemy_zombies_on_screen": False,
        "final_enemy_zombies_on_screen": bool(observation["enemy_zombies_on_screen"]),
        "level_lost_event": level_lost,
        "mower_triggered": mower_triggered,
        "plants": Counter(plant["type"] for plant in observation["plants"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=[30000])
    args = parser.parse_args()
    env = PvZEnv(args.resource_dir)
    for seed in args.seeds:
        result = run(env, seed, args.level)
        print(f"level {args.level} seed {seed}: won={result['won']} result={result['result']} "
              f"reason={result['terminal_reason']} level_lost={result['level_lost_event']} "
              f"mowers={result['mower_triggered']} terminal={result['terminal']} "
              f"wave {result['wave']}/{result['wave_count']} "
              f"tick={result['tick']} actions={result['actions']} "
              f"previews={result['initial_off_board_zombies']} "
              f"enemy_query={result['initial_enemy_zombies_on_screen']}"
              f"->{result['final_enemy_zombies_on_screen']} board={dict(result['plants'])}")
    env.close()


if __name__ == "__main__":
    main()
