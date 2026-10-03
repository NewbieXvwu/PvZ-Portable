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

from pvz_env import PvZEnv, TaskSpec, training_task  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
LEVEL = 7
PEASHOOTER, SUNFLOWER, CHERRY, WALLNUT, POTATO, SNOWPEA = range(6)


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

卡组固定为 {deck}，关卡 {level}。按下面的顺序判断，命中即返回：

  ① 紧急坚果墙
     某一路「最前僵尸的 x < 260」，且该路 col >= 5 处没有坚果墙
     → 在该路 col >= 5 处种坚果墙（取最靠右的合法列）。

  ② 樱桃炸弹清场
     x < 320 的僵尸 >= 4 只
     → 在僵尸最多的那一路种樱桃炸弹（取最靠左的合法列）。

  ③ 经济
     向日葵 < 10 株
     → 在 col <= 2 种向日葵（优先最靠左的列；同列里优先向日葵最少的那一路）。

  ④ 火力（射手 = 豌豆射手 + 寒冰射手，合计 < 14 株时）
     ④a. 寒冰射手 < 4 株且阳光 >= 275
         → 在 col 2..5 种寒冰射手（优先最危险的那一路）。
     ④b. 否则
         → 在 col 2..5 种豌豆射手（优先最危险的那一路；同一路优先最靠右的列）。

  ⑤ 早期地雷
     土豆雷 < 4 株且 wave <= 4
     → 在 col >= 6 种土豆雷（优先最危险的那一路）。

  ⑥ 以上都不满足 → wait 60 tick。

「最危险的那一路」= 该路「最前僵尸的 x」最小；该路没有僵尸时视为 9999。

两条容易误判的行为：
· 选中的落点若被引擎拒绝（例如格子已被占），这一步**退化成 wait 60 tick**，
  策略不会换个落点重试 —— 所以某一步"没种下东西"不代表它没做决策。
· 所有阈值（260 / 320 / 4 / 10 / 14 / 275 / 6）都是写死的常数，
  不随波数、阳光余额或僵尸密度变化。
"""


def decision_rules_text() -> str:
    """把 `DECISION_RULES` 填上卡组/关卡，并**校验描述没有和代码脱钩**。

    校验方式：描述里出现的每一个数字，都必须能在本文件的源码里找到。
    为什么要这一步：这份描述是人写的（不像 `pvz_constants.py` 那样现解析），
    人写的就会过期。改 `choose()` 里的阈值却忘了改描述，模型会照着一份
    错误的规则推理，而且**它没有任何办法察觉**。
    所以宁可在这里硬失败，也不放一份可能过期的描述出去。
    """
    text = DECISION_RULES.format(deck=", ".join(str(c) for c in DECK), level=LEVEL)
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

    # 3. economy first: two columns of sunflowers at the back
    if count[SUNFLOWER] < 10:
        action = pick(SUNFLOWER, lambda item: item["col"] <= 2,
                      lambda item: (-item["col"], rows_with(plants, SUNFLOWER)[item["row"]]))
        if action:
            return action

    # 4. offence: peashooters (with a few snow peas) in the middle columns
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

    # 5. cheap early mines on the lane the first zombies walk down
    if count[POTATO] < 4 and observation["wave"] <= 4:
        action = pick(POTATO, lambda item: item["col"] >= 6, lambda item: -row_threat(item["row"]))
        if action:
            return action

    return {"type": "wait", "ticks": 60}


def run(env: PvZEnv, seed: int, level: int = LEVEL, max_actions: int = 4000,
        task: TaskSpec | None = None) -> dict:
    observation, _ = env.reset(deck=DECK, task=task or training_task(seed, level))
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
