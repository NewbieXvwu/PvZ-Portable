"""A deliberately simple hand-written policy, used as a difficulty probe.

The point is not to write a good bot; it is to answer one question the win rate
cannot: is Adventure-II level 7 (30 waves, 6 base cards) winnable at all, or is
the 0/1024 result explained by the task?  If a 40-line rule bot gets much further
than the search teacher, the teacher is the problem.

    python scripts/scripted_baseline.py --resource-dir <PvZ 1.2.0.1073 dir> --seeds 30000 20000
"""

from __future__ import annotations

import argparse
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

# choose() 的全部阈值。`DECISION_RULES` 的描述从这里填数，所以描述和代码不可能对不上。
WALLNUT_X = 260         # ① 最前僵尸的 x 低于它就补坚果
CHERRY_X = 320          # ② 统计 x 低于它的僵尸
CHERRY_CLUSTER = 4      # ② 达到这么多只就扔樱桃
PRODUCER_TARGET = 10    # ③ 生产者种到这么多株为止
PRODUCER_MAX_COL = 2    # ③ 生产者只种在这一列及以左
MID_COLS = (2, 5)       # ④⑤ 睡莲 / 香蒲 / 射手的列范围
CATTAIL_TARGET = 2      # ④b
SHOOTER_TARGET = 14     # ⑤ 豌豆 + 寒冰合计
SNOWPEA_TARGET = 4      # ⑤a
SNOWPEA_SUN = 275       # ⑤a 阳光不低于它才种寒冰
MINE_TARGET = 4         # ⑥
MINE_MAX_WAVE = 4       # ⑥ 只在这一波及以前
MINE_MIN_COL = 6        # ⑥ 只种在这一列及以右
WAIT_TICKS = 60         # ⑦
NO_THREAT_X = 9999.0    # 没有僵尸的路的「最前僵尸 x」


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


def profile_for_deck(deck: tuple[int, ...], first_aid_owned: bool = False) -> PlayerProfileContext:
    """卡组 -> PlayerProfileContext。seed 槽位数 = 卡组长度；
    卡组里出现的升级植物必须声明所有权，否则模拟器直接拒绝 reset
    （LawnApp.cpp:1329 校验；升级植物集合从 Plant::IsUpgrade 解析）。

    first_aid_owned（坚果包扎术）是**任务属性**，不是卡组属性：打开后
    Board::CanPlantAt 多允许一种落点（坚果 / 高坚果血量 < 2/3 时可以在它身上
    再种一个同种坚果），合法动作由模拟器自己给出，这里只负责把所有权传进
    profile。缺省 False，旧任务的 reset 报文逐字节不变（AGENTS.md §8：
    改任务 = 换任务族，任务清单里用 "first_aid_owned": true 显式打开）。"""
    upgrades = tuple(sorted(set(deck) & set(pvz_constants.upgrade_plants())))
    return PlayerProfileContext(seed_slot_count=len(deck),
                                owned_upgrade_plants=upgrades,
                                first_aid_owned=bool(first_aid_owned))


def rows_with(plants: list[dict], kind: int) -> Counter:
    return Counter(plant["row"] for plant in plants if plant["type"] == kind)


# 决策规则的人话描述。给模型读的，所以**不给源码**、只给规则。
#
# 为什么放在这个文件里：它描述的就是下面那个 `choose()`。放在一起，
# 改代码的人抬头就能看见"还有一份描述要跟着改"。放到别处（README、
# MCP server、文档）必然会漂移，而漂移后的描述比没有描述更坏 ——
# 模型会照着一条**已经不存在**的规则去推理。
#
# 描述里的阈值都是 `{名字}` 占位，由 `decision_rules_text()` 用上面的常量填；卡组那几行由
# `deck_for_level` 现算。所以改阈值只改常量，描述自动跟着变。
# 策略行为修订号：choose() 的**行为**一变就递增。capture 的 meta 里会记录
# 它，whatif 据此判断"重放用的策略和存档当时的策略是否一致"——不一致时
# 明确警告，不静默分叉（2026-10-04 rule ① 几何修复就是第一个跨版本的例子）。
POLICY_REVISION = "v2-2026-10-04-wallnut-front"

DECISION_RULES = """\
── 脚本策略的决策规则（scripted policy）──

一句话：**这是一条固定顺序的 if-else 链，每个决策点只做一件事，没有全局规划。**
它不搜索、不预测、不记历史。当前状态不满足任何一条，就走 wait {WAIT_TICKS} tick。

卡组按关卡地形定（这一关是 {level}；场景判定见 constants 的 terrain 节）：
· 白天草地 1–10：{deck_day}
· 夜间草地 11–20（含 35）：{deck_night}（{SUN_SHROOM} = 阳光菇 25☀，夜里天不掉阳光）
· 白天泳池 21–30：{deck_pool}（{LILY_PAD} = 睡莲；{CATTAIL} = 香蒲 225☀，睡莲的升级植物）
· 夜间泳池有雾 31–40：{deck_fog}
· 屋顶/Boss 41–50：{deck_roof}（{FLOWER_POT} = 花盆；本环境屋顶关开局已预置 c0–c4 的花盆）

按下面的顺序判断，命中即返回：

  ① 紧急坚果墙
     某一路「最前僵尸的 x < {WALLNUT_X}」，且该路僵尸的**必经之路**
     （col ≤ 最前僵尸所在列）上没有坚果墙
     → 在必经之路上种坚果墙（取最靠右的合法列：挡在它走向房子的路上，
     且离房子尽可能远）。2026-10-04 几何修复：旧版种在 col >= 5，
     那在深入腹地的僵尸**身后**，拦不住任何东西。

  ② 樱桃炸弹清场
     x < {CHERRY_X} 的僵尸 >= {CHERRY_CLUSTER} 只
     → 在僵尸最多的那一路种樱桃炸弹（取最靠左的合法列）。

  ③ 经济
     生产者 = 阳光菇({SUN_SHROOM})（夜间优先，或卡组里没有向日葵时）；否则向日葵({SUNFLOWER})。
     生产者 < {PRODUCER_TARGET} 株 → 在 col <= {PRODUCER_MAX_COL} 种（优先最靠左的列；同列里优先生产者最少
     的那一路；仍并列时取合法列表里靠前的那条路，实测即行号较小的）。

  ④ 水路防御（卡组里有睡莲({LILY_PAD})才会触发；草地关没有这张牌）
     ④a. 某条水路上还没有睡莲/香蒲 → 在该路 col {mid_cols} 种睡莲
         （优先最危险的水路，取最靠左的列）。
     ④b. 场上香蒲({CATTAIL}) < {CATTAIL_TARGET} 株，且香蒲有合法位（必须种在睡莲上，225☀）
         → 种香蒲（优先最危险的那一路，取最靠左的列）。

  ⑤ 火力（射手 = 豌豆射手 + 寒冰射手，合计 < {SHOOTER_TARGET} 株时）
     ⑤a. 寒冰射手 < {SNOWPEA_TARGET} 株且阳光 >= {SNOWPEA_SUN}
         → 在 col {mid_cols} 种寒冰射手（优先最危险的那一路）。
     ⑤b. 否则
         → 在 col {mid_cols} 种豌豆射手（优先最危险的那一路；同一路优先最靠左的列）。

  ⑥ 早期地雷
     土豆雷 < {MINE_TARGET} 株且 wave <= {MINE_MAX_WAVE}
     → 在 col >= {MINE_MIN_COL} 种土豆雷（优先最危险的那一路）。

  ⑦ 以上都不满足 → wait {WAIT_TICKS} tick。

「最危险的那一路」= 该路「最前僵尸的 x」最小；该路没有僵尸时视为 {no_threat}。
**x 的单位是像素横坐标**：每列 80 像素、最左列 c0 的左缘 x=40（列 c 的中心
≈ 80*c+40）。所以 ① 的 x<{WALLNUT_X} ≈ 最前僵尸走进 c2 深处，② 的 x<{CHERRY_X} ≈ 刚进 c3。
lane 工具输出的「距割草机 xxx t」是**另一个口径**（用该僵尸自身速度换算的
时间），和 x 不是一回事，别拿 {WALLNUT_X}/{CHERRY_X} 去比它。

三条容易误判的行为：
· 选中的落点若被引擎拒绝（例如格子已被占、那一格是墓碑），这一步**退化成 wait {WAIT_TICKS} tick**，
  策略不会换个落点重试 —— 所以某一步"没种下东西"不代表它没做决策。
· 所有阈值（{all_thresholds}）都是写死的常数，
  不随波数、阳光余额或僵尸密度变化。
· 香蒲({CATTAIL})是睡莲({LILY_PAD})的升级植物：空水格上**没有**香蒲的合法位，
  必须先有睡莲。legal_actions 里看不到香蒲的位置通常就是这两个原因之一
  （没睡莲垫底，或阳光不足 225——付不起的卡整个不出合法位）。
· 环境是确定性的：同 (level, seed, policy, deck) 重抓一局，帧序列与决策
  编号**逐位一致**。重新 capture 后，旧分析里的决策序号可以直接复用，
  不需要重新核对（引擎同状态必给同 legal 列表，策略是固定 if-else 链）。
"""


def decision_rules_text() -> str:
    """把 `DECISION_RULES` 填上关卡、卡组和 choose() 的阈值常量。

    这份描述是给模型读的人话；数字一律从常量填，卡组从 `deck_for_level` 现算，
    所以改 `choose()` 的阈值不会留下一份过期的描述（以前靠"描述里的数字能在源码里搜到"
    来校验，但源码里本来就含这段描述，那个校验永远通过）。
    """
    deck = lambda level: ",".join(map(str, deck_for_level(level)))
    return DECISION_RULES.format(
        level=LEVEL, deck_day=deck(1), deck_night=deck(11), deck_pool=deck(21), deck_fog=deck(31), deck_roof=deck(41),
        SUNFLOWER=SUNFLOWER, SUN_SHROOM=SUN_SHROOM, LILY_PAD=LILY_PAD, CATTAIL=CATTAIL, FLOWER_POT=FLOWER_POT,
        WALLNUT_X=WALLNUT_X, CHERRY_X=CHERRY_X, CHERRY_CLUSTER=CHERRY_CLUSTER, PRODUCER_TARGET=PRODUCER_TARGET,
        PRODUCER_MAX_COL=PRODUCER_MAX_COL, mid_cols="{}..{}".format(*MID_COLS), CATTAIL_TARGET=CATTAIL_TARGET,
        SHOOTER_TARGET=SHOOTER_TARGET, SNOWPEA_TARGET=SNOWPEA_TARGET, SNOWPEA_SUN=SNOWPEA_SUN, MINE_TARGET=MINE_TARGET,
        MINE_MAX_WAVE=MINE_MAX_WAVE, MINE_MIN_COL=MINE_MIN_COL, WAIT_TICKS=WAIT_TICKS, no_threat=int(NO_THREAT_X),
        all_thresholds=" / ".join(map(str, (WALLNUT_X, CHERRY_X, CHERRY_CLUSTER, PRODUCER_TARGET, SHOOTER_TARGET,
                                             SNOWPEA_SUN, MINE_MIN_COL, PRODUCER_MAX_COL))))


def choose(observation: dict) -> dict:
    legal = observation["legal_actions"]["plants"]
    if not legal:
        return {"type": "wait", "ticks": WAIT_TICKS, "rule": "⑦无满足条件→wait"}
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
        # zombie["x"] 是像素横坐标：每列 80 像素，最左列左缘 x=40。
        xs = [z["x"] for z in zombies if z["row"] == row]
        return min(xs) if xs else NO_THREAT_X

    # 1. emergency wall-nut: block in front of the zombie's remaining path.
    #    2026-10-04 几何修复（S6 教师会话发现，见 dsh/pvz-teacher/README.md）：
    #    旧版在 col >= 5 种坚果，但触发条件是僵尸已深入 c0..c2（x < 260）——
    #    僵尸自高列向低列走，col >= 5 在它**身后**，这株坚果对眼前威胁毫无
    #    拦截作用（实测 L21 决策 216：僵尸在 c2，坚果种 (1,8)，lane 判定
    #    "火力被绕过"）。现在把坚果放进僵尸的**必经之路**（col ≤ 最前僵尸
    #    所在列），取最靠右的合法列 = 离房子最远、最贴近僵尸的拦截点。
    for row in range(6):
        threat = row_threat(row)
        if threat < WALLNUT_X:
            front_col = int((threat - 40) // 80)
            if not any(
                    p["type"] == WALLNUT and p["row"] == row and p["col"] <= front_col for p in plants):
                action = pick(WALLNUT, lambda item: item["row"] == row and item["col"] <= front_col,
                              lambda item: item["col"])
                if action:
                    action["rule"] = "①紧急坚果墙"
                    return action

    # 2. cherry bomb on a clustered breach
    cluster = [z for z in zombies if z["x"] < CHERRY_X]
    if len(cluster) >= CHERRY_CLUSTER:
        row = Counter(z["row"] for z in cluster).most_common(1)[0][0]
        action = pick(CHERRY, lambda item: item["row"] == row, lambda item: -item["col"])
        if action:
            action["rule"] = "②樱桃炸弹"
            return action

    # 3. economy first: two columns of sun producers at the back
    #    夜间（天不掉阳光）优先阳光菇(9)，白天优先向日葵(1)；卡组里没有的退而求其次。
    producer = (SUN_SHROOM if by_type.get(SUN_SHROOM) else SUNFLOWER) \
        if observation.get("night") else \
        (SUNFLOWER if by_type.get(SUNFLOWER) else SUN_SHROOM)
    if producer and count[producer] < PRODUCER_TARGET:
        action = pick(producer, lambda item: item["col"] <= PRODUCER_MAX_COL,
                      lambda item: (-item["col"], rows_with(plants, producer)[item["row"]]))
        if action:
            action["rule"] = "③经济"
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
                          lambda item: item["row"] == target and MID_COLS[0] <= item["col"] <= MID_COLS[1],
                          lambda item: -item["col"])
            if action:
                action["rule"] = "④a水路防御·睡莲"
                return action
        cattail_options = by_type.get(CATTAIL)
        if cattail_options and count[CATTAIL] < CATTAIL_TARGET:
            action = pick(CATTAIL, lambda item: MID_COLS[0] <= item["col"] <= MID_COLS[1],
                          lambda item: (-row_threat(item["row"]), -item["col"]))
            if action:
                action["rule"] = "④b水路防御·香蒲"
                return action

    # 5. offence: peashooters (with a few snow peas) in the middle columns
    shooters = count[PEASHOOTER] + count[SNOWPEA]
    if shooters < SHOOTER_TARGET:
        if count[SNOWPEA] < SNOWPEA_TARGET and observation["sun"] >= SNOWPEA_SUN:
            action = pick(SNOWPEA, lambda item: MID_COLS[0] <= item["col"] <= MID_COLS[1],
                          lambda item: -row_threat(item["row"]))
            if action:
                action["rule"] = "⑤a火力·寒冰"
                return action
        action = pick(PEASHOOTER, lambda item: MID_COLS[0] <= item["col"] <= MID_COLS[1],
                      lambda item: (-row_threat(item["row"]), -item["col"]))
        if action:
            action["rule"] = "⑤b火力·豌豆"
            return action

    # 6. cheap early mines on the lane the first zombies walk down
    if count[POTATO] < MINE_TARGET and observation["wave"] <= MINE_MAX_WAVE:
        action = pick(POTATO, lambda item: item["col"] >= MINE_MIN_COL, lambda item: -row_threat(item["row"]))
        if action:
            action["rule"] = "⑥早期地雷"
            return action

    return {"type": "wait", "ticks": WAIT_TICKS, "rule": "⑦无满足条件→wait"}


def run(env: PvZEnv, seed: int, level: int = LEVEL, max_actions: int = 4000,
        task: TaskSpec | None = None, deck: tuple[int, ...] | None = None) -> dict:
    deck = tuple(deck) if deck else deck_for_level(level)
    task = task or TaskSpec(level=level, seed=seed, playthrough=2,
                            profile=profile_for_deck(deck))
    observation, _ = env.reset(deck=deck, task=task)
    initial_plants = observation["plants"]
    initial_sun = observation["sun"]
    initial_off_board_zombies = sum(not zombie["on_board"] for zombie in observation["zombies"])
    initial_enemy_zombies_on_screen = bool(observation["enemy_zombies_on_screen"])
    if initial_enemy_zombies_on_screen:
        raise AssertionError("a new game must not report preview zombies as on-screen enemies")
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
        "initial_enemy_zombies_on_screen": initial_enemy_zombies_on_screen,
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
