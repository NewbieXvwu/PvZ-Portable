#!/usr/bin/env python3
"""生成夜间 L11-L20 原版难度评估面板（task21）。

面板 = 一个任务清单，每关一个评估任务 `night_lN_eval`：
  * 原版难度：`zombie_count_multiplier = 1.0`、`wave_cap = null`（不限波）；
  * 卡组用 `scripts/pvz_tasks.py` 的 `deck_for_level`（本文件不另写卡组规则）；
    夜间草地 = `(0, 1, 9, 2, 3, 4, 5)`（第 3 槽是阳光菇）；
  * 其余字段照白天面板（`scripts/make_day_panel.py`，task17）：
    `sun_start = 50`、`preplanted = []`、`playthrough = 2`；
  * 种子用评估段 30000..31023（与白天面板同一段，1024 个/关）。

特殊关（本环境的口径，源码逐条核对过）：
  * 2-5（L15）是**打地鼠**（`LawnApp::IsWhackAZombieLevel`，Adventure `mLevel == 15`）：
    波次走 `mChallenge->InitZombieWaves()`、`mNumWaves = 8`，坟头由
    `WhackAZombiePlaceGraves(9)` 摆。没有卡组与阳光经济，和 1-5 坚果保龄球同类。
    默认不收；`--include 15` 可强行收进来（能不能跑由冒烟结果决定）。
  * 2-10（L20）是 mini-boss（`IsMiniBossLevel` 含 `mLevel == 20`），
    `Board::HasConveyorBeltSeedBank()` 为真 —— 与白天 1-10 同一语义：
    reset 把卡组写进 seed bank，之后是原生传送带。默认收进来，报告里标注。
  * L15 之外的夜间关都是普通夜间草地（11-14、16-19），坟头数随关卡增长
    （11-13: 4、14/16: 7、17-19: 11、20: 13；`Board::AddGraveStones`）。

同时核对（只报告，不改 `deck_for_level`）：
  * 地形是否都是夜间草地（`pvz_constants.background_for_level`）；
  * 卡组是否只含该关已解锁的植物：原版解锁顺序由
    `LawnApp::GetAwardSeedForLevel` 给出（与白天面板同一判据）。注意
    `playthrough = 2` 时引擎的 `GetSeedsAvailable()` 返回 49，模拟器不拦，
    所以这一列只是“原版首轮解锁”的记录，不是运行约束。

2026-10-22 主控定夜间卡组 n9 = `[0,1,2,3,4,5,7,8,9]`。`--combined` 在同一个清单里写四个族
（eval 对照 / eval_n8 / eval_n9 / train_n9），供采集队列直接读；训练种子
`1900000 + (N-11)*10000` 起每关 8192 个，生成时自查不与评估段及旧训练段重叠。

2026-10-22 task25：夜间卡组改定 n10 = n9 + 墓碑吞噬者 11（10 张 = SEEDBANK_MAX）。
最后一波开始约 200 tick 后每个坟头各冒一只僵尸（`Board::SpawnZombiesFromGraves`），
2-7..2-9 有 11 个坟头；老师（dw4i_s1 推演，beta=1，16 局）n9 → n10：2-7 4 → 11、2-9 1 → 8。
`--combined` 追加 `night_lN_eval_n10`（N = 11-14、16-20）与 `night_lN_train_n10`
（N = 11-14、16-19），训练种子 `2200000 + (N-11)*10000` 起每关 8192；
L20 另出原版开局版 `night_l20_eval_n10_orig` / `night_l20_train_n10_orig`
（`deck: []` + `conveyor_start: "original"`，仿 task23 的 `night_l20_eval_n9_orig`）。

用法：
  python3 scripts/make_night_panel.py                       # 只写对照族（night_lN_eval）
  python3 scripts/make_night_panel.py --combined            # 四个族一起写（主控的采集清单）
  python3 scripts/make_night_panel.py --deck 0,1,2,3,4,5,8,9 --suffix _n8 \
      --out artifacts/stage4/night_panel_n8.json            # 单写 n8 族
  python3 scripts/make_night_panel.py --include 15          # 连打地鼠关一起收
  python3 scripts/make_night_panel.py --skip 20             # 去掉 mini-boss 关
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import pvz_constants  # noqa: E402
from pvz_tasks import conveyor_belt_level, deck_for_level  # noqa: E402
from make_day_panel import SUN_START, award_seed_for_level, unlocked_plants  # noqa: E402

FIRST_LEVEL = 11
LAST_LEVEL = 20
SEED_START = 30000          # 与白天面板（day_l7_*_eval）相同的评估段
SEEDS_PER_LEVEL = 1024

# 2-5 打地鼠：没有卡组与阳光经济（LawnApp::IsWhackAZombieLevel，mLevel == 15）。
EXCLUDED = {15: "whack-a-zombie (LawnApp::IsWhackAZombieLevel, mLevel == 15)"}

# 2026-10-22 主控定卡组：n9 是白天训练卡组的超集（学生已认识其中 7 张）。
N8_DECK = (0, 1, 2, 3, 4, 5, 8, 9)
N9_DECK = (0, 1, 2, 3, 4, 5, 7, 8, 9)
# 2026-10-22 主控定卡组：n10 = n9 + 墓碑吞噬者 11（10 张 = SEEDBANK_MAX）。
# 动机：最后一波开始约 200 tick 后每个坟头各冒一只僵尸（Board.cpp SpawnZombiesFromGraves），
# 2-7..2-9 有 11 个坟头；老师（dw4i_s1 推演，beta=1，16 局）n9 → n10：2-7 4 → 11、2-9 1 → 8。
N10_DECK = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11)

# 训练段：1900000 + (N - 11) * 10000，每关 8192 个。
TRAIN_SEED_BASE = 1900000
TRAIN_SEED_STRIDE = 10000
TRAIN_SEEDS = 8192
# task25 的 n10 训练段：2200000 + (N - 11) * 10000，每关 8192 个。
# eval_n10 / train_n10 都只收普通夜间关（11-14、16-19）；L20 是 mini-boss 传送带，
# 只出原版开局版（night_l20_eval_n10_orig / night_l20_train_n10_orig）。
N10_TRAIN_SEED_BASE = 2200000
N10_TRAIN_STRIDE = 10000
N10_TRAIN_SEEDS = 8192
N10_TRAIN_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
# 已用种子段（自查用）：与这些区间重叠就直接报错，不生成。
RESERVED_SEED_RANGES = (
    (30000, 31023),        # 评估段（白天 / 夜间面板共用）
    (1500000, 1501023),    # night_l12_n1 训练段
    (1600000, 1608191),    # 白天早期训练段
    (1800000, 1878191),    # day_l7..l10 train_d7 / train_d7p
    (1880000, 1888191),    # day_l10_orig_train（task20 传送带）
    (1900000, 1998191),    # night_lN_train_n9
    (2000000, 2088191),    # pool_lN_train_p9k
    (2100000, 2108191),    # day_l1_train_d7（task24）
    (2120000, 2128191),    # day_l3_train_d7
    (2130000, 2138191),    # day_l4_train_d7
    (2150000, 2158191),    # day_l6_train_d7
)


def train_seed_start(level: int) -> int:
    return TRAIN_SEED_BASE + (level - FIRST_LEVEL) * TRAIN_SEED_STRIDE


def n10_train_seed_start(level: int) -> int:
    return N10_TRAIN_SEED_BASE + (level - FIRST_LEVEL) * N10_TRAIN_STRIDE


def train_range(base: int, stride: int, seeds: int) -> tuple[int, int]:
    """一整族训练种子的闭区间（N 从 FIRST_LEVEL 到 LAST_LEVEL 连续排）。"""
    return base, base + (LAST_LEVEL - FIRST_LEVEL) * stride + seeds - 1


def check_seed_overlap(tasks: list[dict]) -> None:
    """自查：任务内种子不重复；训练种子不撞别的已用段，训练任务之间也不重叠。

    不同关卡、不同卡组共用同一段评估种子是**设计如此**（配对比较要同种子），
    所以重复只在“同一个任务内部”才算错。训练族自己那一段（n9 1900000 /
    n10 2200000）不算“已用段”，否则自己的任务会被自己拦下。
    """
    n9_low, n9_high = train_range(TRAIN_SEED_BASE, TRAIN_SEED_STRIDE, TRAIN_SEEDS)
    n10_low, n10_high = train_range(N10_TRAIN_SEED_BASE, N10_TRAIN_STRIDE, N10_TRAIN_SEEDS)
    owner: dict[int, str] = {}
    for task in tasks:
        task_id = task["task_id"]
        if len(set(task["seeds"])) != len(task["seeds"]):
            raise RuntimeError(f"{task_id}: 任务内部有重复种子")
        if "_train_n9" in task_id:
            low, high = n9_low, n9_high
        elif "_train_n10" in task_id:
            low, high = n10_low, n10_high
        else:
            continue
        if min(task["seeds"]) < low or max(task["seeds"]) > high:
            raise RuntimeError(f"{task_id}: 训练种子越出 {low}..{high} 段")
        for seed in task["seeds"]:
            for reserved_low, reserved_high in RESERVED_SEED_RANGES:
                own_family = low <= reserved_low and reserved_high <= high
                if reserved_low <= seed <= reserved_high and not own_family:
                    raise RuntimeError(f"{task_id}: seed {seed} 落在已用段 {reserved_low}..{reserved_high}")
            if seed in owner and owner[seed] != task_id:
                raise RuntimeError(f"{task_id}: seed {seed} 与 {owner[seed]} 重叠")
            owner[seed] = task_id


def task_document(level: int, deck, task_id: str, seeds: list[int]) -> dict:
    return {
        "task_id": task_id,
        "terrain": "night",
        "level": level,
        "deck": list(deck),
        "zombie_count_multiplier": 1.0,
        "wave_cap": None,
        "sun_start": SUN_START,
        "preplanted": [],
        "playthrough": 2,
        "seeds": seeds,
    }


def original_task_document(level: int, task_id: str, seeds: list[int]) -> dict:
    """原版传送带开局（task20 的 conveyor_start）：deck 为空，引擎保留 10 个空槽并原生发免费卡。"""
    return {
        "task_id": task_id,
        "terrain": "night",
        "level": level,
        "deck": [],
        "conveyor_start": "original",
        "zombie_count_multiplier": 1.0,
        "wave_cap": None,
        "sun_start": SUN_START,
        "preplanted": [],
        "playthrough": 2,
        "seeds": seeds,
    }


def check_n10_deck(deck: tuple[int, ...]) -> None:
    """AGENTS §8：长度 6..10（n10 正好 10 = SEEDBANK_MAX）、无重复、不含升级植物。"""
    if not 6 <= len(deck) <= 10:
        raise RuntimeError(f"n10 卡组长度 {len(deck)} 不在 6..10")
    if len(set(deck)) != len(deck):
        raise RuntimeError("n10 卡组内有重复")
    upgrades = sorted(set(deck) & set(pvz_constants.upgrade_plants()))
    if upgrades:
        raise RuntimeError(f"n10 卡组含升级植物 {upgrades}：本族不应出现，需显式核对底座与所有权")


def special_note(level: int) -> str:
    if level == 15:
        return "whack-a-zombie: 8 waves, Challenge wave script, no deck economy"
    if level == 20:
        return "mini-boss: HasConveyorBeltSeedBank, deck seeds the initial packets"
    return ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/stage4/night_panel.json")
    p.add_argument("--seeds", type=int, default=SEEDS_PER_LEVEL)
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--skip", type=int, action="append", default=[],
                   help="不收的关卡（默认只收 2-5 打地鼠）")
    p.add_argument("--include", type=int, action="append", default=[],
                   help="强制收进默认排除的关卡（如 --include 15）")
    p.add_argument("--deck", default="",
                   help="逗号分隔的种子编号，覆盖 deck_for_level（全关卡同一副卡）；默认空 = 用 deck_for_level")
    p.add_argument("--suffix", default="",
                   help="任务名后缀，接在 _eval 之后（如 _n8 -> night_l1_eval_n8）")
    p.add_argument("--combined", action="store_true",
                   help="一份清单里同时写 eval（对照）/ eval_n8 / eval_n9 / train_n9 / "
                        "eval_n10 / train_n10（+ L20 原版开局版）（主控的采集队列读这份）")
    args = p.parse_args()

    fixed_deck = tuple(int(seed) for seed in args.deck.split(",")) if args.deck else None

    skipped = dict(EXCLUDED)
    for level in args.include:
        skipped.pop(level, None)
    for level in args.skip:
        skipped.setdefault(level, "skipped on the command line")

    tasks, checks = [], []
    for level in range(FIRST_LEVEL, LAST_LEVEL + 1):
        if level in skipped:
            continue
        deck = fixed_deck if fixed_deck is not None else deck_for_level(level)
        background = pvz_constants.background_for_level(level)
        unlocked = unlocked_plants(level)
        missing = [seed for seed in deck if seed not in unlocked]
        checks.append({"level": level, "background": background, "deck": list(deck),
                       "unlocked": list(unlocked), "not_yet_unlocked": missing,
                       "special": special_note(level)})
        tasks.append(task_document(level, deck, f"night_l{level}_eval{args.suffix}",
                                   list(range(args.seed_start, args.seed_start + args.seeds))))

    if args.combined:
        for level in range(FIRST_LEVEL, LAST_LEVEL + 1):
            if level in skipped:
                continue
            evaluation = list(range(args.seed_start, args.seed_start + args.seeds))
            tasks.append(task_document(level, N8_DECK, f"night_l{level}_eval_n8", evaluation))
            tasks.append(task_document(level, N9_DECK, f"night_l{level}_eval_n9", evaluation))
            tasks.append(task_document(level, N9_DECK, f"night_l{level}_train_n9",
                                       list(range(train_seed_start(level), train_seed_start(level) + TRAIN_SEEDS))))
        # task25：n10 族（追加在已有四个族之后，已有任务的顺序与内容逐字节不变）。
        # N = 11-14、16-19 出 eval_n10 + train_n10；L20 是 mini-boss 传送带，
        # 只出原版开局版（deck 空 + conveyor_start，仿 task23 的 night_l20_eval_n9_orig），
        # 不再出“卡组写进初始手牌”的 L20 n10 任务。
        check_n10_deck(N10_DECK)
        for level in N10_TRAIN_LEVELS:
            if level in skipped:
                continue
            evaluation = list(range(args.seed_start, args.seed_start + args.seeds))
            tasks.append(task_document(level, N10_DECK, f"night_l{level}_eval_n10", evaluation))
            start = n10_train_seed_start(level)
            tasks.append(task_document(level, N10_DECK, f"night_l{level}_train_n10",
                                       list(range(start, start + N10_TRAIN_SEEDS))))
        if LAST_LEVEL not in skipped and conveyor_belt_level(LAST_LEVEL):
            evaluation = list(range(args.seed_start, args.seed_start + args.seeds))
            tasks.append(original_task_document(LAST_LEVEL, f"night_l{LAST_LEVEL}_eval_n10_orig", evaluation))
            start = n10_train_seed_start(LAST_LEVEL)
            tasks.append(original_task_document(LAST_LEVEL, f"night_l{LAST_LEVEL}_train_n10_orig",
                                                list(range(start, start + N10_TRAIN_SEEDS))))
        check_seed_overlap(tasks)

    document = {
        "schema_version": 1,
        "split": "stage4",
        "purpose": "task21: original-difficulty panel for night levels 11-20 (one model, one deck per level, adventure mode)",
        "note": ("local manifest, not in git; decks come from scripts/pvz_tasks.py deck_for_level, "
                 "seeds 30000.. are the same evaluation block as the day panel"
                 + ("; families: night_l*_eval (control), night_l*_eval_n8, night_l*_eval_n9, "
                    "night_l*_train_n9 (1900000 + (N-11)*10000, 8192 each), "
                    "night_l*_eval_n10 / night_l*_train_n10 (N=11-14,16-19; 2200000 + (N-11)*10000, 8192 each), "
                    "night_l20_eval_n10_orig / night_l20_train_n10_orig (conveyor_start=original, deck empty)"
                    if args.combined else "")),
        "skipped_levels": {str(level): reason for level, reason in sorted(skipped.items())},
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(tasks)} tasks, {args.seeds} seeds each "
          f"({args.seed_start}..{args.seed_start + args.seeds - 1})")
    print(f"skipped: {document['skipped_levels']}")
    print()
    print("level  background             deck                  not unlocked yet  special")
    for row in checks:
        print(f"{row['level']:5d}  {row['background']:<22s} {str(row['deck']):<21s} "
              f"{str(row['not_yet_unlocked'] or '-'):<17s} {row['special']}")
    bad_terrain = [row["level"] for row in checks if row["background"] != "BACKGROUND_2_NIGHT"]
    if bad_terrain:
        print(f"WARNING: not night grass: {bad_terrain}")
    if args.combined:
        for name, deck in (("n8", N8_DECK), ("n9", N9_DECK)):
            bad = {row["level"]: [seed for seed in deck if seed not in row["unlocked"]] for row in checks}
            bad = {level: missing for level, missing in bad.items() if missing}
            print(f"{name} deck {list(deck)}: " + ("all unlocked" if not bad else f"NOT unlocked: {bad}"))
        for row in checks:
            start = train_seed_start(row["level"])
            print(f"train_n9 L{row['level']}: {start}..{start + TRAIN_SEEDS - 1}")
        print(f"n10 deck {list(N10_DECK)}: length {len(N10_DECK)}, upgrades none")
        for row in checks:
            if row["level"] in N10_TRAIN_LEVELS:
                start = n10_train_seed_start(row["level"])
                print(f"train_n10 L{row['level']}: {start}..{start + N10_TRAIN_SEEDS - 1}")
        start = n10_train_seed_start(LAST_LEVEL)
        print(f"train_n10_orig L{LAST_LEVEL}: {start}..{start + N10_TRAIN_SEEDS - 1} (conveyor_start=original)")


if __name__ == "__main__":
    main()
