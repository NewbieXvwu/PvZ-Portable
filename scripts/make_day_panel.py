#!/usr/bin/env python3
"""生成白天 L1-L10 原版难度评估面板（task17）。

面板 = 一个任务清单，每关一个评估任务 `day_lN_eval`：
  * 原版难度：`zombie_count_multiplier = 1.0`、`wave_cap = null`（不限波）；
  * 卡组用 `scripts/pvz_tasks.py` 的 `deck_for_level`（本文件不另写卡组规则）；
  * 其余字段照 `artifacts/stage4/day_l7_hard.json` 里 `day_l7_*_eval` 的写法：
    `sun_start = 50`（Adventure-II 的 reset 值，RESET_V2 不覆盖它，
    见 `scripts/check_task_manifests.py` 的 `sun_start_semantics`）、
    `preplanted = []`、`playthrough = 2`（replay 语义，唯一支持的值；
    reset 会把它翻成 `mFinishedAdventure = 1`，因此所有基础植物都可用）。
  * 种子用与训练不重叠的评估段：30000..31023（与 `day_l7_*_eval` 同一段）。

特殊关：
  * 1-5 是坚果保龄球（`LawnApp::IsWallnutBowlingLevel`，Adventure 模式 `mLevel == 5`），
    没有卡组、没有阳光经济，不收。
  * 1-10 是传送带关（`LawnApp::IsMiniBossLevel` 含 `mLevel == 10` ->
    `Board::HasConveyorBeltSeedBank`），环境能 reset 也能跑，但卡组被传送带取代。
    默认收进来，报告里标注它的语义；`--skip 10` 可去掉。
    task20 起额外收两个**原版开局**任务：`day_l10_orig_eval`（种子 30000..31023）与
    `day_l10_orig_train`（1880000..1888191）。它们 deck 为空、带 `conveyor_start: original`：
    reset 不写卡组、不 UpdateWidth，卡槽保持引擎原版的 10 格空传送带，之后由引擎发卡。
    旧的 `day_l10_eval` 是 task19 变体（卡组写进卡槽），保留但不再新采集。

同时核对（只报告，不改 `deck_for_level`）：
  * 地形是否都是白天草地（`pvz_constants.background_for_level`）；
  * 卡组是否只含该关已解锁的植物：原版的解锁顺序由
    `LawnApp::GetAwardSeedForLevel`（src/LawnApp.cpp）给出，第 L 关结束时拿到
    的奖励种子编号 = 已解锁植物的上界（编号即冒险模式的解锁顺序）。

卡组默认按 `deck_for_level`。`--deck` 可以覆盖成全关卡同一副卡（不碰 `deck_for_level`）：
训练用的白天卡组是 7 张、带双发射手 `7`，而 `deck_for_level` 的白天卡组只有 6 张，
两者对比要用同一副卡时用 `--deck 0,1,2,3,4,5,7 --suffix _d7`（任务名随之变成 `day_lN_d7_eval`）。

用法：
  python3 scripts/make_day_panel.py                 # 写 artifacts/stage4/day_panel.json
  python3 scripts/make_day_panel.py --skip 10       # 不收传送带关
  python3 scripts/make_day_panel.py --deck 0,1,2,3,4,5,7 --suffix _d7 \
      --out artifacts/stage4/day_panel_d7.json      # 7 卡版面板
  python3 scripts/make_day_panel.py --combined \
      --out ~/PvZ-search/artifacts/stage4/day_panel.json
      # 主清单布局：eval / eval_d7 / train_d7（L7-L10 + task24 的 L1/3/4/6），
      # 26 个任务；不含 §14.8 的 1-10 原版开局任务（那两项不在主清单里）
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

LEVELS_PER_AREA = 10
SEED_START = 30000          # 与 day_l7_*_eval 相同的评估段
SEEDS_PER_LEVEL = 1024
# 训练用的白天 7 卡（多双发射手 7）。--combined 用它写 _eval_d7 与 _train_d7。
D7_DECK = (0, 1, 2, 3, 4, 5, 7)
# 已有训练段（task17 / task19）：1800000 + (N-7)*10000，每关 8192。
TRAIN_D7_START = 1800000
TRAIN_D7_LEVELS = (7, 8, 9, 10)
TRAIN_D7_SEEDS = 8192
# task24：L1/3/4/6 从来没有训练任务族（白天训练族只有 L7-L10），补上。
# 2100000 + (N-1)*10000，每关 8192。
EARLY_TRAIN_D7_START = 2100000
EARLY_TRAIN_D7_LEVELS = (1, 3, 4, 6)
# 生成时自查用：训练种子不得落进这些已用段。
RESERVED_TRAIN_RANGES = (
    (30000, 31023),        # 评估段（白天 / 夜间 / 泳池面板共用）
    (1500000, 1501023),    # night_l12_n1
    (1600000, 1608191),    # 白天早期训练段
    (1880000, 1888191),    # day_l10_orig_train（task20 传送带）
    (1900000, 1998191),    # night_lN_train_n9
    (2000000, 2088191),    # pool_lN_train_p9k
)
# task20 原版传送带开局的训练段：1840000–1878191 已被 task19 的 day_l*_train_d7p 占用。
CONVEYOR_TRAIN_START = 1880000
CONVEYOR_TRAIN_SEEDS = 8192
SUN_START = 50              # Adventure-II reset 值；RESET_V2 不覆盖
# 1-5 坚果保龄球：没有卡组与阳光，任务书明确不收。
EXCLUDED = {5: "wall-nut bowling (LawnApp::IsWallnutBowlingLevel, mLevel == 5)"}


def award_seed_for_level(level: int) -> int:
    """`LawnApp::GetAwardSeedForLevel` 的逐行对应（src/LawnApp.cpp:3060）。

    第 L 关的奖励种子编号就是这一关结束时“已有多少种植物”的上界，所以
    `seed_type <= award_seed_for_level(L)` 等价于“第 L 关已解锁”。
    """
    area = (level - 1) // LEVELS_PER_AREA + 1
    sub = (level - 1) % LEVELS_PER_AREA + 1
    count = (area - 1) * 8 + sub
    if sub >= 10:
        count -= 2
    elif sub >= 5:
        count -= 1
    return min(count, 40)


def unlocked_plants(level: int) -> tuple[int, ...]:
    return tuple(range(award_seed_for_level(level) + 1))


def train_seed_start(level: int) -> int:
    """训练种子起点：L7-L10 沿用 task17 的 1800000 段，L1/3/4/6 用 task24 的 2100000 段。"""
    if level in TRAIN_D7_LEVELS:
        return TRAIN_D7_START + (level - 7) * 10000
    if level in EARLY_TRAIN_D7_LEVELS:
        return EARLY_TRAIN_D7_START + (level - 1) * 10000
    raise RuntimeError(f"L{level} 没有训练段配置")


def train_task(level: int) -> dict:
    start = train_seed_start(level)
    return {
        "task_id": f"day_l{level}_train_d7",
        "terrain": "day",
        "level": level,
        "deck": list(D7_DECK),
        "zombie_count_multiplier": 1.0,
        "wave_cap": None,
        "sun_start": SUN_START,
        "preplanted": [],
        "playthrough": 2,
        "seeds": list(range(start, start + TRAIN_D7_SEEDS)),
    }


def check_train_seeds(tasks: list[dict]) -> None:
    """自查：任务内种子不重复；训练种子不落进已用段，也不和别的训练/评估任务撞种子。"""
    eval_seeds = {seed for task in tasks if "_train_" not in task["task_id"] for seed in task["seeds"]}
    owner: dict[int, str] = {}
    for task in tasks:
        if "_train_" not in task["task_id"]:
            continue
        if len(set(task["seeds"])) != len(task["seeds"]):
            raise RuntimeError(f"{task['task_id']}: 任务内部有重复种子")
        for seed in task["seeds"]:
            for low, high in RESERVED_TRAIN_RANGES:
                if low <= seed <= high:
                    raise RuntimeError(f"{task['task_id']}: seed {seed} 落在已用段 {low}..{high}")
            if seed in eval_seeds:
                raise RuntimeError(f"{task['task_id']}: seed {seed} 与评估段重叠")
            if seed in owner and owner[seed] != task["task_id"]:
                raise RuntimeError(f"{task['task_id']}: seed {seed} 与 {owner[seed]} 重叠")
            owner[seed] = task["task_id"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/stage4/day_panel.json")
    p.add_argument("--seeds", type=int, default=SEEDS_PER_LEVEL)
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--skip", type=int, action="append", default=[],
                   help="不收的关卡（默认只收 1-5）")
    p.add_argument("--deck", default="",
                   help="逗号分隔的种子编号，覆盖 deck_for_level（全关卡同一副卡）；默认空 = 用 deck_for_level")
    p.add_argument("--suffix", default="",
                   help="任务名后缀，插在 _eval 之前（如 _d7 -> day_l1_d7_eval）")
    p.add_argument("--combined", action="store_true",
                   help="复现主清单布局：eval / eval_d7 / train_d7（L7-L10 + task24 的 L1/3/4/6），"
                        "不含 task20 的 1-10 原版开局任务")
    args = p.parse_args()

    fixed_deck = tuple(int(seed) for seed in args.deck.split(",")) if args.deck else None

    skipped = dict(EXCLUDED)
    for level in args.skip:
        skipped.setdefault(level, "skipped on the command line")

    tasks, checks = [], []
    for level in range(1, LEVELS_PER_AREA + 1):
        if level in skipped:
            continue
        deck = fixed_deck if fixed_deck is not None else deck_for_level(level)
        background = pvz_constants.background_for_level(level)
        unlocked = unlocked_plants(level)
        missing = [seed for seed in deck if seed not in unlocked]
        checks.append({"level": level, "background": background, "deck": list(deck),
                       "unlocked": list(unlocked), "not_yet_unlocked": missing})
        tasks.append({
            "task_id": f"day_l{level}{args.suffix}_eval",
            "terrain": "day",
            "level": level,
            "deck": list(deck),
            "zombie_count_multiplier": 1.0,
            "wave_cap": None,
            "sun_start": SUN_START,
            "preplanted": [],
            "playthrough": 2,
            "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        })

    # task20：原版传送带开局（1-10）。L10 是传送带关（IsMiniBossLevel），原版是
    # 开局空槽、容量 10，之后由 Challenge::UpdateConveyorBelt 发卡；上面按卡组生成的
    # day_l10_eval 是 task19 的变体（把任务卡组写进卡槽），保留但不再新采集。
    # 原版开局不用任务卡组，所以清单里 deck 写成空数组，并显式带 conveyor_start。
    if 10 not in skipped and conveyor_belt_level(10) and not args.combined:
        tasks.append({
            "task_id": "day_l10_orig_eval",
            "terrain": "day",
            "level": 10,
            "deck": [],
            "conveyor_start": "original",
            "zombie_count_multiplier": 1.0,
            "wave_cap": None,
            "sun_start": SUN_START,
            "preplanted": [],
            "playthrough": 2,
            "seeds": list(range(args.seed_start, args.seed_start + args.seeds)),
        })
        tasks.append({
            "task_id": "day_l10_orig_train",
            "terrain": "day",
            "level": 10,
            "deck": [],
            "conveyor_start": "original",
            "zombie_count_multiplier": 1.0,
            "wave_cap": None,
            "sun_start": SUN_START,
            "preplanted": [],
            "playthrough": 2,
            "seeds": list(range(CONVEYOR_TRAIN_START, CONVEYOR_TRAIN_START + CONVEYOR_TRAIN_SEEDS)),
        })

    # task24：主清单的 _eval_d7 / _train_d7 族。顺序与现有清单一致：
    # eval（全部关）→ eval_d7（全部关）→ train_d7（先 L7-L10，再追加 L1/3/4/6）。
    if args.combined:
        for task in list(tasks):
            d7 = dict(task)
            d7["task_id"] = f"day_l{task['level']}_eval_d7"
            d7["deck"] = list(D7_DECK)
            tasks.append(d7)
        for level in TRAIN_D7_LEVELS + EARLY_TRAIN_D7_LEVELS:
            if level in skipped:
                continue
            tasks.append(train_task(level))
        check_train_seeds(tasks)

    purpose = "task17: original-difficulty panel for day levels 1-10 (one model, one deck per level, adventure mode)"
    if args.combined:
        purpose += (" | d7 variant: training deck [0,1,2,3,4,5,7] (main 2026-10-20)"
                    " | task24: day_lN_train_d7 for L1/3/4/6, seeds 2100000+(N-1)*10000 (2026-10-22)")
    document = {
        "schema_version": 1,
        "split": "stage4",
        "purpose": purpose,
        "note": ("local manifest, not in git; decks come from scripts/pvz_tasks.py deck_for_level, "
                 "seeds 30000.. are the same evaluation block as day_l7_*_eval"),
        "skipped_levels": {str(level): reason for level, reason in sorted(skipped.items())},
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(tasks)} tasks, {args.seeds} seeds each "
          f"({args.seed_start}..{args.seed_start + args.seeds - 1})")
    print(f"skipped: {document['skipped_levels']}")
    print()
    print("level  background             deck                  not unlocked yet")
    for row in checks:
        print(f"{row['level']:5d}  {row['background']:<22s} {str(row['deck']):<21s} "
              f"{row['not_yet_unlocked'] or '-'}")
    bad_terrain = [row["level"] for row in checks if row["background"] != "BACKGROUND_1_DAY"]
    if bad_terrain:
        print(f"WARNING: not day grass: {bad_terrain}")
    if args.combined:
        for level in TRAIN_D7_LEVELS + EARLY_TRAIN_D7_LEVELS:
            start = train_seed_start(level)
            print(f"train_d7 L{level}: {start}..{start + TRAIN_D7_SEEDS - 1}")


if __name__ == "__main__":
    main()
