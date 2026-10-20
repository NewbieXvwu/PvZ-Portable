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
from pvz_tasks import deck_for_level  # noqa: E402

LEVELS_PER_AREA = 10
SEED_START = 30000          # 与 day_l7_*_eval 相同的评估段
SEEDS_PER_LEVEL = 1024
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

    document = {
        "schema_version": 1,
        "split": "stage4",
        "purpose": "task17: original-difficulty panel for day levels 1-10 (one model, one deck per level, adventure mode)",
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


if __name__ == "__main__":
    main()
