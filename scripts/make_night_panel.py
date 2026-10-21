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

用法：
  python3 scripts/make_night_panel.py                       # 写 artifacts/stage4/night_panel.json
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
from pvz_tasks import deck_for_level  # noqa: E402
from make_day_panel import SUN_START, award_seed_for_level, unlocked_plants  # noqa: E402

FIRST_LEVEL = 11
LAST_LEVEL = 20
SEED_START = 30000          # 与白天面板（day_l7_*_eval）相同的评估段
SEEDS_PER_LEVEL = 1024

# 2-5 打地鼠：没有卡组与阳光经济（LawnApp::IsWhackAZombieLevel，mLevel == 15）。
EXCLUDED = {15: "whack-a-zombie (LawnApp::IsWhackAZombieLevel, mLevel == 15)"}


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
        tasks.append({
            "task_id": f"night_l{level}_eval{args.suffix}",
            "terrain": "night",
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
        "purpose": "task21: original-difficulty panel for night levels 11-20 (one model, one deck per level, adventure mode)",
        "note": ("local manifest, not in git; decks come from scripts/pvz_tasks.py deck_for_level, "
                 "seeds 30000.. are the same evaluation block as the day panel"),
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


if __name__ == "__main__":
    main()
