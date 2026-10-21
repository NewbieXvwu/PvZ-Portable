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
from pvz_tasks import deck_for_level  # noqa: E402
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

# 训练段：1900000 + (N - 11) * 10000，每关 8192 个。
TRAIN_SEED_BASE = 1900000
TRAIN_SEED_STRIDE = 10000
TRAIN_SEEDS = 8192
# 已用种子段（自查用）：与这些区间重叠就直接报错，不生成。
RESERVED_SEED_RANGES = (
    (30000, 31023),        # 评估段（白天 / 夜间面板共用）
    (1500000, 1501023),    # night_l12_n1 训练段
    (1600000, 1608191),    # 白天早期训练段
    (1800000, 1878191),    # day_l7..l10 train_d7 / train_d7p
)


def train_seed_start(level: int) -> int:
    return TRAIN_SEED_BASE + (level - FIRST_LEVEL) * TRAIN_SEED_STRIDE


def check_seed_overlap(tasks: list[dict]) -> None:
    """自查：单个任务里种子不重复；训练种子不撞任何已用段。

    不同关卡、不同卡组共用同一段评估种子是**设计如此**（配对比较要同种子），
    所以重复只在“同一个任务内部”才算错。
    """
    for task in tasks:
        if len(set(task["seeds"])) != len(task["seeds"]):
            raise RuntimeError(f"{task['task_id']}: 任务内部有重复种子")
        if "_train_" not in task["task_id"]:
            continue
        for seed in task["seeds"]:
            for low, high in RESERVED_SEED_RANGES:
                if low <= seed <= high:
                    raise RuntimeError(f"{task['task_id']}: seed {seed} 落在已用段 {low}..{high}")
    for task in tasks:
        if task["task_id"].startswith("night_l") and "_train_" in task["task_id"]:
            if min(task["seeds"]) < TRAIN_SEED_BASE or max(task["seeds"]) >= TRAIN_SEED_BASE + 9 * TRAIN_SEED_STRIDE + TRAIN_SEEDS:
                raise RuntimeError(f"{task['task_id']}: 训练种子越出 1900000 段")


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
                   help="一份清单里同时写 eval（对照）/ eval_n8 / eval_n9 / train_n9 四个族（主控的采集队列读这份）")
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
        check_seed_overlap(tasks)

    document = {
        "schema_version": 1,
        "split": "stage4",
        "purpose": "task21: original-difficulty panel for night levels 11-20 (one model, one deck per level, adventure mode)",
        "note": ("local manifest, not in git; decks come from scripts/pvz_tasks.py deck_for_level, "
                 "seeds 30000.. are the same evaluation block as the day panel"
                 + ("; families: night_l*_eval (control), night_l*_eval_n8, night_l*_eval_n9, "
                    "night_l*_train_n9 (1900000 + (N-11)*10000, 8192 each)" if args.combined else "")),
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


if __name__ == "__main__":
    main()
