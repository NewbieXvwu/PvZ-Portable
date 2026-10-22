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

2026-10-22 主控定夜间卡组 n9 = `[0,1,2,3,4,5,7,8,9]`。`--combined` 在同一个清单里写四个族
（eval 对照 / eval_n8 / eval_n9 / train_n9），供采集队列直接读；训练种子
`1900000 + (N-11)*10000` 起每关 8192 个，生成时自查不与评估段及旧训练段重叠。

task25 的 n10 保持追加在旧36任务之后；普通关 n10 含墓碑吞噬者11，L20 的 n10 仅出原版空手开局。

本文件是 `scripts/zone_panel.py`（按区生成的公共核心）的薄包装，CLI 与 task21 时相同。

用法：
  python3 scripts/make_night_panel.py                       # 只写对照族（night_lN_eval）
  python3 scripts/make_night_panel.py --combined            # 四个族一起写（主控的采集清单）
  python3 scripts/make_night_panel.py --deck 0,1,2,3,4,5,8,9 --suffix _n8 \\
      --out artifacts/stage4/night_panel_n8.json            # 单写 n8 族
  python3 scripts/make_night_panel.py --include 15          # 连打地鼠关一起收
  python3 scripts/make_night_panel.py --skip 20             # 去掉 mini-boss 关
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from zone_panel import Family, Zone, main_for  # noqa: E402

# 2026-10-22 主控定卡组：n9 是白天训练卡组的超集（学生已认识其中 7 张）。
N8_DECK = (0, 1, 2, 3, 4, 5, 8, 9)
N9_DECK = (0, 1, 2, 3, 4, 5, 7, 8, 9)
N10_DECK = N9_DECK + (11,)

# 训练段：1900000 + (N - 11) * 10000，每关 8192 个。
TRAIN_SEED_BASE = 1900000
TRAIN_SEED_STRIDE = 10000
TRAIN_SEEDS = 8192
# 已用种子段（自查用）：与这些区间重叠就直接报错，不生成。
RESERVED_SEED_RANGES = (
    (30000, 31023),        # 评估段（白天 / 夜间面板共用）
    (1500000, 1501023),    # night_l12_n1 训练段
    (1600000, 1608191),    # 白天早期训练段
    (30000, 31023),
    (1880000, 1888191),
    (1900000, 1998191),
    (2000000, 2088191),
    (2100000, 2108191),
    (2120000, 2128191),
    (2130000, 2138191),
    (2150000, 2158191),
    (1800000, 1878191),    # day_l7..l10 train_d7 / train_d7p
)

NIGHT = Zone(
    key="night",
    task_prefix="night_l",
    first_level=11,
    last_level=20,
    terrain="night",
    default_out=ROOT / "artifacts/stage4/night_panel.json",
    purpose=("task21: original-difficulty panel for night levels 11-20 "
             "(one model, one deck per level, adventure mode)"),
    excluded={15: "whack-a-zombie (LawnApp::IsWhackAZombieLevel, mLevel == 15)"},
    special_notes={
        15: "whack-a-zombie: 8 waves, Challenge wave script, no deck economy",
        20: "mini-boss: HasConveyorBeltSeedBank, deck seeds the initial packets",
    },
    families=(
        Family(""),
        Family("_n8", N8_DECK),
        Family("_n9", N9_DECK),
        Family("_n9", N9_DECK, "train"),
        Family("_n10", N10_DECK, exclude_levels=(20,), phase=1),
        Family("_n10", N10_DECK, "train", exclude_levels=(20,), phase=1, train_seed_base=2200000),
        Family("_n10_orig", (), include_levels=(20,), phase=1, conveyor_start="original"),
        Family("_n10_orig", (), "train", include_levels=(20,), phase=1,
               conveyor_start="original", train_seed_base=2200000),
    ),
    train_seed_base=TRAIN_SEED_BASE,
    train_seed_stride=TRAIN_SEED_STRIDE,
    train_seeds=TRAIN_SEEDS,
    reserved_seed_ranges=RESERVED_SEED_RANGES,
)


def main() -> None:
    main_for(NIGHT)


if __name__ == "__main__":
    main()
