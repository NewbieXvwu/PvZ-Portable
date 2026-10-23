#!/usr/bin/env python3
"""生成白天泳池 L21-L30 原版难度评估面板（task22）。

面板 = 一个任务清单，每关一个评估任务 `pool_lN_eval`：
  * 原版难度：`zombie_count_multiplier = 1.0`、`wave_cap = null`（不限波）；
  * 卡组默认用 `scripts/pvz_tasks.py` 的 `deck_for_level`（本文件不另写卡组规则）；
    白天泳池 = `(0, 1, 2, 3, 4, 5, 16, 43)`（白天 6 卡 + 睡莲 16 + 香蒲 43）；
  * 其余字段照白天/夜间面板：`sun_start = 50`、`preplanted = []`、`playthrough = 2`；
  * 种子用评估段 30000..31023（与白天/夜间面板同一段，1024 个/关）。

特殊关（源码逐条核对过）：
  * 3-5（L25）是 **Little Trouble**（`LawnApp::IsLittleTroubleLevel`，Adventure
    `mLevel == 25`）：`Board::HasConveyorBeltSeedBank()` 为真 → 没有卡组与阳光经济，
    僵尸点数 ×4（`Board::PickZombieWaves`），僵尸半尺寸、血量 1/4（`Zombie` 构造）。
    和 2-5 打地鼠同类：卡组面板对它没有意义，**默认不收**（`--include 25` 可强行收）。
  * 3-10（L30）是 mini-boss（`IsMiniBossLevel` 含 `mLevel == 30`），
    `HasConveyorBeltSeedBank()` 为真 —— 与 2-10 同一语义：reset 把卡组写进 seed bank，
    之后是原生传送带。默认收进来，报告里标注。
  * 其余（21-24、26-29）是普通白天泳池：6 行，中间两行（row 2/3）是水路，
    非水生植物必须种在睡莲 16 上（`Board::CanPlantAt`），香蒲 43 只能种在睡莲上。
  * 泳池没有坟头（`AddGraveStones` 只对夜间背景 / 挑战模式生效）。

三套卡组（task22，同种子段，任务族名互不覆盖）：
  * 对照 `pool_lN_eval`     = `deck_for_level` = `[0,1,2,3,4,5,16,43]`
  * p9   `pool_lN_eval_p9`  = `[0,1,2,3,4,5,7,16,43]`（学生训练卡组 + 睡莲 + 香蒲）
  * p9k  `pool_lN_eval_p9k` = `[0,1,2,3,4,5,7,16,19]`（学生训练卡组 + 睡莲 + 缠绕水草 19）

2026-10-22 主控定泳池卡组 **p9k**（老师 25/80，p9 23/80 在噪声内），`--combined` 另写训练族
`pool_lN_train_p9k`：普通泳池关（21–24、26–29），种子 `2000000 + (N-21)*10000` 起每关 8192 个。
**训练族不收 3-10（L30）**：它是 mini-boss 传送带，卡组只描述初始手牌，不适合当训练任务；
3-5（L25）本来就整关不收。

AGENTS §8 核对在生成时强制：地形覆盖（水路要有 16 或 43）、升级植物所有权
（香蒲 43 必须在 `PlayerProfileContext.owned_upgrade_plants` 里，由
`pvz_tasks.profile_for_deck` 按卡组自动声明）、长度 6..10、无重复。

本文件是 `scripts/zone_panel.py`（按区生成的公共核心）的薄包装。

用法：
  python3 scripts/make_pool_panel.py                       # 只写对照族（pool_lN_eval）
  python3 scripts/make_pool_panel.py --combined            # 三族一起写（主控的采集清单）
  python3 scripts/make_pool_panel.py --include 25          # 连 3-5 小游戏关一起收
  python3 scripts/make_pool_panel.py --skip 30             # 去掉 mini-boss 关
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from zone_panel import Family, Zone, main_for  # noqa: E402

P9_DECK = (0, 1, 2, 3, 4, 5, 7, 16, 43)
P9K_DECK = (0, 1, 2, 3, 4, 5, 7, 16, 19)

# 训练段：2000000 + (N - 21) * 10000，每关 8192 个（与 1900000 夜间段、1800000 白天段不重叠）。
TRAIN_SEED_BASE = 2000000
TRAIN_SEED_STRIDE = 10000
TRAIN_SEEDS = 8192
# 训练族不收的关：3-10 是 mini-boss 传送带（3-5 整关已排除）。
TRAIN_EXCLUDED_LEVELS = (30,)

# 已用种子段（自查用）：与这些区间重叠就直接报错，不生成。
RESERVED_SEED_RANGES = (
    (1500000, 1501023),    # night_l12_n1 训练段
    (1600000, 1608191),    # 白天早期训练段
    (1800000, 1878191),    # day_l7..l10 train_d7 / train_d7p
    (1900000, 1998191),    # night_lN_train_n9
)

POOL = Zone(
    key="pool",
    task_prefix="pool_l",
    first_level=21,
    last_level=30,
    terrain="pool",
    default_out=ROOT / "artifacts/stage4/pool_panel.json",
    purpose=("task22: original-difficulty panel for day pool levels 21-30 "
             "(one model, one deck per level, adventure mode)"),
    excluded={
        25: ("little trouble (LawnApp::IsLittleTroubleLevel, mLevel == 25): conveyor seed bank, "
             "4x zombie points, half-scale zombies with 1/4 health"),
    },
    special_notes={
        25: "little trouble: HasConveyorBeltSeedBank, no deck economy, 4x zombie points",
        30: "mini-boss: HasConveyorBeltSeedBank, deck seeds the initial packets",
    },
    families=(
        Family(""),
        Family("_p9", P9_DECK),
        Family("_p9k", P9K_DECK),
        Family("_p9k", P9K_DECK, "train", exclude_levels=TRAIN_EXCLUDED_LEVELS),
    ),
    train_seed_base=TRAIN_SEED_BASE,
    train_seed_stride=TRAIN_SEED_STRIDE,
    train_seeds=TRAIN_SEEDS,
    reserved_seed_ranges=RESERVED_SEED_RANGES,
)


def main() -> None:
    main_for(POOL)


if __name__ == "__main__":
    main()
