# CUDA 阶段 1 语料：`artifacts/cuda/corpus_phase1/`（一次性报告，不属于记忆）

生成时间 2026-10-18 00:02 本地；工作区 `/home/newbiexvwu/pvz-cuda`（分支 `cuda-rollout`）。
语料**只在本地磁盘**，未入库（`artifacts/` 被 `.gitignore` 拦下，遵守 AGENTS.md §1）。

## 命令行

```bash
~/.venvs/ml/bin/python3 scripts/cuda_root_corpus.py \
  --manifest ~/PvZ-search/artifacts/stage4/day_l7_hard.json \
  --task-id day_l7_z15_train --seed-offset 11000 --seeds 10 \
  --roots-per-game 30 --root-stride 400 --rollouts-per-root 8 \
  --rollout-policy ~/PvZ-search/artifacts/student/v14/model.pt \
  --value ~/PvZ-search/artifacts/value_net/v2/model.pt \
  --out-dir ~/pvz-cuda/artifacts/cuda/corpus_phase1
```

单进程串行（脚本把 torch 与模拟器子进程都钉成 1 线程），墙钟 47.8 s，峰值内存 < 400 MB，未用 GPU。
种子取 `day_l7_z15_train` 的第 11000–11009 个（值 1611000..1611009）：
生产实验的评估种子是 30000–31023（`*_eval` 任务），采集用训练偏移 ≤ 8191，两处都不相撞。
学生沿用探针语料的 v14，为了与 `corpus_probe/` 可比。

## 产物

| 文件 | 字节 | 内容 |
|---|---|---|
| `roots.cudaroot` | 4 877 696 | 1 × LevelInfo(2420) + 276 × RootRecord(17652)，每条前置 12 字节 tag（PVC4 v4） |
| `corpus.jsonl` | 1 021 879 | 2208 行 = 276 根 × 8 个随机未来（CPU 参照叶子） |
| `roots.json` | 61 082 | 每根的 tick/wave/sun/实体计数 + 汇总 |

`scripts/cuda_layout_check.py` 对同一工作区通过（`layout OK: root 17652 bytes, level 2420 bytes, magic 50564334 v4`），
`python/pvz_cuda_root.py` 的读取器逐字节消费完整个文件，`roots.json` 与解析出的记录零分歧（276/276）。

## supported / 容量

- **276 根全部 `supported = 1`，0 根不支持**，`reason_histogram` 空。协议未改，`src/CudaExport.cpp` 未动。
- 容量实测（上限 → 最大使用）：plants 60→25、zombies 64→9、projectiles 96→8、coins 24→**0**、
  mowers 6→5、packets 10→7（卡组就是 7 张）、income events 160→34。
- 波次表：关卡 7 的 `mNumWaves = 30` **正好等于 `kMaxWaves`**，30 行全部导出且没有触发行数上限
  （`if (mNumWaves > kMaxWaves)` 不触发）；单波最多 24 个僵尸 < `kWaveZombieSlots = 32`。
  每根都带完整的 30 波未来表，10 张不同的表（每局一张），波内僵尸总数 159 上下。
- **没有任何根被截断**（截断即写 `supported=0`，见 `FillIncome` 5000 / plant 1000 / zombie 2000 /
  projectile 3000 / coin 4000 / wave 6000+wave / numWaves 6100 / packet 20000+type）。

## 分布

- 根 tick：min 1020 / median 12376 / max 23426；每 2000 tick 一档的根数
  `[1, 28, 15, 30, 32, 28, 30, 35, 36, 27, 9, 5]`（早段 <6000 共 44 根，中段 120 根，晚段 >14000 共 112 根）。
- 当前波：`{0:1, 1:25, 2:35, 3:33, 4:31, 5:29, 6:24, 7:32, 8:26, 9:40}` —— 只到第 9 波，
  因为 v14 学生在第 9–10 波就把局打完（赢或输），10–29 波只出现在导入的未来表里，不是任何根的当前波。
- 每局根数：8 局满 30 根上限，2 局因提前终局只有 18 根。
- 实体规模：植物 min 1 / 中位 14 / max 25；僵尸 min 0 / 中位 4 / max 9；3 根场上无僵尸。

## 非平凡根的覆盖（276 根里）

| 特征 | 根数 | 占比 |
|---|---|---|
| 植物与僵尸共存 | 273 | 98.9% |
| 种子槽在冷却（`active=0`） | 260 | 94.2% |
| 收入窗口 ≥ 20 个事件 | 229 | 83.0% |
| 受伤僵尸 | 217 | 78.6% |
| 空中有抛射物 | 171 | 62.3% |
| 修剪机已触发（`triggered_lawn_mowers > 0`） | 161 | 58.3% |
| 某行已被修剪（`mWaveRowGotLawnMowered` 是**波号** 3..9，非布尔） | 161 | 58.3% |
| 受伤植物 | 172 | 62.3% |
| 同一行 ≥ 2 个在场僵尸 | 142 | 51.4%（≥3 个 19 根，≥5 个 103 根） |
| 头盔仍有血（路障） | 127 | 46.0% |
| 僵尸正在啃植物 | 63 | 22.8% |
| 僵尸处于死亡中（无头/无臂） | 59 | 21.4% |
| 场上有坚果 / 土豆矿 / 双发射手 | 60 / 38 / 14 | 21.7% / 13.8% / 5.1% |
| 僵尸已过修剪机线（`pos_x < 40`） | 17 | 6.2% |
| **空中有阳光币** | **0** | **0%** |

状态机覆盖：僵尸 phase 直方图 `{0 普通 942, 1 DYING 66, 3 MOWERED 4, 11 撑杆起跳前 41, 12 跳跃中 6, 13 落地后 44}`；
48 根含撑杆跳跃过程的僵尸。抛射物 age 1..149，全部 type 0（豌豆），`chill` 恒 0。

CPU 参照叶子：`ticks` 163–2559（中位 2520）、`decisions` 3–51、`net_calls` 3–47、
`terminal` 50/2208（2.3%）且**全部是输局**（`result` 直方图 `{0:2158, -1:50}`，没有赢），
`plants_planted` 0–7，`mowers_alive` 0–5，`leaf_value` 0.0–0.993（中位 0.654）。
`events` 在 2208 行里都是空对象 —— 与脚本 docstring 说明的 `src/main.cpp` ROLLOUT 不写 counters 一致，
不能当成测得的零。

## 三个实测出来的弱点（阶段 1 之前该知道）

1. **阳光币通道本语料一条都没命中（coins 0/276）。** 原因在 CPU 侧而不是导出器：
   `LawnApp::EnvironmentWait`（src/LawnApp.cpp:1508-1514）每个 tick 开头把场上所有活的阳光
   `ScoreCoin()` 收掉，随后才 `AdvanceLogicTick`；根快照取在决策边界，于是永远在"已收完"的那一刻。
   `CudaExport.cpp:295-297` 把 `ticksToCollect` 硬编码成 1 正是这个语义。
   后果：GPU 的 coin 导入/结算分支只靠这条语料**测不到**。小改动可行的替代：
   造一条合成根（手工填 `RootRecord::coins`），或在语料模式下临时关掉这层自动收阳光；
   两者都不需要动现有协议字段。
2. **`RootProjectile::velX/velY` 是死字段。** 437 个抛射物的两个速度都精确为 0：
   `Projectile` 构造把 `mVelX` 归零（src/Lawn/Projectile.cpp:70），只有 lobed/star/homing 分支才写它，
   直飞豌豆走的是 `Projectile::Update` 末尾的 `else { mPosX += 3.33f; }`（约 :720）。
   内核**不能**从根里读速度，必须用 3.33 px/tick 这个常量（`cuda/SimTick.inc` 由另一个 agent 维护，
   建议核对该常量与 sim_skip 的乘积语义）。
3. **`RootZombie::targetCell` 恒为 -1，包括 88 个正在啃食的僵尸。** `mTargetCol/mTargetRow`
   在原代码里只有蹦极/僵尸先生投放路径写（src/Lawn/Zombie.cpp:1142, 1150, 9782+），
   啃食目标是 `Zombie::mTargetPlantID`（Zombie.h:149），导出器没有导出它。
   内核必须像原代码一样从几何重新推导被啃的植物；如果以后想省这一步，
   正确做法是**追加**一个字段（例如 `eatCell`），不要改现有偏移。

另记：`LevelInfo::zombies[34]` 只有前 33 项被填（导出循环上界是 `NUM_ZOMBIE_TYPES` = 33），
第 34 项整项为零、其 `type` 读到 0，遍历 `zombies[]` 而不看 `allowed[]` 的内核代码会把它当成普通僵尸。
本关卡 `allowed` 只有 {0 普通, 2 路障, 3 撑杆}，语料里的僵尸类型也只有这三种。

## 与探针语料的关系

`artifacts/cuda/corpus_probe/`（30 根 / 2 局，tick 3600..21334，projectiles 15/30、coins 0/30）
没有被改动，读取器仍可完整解析。阶段 1 语料把 tick 下界推到 1020（`--root-stride 400` +
`--seed-offset 11000`），并把带抛射物的根从 50% 提到 62.3%；`root_stride` 从 600 降到 400 是
让每局根数从 15 提到 30 的直接原因（每局的采样点变多，而"有合法可种"过滤只放过约一半的点）。

运行期间 `scripts/cuda_root_corpus.py` 被另一个 agent 加了 `--rollout-max-actions`（默认 3000，
与 `cuda/SimHost.cpp Options::maxActions` 对齐）。本次语料用的是没有该参数的版本，
续走时的实际上界是 `pvz_env.rollout` 的默认 `max_actions=1000000`；
实测每条未来最多 51 次决策，两个上界都远未触达，所以对拍语义等价，无需重跑。
