
---

## 结论（2026-10-27，Claude 接手）

### 1. 生成器与登记表

- `scripts/deck_rules.py`：v5 规则表（只禁零作用；白天蘑菇 ↔ 35 咖啡豆同进同出；白天 42 → 10 + 35；
  升级植物依赖底座；按关对空 / 冰车 / 巨人硬解；屋顶 26 / 22 解禁）。`deck_violations()` 供生成器与测试共用。
- `scripts/make_random_deck_cells.py`：`draw_seed = 137`，白天 / 夜间 / 白天泳池 / 雾夜泳池各 2 关
  （L7、8 / 12、14 / 24、26 / 33、37）× 2 套训练 + 1 套评估 = **16 训练格子 + 8 套评估卡组**。
- `configs/random_deck_cells_v5.json` 进 git（24 套卡组 + draw_seed），卡组可由 `(level, length, draw_seed)`
  逐位重放；训练与评估签名互不重叠，也不与记分板 / 挑选 / 示范已登记卡组重复。
- 面板是派生文件：`artifacts/task137/training_panel.json`（144 任务 = 16 格 × 9 档）与 `eval_panel.json`
  （8 套，无种子），不入 git。

### 2. 引擎的 <1 倍率：支持，但有粒度

`src/Lawn/Board.cpp:4918`（`Board::SpawnZombieWave`）：
`aSpawnCount = std::lround(aBaseCount * multiplier)`。TaskSpec 允许 0.1–10（`LawnApp.cpp:1334`）。
所以 0.5 / 0.75 合法；但 `lround(1 × 0.5) = 1`，**单只波不会变少**，只有 ≥2 只的波才减半。
坟头 / 泳池 / 屋顶的收尾额外刷怪不读倍率（原版行为）。因此不需要改规则、也不需要额外给阳光。

实测（同一关、同种子、同一套 v5 卡组，不放植物，前 8 波；脚本 `scripts/smoke_random_deck_cells.py`）：

| 关卡 | 卡组 | 1.0 档每波 | 0.75 档 | 0.5 档 | 断言 |
|---|---|---|---|---|---|
| L7 白天 | `[1,35,17,4,21,12,15]` | 全 1 | 全 1 | 全 1 | 4 对 |
| L12 夜间 | `[34,6,20,13,22,7,1,8,23]` | 1×4, 2×4 | 1×4, 2×4 | 1×8 | 8 对 |
| L24 泳池 | `[1,43,39,26,4,16,8,35]` | 1×5, 2×3 | 1×5, 2×3 | 1×8 | 10 对 |
| L33 雾夜 | `[7,9,17,32,26,27,16,10,5]` | 1×4,2,3,2,2 | 1×4,2,2,2,2 | 1×4,2,1,1 | 10 对 |

全部等于 `lround(原版 × 倍率)`；四套卡组都被引擎接受（reset 成功、对局推进）。
L7 的早期波都是单只，所以 0.5 在 L7 前 3 波没有效果 —— 低档的收益主要在后段波与高关。

### 3. 课程接入与每轮开销

`curriculum_queue.py --extra-manifest <training_panel.json>`：

- 16 个格子并入现有训练集合，格子阶梯取自清单的 `curriculum_ladder`（0.5 起，到 3.0），
  沿用现有 0.85 / 0.5 升降规则；`mult_index = 0` 时不再降。
- 每轮给随机卡组格一份独立配额 `--random-deck-share`（默认 0.15）：生产参数 768 局/轮、48 局/组，
  保底组 24 局 → **4 格 × 24 = 96 局/轮（12.5%）**，16 格约 4 轮轮完。原有课程每轮少 96 局
  （从 768 里出，不额外加机器时间；按现吞吐约 +0 分钟，只是把 96 局从旧格挪给新格）。
- 随机卡组格在按名字排序的保底轮转里会排在最后（`rd137…`），所以单独做了一轮预留配额，
  否则永远轮不到（已有单测覆盖）。
- 去掉 `--extra-manifest` 即回滚：格子按现有规则退休，`retired_cells` 保留旧窗口与偏移。

### 4. 挑选与门禁

- 挑选面板 `selection-task137-v5`：把 3 个 `*_train_rand`（本身是训练族，等于拿训练过的卡组评估）
  换成这 8 套**没进训练**的 v5 评估卡组；38 冒险 + 8 随机卡组 + 2 score-only 换卡组 = **456/512 局**
  （原 416），不需要改种子块大小。候选的 `selection_random_deck` 胜局单独写进 `candidates.json`。
- **门禁 v5 不动**：`configs/gate_v5_panel.json` 里没有随机卡组族（40 个任务 = 38 冒险 + 2 个固定
  留出监控格），两个监控格本来就是训练外固定卡组。把 8 套评估卡组加进监控要每轮多 8 × 96 = 768 对
  （门禁 +40% 时长、种子池消耗 +40%），本任务不做；需要的话另开任务改门禁版本与合同。

### 5. 部署 `scripts/qc_deploy14l.sh`（未执行，交主控）

- 替换 5 个文件：`scripts/curriculum_queue.py`、`python/pvz_selection_refresh.py`、`scripts/deck_rules.py`、
  `scripts/make_random_deck_cells.py`、`configs/random_deck_cells_v5.json`。
- 未激活 / 回滚分支的 `start_qc` 逐字照抄 `qc_deploy14k.sh`（含 deploy15 `v2sleep` 分支），只在激活时
  追加 `--extra-manifest`。
- **需要重启采集一次**：控制器在启动时读清单，`--extra-manifest` 只能在启动参数里给；走既有 `stop_qc`
  的组边界排空（不 SIGTERM、不丢组）。后台训练 / 打包流水线**不需要排空**。
- 回滚 = 删 `~/pvz-logs/qc_deploy14l.mode` + 恢复 5 个文件 + 不带 `--extra-manifest` 重启，
  state / offsets / 账本全部保留。
- 部署前检查很便宜：控制器 sha256（未变，仍为 admitfix cf94a8cb 的 72e57…）、选择器 sha 属于
  {deploy14k 版, admitfix 版} 之一、5 个新文件与 REV 的 sha256 一致、`--check` 重放登记表、
  面板计数 144 / 16 / 294912 个种子。全是本地哈希与纯 Python，不跑引擎、不读全量清单。

### 6. 测试（Mac）

- `python/test_random_deck_cells.py`：11 项全过（规则、可复现、训练 / 评估不重叠、面板阶梯与种子段）。
- `python/test_curriculum_queue.py`：新增 8 项随机卡组格测试全过；全文件 79 项里只剩
  `test_pack_memory_limit_matches_build_plan_per_builder` 失败 —— 该失败在 `origin/selinc` 上同样存在，
  与本任务无关（打包内存系数）。
- `python/test_selection_holdout.py` 8 项全过（更新了任务数与 456 局）。
- `python/test_student_selection.py::test_select_picks_the_best_total...` 在 macOS 上因 `/private/var`
  路径规范化失败，`origin/selinc` 上同样失败，与本任务无关。

### 7. 冒烟

4 个关卡各 1 个引擎进程串行（≤ 4 进程上限），证据在 `artifacts/task137/smoke_l{7,12,24,33}.json`（不入 git）。

### 8. 遗留

- v5 文档提到的 `asleep` 输入缺口（学生 / 价值网看不到睡着的蘑菇）本任务不涉及，仍按原计划单独排期。
- 低倍率档的收益集中在中后段波；如果 0.5 档在 L7 / L8 仍然输，按现有规则会停在 0.5 而不是更低 ——
  需要更细的难度时再评估“开局阳光”或 `wave_cap`（都会改任务语义，要另开族）。

