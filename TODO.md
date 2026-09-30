# PvZAgent — 执行任务书

本文件是**唯一的任务来源**，覆盖并取代此前所有 TODO / 计划文档。
设计依据见 `DESIGN.md`（必须通读后再动手）。失败复盘见 `FAILURE_ANALYSIS.md`。
**先看 §1 铁律和 §2 门禁规则，再看 §4 任务列表。**

---

## 0. 目标

训练出一个能自主玩 Plants vs. Zombies 的 Agent。

**唯一的成功标准是打赢。** 本项目不存在"先跑通一个最小版本"这种中间产物——一个打不赢的系统和一个不存在的系统价值相同。最终产物必须在**冻结的、多关卡的 held-out 任务集**上达到 **≥ 60% 通过率**（见 T8）。

环境基线与既有资产：

- 资源：PvZ GOTY English `1.2.0.1073`，不入库，通过 `PvZEnv(resource_dir=...)` 指定。
- 本机资源目录：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`；Python 用 mise 3.14。
- 冻结 seed 集（`artifacts/adventure2_level7/seeds/`）：train 0–63、dagger 10000–10063、
  value_bootstrap 20000–20031、value_refinement 21000–21031、development 30000–30255、final_test 40000–41023。
  两两互斥。
- **实测吞吐**：243,893 game-ticks/s；40,000 tick 的完整胜局约 0.16 s；单核约 20,000 局/小时。
  对照：`SearchTeacher.advice()` 一次决策约 150 ms——**搜索比环境推进慢约三个数量级**。
  这条实测是本任务书全部架构决策的依据。

---

## 1. 铁律（不可协商，违反即视为任务未完成）

1. **进度以能力为准，不以 loss 为准。** 任何"loss 下降了 / 准确率提升了"都不是进度证据。
   进度只由 §2 定义的行为门禁判定。
2. **门禁不过就是不过。** 禁止修改门禁定义、禁止放宽阈值、禁止更换评估 seed 集来让门禁通过。
   门禁失败时按 §3 处理，不得继续下一个任务。
3. **不得硬编码策略。** 判据：**这条规则如果错了，梯度能修正它吗？**
   能修正 → 是归纳偏置，可以写进网络结构；不能修正 → 是硬编码，禁止。
   特别禁止：针对特定关卡 / 特定 seed 的分支逻辑（`if level == 7` 之类）、
   写死的种植顺序与时机、写死的列号偏好。
4. **行为来源只有强化学习。** 规则脚本只允许做 T0 的环境可解性验证，**不得进入任何训练数据、
   不得参与蒸馏、不得作为行为来源**。
5. **禁止删除或跳过失败 seed。** 评估必须跑满全集；单个 seed 崩溃必须报出来并修好，
   不得从结果里剔除。
6. **禁止用 dev 集调参后又把 dev 成绩当作泛化成绩报告。** 模型选择只能用训练集与 dev 集，
   泛化结论只能来自 held-out 集。
7. **每个任务必须留下可复核的证据**（§2.3）。没有证据文件的任务视为未完成，
   无论口头声称完成与否。
8. **禁止写未经实测的结论。** 不得出现"应该可以工作"、"预计能够"这类表述。
   每个断言必须附命令与实测输出。

---

## 2. 门禁规则

### 2.1 什么算通过

一个任务通过，必须同时满足：

- 该任务列出的**每一个**量化指标都达标（不是"大部分达标"）；
- 指标由**冻结评估集 + 独立评估脚本**产出，不是临时脚本、不是手算、不是抽样估计；
- 证据文件 `gates/<task_id>.json` 已落盘且内容完整（§2.3）；
- 受保护资产（§2.4）未被修改，并在证据文件中声明。

### 2.2 指标定义

- **pass rate**：在指定任务与 seed 全集上 `won == True` 的比例。分母是全集，不是成功跑完的子集。
- **Wilson 95% 区间**必须与 pass rate 一并报告，样本量 < 64 时结论不采信。
- **完整分布**：必须记录终局波次/存活 tick 的直方图，不只报均值。均值会被少数长局拉高。
- **泛化差距**：同一 checkpoint 在训练集任务与 held-out 任务上的 pass rate 之差。
  这是**防硬编码的核心指标**——差距过大说明在背题。

### 2.3 证据文件

每个任务完成后写入 `gates/<task_id>.json`，至少包含：

```json
{
  "task_id": "T7",
  "commit": "<git rev-parse HEAD>",
  "worktree_clean": true,
  "reproduce_command": "<一条可复现的完整命令>",
  "gate_result": "pass | fail",
  "metrics": { "...": "门禁要求的每个指标的实际值" },
  "thresholds": { "...": "门禁要求的阈值" },
  "raw_seed_results_path": "<完整 seed 级结果的路径>",
  "protected_assets_unmodified": true,
  "notes": ""
}
```

`worktree_clean` 必须为 `true`（先提交再评估）。`protected_assets_unmodified` 必须为 `true`。

**证据必须回到远程**：写完证据文件后 `git commit` 并 **`git push origin pvz-env`**。
执行机与审阅机不是同一台（见 §5.1），不推送的证据等于不存在——无法核验、也无法据此放行下一任务。
`commit` 字段填的 HEAD 必须是**已推送到 origin** 的那个。

### 2.4 受保护资产（禁止修改，除非任务明确要求）

- `artifacts/**/seeds/**` — 全部冻结 seed 集
- `gates/**` — 已通过的门禁证据
- 评估脚本的**判定逻辑**（可以优化性能，不得改变指标定义与阈值）
- `DESIGN.md`、`FAILURE_ANALYSIS.md`、本文件

若确有必要修改受保护资产，必须先在 `notes` 中说明理由并取得人工确认，**不得先改再报**。

### 2.5 变异测试要求

凡新写的自动化检查（尤其是 T1 的冒烟测试），必须证明它**真的会失败**：
人为注入一个已知缺陷，测试必须报错。测不出来的检查等于没有检查，不算完成。

---

## 3. 门禁失败时的处理

**立刻停止，不要继续下一个任务。**

1. 把当前状态、实测数字、已尝试的方法写进 `gates/<task_id>.json`，`gate_result` 填 `fail`。
2. 写一份诊断：定位到**具体的失效层**（环境 / 数据 / 算法 / 工程），给出实测证据，
   不要写"可能是 X 的问题"这种无证据的猜测。
3. 提出下一步的**具体**改动，并与现状对比说明为什么这个改动能突破当前瓶颈。
4. 等人工确认后再继续。

上一轮的教训写在这里：**教师 dev 0/256 之后没有停下，继续跑了 DAgger 和最终验收，
把几小时算力投在已知 0 胜的基座上。** 同样的错误不允许再犯一次。

允许重试，但连续 3 次未达标后必须停下来诊断，不得靠"再跑一遍希望这次行"推进。

---

## 4. 任务列表

任务按顺序执行。**前一项门禁未通过，不得开始后一项。**

---

### T0 · 修复胜利判定（L0）

**目标**：让"赢"在环境层变成可达的。

**根因**（已定位，直接用）：`LawnApp::EnvironmentReset` → `StartLevelIntro()` → `CancelIntro()`
→ `PlaceStreetZombies()` 放置开场装饰僵尸（`mFromWave == ZOMBIE_WAVE_CUTSCENE`）。
`Zombie::IsOnBoard()` 对它们返回 false，但 `Board::AreEnemyZombiesOnScreen()`
（`src/Lawn/Board.cpp:261`）**不检查 `IsOnBoard()`**，导致恒为 true，
而唯一通关判定 `Zombie::TrySpawnLevelAward()` 要求 `!AreEnemyZombiesOnScreen()` → 永不成立。
同文件 `CountZombiesOnScreen()`（约 275 行）才是带 `IsOnBoard()` 的正确版本。

**做法**：优先给 `AreEnemyZombiesOnScreen()` 补 `IsOnBoard()` 条件，与 `CountZombiesOnScreen()` 对齐。
备选：`EnvironmentReset` 在 `CancelIntro()` 后 `RemoveAllZombies()`。

**T0 验证的是环境，不是脚本强度。** 规则脚本只是一个难度探针：它赢几个 seed 无所谓，
**它是否会被"打不完的僵尸"卡住才有关系**。一个 40 行的手写脚本在 30 波完整关卡上输掉几个 seed
是完全正常的，不构成环境缺陷。

**门禁（全部满足）**：

| 指标 | 阈值 |
|---|---|
| level 7 / seed 30000 | `won == True`，且终局 tick ≤ 120,000 |
| level 7 / seed 30001、30002、30003 | **全部正常终局**（`won` 或 `lost` 均可），终局 tick ≤ 120,000 |
| 每个失败 seed | 必须记录终局 tick 与终局原因，并确认是**正常游戏失败**（防线被突破），不是"有僵尸却打不完" |
| level 1 / 8 / 20 各 1 个 seed | 能终局（胜负均可），终局 tick ≤ 120,000 |
| 直接环境断言 | 获胜局终局时，场上不存在 off-board 僵尸参与胜负判定（提供观测证据，不靠脚本结果反推） |

**明确禁止**：为了让脚本多赢几个 seed 而去调优脚本。原因有三——
脚本是探针不是产品，调优它不产生任何后续价值；调优方向会滑向针对关卡的硬编码，违反铁律 3；
后续 RL Agent 的能力与脚本强度无关。**花在这上面的算力全部是浪费。**

若某个 seed 未能终局（tick 远超阈值仍未结束），那才是环境缺陷，必须修环境而不是改脚本。

**交付**：C++ 改动 + 复现命令 + `gates/T0.json`

---

### T1 · 「能赢吗」冒烟测试进 CI

**目标**：把 T0 的结论固化成一道永久闸门。这个测试比任何胜率指标都更早、更便宜地拦住同类问题。

**门禁**：

- 一条命令即可运行，失败时非零退出。
- **变异测试**：临时回退 T0 的修复，测试**必须失败**；恢复后必须通过。
  把两次运行的输出都贴进证据文件。测不出回归的检查不算完成。
- 测试覆盖：至少 3 个关卡 × 2 个 seed，断言 `won == True` 且终局 tick 在合理范围内。

**交付**：`gates/T1.json`（含变异测试的两次输出）

---

### T2 · env 增加 `wave_cap` 与 `preplanted`

**目标**：给课程与能力拆解提供物理前提。**没有 `wave_cap`，后面所有任务都无法做。**

**下面的语义与实现规范已经定死，按此执行，不要自行设计。**

#### 2.1 语义

**`wave_cap: int | None`** — 本次对局的波数上限。

- 对局只生成 `min(mNumWaves, wave_cap)` 波；最后一波自动成为 final wave（有旗帜），走**正常通关判定**。
- `None` 或 `0` = 不截断，等于现有行为。
- **只允许截断，不允许放大**：`wave_cap > mNumWaves` 时取 `mNumWaves`，不得扩展波表。

**`preplanted: tuple[(seed_type, row, col), ...]`** — 开局预置植物。

- 按列表顺序逐株种植；**不消耗阳光、不消耗卡片、不触发冷却、不受 deck 限制**。
- **不自动配底座**：泳池/屋顶需要 LilyPad / FlowerPot 时，调用方必须在列表里显式先声明底座、
  再声明植物。
- 任一株因地形非法无法种植 → **整次 reset 失败并报错**，不得静默跳过。

#### 2.2 实现规范

**C++ 侧**

1. `EnvironmentTaskSpec` 增加 `waveCap`（`0` = 不限）与预置植物列表。
2. `wave_cap` 的实现**必须同时满足 (a) 和 (b)，缺一不可**——只做 (a) 是错的。

   **(a) 截断总波数**：在 `Board::PickZombieWaves()`（`src/Lawn/Board.cpp:575`）中，
   `mNumWaves` 赋值段（581–617）之后、`PVZP_ASSERT`（617 行）之前、波次生成循环（621 行起）之前：

   ```cpp
   const int aFullWaves = mNumWaves;                                  // 先存完整波数
   if (mWaveCap > 0 && mNumWaves > mWaveCap) mNumWaves = mWaveCap;    // 后截断
   ```

   **(b) 波次组成必须用完整波数计算**（第一版实现漏掉的关键点）：
   循环内所有依赖"总波数"的判定改用 `aFullWaves`，**不是** `mNumWaves`：

   - `aIsFinalWave`（630 行）：`aWave == aFullWaves - 1`
   - 新僵尸引入判定（706–711 行）的 `aWave == mNumWaves / 2`：`aWave == aFullWaves / 2`
   - `IsFlagWave(aWave)`（528 行）内部 `GetNumWavesPerFlag()`（525 行）：按完整波数取

   **为什么必须做 (b)**：只做 (a) 会让第 `wave_cap` 波变成最终旗帜波——带旗帜、
   触发新僵尸引入的大波。实测后果：level 7 / seed 30000 在完整 30 波下能赢（tick 65,401），
   在 `wave_cap=3` 下反而于 **tick 11,777 被打穿**。原因是发育时间从约 75,000 tick 砍到
   约 7,500 tick，却要立刻面对终局大波——**短任务比长任务还难，课程设计直接失效。**

   `wave_cap` 的正确语义是**"打满前 N 波就结束"**，不是**"把整关压缩成 N 波"**。

3. 预置植物用 `Board::AddPlant(col, row, seedType, SeedType::SEED_NONE)`
   （已有接口，`Board.cpp:2079`），在 `EnvironmentReset` 完成后、返回观察前执行。
   **不要走 `EnvironmentPlant`**——它会 `packet.Deactivate()` 消耗卡片并进入冷却。

**协议**：`RESET_V2` 末尾追加 `{wave_cap} {preplanted}`。

- `wave_cap`：整数，`0` = 不限。
- `preplanted`：逗号分隔的 `type:row:col`，`-` = 无。
- 注意 `src/main.cpp:303` 的 `if (input >> extra) parsed = false;` 多余参数检查——
  扩展后必须把它移到新参数**之后**，否则新参数会被判为多余而导致解析失败。

**Python 侧**：`TaskSpec` 增加 `wave_cap: int | None = None` 与
`preplanted: tuple[tuple[int, int, int], ...] = ()`；`reset()` 校验
（`wave_cap` 为 `None` 或 1–50；每项 `seed_type ∈ [0,48]`、`row ∈ [0,5]`、`col ∈ [0,8]`）。

#### 2.3 门禁（全部满足）

| 指标 | 阈值 |
|---|---|
| **波次组成一致性**（验证 (b) 确实做对了） | `wave_cap=3` 的第 1/2/3 波僵尸组成，与 `wave_cap=None` 同一 seed 的前 3 波**逐波一致**；不一致即判失败，即使恰好能赢 |
| level 7 `wave_cap=3`，3 个 seed | 第 3 波后**正常通关**（`won == True`），终局 tick 应 < 30,000 |
| level 7 `wave_cap=1` | 第 1 波后终局 |
| level 7 `wave_cap=None` | 仍为 30 波，与改动前一致 |
| `preplanted` 2 株向日葵（`seed_type=1`） | 开局植物数 = 2，位置精确匹配，`won` 不受影响 |
| 泳池关卡无水莲处预置豌豆 | **reset 失败并报错**，不得静默成功 |
| **回归** | 不指定新参数时 level 7 / seed 30000 结果与 T2 之前一致 |
| `zombie_count_multiplier` | 保持 [1.0, 10.0]，不得放宽下界 |

**禁止**：

- 不得把 `wave_cap` 实现成"到达波数后强行判负"——必须是正常通关判定（会 `won`）。
- 不得自动补 LilyPad / FlowerPot。
- 不得静默跳过种植失败。

**交付**：`gates/T2.json`

---

### T3 · 任务族定义与冻结任务集

**目标**：把"一个关卡"换成"一个任务族"。

**做法**：任务由 `θ = (level, deck, zombie_count_multiplier, wave_cap, sun_start, preplanted)` 描述。
定义并冻结：

- **训练任务池**：可枚举、可变异生成。
- **held-out 任务集**：**从未参与训练**，至少覆盖白天 / 夜晚 / 泳池 / 浓雾 / 屋顶五种地形，
  每地形 ≥ 4 个任务，每任务 ≥ 16 seed。

**门禁**：

- 任务 manifest 落盘（JSON），含每个任务的完整参数与 seed 列表。
- 训练池与 held-out 集**零重叠**，用脚本断言（不是肉眼检查）。
- 现有冻结 seed 集（development / final_test）与 held-out 集的关系在 manifest 中明确声明。

**held-out manifest 一经落盘即纳入 §2.4 受保护资产**，后续任何任务不得修改——
它是全部泛化结论的基准，改它就等于改答案。

**交付**：manifest 文件 + 重叠性断言脚本 + `gates/T3.json`

---

### T4 · 能力剖面评估器

**目标**：建立唯一的进度度量。这个评估器是后面所有门禁的执行者，必须先建好。

**做法**：对给定 checkpoint 与任务集，产出：

```
per_task: { pass_rate, wilson_95, mean_terminal_wave,
            mean_survival_ticks, peak_offense, economy_curve }
aggregate: { train_mean_pass, heldout_mean_pass, generalization_gap }
```

**门禁**：

| 指标 | 阈值 |
|---|---|
| 随机策略在 `wave_cap=3` 任务上 | pass rate ≤ 10%（评估器必须能识别"什么都没学会"） |
| T0 规则脚本在 `wave_cap=3` 任务上 | pass rate ≥ 80%（评估器必须能识别"会玩"） |
| 同一 checkpoint 两次运行 | 结果一致（确定性） |
| 单 seed 崩溃 | 必须报错退出，不得静默跳过 |

**这两个对照是防"评估器写错了但看起来在跑"的关键。** 两个都测不出来，评估器不合格。

**交付**：`gates/T4.json`（含随机策略与规则脚本两组对照数字）

---

### T5 · 并行 rollout 与 PPO 闭环

**目标**：跑通"环境 + 策略 + 更新"的完整闭环，并证明它能学到东西。

**做法**：

- 并行环境：`multiprocessing` + `spawn` 启动（不得 fork）。每 worker 独立创建 env。
- 可恢复：逐 seed / 小 shard 原子落盘（复用 `pvz_seed_jobs.atomic_write`）。
- 算法：PPO + GAE。奖励 = 终局胜负 + **potential-based shaping**（`F = γΦ(s') − Φ(s)`，
  Ng/Harada/Russell 1999）。**不得使用任意常数惩罚**——当前 `_transition_reward` 那种
  0.05/株的写法会污染目标，必须替换而不是沿用。

**实现约束（T5 专用，防止设计漂移）**：

- **动作空间沿用现有的 factored 分布**（type → packet → cell / wait 三档）。
  意图空间与分层是 T6 的事，T5 不做——T6 的门禁需要 T5 的扁平架构作对照基线。
- **critic 允许使用特权信息**（`privileged_state` 的 wave_timer / zombies_in_wave），
  即不对称 actor-critic：训练时 critic 可以看执行时看不到的信息，这是标准且合法的做法。
  `DESIGN.md` §4.4 中"privileged 拿去当 critic 是错用"的表述**在此修正**——错用的是
  学生 value 头的蒸馏标签来源，不是 PPO critic 本身。actor 侧仍然只用普通观测。
- **Φ 的具体形式允许调整**（如 `sun/1000 + 火力血量占比 + wave/wave_count` 的加权），
  因为势函数形式在理论上不改变最优策略；但**必须在证据文件中记录实际使用的 Φ**。
- **不得把规则脚本的任何轨迹、动作或候选排序混入训练数据**（铁律 4）。
  脚本只作为 T4 已建好的评估对照。
- **T4 的评估器是 T5 门禁的唯一裁判**，训练中不得修改它（已冻结）。

#### 无人值守执行细则（本节是指令，逐条执行）

**执行边界**：只做 T5。门禁通过、或触发下方任一硬停止条件后，立即停止并等待确认，**不得开始 T6**。

**机器与并发（台式机执行）**：

- 机器：**台式机**（i7-12700F + RTX 5080，WSL Ubuntu）。代码用 git 同步到台式机仓库后执行，
  产物落在台式机 `artifacts/`；资源目录与本机不同，见 §5。
- 训练与推理设备：`--device cuda`。
- worker 用 `spawn` 启动。**必须实测 `workers × torch_threads` 组合**（建议起点：
  6 workers × 2 threads、8 × 2、12 × 1）再定正式配置——x86 上多线程有约 2× 收益，
  但多 worker 叠加多线程会发生超订阅，组合必须实测而不是照抄。
- 吞吐实验数据留作门禁证据。

**吞吐陷阱（第一轮实测踩到，务必先看）**：rollout 快不等于训练快。
第一轮实测 rollout 达 82,472 局/小时（10 workers），但**端到端只有约 1,000 局/小时**——
每 100 局就做一次完整序列 BPTT，更新耗时约 350 秒而 rollout 仅约 5 秒，**更新占 95%**。
按该速度两小时仅训 2,000 局，离 20,000 局判定点还需 20 小时。

因此开工前**必须实测并报告 rollout / 更新的耗时拆分**（写进证据文件）。
这一条修正了此前"T5 锁定 Mac"的决定：那个判断基于"rollout 占大头"的假设，
而本机实测是更新占 95%。

**但"更新占 95% → 批量矩阵运算 → RTX 5080 强项"这一步推理是错的，已推翻**——见 `MEMORY_BUDGET.md` §3.3/§6。
更新慢的真实原因是 **batch=1 串行循环 + 每个 epoch 重新做 Python 侧 token 化**
（`model.step` 里 `x.unsqueeze(0)`、hidden 写死 batch=1；`model.step` 每次都从 42 KiB 的
observation dict 重建张量）。这种负载下 GPU 大部分时间在等 kernel launch 与 CPU→GPU 搬运。
i7-12700F 单核约为此机一半，**若 GPU 红利不成立，台式机未必更快**——必须实测，不得再靠推理定机器。

正确顺序是**先做 batch 化，再选机器**。当前台式机那轮跑完不打断（它验证的是学习信号，与吞吐无关），
但下一轮开始前必须先完成 `MEMORY_BUDGET.md` §3.1–§3.3。

无论实测结果如何，两条硬要求不变：

- **每次 PPO 更新的样本量不得低于 2,000 局**（避免更新频率过高把开销放大）。
- 用 **truncated BPTT（8–16 步窗口）** 控制序列反向传播成本。

**执行边界（本次无人值守）**：完成 T5 的**阶段 0 验证门 + 阶段 1 正式训练**后停止等待确认，
**不得开始 T6**。T6 要重写网络结构（分层 + 意图空间），需要设计判断，不允许无人值守推进。

**课程起点**：先用 `train.json` 中 `wave_cap=1` 且 `multiplier=1.0` 的 5 个任务开训——
最短、最易、学习信号最密。确认 pass rate 明确上升后再扩展到全部 20 个任务。
不要一上来就在混合难度（含 2.0×、cap5）上训练，那会稀释早期信号。

**阶段 0：学习信号验证门（必须先过，再进正式训练）**

在投入大规模训练前，先用**最便宜的实验**证明 RL 链路真的能学到东西。
目的是避免在错误配置上烧掉几十万局才发现什么都没学会。

- 任务：`train.json` 中 `wave_cap=1` 且 `multiplier=1.0` 的 5 个任务（最短、最易、信号最密）。
- 预算：**10,000 局以内**（按修复后的端到端吞吐约 1–2 小时）。
- 判据：评估集 pass rate 由训练前基线（< 5%）**上升到 > 50%**。
- 通过 → 进入正式训练（全部 20 个任务，门禁 90%）。
- **不通过 → 停止并按 §3 写诊断。** 不许用"样本还不够"当理由继续加量：
  在最简单的任务上 10,000 局仍学不动，问题一定出在奖励 / 塑形 / 探索 / 网络四者之一，
  继续加样本只是把同样的失败放大。

这个门的价值：用 1–2 小时买一个确定性，而不是赌几十万局。

**网络要求（阶段 0 之前完成）**

验证前先补两处观测编码改动。二者都是低风险、高收益，且不改变训练流程：

1. **显式派生特征**——不是硬编码策略，是**状态的可计算函数**：
   - 每行：僵尸威胁度、最近僵尸距离、该行射手数、该行植物总血量
   - 全局：阳光收入速率（滑动窗口）、经济植物数与火力植物数之比、波次进度、距下一波进度
2. **lane 级聚合 token**——每行为一个 token 聚合该行实体，让注意力直接在"行"这一层发生。

这两条属于归纳偏置（错了梯度能修正），不是硬编码。它们把模型原本要从 54 个 cell token 里
自己学出来的结构直接摆出来，能显著降低学习难度。

容量：若在台式机（RTX 5080，16 GB）执行，width 可由 192 提到 256–384，层数 4 → 6。

**任务集（冻结，不得更改）**：

- 训练：`artifacts/task_family/train.json` 全部 20 个任务（自带难度阶梯：
  cap1/1.0×5、cap3/1.5×5、cap3/2.0×5、cap5/1.0×5），按各任务近期 pass rate 做自适应采样调权。
- **门禁评估集**：`heldout.json` 中 `wave_cap=3` 且 `zombie_count_multiplier=1.0` 的**全部**任务
  （每任务 64 seed）。这个子集与 T4 脚本对照（89.06%）同难度，90% 的阈值是参照它定的。
- 参考项（**报告但不设阈值**）：heldout 全部 `wave_cap=3` 任务（含 1.5×）。1.5× 的成绩供 T6/T7 参考。

**调参纪律**：

- 允许调整：学习率、GAE λ、entropy 系数、PPO clip、Φ 权重、worker 数、rollout 批大小。
- **正式训练运行最多 8 次**。每次必须记录：完整超参、本次改动动机、与上一次的学习曲线对比。
  第 8 次后无论结果如何，停止并汇报。
- 禁止：修改门禁、评估器、任务定义、冻结 seed；禁止为过门禁把 Φ 调到扭曲的量级——
  Φ 在理论上不改变最优策略，但扭曲的 Φ 会让训练本身不稳定，那是实现问题，不是捷径。

**训练进度判读（防止无效硬堆烧算力）**：

- 学习曲线每 30 分钟落盘一次（时间戳、累计局数、各任务 pass rate）。
- **硬停止条件**：累计训练满 20,000 局后，评估集 pass rate 仍 < 20%，且最近 5,000 局
  上升不足 5 个百分点 → 停止训练，按 §3 写诊断。**不许"再跑一轮看看"。**
- 曲线已明显平台化但低于门禁 → 同样停止诊断。

**常见失败模式的诊断顺序**（按此排查，不要跳步）：

1. pass rate 卡 0%：先查奖励信号——终局 ±1 是否真的进入回报、Φ 塑形项是否在量纲上淹没主信号、
   advantage 是否做了归一化。
2. 学到了但低于 90%：把失败 seed 的终局波次分布与脚本失败 seed 对比，判断是同一批难 seed
   （可接受）还是策略缺陷（需改进）。
3. 训练崩溃 / NaN：查梯度裁剪是否生效、value loss 与 policy loss 的量级比例。
4. 吞吐不足 5,000 局/小时：profile 环境推进与模型推理各占多少，用数据说话，不要猜。

**门禁**：

| 指标 | 阈值 |
|---|---|
| **端到端训练吞吐**（rollout + PPO 更新，**含更新**） | ≥ 5,000 局/小时（实测，不是估算） |
| rollout 吞吐（单独测量，仅作参考） | ≥ 20,000 局/小时（10 workers 实测 82,472） |
| `wave_cap=3` 任务，训练前 | pass rate < 10%（**T4 已实测 0%**，直接引用 T4 证据，不必重测） |
| `wave_cap=3` 任务，训练后 | pass rate **≥ 90%**（评估集 = 上述冻结的 held-out 1.0× 子集，≥ 16 seed） |
| 学习曲线 | 必须给出 pass rate 随训练局数的序列，**必须上升** |

**注意**：吞吐门禁和胜率门禁都要。吞吐不达标说明工程有问题，胜率不达标说明算法有问题，
不要用其中一个掩盖另一个。

**交付**：`gates/T5.json`（§2.3 字段齐全）+ 学习曲线数据文件 + checkpoint 与完整超参 + 吞吐实验数据 + 逐 seed 结果

---

### T6 · 分层网络与意图动作空间

**目标**：解决 horizon 问题（900 tick 跨不过一波），并消除零 tick 退化。

**做法**（详见 `DESIGN.md` §4）：

- **Strategist（慢）**：每波或每 K 决策运行一次，输入全局 + lane 摘要 + 上一波结果，
  输出目标向量 `g` 与每行资源权重，GRU 跨波记忆。
- **Tactician（快）**：每决策运行，输入全 token + `g`，输出意图分布，GRU 短程记忆。
- **意图空间**：`BUILD_ECONOMY` / `BUILD_OFFENSE(lane)` / `FORTIFY(lane)` /
  `ANSWER_THREAT(lane)` / `SHOVEL(cell)` / `WAIT_UNTIL(event)`。
  执行器把意图映射到合法具体动作——**执行器是确定性的，不含任何策略判断**。
- **值头**：`P(win)` 分类 + `E[剩余波数 | 会输]` 回归，取代当前的绝对 MSE 回归。
- **辅助头**：`lane_threat` / `next_spawn` / `wave_timer`，**用特权信息训练，推理时丢弃**。
  当前 `GameplayModelV1.privileged_value` 把特权信息拿去当 critic 是错用，改掉。

**门禁（消融对比必须做）**：

| 指标 | 阈值 |
|---|---|
| `wave_cap=5` 任务，分层架构 vs 扁平架构（同等训练预算） | 分层 **≥** 扁平，且差异有 Wilson 区间支撑 |
| 零 tick 退化 | 单位时间内的"种后即铲"次数较当前教师**下降 ≥ 80%**（当前基线：79% 阳光被自己铲掉） |
| 推理成本 | 单决策前向 ≤ 10 ms（batch=1，CPU） |

**注意**：T6 的门禁是"不比扁平差"，不是"必须大幅提升"。若分层架构更差，**如实报告**，
这是重要发现，不要粉饰。

**交付**：`gates/T6.json` + 消融对比数据

---

### T7 · 完整关卡

**目标**：从短任务推进到完整关卡（30 波）。这是**第一个高能力门禁**。

**做法**：`wave_cap` 全开，先用 level 7，再用 T3 定义的任务池做自适应课程
（采样权重偏向 pass rate ∈ [0.2, 0.8] 的任务）。

**门禁**：

| 指标 | 阈值 |
|---|---|
| level 7，development 30000–30255（256 seed） | pass rate **≥ 60%** |
| 平均终局波次 | ≥ 25 / 30 |
| 学习曲线 | 必须显示从短任务到完整关卡的迁移过程 |
| 泛化差距（level 7 vs 同地形 held-out 关卡） | ≤ 15 pp |

**这是硬门禁。** 60% 是"真正会玩这一关"的下限，不是可以商量的数字。
若长时间卡在 30–50%，问题几乎一定在**长程信用分配**或**课程设计**，回到 T5/T6 重新设计，
不要靠加算力硬堆。

**交付**：`gates/T7.json` + checkpoint + 完整 seed 级结果

---

### T8 · 多关卡泛化（最终门禁）

**目标**：证明系统学的是"玩 PvZ"，不是"背下 level 7"。

**做法**：在 T3 定义的完整任务池上训练（deck 随机化、multiplier 扰动、多种地形），
在 **held-out 任务集**上验收。

**门禁（全部满足，这是最高门禁）**：

| 指标 | 阈值 |
|---|---|
| held-out 全任务平均 pass rate | **≥ 60%** |
| 五种地形中每一种的 pass rate | **≥ 45%**（不允许靠刷擅长的地形拉高均值） |
| 泛化差距（训练任务 vs held-out） | **≤ 12 pp** |
| 单任务最低 pass rate（held-out 内） | ≥ 25%（不允许完全放弃某些任务） |
| 卡组扰动（换 2 张卡）后 pass rate 下降 | ≤ 20 pp |

**泛化差距 ≤ 12pp 这条是防硬编码的总闸。** 一个靠背题过 T7 的系统在这里必然失败。

**交付**：`gates/T8.json` + checkpoint + 每任务每 seed 的完整结果

至此才算"能自主玩 PvZ"。T8 之后才是可选的加固（更高 multiplier、更恶劣 card draw）。

---

## 5. 机器配置（实测，勿重复摸索）

| 机器 | 推荐配置 | 实测最优 |
|---|---|---|
| Apple M5 Pro | `--device cpu`（`auto` 即为此） | 11.3 min（值模型 CPU + 学生网络 MPS 拆设备） |
| i7-12700F + RTX 5080 | `--device cuda --threads 4` | **15.4 min** |

`--device auto` 在两边都会选对。需要逐位对齐旧产物时才显式加 `--threads 1`。

### 台式机执行细节（T5 起在台式机上跑）

| 项 | 值 |
|---|---|
| 接入 | `ssh` 到 Windows 后 `wsl -d Ubuntu -e bash`（助手脚本 `scripts/win_ssh.py`） |
| 仓库 | WSL 内 `~/PvZ-Portable`；代码用 **git** 与本机同步，不要用拷贝 |
| 资源目录 | `~/.cache/pvz-research-resources`（**与本机不同**，不要把本机路径带过去） |
| Python | `~/.venvs/ml`（torch 2.14.0+cu132，CUDA 可用）。**不要再新建 venv、不要再装 torch** |
| 资源一致性 | main.pak 与 partner.xml 的 sha256 已确认与本机一致，`task_signature` 跨机对齐，不必重验 |

本机资源目录（仅在 Mac 上跑时用）：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`。

### 5.0 内存预算与轨迹表示（已实施，2026-09-29）

诊断见 **`MEMORY_BUDGET.md`**（含 §7 实施结果）。**三项改动已全部落地并验证**，
台式机直接 `git pull` 即得：

- transition 由 184.6 KiB 降到 **13.1 KiB（−93%）**：privileged_state 只存 critic 用的
  16 维向量，observation 只存打包 token，legal_actions 存 bitmask。
- 2,000 局批的轨迹缓冲从 27.8 GiB 降到约 1.8 GiB，实测批更新峰值 0.9 GB —— **OOM 根因已消除**。
- 训练前向已批量化（chunk 内一次 encoder + GRU 序列化 + 批量重放），语义与旧版一致
  （等价性已验证：前向 2.7e-7、重放 4.8e-7、critic_extra 逐位相同）。
- **新增 `--minibatch-chunks` 参数（默认 1）**：>1 时跨 episode 分层批、优化器按批 step，
  **lr 必须上调**（建议 1e-4 起）。CPU 实测分层批是负优化，但 GPU 上大 batch 才能摊
  kernel launch —— 台式机开工时必须实测 `--minibatch-chunks` 1/16/64/128 四档
  （`--device cuda`），报告每档更新耗时与峰值显存，据此定正式配置（见 MEMORY_BUDGET.md §7）。

**2026-09-29 台式机实测结论（已合入，HEAD `02125b9`）**：CPU 上分层批是负优化（2000 局
×51 步×2 epoch：逐条 773 s → 单 chunk 750 s → 分层批 16/128 均为 852 s）——
**这是 Mac CPU 的结论，不要外推**。CUDA 上大 batch 的红利**确实存在但来源是 attention kernel
而非"矩阵更大"**：`--minibatch-chunks 16` + exact FlexAttention 使 2000 局更新
**404 s → 188 s（2.15×）**，峰值显存分配 2.07 GB/预留 5.88 GB（dense 为 2.49/6.44 GB），
loss 与梯度范数与 dense 一致（policy 0.12451 vs 0.12455，grad norm 1.488 vs 1.489）。
证据：`artifacts/t5/perf/ppo_update_2000_{flex,dense}.json`、`attention_sweep.json`。

> **注意（2026-09-29 晚核对代码）**：台式机提交 `771700d` 已把脚本默认值改成
> `--minibatch-chunks 16`、`--learning-rate 1e-4`、`--attention-backend auto`。
> 即**直接运行 `train_pvz_ppo_task_family.py` 得到的就是 CUDA 吞吐最优配置**。
> 但该配置**只验证过吞吐，从未验证过学习信号**——见下方 §5.2。

### 5.1 交接状态（执行环境已搬到台式机，笔记本离线）

**2026-09-29 交接**：笔记本上的全部 30 个提交已推送到 `origin/pvz-env`，
HEAD = `f07bd9e`（含 T0–T4 全部代码、`gates/T0..T4.json` 与 seed 级结果、
`artifacts/task_family/{train,heldout}.json`、本文档、`DESIGN.md`）。
**笔记本不再参与执行**，之后所有代码与证据的往返都走 `origin`，不要再尝试从笔记本拷贝文件。

台式机开工第一步（**逐条执行，不要跳**）：

```bash
cd ~/PvZ-Portable
git fetch origin
git branch --show-current          # 记下当前分支
git status -sb                     # 看是否与 origin 有分歧、是否有未提交改动
```

- 若当前分支**不是** `pvz-env`：
  - 有未提交改动 → **先归档，不要丢弃**：`git stash push -u -m "wsl-local-<日期>"`
    （或提交到本地分支 `wsl-local-backup`），并在报告里说明归档了什么。
  - 再 `git checkout pvz-env`；若分支不存在则 `git checkout -b pvz-env origin/pvz-env`。
- 然后 `git pull --ff-only origin pvz-env`，确认 HEAD 为 **`9ee8756`**（`git log --oneline -1`；
  至少应包含 `02125b9` 那 7 个优化提交与本次文档回填）。
- **`--ff-only` 失败时不要用 `--force`、不要 `reset --hard`**：说明本地有提交历史分歧，
  按 §3 停下报告，由人工判断哪些本地产物要保留。
- 同步完成后工作区必须是干净的（`git status -sb` 只显示分支行）。带脏工作区开跑，
  门禁的 `worktree_clean` 会直接判 false，等于白跑。

**每次提交门禁证据后必须 `git push origin pvz-env`**（见 §2.3），这是结果能回到审阅方的唯一通道。

#### 交接已闭环（2026-09-29 晚）

台式机那一轮**只做吞吐/协议/attention 优化，没有跑正式 T5**（无 `gates/T5.json`）。它的 7 个提交
已推送并合入，两边 HEAD 一致为 `02125b9`：

- `a762deb` PPO 轨迹 shard 紧凑化 + NPZ（60 步恢复对象 956 KB → 496 KB，序列化比 gzip JSON 快 4.9×）
- `88a5d30` GAE 测试期望修正（**顺带修好本机原有 3 个旧设计测试失败，现在 113 项全绿**）
- `f6f8b8b` `Board.cpp` 单波进度条除零
- `5fb13b6` `CRITIC_INPUTS` 协议 4 + CUDA FlexAttention
- `771700d` 按实测配置 T5 rollout worker 与 PPO batch
- `9a043b3` 三个 benchmark 脚本
- `02125b9` 两份报告（`T5_PERFORMANCE_REPORT.md`、`OPTIMIZATION_TASKS.md`）

**worker 选型（正式 `run_seed_jobs` 同一批 2,000 局实测）**：18×1 为 161.6 s / **44,565.9 局/h** 最快，
16×1 为 163.5 s / 44,024.6，20×1 反降到 174.7 s / 41,221.3。**单核门禁重跑 6,273.8 局/h（>5,000 通过）**。
配置写入 `artifacts/t5/throughput.json`，正式 T5 默认 **CPU/18 worker/每 worker 1 线程**。

**WSL OOM 的第二个根因（本机之外）**：18 个 worker 进程树 RSS 合计峰值 **19,168 MB**，与训练的
约 3.8 GB CPU RSS、约 2 GB CUDA 分配**同时**存在。进程池目前每次 2,000 局更新重建一次；报告明确
**不提交持久池**（省下的启动开销不值这份内存）。在 16 GB WSL 上跑正式 T5 前，先按 `MEMORY_BUDGET.md` §4
把 `.wslconfig` 上限调够，或接受 18 worker 的峰值需求。

### 5.2 阶段 0 的三个阻塞点（2026-09-29 晚核对代码发现，**开工前必须解决**）

> **彻夜执行请直接读 `T5_OVERNIGHT_ORDER.md`** —— 它是给执行 Agent 的指令，
> 含可执行的硬阻断机制（退出码 + 代码级 `raise`）、三个缺口的实现规格与验收条件、
> 判据预注册模板、失败处理流程。本节只是缺口的背景说明。
>
> **2026-09-30 更新：三个缺口已由台式机补齐并推送**（`3113d1b` 课程过滤、
> `0193d36` 阶段 0 评估 + 硬门禁、`754dbbd` 派生 lane 特征、`de294e2` 判据预注册）。
> 核验结论见 §5.3。**阶段 0 尚未产出结果**（`gates/T5.json` 仍不存在）。

§4 的 T5 细则要求"先过阶段 0 学习信号验证门"。核对 `python/train_pvz_ppo_task_family.py`
后发现，**阶段 0 按现有代码无法按要求执行**，有三处缺口：

**缺口 1：课程起点无代码路径。** §4 要求"先用 `train.json` 中 `wave_cap=1` 且 `1.0×` 的 5 个
任务开训"，但 `_task_family()`（第 66–72 行）**硬性断言训练集恰好 20 个任务**，
`TRAIN_PATH` 是模块常量，**没有任何 `--train-manifest` 或课程过滤参数**。
→ 需要新增课程入口（过滤清单或 `--curriculum` 参数）。

**缺口 2：阶段 0 的评估判据无处落地。** `_evaluate()`（第 188–234 行）**只评估 heldout 的
cap3×1.0 子集**（10 个任务），`_curve_rises()` 也只看 `heldout_cap3_x1_pass_rate`。
而 §4 阶段 0 判据写的是"评估集 pass rate 由训练前基线（< 5%）上升到 > 50%"——
这个"评估集"若指 cap1 训练任务，**代码里没有独立评估路径**（只有 `_curve_row` 里
训练过程的滚动 64 局值，不是独立评估）；若指 heldout cap3，则基线是 **0%**（T4 实测）而非 <5%。
→ 需要裁定阶段 0 到底在哪个集合上判定，并补上对应的独立评估。

**缺口 3：网络要求（§4 明写"阶段 0 之前完成"）未实现。** §4 要求两处观测编码改动：
显式派生特征（每行僵尸威胁度/最近僵尸距离/该行射手数/该行植物总血量；全局阳光收入速率/
经济与火力比/波次进度）与 lane 级聚合 token。代码中**只有 T4 时代的 `aux_lane_threat`
辅助预测头**（`pvz_agent_model.py:448`），既不是 lane 聚合 token，也没有上述派生特征。
→ 需要实现。

**另外两个必须注意的行为（不是缺口，是设计约束）**：

- **每次调用脚本都会消耗一个正式 run 名额**（`run_number = state["formal_runs"] + 1`，
  `MAX_FORMAL_RUNS = 8`）。run > 1 必须给 `--motivation`，且**默认从上一 run 的
  `runs/run_{N-1}/gameplay_model_v1_ppo.pt` 续训**。因此若先跑 run 1 再改网络结构，
  run 2 会因 `model_architecture_version` 不匹配而直接报错 ——
  **顺序必须是"先做完缺口 1–3，再开 run 1"**。
- run 1 的随机初始化必须与 T4 冻结的 seed-0 `state_sha256` **逐位一致**，否则 `_baseline()`
  与初始化检查都会报错。这条保证了基线的可比性，不要绕过。

**已确认可用的部分**：硬停止条件已实现（每 5,000 局或 30 分钟评估一次；累计 20,000 局后
pass rate < 20% 且最近 5,000 局改善 < 5pp → 自动停止并写 `stop_reason`）；
`--max-episodes-per-run` 默认 20,000，阶段 0 需显式传 `10000`。

### 5.3 台式机补齐缺口的核验结论（2026-09-30）

三个缺口 + 判据预注册全部落实，且**硬阻断机制按规格写进了代码**（不只是文档）：

| 工作令条款 | 落实位置 | 核验 |
|---|---|---|
| 缺口 1 课程过滤（过滤器而非换清单） | `3113d1b`，`_curriculum_tasks()` | ✅ `TRAIN_PATH` 未动，`_baseline()` 的 sha256 校验仍通过 |
| 缺口 2 阶段 0 独立评估 | `0193d36`，新增 `stage0_set` 字段 | ✅ 与 `gate_set` 并列，未改后者语义 |
| 缺口 3 派生特征 + lane token | `754dbbd`，模型 + env + `LawnApp.cpp` | ✅ `MODEL_ARCHITECTURE_VERSION` 4→5，`OBSERVATION_VERSION` 2→3 |
| **M1** 早期零信号硬停止 | `_stage0_has_no_signal()` + `stop_reason="stage0_no_signal"` | ✅ 含 `stage0_diagnosis` 诊断负载 |
| **M2** 阶段 0 门禁代码级阻断 | `_check_stage0_gate()` | ✅ 缺 `stage0_gate.json` 或非 pass 直接 `raise`；逃生口需 `--motivation` |
| **M3** 变异测试 | `python/test_t5_overnight.py`（195 行） | ✅ 有独立测试文件 |
| **M4** 不许跳步 | 判据预注册先于开跑（`de294e2`，16:43:45Z） | ✅ |
| **R7** seed-0 初始化校验 | `_seed0_initialization_baseline()` | ✅ 不给 note 就 `raise`；给了则记录 `{status: superseded, actual, t4, note}` |
| 额外 | `protected_paths` 校验含 `TODO.md`/`DESIGN.md` | ✅ 超出要求 |

**本机独立验证发现并修掉一处缺陷**（提交 `6095c24`）：台式机升了 `OBSERVATION_VERSION`
（正确——env 确实新增了 `sun_income_rate`），但漏改钉住该常量的
`test_shared_helpers.test_version_constants_are_the_documented_values`，导致测试失败。
修正后 `test_agent_model / test_training_semantics / test_seed_jobs /
test_shared_helpers / test_t5_overnight` 共 **99 项全通过**。

**仍未验证**：`test_env_protocol` 未跑（本机 PvZ 二进制仍是协议 3，需 `cmake --build build`）；
**阶段 0 没有任何结果产出**，`gates/T5.json` 与 `artifacts/t5/stage0_gate.json` 均不存在。

### 5.4 注意力机制移植性分析（2026-09-30，独立方向）

见 **`ATTENTION_TRANSFER_ANALYSIS.md`**。结论：LLM 侧的稀疏/压缩注意力
（CSA2 / QSA / MSA / SGA / IndexShare / KDA / Gated DeltaNet-2）**针对的是 PvZ 模型
根本不存在的问题** —— 上下文 L 仅 74–116 token、无 KV Cache、注意力只占单层算力 4.2%。

但核查发现了真正的瓶颈：**relation-bias 构造占 attention 时间 67.1%、每决策总耗时约 35%，
一次派发 117 个算子**。已实测 `torch.compile` 融合 **1.83–1.86×，`torch.equal` 逐位相同**。
复现脚本：`scripts/attention_cost_profile.py`。**这些优化不得在阶段 0 通过前进主分支。**

### 5.5 全流程热点优化已落地（2026-09-30，用户裁定"立刻全做并合回 env 再重跑"）

见 **`TRAINING_HOTSPOT_ANALYSIS.md` §9**（含两处对 §7 投影的更正）。已实施：

| 项 | 内容 | 验证 |
|---|---|---|
| P0-1 | `episode_digest`：字节摘要取代 JSON 物化 | `test_episode_digest.py` 13 项；2 个变异被捕获；45× |
| P0-2 | `_evaluate` 改 spawn 进程池 | 真实模拟器**逐条逐位相同**；128 局 28.8 s → 10.4 s |
| P1-1 | 中间评估跳过 `reference_set`，每轮末必补 | `test_t5_overnight.EvaluationTests` |
| P1-2 | 关系偏置融合默认开启（`PVZ_RELATION_BIAS_FUSION=0` 可关） | 稳态 1.14–1.31×；编译成本 440 ms/进程 |

**§5.4 末尾"这些优化不得在阶段 0 通过前进主分支"这条约束已被用户明确解除** ——
裁定是"优化立刻全做，合并回 env 分支再重跑"。理由：这些改动不改变任何训练语义或判据
（逐位相同已证），且 `episode_digest` 会改变 `trajectory_sha256` 的取值，
**必须在 T5 产出任何结果之前定死**，否则证据不可比。

**同时新增 M5 硬阻断**（见 `T5_OVERNIGHT_ORDER.md` §M5）：`resolve_device("auto")` 在
CUDA 掉线时会静默回退 CPU，导致"进程活着但一个更新都落不下盘"。
首次阶段 0 就这样空转 75 分钟。现在非 CUDA 更新会直接 `raise`。

### 已排除的路线（有实测数据，勿重复尝试）

- **降精度换速度**：fp16 / bf16 / 稀疏首层 / float32 累加器**全部比全精度更慢**
  （0.09×–1.00×）。Apple 的 AMX 是 fp32 单元，M 系列没有更高吞吐的半精度路径。结构性，非配置问题。
- **值模型放 MPS**：batch=1 形状上 MPS 全面更慢（`predict` 68 µs → 451 µs）。
  `resolve_device("auto")` 已不再考虑 MPS。
- **叶子估值批量化**：每批仅约 4.3 行，`F.linear` 非连续转置权重的 ~44 µs 固定开销吃光收益。
- **特征向量稀疏化**：特征仅 1.1%–3.3% 非零，但 `Linear(4116,128)` 在 batch=1 只要 6.59 µs，
  索引构造开销远超省下的乘加，实测慢 5.6×。
- **调 CPU 线程数（Apple）**：M5 Pro 上 threads 1/2/5/10/15 无差异，Accelerate 已吃满。
  **但 x86 上相反**：threads=4 比 1 快 2.0×。这条不是普适结论，按机器区分。
- **继续优化 `advice()` 吞吐**：真实环境下 Python 侧全部优化到零端到端也只有 2.9×，
  而搜索本身就该被移除（见 `DESIGN.md` §1）。**不要在这里花时间。**
- **在 0 胜基座上继续扩大 BC / DAgger 规模**：已实测无效（64 集 → 6400 集只会更精确地复制失败）。

---

## 6. 保留与删除（改动边界）

| 模块 | 处置 |
|---|---|
| `pvz_env.py` | **保留，扩展**（720 行仅 5 个魔数，全仓最干净） |
| `GameplayModelV1` 的 token 化与关系注意力 | 保留，改进（加 lane 摘要与派生全局量） |
| `SearchAdvice` 预算记账 / `_expand_paired` / `one_step_children` 防错位接口 | 保留（设计正确） |
| `pvz_seed_jobs` / `atomic_write` / spawn 并行采集 | 保留 |
| `diverse_plant_groups` | 删除重写（几何多样性，非语义） |
| `discounted_terminal_value` / `train_search_value` | 删除重写（回归目标 vs 排序用途，错配） |
| `_transition_reward` | 删除重写（任意常数惩罚，非势函数形式） |
| successive halving 硬剪枝 | 删除重写（用噪声做不可逆剪枝） |
| BC / DAgger 主链路 | 降级为可选，不再是行为来源 |
| `scripts/scripted_baseline.py` | 保留，仅作 T0 环境可解性验证器 |

现有 192 个 Python 测试中，`test_search_teacher.py`（49）与 `test_search_value_features.py`（15）
的断言编码了旧设计，重写时必然全红——**这是预期的，不是 bug**。
先定新设计再写新测试，**不得为了让旧测试通过而保留旧设计**。

---

## 7. 当前状态

- 上一轮（教师蒸馏路线）已确认失败并停止：`FAILURE_ANALYSIS.md`。
  教师 dev 0/256，学生 final-test 0/1024。**不要试图挽救这条路线。**
- 新设计已定稿：`DESIGN.md`。
- **T0 的 L0 修复已成功**（提交 `a871927`）：level 7 / seed 30000 在 **tick 65,401 获胜**
  （修复前 1,796,340 tick 永不终局），seed 30001 亦获胜。seed 30002 / 30003 分别在第 10、7 波
  正常失败——已确认是脚本策略强度问题，**不是环境阻塞**。
- **执行环境已搬到台式机**（2026-09-29）：同步与证据回传全部走 `origin/pvz-env`。
- **内存诊断 + 三项优化已落地并合并**（2026-09-29，`MEMORY_BUDGET.md`）：WSL 吃满 16 GB 的根因有两条——
  轨迹缓冲以 Python 对象驻留（185 KiB/transition，2,000 局批需 27.8 GiB）与 18 worker 进程树 RSS 19 GB。
  前者已修（transition → 13.1 KiB，−93%，`torch.equal` 逐位等价；批缓冲 → 约 1.8 GiB，实测峰值 902 MiB），
  后者按实测**不引入持久池**（见 §5.1）。
- **"更新占 95% → GPU 强项"的推理已被部分修正**：真实瓶颈是 batch=1 串行 + 每 epoch 重做 token 化，
  但 CUDA 大 batch 确有效益——**来源是 attention kernel**（exact Flex 2.15×），不是"矩阵更大"。
- 当前待办：**T5 阶段 0 尚不能开工** —— 核对代码发现三个缺口（见 §5.2）：
  ①课程起点（cap1 5 任务）无代码路径（训练集硬断言 20 个任务）；
  ②阶段 0 的评估判据无处落地（`_evaluate` 只评估 heldout cap3×1.0，基线 0%）；
  ③§4 明写"阶段 0 之前完成"的网络要求（派生特征 + lane 聚合 token）未实现。
  **必须先补完 ①–③ 才能开 run 1**，否则 run 1 的名额会被一个改不了结构的配置占掉
  （run > 1 默认从上一 run 续训，网络一变就 `model_architecture_version` 不匹配报错）。
  补齐后的顺序：确认 HEAD → 清 `artifacts/t5/training_state.json` → `--device cuda`
  `--max-episodes-per-run 10000` 跑阶段 0 → 判据 >50% → 再进阶段 1（20 任务，heldout
  ∩(cap3,×1.0) ≥90%）→ **停止等待确认，不得开始 T6**。
  另注意脚本默认已是 `chunks 16 / lr 1e-4`，该组合**只验证过吞吐、未验证过学习信号**。
  台式机 smoke 显示"一次 2,000 局更新后仍 0/20 胜出"，样本太小不构成证据，
  阶段 0 才是真正的裁判。
- T6 及以后（分层网络 + 意图动作空间）需要设计判断，本次无人值守不推进。
- **T2 的语义与实现规范已在本文档 2.1–2.3 定死**（`preplanted` 走 `Board::AddPlant`，
  不自动配底座、失败即报错）。执行时按文档实现，不要另行设计。
- **T2 第一版实现的缺陷已定位并修正**（规范已更新）：只截断 `mNumWaves` 而不改
  `aIsFinalWave` / 新僵尸引入 / `IsFlagWave` 的判定基准，会让第 `wave_cap` 波变成终局大波，
  导致 `wave_cap=3` 在 seed 30000 上 tick 11,777 被打穿（同一 seed 完整 30 波反而能赢）。
  必须按 2.2(b) 用完整波数 `aFullWaves` 计算波次组成。
- 遗留独立任务：裸指针序列化修复—**已完成**（v4 字段逐项序列化，双进程逐字节一致，
  `verify_env_equivalence.py` 7 关卡全通过）。无需再做。
