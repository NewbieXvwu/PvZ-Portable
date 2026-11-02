# task217b — 单次决策的时间分布、提速结论与阶段 2 结果

（用户 2026-11-01 的任务书。测量用 `scripts/csearch_profile.py`：真实学生整局，逐决策跑一次生产
预算的 `SEARCH`，读引擎自己的分段计时；同一时刻只有一个进程、`nice -n 19`、`PVZ_SEARCH_PROFILE=1`。）

## 结论先说（三条，都影响后续决策）

1. **时间花在模拟和网络前向，不花在来回通信上。** 一次决策的中位数（L28/L29/L33/L44 各 10 个决策）
   是 19–37 ms：引擎模拟 36–57 %、学生前向 32–53 %、静态打分 3.5–7.6 %，其余（恢复、合法格扫描、
   叶值、记账）合计 <3 %。阶段 1 把整棵树搬进 C++ 只赚到 1.0–1.08×，正是因为每条命令本来就要跑
   这么多模拟，而一次往返只有约 75 µs（20 条命令合计 1.5 ms，不到一次决策的 2 %）。
2. **"少通信"的空间已经榨干，2.2× 也没有拿到。** 任务书列的两条路（候选记账去字符串化、学生前向
   批量化）实测都不成立：记账在总时间里 <1 %（已按无字符串键重写，见下），而学生前向是**每块棋盘
   每 tick 一次**，一条 lane 一块棋盘，批不起来。**要 2× 只能少算**（减少 rollout/验证、降低 tick
   数、或把 sim 与 net 本身做快），其中任何一条都会改变搜索结果，必须单独做一致性/质量对比。
3. **阶段 2 的迁移全部做完，Python 搜索已删除。** 保存探针、组合候选（含 `--combo-buffer` /
   `--combo-interval`）、student-topk 都在 C++；宽范围一次性对比（白天/夜晚/泳池/雾/屋顶 5 关 ×
   2 种子，89 个决策）动作、候选值、计数器、验证全同，随后删掉 Python 版与 `--search-impl`。

## 1. 单次决策的时间分布（中位数，生产预算，无组合/探针）

| 关卡 | 决策 ms | 模拟 | 学生前向 | 静态打分 | 叶值 | 其他 |
|---|---|---|---|---|---|---|
| L28（泳池） | 37 | 52 % | 41 % | 3.5 % | 0.7 % | 1.6 % |
| L29（泳池） | 33 | 57 % | 32 % | 7.6 % | 0.8 % | 2.0 % |
| L33（雾） | 19 | 36 % | 53 % | 5.4 % | 1.1 % | 2.3 % |
| L44（屋顶） | 28 | 46 % | 39 % | 7.6 % | 1.1 % | 2.4 % |

每个决策实际干了多少活（中位数）：

| 关卡 | 模拟 tick | 学生前向 | 叶值前向 | rollout | 静态打分 |
|---|---|---|---|---|---|
| L28 | 23 940 | 429 | 9.5 | 9.5 | 18 |
| L29 | 26 460 | 463 | 10.5 | 10.5 | 37 |
| L33 | 20 258 | 341 | 8 | 8 | 17 |
| L44 | 30 240 | 548.5 | 12 | 12 | 31 |

**带验证的决策贵一倍以上**：L28 第一个决策（90 次打分 + 18 个验证 rollout）98 ms，同关普通决策
25–30 ms。所以"决策有多贵"主要由验证轮决定，不由通信决定。

## 2. 任务书两条提速路线为什么没成

**候选记账去字符串化（已做，收益 <1 %）。** 原来每个候选构一个 `"P <slot> <col> <row>[ u<up>]"`
字符串塞进 `std::map<std::string,double>`，每次决策还要重建一遍 `present()`。现在候选直接带
`staticScore`，去重用 `SameCandidate`/`HasCandidate` 比整数字段。实测这两处的耗时（单独计时）
是 0.000 ms 量级，**只值得为形状做，不值得为速度做**。这一条是本任务里唯一"按计划做完但收益远低于
预期"的项，写在这里以免下次再被同一直觉误导。

**学生前向批量化（不可行，未做）。** 学生每 ~56 tick 决策一次，一次 `Predict` 约 20 µs，占 32–53 %。
批量化需要同一时刻多块棋盘：一条 rollout lane 就是一个 `LawnApp` 单例，要让 N 条 lane 同步推进得
重写整个续走循环并复制 N 份棋盘（内存与恢复成本会吃掉收益）。**逐 tick 的前向无法在一条 lane 内批。**

## 3. 真要 2× 的话（都改变结果，都只是提案）

| # | 选项 | 估计收益 | 代价 |
|---|---|---|---|
| 1 | `--verify-samples 9→5`，或只验证前 2 名 | 带验证决策 −30…−50 % | 验证均值方差变大，翻转率上升 |
| 2 | 复用减半阶段已跑过的未来做验证 | 少 9 个 rollout/次 | 种子变了，值不可比，必须重标 |
| 3 | `--samples 3→2` 或 `--horizon 2500→1800` | 模拟与网络按比例降 | 视野变短，长线卡（升级、双发）被低估 |
| 4 | 抬高 `--rollout-sim-skip`（61→93/125，加 `ENV_SPARSE_TRACKS`） | 实测 61→317 只快 4 % | sparse tracks 不改结果；其它位改 RNG 会改对局 |
| 5 | 把 sim 本身做快（热点函数、SIMD、每 tick 少扫一遍僵尸） | 模拟 36–57 % 是上限 | 工程量大，逐位一致，需要单独的等价性验证 |

**没有一项是"免费"的 2×。** 免费那侧（通信、记账、快照）加起来不到 5 %。

---

# task217 — 把老师搜索整棵树搬进 C++

目标（用户 2026-11-01）：候选生成、逐卡选格、减半、验证、保存探针、组合全部在 Python，只有单个续走
ROLLOUT / ROLLOUT_VALUE / LEAF_VALUES / STATIC_SCORES / BRANCH_SNAPSHOT_FAST / SIM_SKIP 走 C++，
**一次决策十几次往返**（违反 AGENTS.md §6.6 规则 2），也把后续加深搜索的空间压住了。
本任务把整次决策变成一个 C++ 命令 `SEARCH`。

本文件分三部分：**A 阶段 0**（把当时的 Python 算法完整写成规格 + C++ 侧设计 + 迁移后 Python 薄面），
**B 阶段 1**（实现与一致性/性能验收），**C 阶段 2 结果**（保存/组合/topk 迁移、宽范围对比、
删除 Python 版、实测性能）。阶段 2 已全部完成，本文件的 A 部分保留为历史规格。

---

## A1. 输入（一次决策的全部输入）

决策点：`legal_actions.plants` 非空。生产参数（`MEMORY/docs/task214-teacher-upgrades.md` 之后的
52 号生产配置）：`--verify-samples 9 --student-candidates 2 --samples 3 --horizon 2500 --per-card 1
--halving --halving-first-samples 1 --min-gap 300 --rollout-sim-skip 61 --margin-relative 0`
（`MARGIN = 0.02`、`MARGIN_FLOOR = 0.001` 只在 `margin_relative > 0` 时参与；`--save-candidates 0`、
`--combo-candidates 0` 关闭）。常量 `NOOP = {"type":"wait","ticks":60,"until":"timeout"}`。

1. 决策根快照 `root`（`SNAPSHOT_FAST`），以及根上的 `sun`、`sun_income_rate`、`packets`（槽位/费用/类型）、
   `tick`、合法动作列表；根的太阳产量历史（`history_start` + `(tick, produced)` 对）。
2. 学生先验：`policy_action`（学生在根上的选择）、`_W["student_frame"]`、`_W["student_previous_frame"]`
   （学生刚刚读过的两帧）、`student_history`（v2 历史）、`context`。
3. `stats`：`last_search_tick`（min-gap 时钟）、`sun_reserve`（组合候选留钱）、`searched`、计数器。
4. `decision_seed = seed * 10000 + decisions`；第 k 个未来种子 `(decision_seed*1000 + k) % 2**32`。
5. `--leaf-probe` / `--leaf-probe-limit`：可选诊断文件。

## A2. 步骤（严格顺序）

1. **门槛**：无合法动作用 `policy_action`；`sun_reserve` 非零时滤掉会让余额低于 reserve 的植株；仍为空则 `NOOP`。
   `min_gap` 生效条件：`policy_action` 不是 plant 且 `tick - last_search_tick < min_gap` → 直接返回
   `policy_action`（不搜索、不更新时钟）。否则更新 `last_search_tick`、`searched += 1`。
2. **候选集**（`candidate_mode = static`，生产唯一路径）：
   a. 全部合法植株逐个 `branch_snapshot` 后静态网打分（`STATIC_SCORES`，收入=根的收入率，sim_skip=0），按分数降序；
   b. 每张卡只保留最好的 `--per-card` 格（`by_card`），按出现顺序连成列表，再追加 `NOOP`；
   c. `policy_action` 若是 plant 且不在集合里，追加；
   d. `--student-candidates 2`：学生 `ranked_plants` 按概率降序，跳过已在集合里的，最多追加 2 个，记 `student_candidate_added`；
   e. `--combo-candidates`（关闭）：底座植物 + `combo` 标记，两步走；
   f. `--demo-actions`：当前轮到的示范动作若此刻合法且不在集合里，追加，记 `demo_candidate_added`。
   集合里绝不含重复（全部用 `json.dumps(sort_keys=True)` 去重）。
3. **续走**（`--halving` 关闭时）：每个候选 `--samples` 个未来，**逐个候选**串行，每个未来
   `ROLLOUT_VALUE`（`first` = 候选，`until = tick + horizon`，`root`，种子如上，`sim_skip=61`，event_wait，
   叶子带 `student_history`）；结束局面的返回先按终局值，其余按 `LEAF_VALUES` 的顺序累加，
   `total = sum / len`。
4. **减半**（`--halving`，生产开）：第一次 `target = halving_first_samples = 1`。
   每轮给存活候选补到 `target` 个未来（`ROLLOUT_VALUE`，一次一个），落盘顺序按“终局结果在前、其余在后”
   稳定排序；`--samples` 轮满了就停；否则保留均值最好的一半（至少 2 个，学生自己的选择强制留下），
   `target = done * 2`。被剪掉的候选 `total = -inf`。
5. **保存探针**（`--save-candidates` 0）：先试“等够太阳再种贵卡”的计划，超过所有候选 + margin 就返回一个 wait。
6. **验证**（`--verify-samples 9`）：`best != policy_index` 且 `totals[best] - totals[policy] > margin`
   时，对学生与挑战者各补 9 个未来（种子从 `samples` 起），用加权公式更新两个 total；
   记录 `last_verification`（9 个样本、覆盖前值、覆盖后值、`original_samples` 等）。
7. **决策**：`totals[best] - totals[policy_index] > margin` 且 `best != policy_index` → 选 `candidates[best]`；
   否则用 `policy_action`。统计 `overrides`、`override_kinds`、`student_candidate_best/override`、
   `last_targets`、`last_best`，`--leaf-probe` 落一行。
8. 收尾 `restore_fast(root)`。

`best` 用 `max(alive, key=...)`，并列取**索引最小**者。

## A3. 输出

返回“最终动作”（plant 或 `policy_action`）。训练帧里只出现最终动作；诊断字段
（`last_targets` = [(动作, 值)]、`last_verification`、计数器）只进 stats / leaf-probe。

## A4. C++ 侧设计

新命令 `SEARCH`，一条命令跑完整个决策，ready 行广告 `search_version: 1`：

```
SEARCH rootId decisionSeed income previousTick previousPlant mask horizon perCard studentCandidates
       samples verifySamples halving halvingFirstSamples marginRelative minGap lastSearchTick reserve
       policyToken demoTarget historyStart historyCount [tick produced]*
       V2H wave waveTick plants... [TRACE "path"]
```

`policyToken` / `demoTarget` 用 branch 记号：`W`（NOOP）或 `P:packet:col:row`；无示范用 `-`。
回包
`{ok, searched, action, best, policy_index, overridden, stats{...}, targets[[packet,row,col,value]],
verification}`；非有限 total 的 value 为 `null`（Python 侧还原成 `-inf`）。

实现要点：

- `class EnvironmentSearch`（`src/main.cpp`，与 ROLLOUT 共用文件级 helper）：`ScoreStatic` / `Continue`
  / `TerminalLeaf` / `AppendBatch` / `HalvingChoice` / `FullChoice` / `VerifiedChoice` / `Run`。
  进程内复用 `EnvironmentRollout::Predict/Features/History`、`EnvironmentValueNet::Predict`、
  `SaveEnvironmentSnapshot` / `RestoreEnvironmentSnapshot` / `ExecuteEnvironmentAction` / `EnvironmentWaitUntil`。
- 从 ROLLOUT 分支抽出共享的 `RunRollout()`，`ROLLOUT` 与 `SEARCH` 用同一个续走循环（响应字段不变）。
- 学生排名：`EnvironmentRollout::Rank()` 复刻 `pvz_rollout_policy.ranked_plants`
  （sigmoid(plant) × packet softmax × cell softmax，按 `(-score, kind, row, col, packet)` 排序），
  保证 `rank[0]` 等于 `act` 的选择。合法植株枚举收敛到 `LawnApp::EnvironmentLegalPlants()`
  （`EnvironmentLeanObservation` / `EnvironmentAnyLegalPlant` / 搜索共用）。
- 种子：同一 `(decisionSeed, k)` 公式在 C++ 里算，`ReseedEnvironment` 语义不变；RNG 仍在同一位置消耗。
- `min_gap` 与 reserve 过滤在 C++ 内判定，`searched=false` 时回 `policy_action`；Python 只在
  `searched=true` 时推进时钟和计数，因此未搜索决策不会漂移。
- 内存/线程：搜索状态（候选、totals、快照 id）全在栈上，决策结束即释放；每个 worker 一个进程，
  仍只有单线程引擎，和 Python 版相同。根快照与候选快照沿用现有 snapshot 表与上限。
- 诊断：`PVZ_SEARCH_DEBUG=1` 时把排名前 8 打到 stderr；另留 `RANK_PROBE income previousTick previousPlant
  V2H...` 小命令，方便阶段 2 单独比对排名。

## A5. 迁移后留下的 Python 薄面

阶段 2 之后留在 Python 的只有：`_search()`（snapshot → 一条 SEARCH → restore/release → 把计数器
差值、`last_targets`、`last_verification` 抄进 stats）、`pvz_env.PvZEnv.search()`（拼命令）、
`_demo_candidate()`、`_probe_decision()`（leaf-probe 文件）。候选/续走/减半/验证/保存/组合全部在引擎。

---

## B1. 已实现（阶段 1）

生产路径全覆盖：静态候选、逐卡取格、`NOOP`、策略动作、学生 top-k、示范候选、减半（含首轮多样本）、
验证 9 个未来、margin 判定、reserve、min_gap、`null` 值、全部计数器、`targets`/`verification` 回包、
`RANK_PROBE` / `PVZ_SEARCH_DEBUG` 诊断。`--leaf-probe` 仍由 Python 侧落地（传入 candidates/totals）。

仍是 Python/未覆盖：`--save-candidates`（保存探针）、`--combo-candidates`、`candidate_mode=student-topk`
（task42 对照臂）、`_W["fast"]` 的 torch 前向（学生先验本身仍由 Python 算）。

## B2. 一致性验收（Mac，`scripts/csearch_consistency.py`）

同一局面、同一 decision_seed、两侧各跑一次，交替先后顺序（否则第二名吃热缓存，实测差 5%），
比对动作、`targets`（候选序列 + 值，容差 `1e-6` 相对）、`verification` 全部字段、结构计数器。

| 关卡/预算 | 决策数 | 动作一致 | targets 一致 | 计数器一致 | 验证一致 | 决策耗时 (py → cpp) |
|---|---|---|---|---|---|---|
| L28 生产预算 | 40 | 40/40 | 40/40 | 40/40 | 10/10 | 52.2 → 48.4 ms (**1.08×**) |
| L28 深预算（samples 6 / verify 15 / top-4 / horizon 5000） | 14 | 14/14 | 14/14 | 14/14 | 6/6 | 75.1 → 72.2 ms (1.04×) |
| L29 深预算 | 30 | 30/30 | 30/30 | 30/30 | —（本关无覆盖） | 90.9 → 88.1 ms (1.03×) |
| L29 生产预算 | 40 | 40/40 | 39/40* | 40/40 | — | — |

\* 唯一一次不符合是学生在两格之间近似并列、候选顺序互换（值本身相同），属允许的浮点细节。

整局同种子对照（3 个种子，L28）：`won` / `final_wave` / `decisions` / `searched` / `overrides`
**完全相同**（第 3 种子 8 波 377 决策 63 搜索 12 覆盖），墙钟 5.5/4.5/2.3 s（Python）对
5.4/4.2/2.1 s（C++）。

## B3. 已发现的差异与修复

`Rank()` 初版把 `previousTick` 传成 `-1`，学生特征少了“上一次动作”上下文，前 6 名分数差 10% 以上、
候选集不同 → 决策不同（第一次深预算跑出 11/12）。补上 `previousTick`/`previousPlant` 后排名与
`ranked_plants` 前 9 位逐位一致、分数 7 位有效数字相同。

## C. 阶段 2 结果（已完成，2026-11-01）

### C1. 保存探针 / 组合 / student-topk 全部进 C++

| 功能 | 状态 | 说明 |
|---|---|---|
| 保存探针 `--save-candidates k` | 已迁 | 从根等待到卡可付（`--save-share`/`--save-interval`/`--save-chunk`/`--save-max-wait`），按静态网选格，再跑 `samples+verify` 个未来取均值；`save_probes` 按**成功**的计划计、`save_override` 只在超过 `max(有限值)+margin` 时计 |
| 组合 `--combo-candidates k` | 已迁，含 task214d 改版 | 见 C2 |
| `student-topk` 候选模式 | 已迁 | `candidate_mode=1`：学生自己的全局 top-k + NOOP + 自己的动作，不再走静态网；`candidate_mode/student_topk` 两个新参数 |
| 演示候选（`_demo_candidate`） | 已迁（阶段 1） | 记录动作在当前 tick 仍合法时加入候选 |

### C2. 组合候选：从"留钱"改成 task214d 的 buffer/interval

阶段 1 的 C++ 组合沿用旧的"把升级全价压在银行里"的设计。而 `de6b1561`（task214d）已经把它改掉了：
不再留银行，`--combo-buffer 25`（买完还剩 25 阳光才强制升级）、`--combo-interval 600`（两张底座
之间至少 600 tick）。旧设计在夜间雾关会把 50 阳光的银行锁死（L33 组合臂 0/8、8.8 波），新设计
桌面 056/059 从 8 波输变成 20 波胜。

C++ 侧因此重写：`SearchState` 用 `lastComboTick` 取代 `sunReserve`，`ForcedUpgrade` 用 buffer 门控，
"待升级"由选择的那次决策**登记**、下一次决策**盖时间戳并尝试强制**（底座在两次决策之间才种下）。
这一条是阶段 2 里唯一一处**语义**变化，不是纯移植；L28 用 `--combo-candidates 2` 对比 30/30 决策一致、
72 行 targets 一致、计数器全同。

### C3. 宽范围一次性对比（`AGENTS.md` §6.5：只做一次，不常驻）

工具 `scripts/csearch_consistency.py`（两侧同局面、同 `decision_seed`、交替先后顺序），每关 2 个
种子 × 8 个决策：

| 关卡 | 地形 | 决策 | 动作 | targets | 计数器 | 验证字段 |
|---|---|---|---|---|---|---|
| L1 | 白天草地 | 16 | 16/16 | 16/16 | 16/16 | 10/10 |
| L17 | 夜晚草地 | 9 | 9/9 | 9/9 | 9/9 | —（无覆盖） |
| L28 | 白天泳池 | 16 | 16/16 | 40/40 | 16/16 | 4/4 |
| L33 | 夜池有雾 | 16 | 16/16 | 64/64 | 16/16 | 3/3 |
| L44 | 屋顶 | 16 | 16/16 | 45/45 | 16/16 | —（无覆盖） |
| L33（组合+探针开） | 夜池有雾 | 16 | 16/16 | 64/64 | 16/16 | 3/3 |

**合计 89 个决策：动作、候选序列与值、全部计数器、全部验证字段逐位相同**（值容差 `1e-9+1e-6·|v|`）。
整局同种子对照（L28 三个种子）：`won`/`final_wave`/`decisions`/`searched`/`overrides` 全同。

### C4. 删除 Python 搜索

`de6b1561` 之后 Python 侧的组合语义与 C++ 一致、宽范围对比全绿，于是同一次改动里删掉：
`_search`、`_halving_choice`、`_verified_choice`、`_save_plan`/`_save_probe`/`_save_value`、
`_combo_rollout`、`_forced_upgrade`、`_student_topk_candidates`、`_static_scores` 及只被它们用的
辅助函数、`--search-impl` 开关，以及 4 个只测这套实现的单测文件（`test_search_margin_halving`、
`test_combo_candidates`、`test_save_candidates`、`test_verify_override`）。`_cpp_search` 改名
`_search`，`scripts/csearch_consistency.py` 随对比对象一起删除。`scripts/csearch_profile.py` 保留，
它测的是仍然存在的引擎。

生产命令不变（不含 `--search-impl` 时原本也是 Python；现在引擎是唯一实现），需要
`search_version:1` 的新二进制。

### C5. 性能（实测，不粉饰）

| 场景 | 阶段 1（Python 决策） | 阶段 2（C++ 决策） | 倍数 |
|---|---|---|---|
| L28 生产预算，每决策中位 | 41.5 ms | 40.7 ms | 1.02× |
| L28 深预算 | 75.1 ms | 72.2 ms | 1.04× |
| 整局（L28 种子 1/2/3） | 5.5/4.5/2.3 s | 5.4/4.2/2.1 s | 1.02–1.08× |

**没有到 2.2×。** 原因见文首第 2 条：往返 75 µs、记账 <1 %、快照恢复 <2 %，能"免费"拿的只有不到
5 %；剩下 95 % 是模拟 tick 与每 tick 一次的学生前向，两项在阶段 1 就已经在 C++ 里跑，搬动调用方
不会让它们变快。真正的 2× 属于"少算"或"把 sim/net 本身做快"，见第 3 节的表。
