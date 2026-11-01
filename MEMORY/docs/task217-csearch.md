# task217b — 单次决策的时间分布与提速选项

（用户 2026-11-01 的任务书第 1 条。测量用新脚本 `scripts/csearch_profile.py`：真实学生整局，逐决策跑
一次生产预算的 `SEARCH`（C++ 实现），读引擎自己的分段计时。同一时刻只有一个进程、`nice -n 19`。）

## 结论先说

**时间花在模拟和网络前向，不花在来回通信上。** 一次决策的中位数（L28/L29/L33/L44，各 10 个决策）是
50–140 ms，其中引擎模拟 21–31 %、学生前向 14–36 %、静态打分 4–7 %、候选记账 14–55 %、快照恢复 <1 %。
把整次决策搬进 C++ 只赚到 1.03–1.08×，正是因为每条命令本来就要跑 50–90 ms 的模拟，而一次往返只有
约 75 µs；20 条命令合计约 1.5 ms，不到一次决策的 2 %。**继续靠"少通信"提速的空间已经基本榨干**，
剩下的 2× 只能从"少算"里拿。

每个决策实际干了多少活（中位数）：

| 关卡 | 决策数 | 模拟 tick | 学生前向 | 叶值前向 | rollout | 其中验证 rollout | 静态打分 |
|---|---|---|---|---|---|---|---|
| L28 | 10 | 23 940 | 429 | 9.5 | 9.5 | 0（坏情况 18） | 68.5 |
| L29 | 10 | 26 460 | 463 | 10.5 | 10.5 | 0 | 73 |
| L33（雾） | 10 | 25 326 | 426 | 10 | 10 | 0 | 17 |
| L44（屋顶） | 10 | 30 240 | 548 | 12 | 12 | 0 | 47.5 |

带验证的决策贵一倍：L28 第一个决策 90 次打分 + 18 个验证 rollout = 214 ms（同关无验证的决策 47–49 ms）。
**所以"决策有多贵"主要由验证轮决定，不由通信决定。**

## "候选记账"到底是什么

它和静态打分次数强相关（17 次打分→5–8 ms；50–120 次打分→60–150 ms），但**不在**打分本身（打分含
plant 与 net 前向，中位 0.12 ms/次）。逐项看是每格一次的 `CandidateKey`（`"p<slot> c<col> r<row>[ u<up>]"`）
字符串构造 + `unordered_map` 插入/查找、`BreaksReserve` 对 `legal` 的线性扫描、`TerminalLeaf`、
候选/取值向量的反复扩容。**这部分是纯记账，一次 0.7–1.2 ms，比它保护的 net 前向贵 6–10 倍。**

## 能拿到 2× 的选项（按收益排序，都只是提案）

| # | 选项 | 估计收益 | 对结果的影响 |
|---|---|---|---|
| 1 | 候选记账去字符串化：`CandidateKey` 换成候选下标并列数组 + `reserve()`，去掉 `unordered_map<string,…>` 与 `BreaksReserve` 的重复扫描 | 记账 14–55 % → 约 5 %，**单项约 1.5×** | 纯重构，逐位一致 |
| 2 | 学生前向批量化：同一候选的各格、同一轮的各候选凑成一次 batch 前向 | 学生 14–36 % → 约 4 %，**单项约 1.25×**；与 1 合用约 2.2× | 批内互不影响时逐位一致；若连带改"每 tick 一次前向"的顺序则会改对局 |
| 3 | 抬高 `--rollout-sim-skip`（61→91/121）或只在有僵尸的格子附近逐 tick 推进 | 模拟 21–31 % 砍 30–40 %，整决策 −7…−12 % | 特征混叠加重，取值噪声变大，接近平局时可能换动作（阶段 0 已量化过 skip 的影响） |
| 4 | 减少验证：`--verify-samples 9→5`，或只验证前 2 名，或复用减半阶段已跑过的未来 | 带验证决策 −30…−50 % | 验证均值方差变大，翻转率上升 |
| 5 | 静态打分抽样（每包只打 1/2…1/4 的格子），或直接用学生在根上对同一动作的值 | 静态+记账 −25…−50 % | 抽样会改选格，平局处可能改动作；用学生值无需额外前向但语义变了 |
| 6 | 加大 `--min-gap`（300→600） | 决策频率减半，整局 −30 % 左右 | 两次搜索之间学生自由发挥更久，质量影响未测 |
| 7 | 快照复用（plant 后 undo 代替 restore） | **<1 %**，因为恢复本来就只有 0.3–2 % | 逐位一致 |
| 8 | 叶值前向批量化 | **<1 %**，每次 rollout 只有 1 次 | 逐位一致 |

**没有任何单项能到 2×；1+2 合起来能到 2.2×，且都不改结果。** 3/4/5 是另一条轴：它们换质量换速度，
必须单独做一致性对比才能上线。

---

# task217 — 把老师搜索整棵树搬进 C++

目标（用户 2026-11-01）：候选生成、逐卡选格、减半、验证、保存探针、组合全部在 Python，只有单个续走
ROLLOUT / ROLLOUT_VALUE / LEAF_VALUES / STATIC_SCORES / BRANCH_SNAPSHOT_FAST / SIM_SKIP 走 C++，
**一次决策十几次往返**（违反 AGENTS.md §6.6 规则 2），也把后续加深搜索的空间压住了。
本任务把整次决策变成一个 C++ 命令 `SEARCH`。

本文件分三部分：**A 阶段 0**（把现行 Python 算法完整写成规格 + C++ 侧设计 + 迁移后 Python 薄面），
**B 阶段 1**（实现与一致性/性能验收），**C 阶段 2 计划**（全量对比 + 删除 Python 版）。
任务书只要求阶段 0 + 阶段 1，阶段 1 在 `--search-impl` 开关后面，默认仍是 Python。

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

留在 Python 的只有：`--search-impl {python,cpp}` 开关、`_cpp_search()`（snapshot → 一条 SEARCH →
restore/release → 把计数器差值、`last_targets`、`last_verification` 抄进 stats）、
`pvz_env.PvZEnv.search()`（拼命令）、`_demo_candidate()`、`_probe_decision()`（leaf-probe 文件）。
阶段 2 对比通过后删除：`_search` 的候选/续走/减半/验证主体、`_halving_choice`、`_verified_choice`、
`_save_plan`、`_save_probe`、`_save_value`、`_combo_rollout`、`_static_scores`、`_leaf_values`、
`_continuations`、`_student_topk_candidates`、`_score_terminal_or_value`、`_margin_threshold`、
`_rollout_policy` 以及 `--search-impl` 开关本身；`_search` 只留入口壳或直接改叫 `_search` 调 C++。

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

## C. 阶段 2 计划（未开始）

1. `--search-impl cpp` 变成默认；`scripts/csearch_consistency.py` 保留为一次性对比工具。
2. 扩对比面：多关卡（L28/L29/L30/L42）、多卡组、`--save-candidates 2`、`--combo-candidates 2`、
   `candidate_mode=student-topk`；整局同种子 A/B 至少 30 局，要求胜负、波数、决策序、覆盖数全同。
3. 性能：先按 §6.6 规则 3 补测更大预算（samples 8 / horizon 6000 / per-card 2）下的每决策时间与
   每 1000 tick 时间，确认往返开销消失后剩下的全在模拟与网络前向内；再决定是否把快照恢复
   （每候选一次 `RestoreEnvironmentSnapshot`）与 leaf 评估也内联到无拷贝路径。
4. 全绿后在同一次改动里删掉 A5 列出的 Python 函数与 `--search-impl`，然后才更新文档与生产命令。
