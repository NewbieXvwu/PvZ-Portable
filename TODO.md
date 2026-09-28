# PvZEnv

## 当前目标

构建可复现、可批量运行、可由 Python 控制的 Plants vs. Zombies 环境，并用高保真模拟器搜索生成教师数据，训练最终自主控制器。

当前训练基线使用 Adventure-II、`playthrough=2`。策略观测包含场上全部存活僵尸；未来出怪表、随机数状态和隐藏计时器只属于研究用完整状态。环境只返回客观状态、事件和胜负，奖励与搜索评价由训练端定义。

标准资源基线采用 PvZ GOTY English `1.2.0.1073`。游戏资源不进入仓库，Python 入口通过 `PvZEnv(resource_dir=...)` 指定资源目录。

## 已完成

- 无画面环境与可视模式共用 `AdvanceLogicTick()`，支持确定性 reset、plant、shovel 和固定 tick 等待。
- 环境协议升级到 v2；Python 只接受 v2，reset 使用 `RESET_V2`，旧二进制会在握手阶段直接失败。
- 结构化观测覆盖格子、植物、僵尸、阳光、投射物、卡片、波次、合法动作与客观事件。
- 内存 snapshot/restore 保存棋盘、随机状态、计数器和动画相关状态；`verify_env_equivalence.py` 用于可视/无画面与恢复后的逐 tick 等价性验证。
- 搜索路径支持轻量快照命令和 `BRANCH_SNAPSHOT_FAST`；批量分支同时返回完整快照状态哈希，用于 transposition 去重。
- SearchTeacher 完全独立于学生网络。候选只来自合法动作、lane pressure、空间多样性、铲除与时间动作；学生策略和值头不会参与教师候选或叶子打分。
- 搜索同时受 `horizon_ticks`、`max_decisions` 和整次决策共享的 `simulation_budget` 限制，并用 successive halving 把更多模拟分配给更有希望的根动作；`max_same_tick_actions` 也会阻止零 tick 动作无限展开。
- 搜索结果按 `确定胜利 > 未终局 > 确定失败` 排序；BC、搜索与 PPO 使用统一的按游戏 tick 折扣终局价值语义。
- 独立 `SearchValueModel` 只学习模拟器轨迹的真实折扣终局结果。冷启动、value refinement、最终策略教师数据使用互斥 seed；最终 SearchTeacher 使用冻结的独立 value 模型。
- 训练期模型选择只使用冻结 development 256 seeds；final-test 1024 seeds 与训练、DAgger、value bootstrap/refinement、development 全部隔离，只用于最终验收。
- BC、DAgger、PPO 训练入口和冻结 seed benchmark 已接入当前模型与搜索教师。
- 训练采集与 benchmark 共用 spawn worker 和逐 seed 原子分片；任务 metadata 匹配时可跳过已完成 seed，汇总后统一重算指标。训练入口有 `--value-only`，避免 value checkpoint 后误跑普通搜索采集。
- 搜索决策落盘时区分**筛选**与**深度**两段模拟预算（`screening_simulations` /
  `depth_simulations` / `effective_depth_budget`），因为 `simulation_budget` 不是搜索深度：
  每个根候选在任何一条线展开前都要先花 1 次模拟。
- BC 除 argmax 行为克隆外，还用 `--soft-label-weight`（默认 0.5）蒸馏搜索教师在根候选集上的
  完整分布；`0` 精确还原旧行为。
- `python/pvz_search_audit.py` 在 development 种子上审计两件设计无法自证的事：
  值模型在**反事实一步子节点**上的兄弟排序是否与更深搜索一致（`sibling_ranking`），
  以及候选生成 + 筛选两级各自丢掉了多少合法动作（`candidate_recall`）。
- 存档改为只使用显式字段保存状态，移除了旧版整块对象布局存档和裸指针恢复路径；稳定化消息文本尾部与运行时 `mGameID`，补齐棋盘计数器字段。
- 两个全新进程的 30 个快照样本、240 个分支哈希逐字节一致；`verify_env_equivalence.py` 全量通过，SearchTeacher 在 development seeds 30000–30001 上等价。Linux g++ 构建通过，SDL-Mixer-X 的 C++ `mp3utils_test` 为 1/1，Python unittest 为 192/192。
- 台式机已完成 SearchValue v2 正式管线：bootstrap seeds 20000–20031、refinement seeds 21000–21031，各 32 集；两段各训练 8 epoch。loss 历史分别从 0.152216 降至 0.009705、从 0.049708 降至 0.008806，完整历史已写入 checkpoint。
- `artifacts/adventure2_level7/search_value_v2.pt` 已通过 CPU 重载与版本、feature、task signature 核验；protocol=3、search labels=3、feature=2、value semantics=`discounted_terminal_v1`。bootstrap/refinement、train 0–63、DAgger 10000–10063、development、final-test seed 集两两无交叉。
- SearchValue v2 默认教师完成 development 256：0 胜，Wilson 95%=[0, 1.48%]，平均终局波次 3.570，75.82 动作/局，11.31 s/局，吞吐 1208 episodes/hour；筛选/深度模拟 374,511 / 2,405,303，平均 `effective_depth_budget` 236.67。
- 搜索消融报告位于台式机 `artifacts/adventure2_level7/search_ablation_dev256_comparison.json`。候选上限 12 全量 256 仍 0 胜，配对终局波次 57 高于默认、51 低于默认，均值仅 +0.016；simulation budget 128 全量 256 仍 0 胜，吞吐 1963 episodes/hour，但平均终局波次降至 3.355。budget512、width4 仅完成 32-seed 筛选，均无胜利且无明确收益。#10 采用默认 width3 / candidates8 / budget256。
- #10 完成：train 0–63 搜索监督轨迹 64 集，DAgger 10000–10063 轨迹 64 集；`search_trajectories.json.gz` 与 `dagger_search_trajectories.json.gz` 的 checkpoint SHA256 均匹配。最终 `gameplay_model_v1.pt` 可重新加载，train、DAgger、value bootstrap/refinement、development、final-test seed 两两互斥。
- DAgger 重训 8 epoch loss 为 2.48755、2.42823、2.39204、2.35809、2.32229、2.28290、2.24175、2.20700。最终模型在 development 256 上 0 胜，平均终局波次 3.398、平均动作 55.86；checkpoint 与 `training_summary.json` 均已落盘于台式机 `artifacts/adventure2_level7/`。

## 当前验收条件

- Python 无需菜单交互即可重置普通关卡并完成完整自动对局。
- 固定 seed 与固定动作序列可复现；snapshot 恢复后重放轨迹一致。
- 搜索教师在固定 `horizon_ticks` 下比较分支，整次决策的模拟总量不超过 `simulation_budget`。
- 搜索中相同 `(state_hash, elapsed_ticks, decision_count, same_tick_actions)` 状态只保留得分更高的路径。
- 训练、DAgger、value bootstrap、value refinement、development 和 final-test seed 集各自唯一且两两互斥。
- checkpoint 必须声明并匹配当前协议、模型架构、观测版本、任务版本、搜索标签版本和 value semantics。
- final-test 1024 seeds 在最终验收前不得用于训练、调参或模型选择。

## 接下来

- 台式机开发样本 4 seeds 吞吐矩阵：1×1 为 35.2 s、2×1 为 21.4 s、4×1 为 14.7 s、4×2 为 14.6 s，RSS 约 231 MiB/worker；采集采用 `workers=4`、`collection_threads=1`。
- 在冻结 final-test 40000–41023 上对最终模型做一次验收，随后检查多地形与不同卡组的候选覆盖、SearchValueModel 泛化。
- 剖析搜索吞吐；多分支 rollout 已批量下沉到 C++，后续只针对实际 profile 中仍占主要成本的保存/恢复或状态序列化继续优化。
  - **已完成（Python 侧）**：`search_value_features` 改为 numpy float64 累加器 + 视图，
    真实观测上 122.8 µs → 23.2 µs（5.3×），逐位等价（见 `CODE_REVIEW.md` §5.2 与
    `python/test_search_value_features.py`、`scripts/real_env_equivalence.py`）。
    参考模拟器下 `advice()` 30.0 ms → 8.6 ms（3.49×）；真实模拟器下 186.5 ms → 164.6 ms（1.13–1.17×）。
  - **下一步（C++ 侧）**：真实环境下 `advice()` 有 66% 花在等待 C++ 返回。已实测每分支固定开销
    ~290 µs = `EnvironmentObservation` 90 µs + `LawnSaveGameToMemory` 101 µs +
    `LawnLoadGameFromMemory` + 对整块 `board` 做 FNV-1a 的 `EnvironmentSnapshotHash`，
    另有 1.7 µs/tick 的模拟成本。动 `BRANCH_SNAPSHOT_FAST` 前需先把
    `verify_env_equivalence.py` 变成可自动化的回归。**该回归已就位**（mutation 验证见
    `CODE_REVIEW.md` §5.6）。
  - **已完成（C++ 侧，两轮）**：新协议命令 `BENCH_SNAPSHOT <reps>` 给出进程内分项剖面
    （`scripts/branch_benchmark.py` 给的是含传输与 JSON 解析的端到端口径）。在 level 8 /
    seed 30000 / tick 2192 的板面（4 植物 / 10 僵尸 / 19 动画，载荷 86,424 B）上：

    | 成分 | 轮次前 | 轮次后 |
    |---|---|---|
    | `EnvironmentSnapshotHash` | 77.2 µs | **9.7 µs**（8.0×） |
    | `LawnSaveGameToMemory` | 89.1 µs | 86.8 µs |
    | `LawnLoadGameFromMemory` | 68.7 µs | 68.4 µs |
    | `EnvironmentObservation` | 23.2 µs | 22.2 µs |
    | **端到端单分支** | **343.8 µs** | **256.1 µs**（−27%） |

    * 轮次一：`SaveGame.cpp` 的 `WriteChunkV4` 直接写进 payload，去掉「字段 writer → chunk
      writer → 临时 vector → AppendChunk」的三遍拷贝；48 个板面状态新旧写入器逐字节相同
      （`identical: true` / `stable: true`）。
    * 轮次二：`EnvironmentSnapshotHash` 从逐字节 FNV-1a 改成 **64 位字折叠 + MurmurHash3
      fmix64 雪崩**。逐字节版的每个输入字节都是串行 xor/乘法链上的一环，86 KB 上花 77 µs。
      **哈希值本身是不透明的转置键，改写不算兼容性破坏**；雪崩实测（3000 次真实载荷单比特翻转）
      反而更好：均值 32.08 bits（旧 30.74）、最低 20 bits（旧 19）、碰撞 0。
    * 等价性：`scripts/binary_search_equivalence.py` 用两个二进制在同一真实状态下跑
      `SearchTeacher.advice`，比较动作、全部候选打分、蒸馏策略、margin、终局、模拟计数与
      筛选/深度预算拆分 —— **3/3 种子逐位一致**，真实 `advice()` 159.8 → 142.7 ms（1.12–1.16×）。
      该脚本刻意**不比较 `state_hash`**：它是转置键，改写哈希必然改变数字但不应改变任何去重决策，
      比较它会在一次合法优化上误报。
  - **指针修复后的实测**：同一 level 8 / seed 30000 / tick 2192 板面载荷为 86,976 B，较修复前 86,424 B 增加 552 B（0.64%）；save 84.1 µs、restore 70.2 µs、hash 9.7 µs、observation 21.3 µs，与此前测量处于同一量级。
  - **剩下的成本（未动）**：`LawnSaveGameToMemory` 与 `LawnLoadGameFromMemory`
    仍是 C++ 侧大头；另有约 73 µs/分支落在 Python 侧的 JSON 解析与管道传输上
    （`OBS` 端到端 83 µs 而 C++ 内只 22 µs）。单个叶子之外还有一个结构性机会：
    一批 32 个分支里真正被保留下来的快照远少于 32，为「先不存快照地评估、只重算保留下来的那几个」
    加一条命令可以省掉被丢弃分支的 86.8 µs/个；这是协议与搜索的改动，不是纯 C++ 优化，
    先量清楚存活率再决定。
- **已完成（指针序列化修复）**：v4 的 Board 及子对象字段逐项序列化，不再保存对象内存布局或裸指针；旧版 raw-layout 加载器已删除（允许不兼容旧存档）。`mGameID` 写固定值，消息标签零填充未初始化尾字节，并补齐 `mTextReanimByteOffset`、`mTextReanimCount`、`mBoardUpdateCounter`、`mZombiesKilled`、`mSunMoneyProduced`。同一板面双进程快照逐字节一致；`verify_env_equivalence.py` 7 个关卡全部通过（包括 rewind stress、snapshot restore、`BRANCH_SNAPSHOT_FAST`），SearchTeacher 两颗开发种子所有候选分数、动作、策略分布、margin、终局及模拟预算拆分一致。Linux 项目构建通过；CTest 根目录没有登记项目测试，SDL-Mixer-X 子目录 `mp3utils_test` 1/1 通过。
- 扩展到白天、夜晚、泳池、迷雾和屋顶等地形，检查状态候选覆盖和 SearchValueModel 在不同卡组上的泛化。

## 已排除的路线（有实测数据，勿重复尝试）

- **降精度换速度**：真实模拟器 160 次决策的误差预算约 1e-5~1e-4（margin 中位 6.2e-4，
  排除 10.6% 的精确并列后没有任何一次低于 1e-7）。但 fp16/bf16/稀疏首层/float32 累加器
  **全部比全精度更慢**（0.09×~1.00×）。**在 Apple M5 Pro 上用真实负载形状重测过**：
  fp16 全 half 0.24×、bf16 首层 0.33× —— Apple 的矩阵协处理器（AMX）是 **fp32** 单元，
  M 系列没有比 fp32 吞吐更高的 fp16/bf16 路径。这是结构性的，不是配置问题。
  见 `CODE_REVIEW.md` §6.2。换到非 Apple 硬件仍需重测这一条。
- **调 CPU 线程数**（**结论已更正：这条只在 Apple Silicon 上成立，不是普适结论**）：
  M5 Pro 上 threads=1/2/5/10/15 下 batch=1 恒为 15.1–15.3 µs、batch=256 恒为
  272.7–275.2 µs，Accelerate/AMX 已在单线程吃满这条路径；`torch.get_num_threads()`
  默认返回 5 是 P 核数，不是配置错误。**但在 x86 上完全相反**：i7-12700F 上
  threads=4 比 threads=1 快 2.0×（leaf）/1.7×（64 步训练），一次完整训练
  32.0 min → 21.3 min。因此这**不再是「已排除」**，而是改成了可配参数
  `--threads`（默认 4，`--threads 1` 复现旧产物），数值扰动 4e-7~6e-7，
  低于误差预算两个数量级。见 `CODE_REVIEW.md` §6.6。
- **把值模型放到 MPS**：batch=1 的形状上 MPS 全面更慢 —— `SearchValueModel.predict`
  68 µs → 451 µs，真实模拟器上单次 `advice()` 162 ms → 282 ms，整集 rollout 13.2 s → 22.6 s
  （1.71×）。一次完整训练净亏约 28 分钟。**`resolve_device("auto")` 已改为不再考虑 MPS**；
  显式 `--device mps` 仍可用。见 `CODE_REVIEW.md` §6.5。
- **指望 `encoder batch=64` 的 MPS 收益（2.44×）**：这个形状在真实训练路径里**不存在**。
  `pvz_imitation.train` 的 `for start in range(0, len(steps), 64)` 只是梯度累积窗口，
  循环体里仍是逐条 `model.step()`（GRU 隐状态串行传递），张量维度恒为 1。
  除非把 encoder 跨 episode 批量化（大改，且 GRU 仍串行），否则拿不到。
- **叶子估值批量化**：每批只有约 4.3 行，被 `F.linear` 非连续转置权重在 M≥2 时的
  ~44 µs 固定开销吃光（14.79 vs 15.07 µs/row）。见 `CODE_REVIEW.md` §5.4。
- **对特征向量做稀疏化**：特征只有 1.1%–3.3% 非零，但 `Linear(4116,128)` 在 batch=1 只要
  6.59 µs（权重命中 L2/L3），`torch.nonzero` + fancy indexing 的索引构造开销远超省下的乘加，
  实测慢 5.6 倍。

**要提速只能「少算」，不能「算粗」**：减少叶子估值次数、更激进的剪枝、
把 C++ 往返批得更狠。真实环境下 Python 侧全部优化到零端到端也只能 2.9×。

## 两台机器的推荐配置

实测于 Apple M5 Pro 与 i7-12700F + RTX 5080（`scripts/thread_effect.py` 的决策矩阵，
按「192 集 rollout + 32768 步训练」折算）：

| 机器 | 推荐 | 实测最优 | 实测最差 |
|---|---|---|---|
| Apple M5 Pro | `--device cpu`（默认即是），不要用 MPS | 11.3 min（值模型 CPU + 学生网络 MPS 拆设备） | `--device mps` 64.7 min |
| i7-12700F + RTX 5080 | `--device cuda --threads 4` | **15.4 min** | `--device cpu --threads 1` 32.0 min |

`--device auto` 在两边都会选对（CUDA 优先，否则 CPU），所以台式机上真正需要显式加的
只有 `--threads 4` —— 而它已经是默认值。**唯一需要显式指定的是「要与旧产物逐位对齐」时的
`--threads 1`。**
