# PvZEnv

## 当前目标

构建可复现、可批量运行、可由 Python 控制的 Plants vs. Zombies 环境，并用高保真模拟器搜索生成教师数据，训练最终自主控制器。

当前训练基线使用 Adventure-II、`playthrough=2`。策略观测包含场上全部存活僵尸；未来出怪表、随机数状态和隐藏计时器只属于研究用完整状态。环境只返回客观状态、事件和胜负，奖励与搜索评价由训练端定义。

标准资源基线采用 PvZ GOTY English `1.2.0.1073`。游戏资源不进入仓库，Python 入口通过 `PvZEnv(resource_dir=...)` 指定资源目录。

## 已完成

- 无画面环境与可视模式共用 `AdvanceLogicTick()`，支持确定性 reset、plant、shovel、固定 tick 等待和事件等待。
- 环境协议升级到 v2；Python 只接受 v2，reset 使用 `RESET_V2`，旧二进制会在握手阶段直接失败。
- 结构化观测覆盖格子、植物、僵尸、阳光、投射物、卡片、波次、合法动作与客观事件。
- 内存 snapshot/restore 保存棋盘、随机状态、计数器和动画相关状态；`verify_env_equivalence.py` 用于可视/无画面与恢复后的逐 tick 等价性验证。
- `WAIT_DECISION` 使用自适应反应窗口和紧凑状态签名；危险状态可更快返回。
- 搜索路径支持轻量快照命令和 `BRANCH_SNAPSHOT_FAST`；批量分支同时返回完整快照状态哈希，用于 transposition 去重。
- SearchTeacher 完全独立于学生网络。候选只来自合法动作、lane pressure、空间多样性、铲除与时间动作；学生策略和值头不会参与教师候选或叶子打分。
- 搜索由 `horizon_ticks` 定义游戏时间范围，由整次决策共享的 `simulation_budget` 定义计算预算，并用 successive halving 把更多模拟分配给更有希望的根动作；`max_same_tick_actions` 只负责阻止零 tick 动作无限展开。
- 搜索结果按 `确定胜利 > 未终局 > 确定失败` 排序；BC、搜索与 PPO 使用统一的按游戏 tick 折扣终局价值语义。
- 独立 `SearchValueModel` 只学习模拟器轨迹的真实折扣终局结果。冷启动、value refinement、最终策略教师数据使用互斥 seed；最终 SearchTeacher 使用冻结的独立 value 模型。
- 训练期模型选择只使用冻结 development 256 seeds；final-test 1024 seeds 与训练、DAgger、value bootstrap/refinement、development 全部隔离，只用于最终验收。
- BC、DAgger、PPO 训练入口和冻结 seed benchmark 已接入当前模型与搜索教师。

## 当前验收条件

- Python 无需菜单交互即可重置普通关卡并完成完整自动对局。
- 固定 seed 与固定动作序列可复现；snapshot 恢复后重放轨迹一致。
- 搜索教师在固定 `horizon_ticks` 下比较分支，整次决策的模拟总量不超过 `simulation_budget`。
- 搜索中相同 `(state_hash, elapsed_ticks, same_tick_actions)` 状态只保留得分更高的路径。
- 训练、DAgger、value bootstrap、value refinement、development 和 final-test seed 集各自唯一且两两互斥。
- checkpoint 必须声明并匹配当前协议、模型架构、观测版本、任务版本、搜索标签版本和 value semantics。
- final-test 1024 seeds 在最终验收前不得用于训练、调参或模型选择。

## 接下来

- 跑完整 C++ 构建以及逐 tick 可视/无画面差分，覆盖舞王、墓碑、迷雾、Boss、卡片冷却和动画中途 snapshot 恢复。
- 在 development 256 seeds 上校准 search horizon、总 simulation budget、beam width 与 SearchValueModel；只依据 development 结果做选择。
- 锁定配置后生成大规模搜索监督与 DAgger 数据，并在 final-test 1024 seeds 上做一次最终验收。
- 剖析搜索吞吐；多分支 rollout 已批量下沉到 C++，后续只针对实际 profile 中仍占主要成本的保存/恢复或状态序列化继续优化。
- 扩展到白天、夜晚、泳池、迷雾和屋顶等地形，检查状态候选覆盖和 SearchValueModel 在不同卡组上的泛化。
