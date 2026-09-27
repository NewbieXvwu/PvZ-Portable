# PvZEnv

## 当前目标

构建可复现、可批量运行、可由 Python 控制的 Plants vs. Zombies 环境，并用高保真模拟器搜索生成教师数据，训练最终自主控制器。

当前训练基线使用 Adventure-II、`playthrough=2`。策略观测包含场上全部存活僵尸；未来出怪表、随机数状态和隐藏计时器只属于研究用完整状态。环境只返回客观状态、事件和胜负，奖励与搜索评价由训练端定义。

标准资源基线采用 PvZ GOTY English `1.2.0.1073`。游戏资源不进入仓库，Python 入口通过 `PvZEnv(resource_dir=...)` 指定资源目录。

## 已完成

- 无画面环境与可视模式共用 `AdvanceLogicTick()`，支持确定性 reset、plant、shovel、固定 tick 等待和事件等待。
- 结构化观测覆盖格子、植物、僵尸、阳光、投射物、卡片、波次、合法动作与客观事件。
- 内存 snapshot/restore 保存棋盘、随机状态、计数器和动画相关状态；`verify_env_equivalence.py` 用于可视/无画面与恢复后的逐 tick 等价性验证。
- `WAIT_DECISION` 使用自适应反应窗口和紧凑状态签名；危险状态可更快返回。
- 搜索路径支持轻量快照命令和 `BRANCH_SNAPSHOT_FAST`，减少搜索中的 IPC 与无用观测序列化。
- SearchTeacher 只依赖模拟器、合法动作、通用状态几何和可选模型候选提案。搜索由 `horizon_ticks` 定义游戏时间范围，由 `simulation_budget` 定义计算预算，并用 `max_same_tick_actions` 防止零 tick 动作无限展开。
- 搜索结果按 `确定胜利 > 未终局 > 确定失败` 排序；BC、搜索与 PPO 使用统一的按游戏 tick 折扣终局价值语义。
- BC、DAgger、PPO 训练入口和冻结 seed benchmark 已接入当前模型与搜索教师。

## 当前验收条件

- Python 无需菜单交互即可重置普通关卡并完成完整自动对局。
- 固定 seed 与固定动作序列可复现；snapshot 恢复后重放轨迹一致。
- 搜索教师在固定 `horizon_ticks` 下比较分支，计算预算不会改变游戏时间语义。
- 教师、DAgger 和评估 seed 集唯一且互斥。
- 冻结测试集至少包含 256 个唯一 seed，训练数据不得使用这些 seed。
- checkpoint 必须声明并匹配当前模型架构、观测版本、任务版本和 value semantics。

## 接下来

- 从当前 SearchTeacher 生成搜索监督与 DAgger 数据。
- 在冻结 256 seeds 上测 SearchTeacher 与训练后模型的胜率、失败波次、植物损失、割草机触发、搜索模拟量、策略熵和吞吐。
- 跑完整 C++ 构建以及逐 tick 可视/无画面差分，覆盖舞王、墓碑、迷雾、Boss、卡片冷却和动画中途 snapshot 恢复。
- 剖析搜索吞吐；多分支 rollout 已批量下沉到 C++，后续只针对实际 profile 中仍占主要成本的保存/恢复或状态序列化继续优化。
- 扩展到白天、夜晚、泳池、迷雾和屋顶等地形，检查候选生成与模型动作头在不同卡组上的覆盖率。
