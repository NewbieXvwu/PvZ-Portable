# PvZEnv

## 目标与边界

先交付一个可复现、可由 Python 调用的 PvZ 游戏环境，为后续训练自主控制器和研究不同决策方法打底。首个完整闭环使用一关普通白天冒险关，再验证白天、夜晚、泳池、迷雾和屋顶场景的状态与动作接口。

策略观测包含场上全部存活僵尸，即使僵尸位于雾后也可见；雾仍按游戏规则影响画面和战斗。未来详细出怪队列、随机数状态及隐藏计时器只通过研究用完整状态接口提供。环境返回客观事件和胜负结果，奖励由训练端定义。

标准资源基线采用用户提供的 PvZ GOTY English `1.2.0.1073`：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN/main.pak`，SHA-256 为 `7a50e3b247e1b678fa034e0fabe305f419c38a59cdfc5dd379c2f434b5d4a421`；`properties/partner.xml` 的 SHA-256 为 `69d1423cc849e3c4f231db6aa47038efc67bdea677ccee96200911720eadafc3`。资源留在本机下载目录，不进入仓库；Python 入口把资源所在目录传给 `-resdir`。

PvZ 的部分战斗后果由 Reanimation 动画进度、轨道变换和关键事件触发。桌面与环境模式现在共用 `AdvanceLogicTick()`；快照恢复 `mAppCounter`、随机状态、实体和动画。旧的 6 个检查点只覆盖此前的环境循环，当前代码的逐 tick 可视/无画面验证由 `python/verify_env_equivalence.py` 执行。

GitHub fork 为 [NewbieXvwu/PvZ-Portable](https://github.com/NewbieXvwu/PvZ-Portable)。Python 入口 `python/pvz_env.py` 通过 `PvZEnv(resource_dir=...)` 使用游戏资源。训练基线固定为 `playthrough=2`，环境拒绝 `playthrough=1`，直到首次冒险教程流程得到完整实现。

仓库已有 Teacher、BC、DAgger 和 PPO 训练入口，旧训练数据与评估结果列在 `artifacts/data_status.json` 并标记为修复前记录。多进程环境池和批量推理要在基准剖析后实现；教师数据通过完整自动对局生成。

## 阶段一：Fork 上游并建立基线

- [x] 从 PvZ-Portable 官方仓库 fork；`pvz-env` 分支固定基于 `0.2.4`（提交 `a0f676a`），保留官方 `upstream` 远端并记录基线提交。
- [x] 核对并遵守上游 LGPL-3.0-or-later 许可；游戏资源由使用者自行提供，不将资源文件提交到仓库。
- [x] 在 macOS 构建并启动普通可视程序，4 秒渲染循环后干净退出；记录 CMake 3.31.10、Apple Clang 21.0.0、资源路径及启动命令。桌面锁定期间无法目视确认窗口内容。
- [x] 定位游戏主循环、逻辑更新、关卡初始化、随机数、存档和录像回放的实现，确认可复用的入口与需要修改的最小范围。

参考：[PvZ-Portable 发布页](https://github.com/wszqkzqk/PvZ-Portable/releases)、[项目说明与许可](https://github.com/wszqkzqk/PvZ-Portable)。

## 阶段二：在 fork 中实现无画面运行和游戏控制入口

- [x] 增加无画面运行路径，跳过窗口创建、绘制和声音输出，同时完整加载并更新 Reanimation；只批量推进游戏逻辑 tick。已用 1.2.0.1073 资源验证向日葵、僵尸和豌豆发射能在无窗口模式运行。
- [x] 保留可视运行方式；桌面和环境路径共用逐 tick 逻辑入口。`python/verify_env_equivalence.py` 提供逐 tick 观测、事件、完整状态和快照恢复差分验证。
- [x] 增加无需菜单交互的关卡重置入口，接受关卡、随机种子和卡组。
- [x] 暴露游戏语义动作：种植、铲除和等待指定 tick；首版默认自动收集阳光，不模拟鼠标移动与点击。
- [x] 明确非法动作的结果，并提供当前合法动作集合或动作掩码。
- [x] 提供推进到下一个决策时刻的快速接口；完整 1-1 胜局推进 34,475 tick，共 134 次策略动作。300 tick 最短推进下该规则策略获胜，提高到 360 tick 会在第 3 波失守，因此保留 300 tick 响应设置。

## 阶段三：实现环境状态、快照与 Python 接口

- [x] 提供算法无关的 C++ 环境入口，支持重置、动作执行、逻辑推进、观测、完整状态、快照和恢复。
- [x] 提供薄 Python 封装，采用 Gymnasium 风格的重置与步进返回值，不将具体学习算法写入环境核心。
- [x] 结构化观测包含格子、植物、僵尸、阳光、投射物、卡片、波次与合法动作；多地形结构仍待核验。
- [x] 分离策略观测与研究用完整状态。策略观测显示雾后的存活僵尸，但不含未来详细出怪表、RNG 状态和隐藏计时器；研究接口保留这些信息。
- [x] 事件覆盖击杀僵尸、植物被吃、阳光产生与消耗、小推车触发、波次开始及胜负。
- [x] 实现内存 `snapshot/restore`，复用 `.v4` 状态序列化并单独保存随机数发生器和环境事件计数；已核对序列化包含 Reanimation 动画时间、轨道实例及挂接，并在豌豆投射物飞行时恢复快照，后续观测逐项一致。

## 阶段四：可复现记录与场景验证

- [x] 固定输入的冒烟轨迹可复现并生成状态摘要；完整对局的跨进程复现通过。
- [x] 已核验保存快照、执行动作、恢复快照并重放后，观测完全一致。
- [x] v3 回放采用 JSON Lines，可选 gzip；每个实验只写一份含源码修订、脏工作区补丁、构建项、资源与可执行文件摘要的清单。回放逐操作记录实际动作、事件、tick、快照与终局；仍能读取 v2。
- [x] 已在白天、夜晚、泳池、迷雾和屋顶各重置一关，核验背景编号、6×9 网格、投射物字段及合法动作；屋顶卡组使用花盆。
- [x] 在普通白天 1-1 完成完整对局并核验最终胜负：20/20 波、34,475 tick、胜利；新进程逐操作重放 134 条记录，最终状态摘要一致。
- [ ] 修复后重建 Teacher、BC、DAgger、PPO 数据与冻结基准；旧数据状态见 `artifacts/data_status.json`。
- [ ] 运行逐 tick 可视/无画面差分，覆盖舞王、墓碑、迷雾、Boss、卡片冷却和动画中途快照恢复。
- [ ] 测环境 tick 与完整策略吞吐、reset 耗时和 worker 内存，再决定是否引入常驻 worker 池、批量推理或紧凑 IPC。

## 当前验收条件

- Python 无需点击菜单即可重置普通关卡、读取结构化观测并执行种植、铲除、定 tick 等待和推进到决策时刻。
- 策略观测与研究用完整状态隔离，环境不内置奖励函数。
- 固定输入可复现；快照恢复后轨迹一致；录像记录可跨进程重放。
- 可视与无画面逐 tick 状态差分通过。
- 至少覆盖五种常见地形的状态与动作接口，并完成一关普通白天关的完整对局闭环。

## 后续扩展

根据吞吐剖析评估常驻子进程和批量推理；`WAIT_DECISION` 签名优化、资源预派生共享、低内存和仅加载图像元数据只在剖析确认瓶颈后推进。Cob Cannon 控制、快照释放接口、环境专用构建目标及 `.dmo` 与语义回放关联作为可选项。验证与 PopCap 原版的等价性需要对应原版程序和 `properties` 作为外部参照。

## 本机运行记录

- 构建：`cmake -G Ninja -B build`，然后 `cmake --build build --parallel 4`；本次使用 CMake 3.31.10 和 Apple Clang 21.0.0。
- 普通可视程序：`./build/pvz-portable -resdir "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN" -savedir /tmp/pvz-visible-startup`。
- Python 环境：`PYTHONPATH=python` 后导入 `from pvz_env import PvZEnv`，并以 `PvZEnv(resource_dir="/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")` 创建实例；`headless=False` 可切换到可视控制模式。
- 资源 SHA-256：`main.pak` 为 `7a50e3b247e1b678fa034e0fabe305f419c38a59cdfc5dd379c2f434b5d4a421`，`properties/partner.xml` 为 `69d1423cc849e3c4f231db6aa47038efc67bdea677ccee96200911720eadafc3`。
