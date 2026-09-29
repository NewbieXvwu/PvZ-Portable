# 会话上下文压缩摘要

## 1. 项目背景

**PvZAgent / PvZ-Portable**：基于 PvZ1 模拟器（`pvz-portable`，C++20 + Python）魔改的 AI 训练环境。核心组件：`SearchTeacher`（纯状态 beam search 教师）、`SearchValueModel`（4116 维特征 v2）、BC/DAgger/PPO 训练入口、冻结 seed benchmark。最终目标：教师数据 → 训练自主控制器。

- **本机仓库**：`/Users/newbiexvwu/PvZAgent`，分支 `pvz-env`，origin `github.com/NewbieXvwu/PvZ-Portable`（`upstream` 是 wszqkzqk/PvZ-Portable）。
- **台式机仓库**：`~/PvZ-Portable`（WSL），已同步到最新提交。
- 本机游戏资源：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`。
- 标准基线：Adventure-II，level 7，`playthrough=2`，六卡槽无商店。
- 冻结种子集（`artifacts/adventure2_level7/seeds/`）：**train 0–63、dagger 10000–10063、value_bootstrap 20000–20031、value_refinement 21000–21031、development 30000–30255（256）、final_test 40000–41023（1024）**，两两互斥。
- 最终验收只在 final-test 1024 上做一次，不得用于训练/调参。

## 2. 用户的常设指令

1. **所有上报必须中文**，包括干到一半的短反馈。
2. 持续工作直到需要 GPU；现在 GPU 已可用（台式机）。
3. **一切改动自由授权，无需任何向后兼容**。
4. **传文件走 git**：提交并推送。台式机无法直接连接 GitHub 时，可以在本机和台式机之间通过 git bundle 等 Git 机制同步。
5. 本机任何 Agent 统一用 mise 的 Python，已写入用户级 `~/.workbuddy-ai/AGENTS.md`。
6. 台式机 Python 用**用户自备的 `~/.venvs/ml`**（torch 2.14.0+cu132，CUDA 可用）。**不要再装 torch、不要再建 venv**（我曾误建 `~/ml`，已删，并清掉我产生的 3.4 GB pip 缓存）。
7. 今晚属于长时间无人值守推进。对于普通实现取舍、性能参数选择、失败重试、测试修复等事项，自行分析后继续，不要因这些问题停下来等待人工确认。
8. 每完成一个可以独立验证的阶段就提交，避免把一整夜成果堆成一个巨型未提交工作区。

## 3. 台式机接入（关键操作知识）

- 链路：`ssh newbiexvwu@10.81.68.81` → Windows PowerShell → `wsl -d Ubuntu -e bash`。
- 助手脚本：`scripts/win_ssh.py`，`--host` 或 `$PVZ_DESKTOP_HOST`；`ps` / `wsl` 两种模式；`wsl` 可带 `--venv /home/newbiexvwu/.venvs/ml --interpreter python3`（**必须传 WSL 绝对路径**，Mac shell 会把 `~` 展开掉）。密钥登录、BatchMode，不弹密码。
- 需要在本机和台式机间同步代码时，请使用Git。本机和台式机与 Github 间的连接良好。
- 桌面资源：`~/.cache/pvz-research-resources`，main.pak 与 partner.xml 的 sha256 与本机**完全一致** → `task_signature` 跨机对齐。不要再花时间校验它们。
- 桌面 C++ 已重建（Release，g++，39 s），**191 个单测在 Linux 构建上全绿**。
- 本机 Python：`/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3`

## 4. 本会话完成的工作

**两个提交已推送**：
- `2e62d07` Make snapshot restore lossless and provably so — 修 `DataArray::mNextKey` 跨代不重置（同进程第二个 episode 不可复现的根因）；修 restore 丢状态（SIGSEGV）；重写 `verify_env_equivalence.py` 为可自动化回归并用变异测试证明有效。
- `8616b30` Cut BRANCH_SNAPSHOT_FAST cost 27% and prove the search is unchanged。

**任务 #7 完成**（level 8 / seed 30000 / tick 2192，载荷 86,424 B）：

| 成分 | 前 | 后 |
|---|---:|---:|
| `EnvironmentSnapshotHash` | 77.2 µs | **9.7 µs** |
| `LawnSaveGameToMemory` | 89.1 µs | 86.8 µs |
| `LawnLoadGameFromMemory` | 68.7 µs | 68.4 µs |
| `EnvironmentObservation` | 23.2 µs | 22.2 µs |
| **端到端单分支** | **343.8 µs** | **256.1 µs** |

- 轮次一：`WriteChunkV4` 直接写进 payload，去掉三遍拷贝，48 个板面逐字节相同。
- 轮次二：逐字节 FNV-1a → **64 位字折叠 + MurmurHash3 fmix64**。雪崩实测（3000 次单比特翻转）均值 32.08 bits（旧 30.74）、最低 20（旧 19）、碰撞 0 —— 比旧的混得更好。
- 新命令 `BENCH_SNAPSHOT <reps>`（保留的进程内分项剖面）。
- 新脚本 `scripts/binary_search_equivalence.py`：双二进制跑 `advice()`，比动作/候选分/策略/margin/终局/模拟计数/预算拆分，**3/3 种子逐位一致**，真实 `advice()` 159.8→142.7 ms（1.12–1.16×）。**刻意不比 `state_hash`**。
- `TODO.md` 已更新实测数据；记忆文件 `2026-09-28.md` 有第十、十一轮记录。

## 5. 待修复：存档载荷里有裸指针（必须做）

`SyncBoard` 用 `SyncBytes(&theBoard->mPaused, sizeof(Board) - offset)` 整块序列化 `Board`，另有 `sizeof(SeedBank)` / `sizeof(Challenge)` / `sizeof(CursorObject)` / `sizeof(CursorPreview)` / `sizeof(MessageWidget)` / `sizeof(Music)`（`src/Lawn/System/SaveGame.cpp` ~2710–2752 行）。两个后果：

- `state_hash` **只在单进程内可用**（换进程指针值就变，跨进程比对无意义）。
- 恢复时把上一次的指针值写回 `Board`，**潜在 use-after-free 地雷**。

用户已授权自由改动、无需向后兼容，所以可以直接改格式。这是一条独立任务。

修复时禁止继续依赖“整个 C++ 对象内存布局直接 `SyncBytes(sizeof(T))`”这种方式保存含指针对象。可以重新设计 snapshot/save 格式。

该任务完成的验收至少包括：

1. 两个全新 simulator 进程，在相同资源、task、seed 和动作序列下，产生可跨进程稳定比较的 snapshot/state hash。
2. `verify_env_equivalence.py` 全量通过，包括 episode-to-episode reproducibility、rewind stress、snapshot restore 和 `BRANCH_SNAPSHOT_FAST` 等价。
3. 固定 development 小样本上，用 `scripts/binary_search_equivalence.py` 或等价检查证明 SearchTeacher 的动作、全部候选分、search policy、margin、terminal outcome、simulation count、screening/depth budget 拆分保持一致。由于序列化格式会变化，`state_hash` 数字本身允许改变。
4. Linux C++ 单测继续全绿。
5. 新格式如果使 snapshot payload 尺寸或 save/restore 性能明显变化，要记录修改前后数据。

## 6. 剩余待办（任务清单）

| # | 任务 | 状态 |
|---|---|---|
| 7 | 优化 `BRANCH_SNAPSHOT_FAST` 并证明等价 | ✅ 完成（−27%，决策一致） |
| 8 | 生成 `SearchValueModel` checkpoint（`search_value_v2.pt`）：bootstrap（20000–20031）→ 训练 → refinement（21000–21031）→ 再训练 → `save_search_value` | ⬜ **现在可在台式机做** |
| 9 | dev 256 教师成绩单 + 搜索配置消融（用 `effective_depth_budget` 表述） | ⬜ |
| 10 | 大规模搜索监督 + DAgger 闭环训练 | ⬜ |
| 11 | final-test 1024 seeds 最终验收 | ⬜ |
| 12 | 多地形扩展与候选/值模型泛化检查 | ⬜ |
| — | **修复存档载荷里的裸指针**（见上 §5） | ⬜ **用户点名要求** |

优化遗留（非阻塞，已量化记录）：save 86.8 / restore 68.4 µs 仍是 C++ 大头；每分支约 73 µs 落在 Python JSON 解析与管道；一个结构性机会——一批 32 分支里真正存活的快照远少于 32，加一条「先不存快照地评估、只重算存活者」的命令可省掉被丢弃分支的 86.8 µs/个（这是协议+搜索改动，先量存活率再决定）。

## 7. 计划中的下一步（接手者从这里开始）

### 7.1 并行化是关键使能器

单集教师 episode ≈ 30 s（`advice` ~150 ms × ~200 决策，其中约 66% 等待 C++；值模型只占 ~3 ms），因此采集阶段真正重要的是**按 seed 横向并发**。

计划用 `multiprocessing`，启动方式使用 `spawn`，避免 fork 继承带活子进程的 `PvZEnv` / simulator 状态。每个 worker 必须独立创建：

- simulator 子进程
- `PvZEnv`
- `SearchTeacher`
- CPU 上的 `SearchValueModel`

不能在父进程创建环境后 fork。

`benchmark_pvz_agent.py` 支持分片 seed 文件（schema 1 / level 7 / playthrough 2 / role 匹配），输出可以合并后重新运行统一 `summarize`。

当前 `train_pvz_agent.py` 和 `benchmark_pvz_agent.py` 的 seed 执行仍以串行为主，因此这里允许直接重构公共的多进程采集层，让训练数据采集和 benchmark 复用。

并行实现必须先用少量 seed 做吞吐实验，再决定整夜 worker 数量。台式机是 i7-12700F，已有测量表明 x86 上 torch 多线程有收益，但多 worker × 每 worker 多 torch threads 会发生过度订阅，因此需要实测 `workers × torch_threads` 组合。目标是最大化稳定 episodes/hour，同时避免明显 swap、频繁 OOM 或系统失去响应。

### 7.2 长任务必须可恢复

任何需要数小时的按 seed 任务都不能等全部 seed 完成后才拥有唯一有效输出。

优先采用以下形式之一：

- 每个 seed 独立结果文件。
- 小 shard 原子落盘。
- 临时文件完整写入并校验后 `rename` 到最终文件。


重新启动同一任务时，应：

- 检查已有结果。
- 验证 metadata 与当前任务一致。
- 自动跳过已经正确完成的 seed。
- 只重跑缺失或损坏部分。

单个 seed 崩溃或 simulator 异常不能导致此前几小时结果丢失。

最终合并 shard 时必须用统一 `summarize()` 对所有 episode records 重新计算汇总。不要直接平均 shard 的 win rate、Wilson interval 或其他派生统计量。

### 7.3 完成任务 #8

跑 `train_pvz_agent.py` 相关管线完成 SearchValueModel v2：

1. bootstrap：20000–20031
2. 用 bootstrap 数据训练 SearchValueModel
3. refinement：21000–21031，使用 bootstrap 后的 value model 进行搜索
4. 使用 bootstrap + refinement 再训练
5. `save_search_value` 生成 `search_value_v2.pt`

文件契约是 `--output-dir` 下：

- `search_value_bootstrap.json.gz`
- `search_value_refinement.json.gz`
- `search_trajectories.json.gz`
- `dagger_search_trajectories.json.gz`
- `search_value_v2.pt`

当前 `--collect-only` 会在 value 阶段之后继续采集普通 search trajectories，因此如果它不适合作为真正的“只生成 value checkpoint”入口，可以直接重构 `train_pvz_agent.py`，增加清晰的分阶段/value-only/resume 能力，无需维持旧 CLI 向后兼容。

建议把“搜索采集用什么 device”和“批量训练用什么 device”分开。采集阶段每个叶子的 value inference 是 batch=1，优先使用 CPU；批量训练使用 CUDA。不要为了让 value inference 上 CUDA 而让多个采集 worker 争抢 GPU。

生成 checkpoint 后核验：

- bootstrap seed provenance 正确。
- refinement seed provenance 正确。
- 两组 seed 与 train/dagger/development/final-test 均无交叉。
- task signature 正确。
- feature/search-value/protocol/search-label/value-semantics 版本正确。
- checkpoint 可以重新 load。
- 记录 bootstrap/refinement loss 历史。

### 7.5 任务 #9：development 256 教师成绩单和搜索配置消融

有稳定 `search_value_v2.pt` 后，在 development 30000–30255 上运行 SearchTeacher benchmark。

所有搜索深度相关汇报必须以 `effective_depth_budget` / screening 与 depth simulations 的实际拆分解释，不能直接把输入的 `simulation_budget` 称为“搜索深度”。

先完成当前默认配置的完整 dev 256 成绩单，再考虑消融。消融以信息增益为目标，不需要机械地扫巨大的笛卡尔积。优先研究可能真正改变质量/吞吐折中的变量，例如：

- simulation budget
- beam width
- candidate limit
- horizon ticks / max decisions

每组配置需要记录：

- win rate / Wilson 95%
- failure wave distribution
- mean actions
- wall time / throughput
- search simulations
- screening/depth split
- mean effective depth budget
- candidate count / margin / entropy 等已有指标

可以先用 development 子集筛掉明显差的配置，再把少量候选跑完整 development 256。用于选择最终配置的证据只能来自 development。

**final-test 40000–41023 保留给最终一次验收。今晚推进 #8/#9/#10 时不要提前使用它做调参、模型选择或普通 smoke benchmark。任务 #11 到整个训练/配置选择完成后再执行。**

### 7.6 任务 #10：大规模搜索监督 + DAgger

完成 #8、拥有可信 SearchValueModel，并对 #9 的教师配置有足够 development 证据之后，开始：

- train 0–63 搜索监督数据
- BC
- dagger 10000–10063
- DAgger 重训练

优先确保整个流水线可恢复，再扩大运行量。

如果 #9 的完整 dev benchmark 正在长时间运行，可以利用剩余 CPU/GPU 资源并行推进与其互不污染的工程任务，但不要让多个重型任务互相抢资源到总吞吐下降。

### 7.7 裸指针任务如何穿插

这条是必须完成的独立任务，但它风险较高，不要因为进入 SaveGame 重构后遇到复杂问题，就让整夜的数据采集和训练全部停摆。

可以使用独立 worktree / 独立提交开发裸指针修复。

适合的执行方式：

- 先启动已经验证过的长时间采集/benchmark。
- 在机器仍有足够资源时开发裸指针修复。
- 修复需要重建 C++ 或跑重型 equivalence 时，协调资源，避免破坏正在运行的数据任务。
- 修复达到明确的可验证阶段就提交。
- 若连续长时间卡在同一点，保留最小复现、当前发现和有效修改，切换到其他仍可推进任务，之后再回来。

## 8. 本会话踩过的坑

- `BENCH_SNAPSHOT` 里纯函数被编译器删除（哈希测出 0 µs）→ 必须喂 `volatile` 汇聚点。
- Mac shell 把 `--venv ~/.venvs/ml` 的 `~` 展开成 `/Users/newbiexvwu` → 传 WSL 绝对路径。
- 误建 `~/ml` venv 装 torch → 已删（14 MB）并 `pip cache purge`（3.4 GB）。
- `python3` 在 Mac 上解析到 WorkBuddy 托管 3.13（无 torch）→ 必须用 mise 路径。
- SearchTeacher / simulator 环境不能依靠 `fork` 继承已有子进程；多进程采集使用 `spawn`。
- 长时间 seed 批任务不能只在全部完成后一次性保存结果，否则尾部一个异常可能浪费数小时计算。

## 9. 今晚无人值守的成果优先级

今晚的目标是让第二天查看时能看到**大量已经完成、可验证、可继续恢复的实质进展**，不追求为了形式把所有任务编号都勾掉。

优先级如下：

1. 保证台式机环境和 equivalence regression 正常。
2. 建成可靠的多进程、可恢复 seed 采集基础设施。
3. 完成 `SearchValueModel v2` 的 bootstrap → refinement → checkpoint。
4. 启动并尽可能完成 development 256 教师 benchmark。
5. 根据 development 数据做有信息量的搜索配置消融。
6. 开始大规模 search supervision / DAgger。
7. 并行推进裸指针序列化修复，并达到明确可验证的阶段。

不要为了过早开始更靠后的任务而跳过前面的正确性验证。

## 10. 晨间交付要求

第二天用户查看时，应尽量留下以下可直接检查的成果：

- 清晰的 git commits，每个提交有明确目的和验证结果。
- 多进程 seed 采集/benchmark 已能使用 `spawn` 稳定运行。
- 长任务具有 resume 能力，已有 seed 结果不会因重新启动丢失。
- 实测过 worker/thread 组合，并记录最终采用的并发配置和 episodes/hour。
- `search_value_bootstrap.json.gz`
- `search_value_refinement.json.gz`
- `search_value_v2.pt`
- SearchValue checkpoint 的task signature、seed provenance、训练 loss。
- 至少已经开始 development benchmark；理想状态是完整 dev 256 教师成绩单已经完成。
- 若做了搜索消融，留下结构化结果，包含 `effective_depth_budget` 和真实吞吐。
- 若已经进入 search supervision / DAgger，留下完成 seed 数、剩余 seed 数和可恢复状态。
- 裸指针修复若完成，应有跨进程 hash 稳定性、equivalence、SearchTeacher 等价和单测证据。
- 裸指针修复若仍在推进，应留下独立 commit 或最小复现、已经确认的根因/字段范围，以及下一个明确技术步骤，不能只留下不可解释的脏工作区。
- 更新 `TODO.md` 和研究记录，写清实际数据、已完成工作、仍在运行的任务和真实阻塞点。

普通失败自行处理。一个方案失败后先定位原因、保留证据、尝试合理替代方案；避免重复做已经被本会话实测排除的路线。

最终的衡量标准是：**一夜之后仓库、模型产物、benchmark 数据和实验结论都有可验证的净进展，而且所有长任务都能从现有状态继续运行。**