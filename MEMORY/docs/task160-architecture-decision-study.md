# task160：架构 / MoE / 数据配比 / 采集方向的决策研究（分 4 份，交外部代理）

用户 10-29 指示：这是关键转折点，要一次想清楚以下五件事，尽量避免以后返工：
- 网络结构；
- 是否用 MoE；
- 用什么样的 MoE；
- 数据占比结构；
- 后续采集的方向和方法。

各份结论写在 `MEMORY/docs/task160-<字母>-*.md`，推送到各自分支，主控汇总。

主控已有的事实：
- `task153-entity-attention-student.md`（`origin/entityattn` 上的版本含修订 1）；
- `task158-terrain-interference-moe.md`；
- `MEMORY/notes/student-architecture.md`；
- 计划文件 10-29 各条。

r36 训练包的构成：
- **按来源：** 课程前历史 DAgger 60%（L7 一关占 47%），随机卡组格 0.2%，示范起点格 3.2%；
- **按地形：** 白天 66%，屋顶 1.8%；
- **B 配方：** 自 r27 起从未被选中。

通用约束：
- **只读：** 除非本份写明要改代码。
- **测试：** 遵守 AGENTS §6.5 第 7、8 条。
- **不部署。**
- **台式机是生产机：** 只读日志和小文件，不碰 qc / 看门狗 / `stop_curr`，不 `pkill -f`。
- **安全：** 不打印凭证。

## A. 仓库历史综述（台式机 CodeBuddy）

范围：把所有**架构 / 输入**与**数据 / 配方 / 课程 / 采集**的实验和决定，整理成两张按时间排列的表。

资料来源：
- `MEMORY/2026-*.md`
- `MEMORY/docs/student-bottleneck-plan.md`（最重要）
- `MEMORY/docs/*`、`MEMORY/notes/*`、`learn/*`、DESIGN §4 / §13 / §14
- 旧模型：`git show pre-cleanup-20261019:python/pvz_agent_model.py`

表格列：日期 | 改动 | 数据规模 | 指标 | 结果数字 | 决定 | 注意事项。

必须覆盖：
- **架构类：**
  - GameplayModelV1 及其退役原因；
  - identity / slot 卡包头、row_pool、aux 辅助头；
  - 倍率输入的外推问题；
  - 宽度 128（task23）；
  - v2、v2timer 输入；
  - 帧堆叠 / 记忆讨论；
  - TRM / DRC 讨论；
  - MoE 的四次讨论；
  - task153 的两轮对照；
  - 价值网 v1–v5；
  - 前向耗时（µs）。
- **数据类：**
  - 自我标签稀释 H2（task15）；
  - 重标注；
  - 软标签与 spread；
  - task36、task104、task132、task135、task127、task136、task129、task137；
  - llmplay 和 GPT Pro；
  - 门禁 v1–v5。

另外列出：最关键的 10 条教训（带 file:line）、定了但没执行的事、历史回答不了的问题。

## B. 文献调研（Mac Qoder，可联网）

调研以下 9 个主题，每条结论要写明：引用（题目 / 年份 / arXiv 号 / 链接）、可信度、对我们意味着什么。

1. 实体加空间输入的游戏网络：
   - AlphaStar：entity transformer、scatter connections、指针头；
   - OpenAI Five、Gato、MuZero 的网络规模；
   - Transformer 与 CNN 在 1000 万到 3000 万行数据规模下的数据效率。
2. 强化学习 / 模仿学习中的 MoE：
   - Obando-Ceron 2024，“Mixtures of Experts Unlock Parameter Scaling for Deep RL”（Soft MoE）；
   - 多任务 RL 的 MoE：MOORE、CARE、soft modularization；
   - 小规模下的路由塌缩；
   - MoE 对分布外泛化是帮还是伤。
3. 多任务之间的干扰及对策：PCGrad、FiLM、适配层。
4. 模仿学习的缩放律（如 NetHack，Tuyls 2023）：怎么判断是数据不够还是容量不够。
5. 对没见过的组合怎么泛化：
   - Procgen / CoinRun 中训练关卡数的影响；
   - 卡组 / 配置随机化；
   - 卡牌游戏里的组合泛化。
6. 大小两个网络的分工：AlphaGo 的快速走子网络、蒸馏、int8 CPU 推理。
7. 数据配比：DoReMi、data mixing laws、温度采样。
8. 专家迭代的数据管理：
   - KataGo / Leela 的回放窗口、样本加权、策略目标剪枝；
   - 旧老师标签是否过时；
   - DAgger 聚合与按时间加权。
9. 课程 / 无监督环境设计（PLR、ACCEL），以及示范状态当作起点（Go-Explore、Salimans 2018），等价动作多时的标签噪声。

最后给出：
- 架构选项排序（含“保持 CNN、修数据”）；
- MoE 选项排序；
- 数据配比与采集策略排序。

每项写优缺点。查不到原文的说法要标注出来。

## C. 对局评测工具（Mac Qoder，写代码）

工作树 `.claude/worktrees/archbench`（分支 `archbench`），只在本地提交，不推送。

做一个按真打胜率评测学生的工具 `scripts/bench_play.py`：
- **评测对象：** 纯学生，不带搜索。
- **两条执行路径：**
  - 能导出 C++ 的模型（现行 RolloutNet、PVZRN006）走 C++ 原生，最快；
  - 其他结构在 Python 里出手，作为原型路径。
- **参数：** 面板（复用 `configs/gate_v5_panel.json` 或其子集，五个地形都要有，并含换卡组留出）、种子（各模型相同，便于配对比较）、进程数（≤ 8）。
- **输出：** 分任务、分地形、总体胜率与 Wilson 区间；`--compare` 模式给出分地形的配对 McNemar 检验。

验收方法：
1. 从台式机只读拷 r36 学生：
   ```
   ssh -o ConnectTimeout=40 newbiexvwu@100.120.155.85 wsl -d Ubuntu -e cat /home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000036/student/model.pt > 本地文件
   ```
2. 在小面板上跑一遍，与它的门禁 v5 结果（`round_000036/evaluations/*/decision.json`）对照。
3. 核对 Python 路径与 C++ 路径在同种子下的动作是否一致。

Mac 引擎：先看已有构建能不能用（`build/`、其他工作树的 `build/`），不能用就在本工作树编 Release。

文档写到 `MEMORY/docs/archbench-harness.md`，内容包括：用法、速度（局 / 小时 / 进程）、两条路径的一致性、怎么接入新结构。

## D. 生产流程画像（台式机 CodeBuddy，做完 task156 后接着做）

读 `~/pvz-logs/curriculum.log` 和 `curriculum_state.pipeline/round_*/`，回答：
1. **每轮耗时：** 采集、打包、每个候选的训练、挑选、门禁各花多久；近 48 小时显卡忙碌占比。
2. **门禁记录：** r20–r36 各轮的门禁决定，分地形的数字与否决原因；已晋升学生的分地形胜率走势。
3. **课程现状：** 当前活跃的格子，每轮各地形的局数；示范起点格、随机卡组格、开局面板各占多少组。
4. **老师胜率：** 近 3 轮按地形 / 关卡 / 倍率统计。
5. **A/B 挑选：** 每轮 A 和 B 的胜负与原因。
6. **新鲜数据占比：** 每轮新增行占训练包的比例。

最后按影响大小排出瓶颈。

## 主控对 C++ 前向的决定（10-29，用户建议）

生产中的前向一律走 C++：采集、续走、门禁都是。Python 只用于原型和小实验。

为避免每换一次结构就重写 C++，task157 第 4 步改为：在 C++ 里写一套**按配置组装的通用模块**，包括 卷积 / 注意力（含关系偏置）/ FFN / MoE / 指针头，权重格式自描述。这样以后换结构只需要改配置和训练代码。

## E. 汇总与决策备忘（Mac Qoder，主控只审）

用户 10-29：Claude 额度紧张，分析和汇总全部外包。

### 输入

- A、B、C、D 的结论；
- task153–159 的结论（各分支上的任务书末尾；`git fetch` 后看 `origin/<分支>`）；
- 计划文件。

A–D 还没出齐时，先读已有的，定时（每 30 分钟）再查一次。

### 产出

文件 `MEMORY/docs/task160-decision-memo.md`，推送 pvz-env，只提交这一个文件。内容如下。

1. **五个问题逐一分析。** 网络结构、是否用 MoE、用什么 MoE、数据配比、采集方向。每个问题列出全部可选路径，每条路径写明：
   - 证据（file:line 或文献）；
   - 优点、缺点、风险；
   - 训练与推理成本（C++ 前向）；
   - 返工风险。
2. **给出推荐。** 分两类写：
   - 现在就能定、不管实验结果如何都不会返工的部分。例如：采集时把富输入随帧写入；C++ 通用前向模块；以对局胜率为准的评测。
   - 要靠实验来定的部分。每项写预登记的实验设计：对照组、数据、种子数、指标、判定规则、用哪台机器、耗时。
3. **实验排期。** 机器：Mac MPS / CPU、台式机显卡空档；代理：Qoder / CodeBuddy / Codex。

### 实验执行

备忘写完后，接着把 Mac 上能做的实验跑起来，结果追加到备忘末尾。

- **数据：** r42 v2timer 训练包已拷到 Mac `~/pvz-bench-data/r42pack/`（读法见 `python/pvz_training_pack.py`）。
- **评测：** 用 C 节的 `scripts/bench_play.py`。

约束同上。不部署，不碰台式机生产。
