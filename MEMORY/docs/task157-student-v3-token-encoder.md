# task157：学生 v3——移植旧 GameplayModelV1 编码器 + 富输入

分支 `studentv3`，基于 `origin/prodint16`。在 Mac 上做（Codex 账号 A）。这是长期主线，分阶段提交推送。

## 为什么

- 现行学生是 6×10 格子 CNN，输入是紧凑帧（植物只有行列种类血量；僵尸按 80 像素落格）。
- 卡组泛化（老师 14.6%）、屋顶、空间关系关卡都是短板。用户决定花大功夫改进网络结构和输入。
- task153（`origin/entityattn`）的实体分支是半成品：单头、两层、线性距离偏置、卡包和植物共用嵌入、卡包混在均值池化里 → 在未见卡组上出手偏置漂移（见该任务书“主控复盘”）。
- PPO 时期的 `GameplayModelV1`（tag `pre-cleanup-20261019`，`python/pvz_agent_model.py`）设计完整得多：
  - token 种类 12 种（global / profile / cell×54 / plant / zombie / projectile / defense / grid_item / seed_packet / zombie_roster / lane / lane_enemy），每个 token = 种类 + 类别 + 变体嵌入 + 32 维特征投影 + 行 / 列嵌入；
  - 植物带状态倒计时、发射 / 射击计数、睡眠、目标僵尸；僵尸带精确 x/y、头盔 / 盾牌血量、相位、速度、减速 / 黄油 / 冰冻、啃食、目标格；投射物带 x/y/z 与速度；
  - 4 层、宽 192、6 头，pre-LN，SwiGLU；关系偏置 = 每头的“种类对”表 + 行差分桶 + 列差分桶 + 同格 + 目标链接；
  - 卡包用指针头（状态 query 对卡包 token 的 key），格子也用指针头（以选中卡包为条件）。卡包不进全局池化。
- 它当年被小 CNN 追平只在 L7 固定卡组上（`learn/05`、`learn/06`）；没在未见卡组 / 多地形上比过。

## 要做的（按阶段，每阶段提交推送）

1. **输入（obs v3）**：定义只含公开信息的富 token 输入，参照旧 `observation_tokens`，按现在的引擎字段补齐（v2timer 的计时、雾 / 风暴可见性遮挡、传送带、滚动坚果、蹦极落点等）。
   - C++ 直接产出 token 数组（不走 Python 逐个组装，旧版 3.3 ms/步的主要开销就在这里），采集时随帧写入，旧紧凑帧不变。
   - 不放未来或隐藏信息；不手写轨迹 / 策略公式。旧版的派生量（阳光收入率、经济火力比、距下一波）逐项判断：是可见量的简单汇总可以保留，任何“该怎么打”的公式不要。写清取舍。
   - 评估回填：存档有种子和动作，能否在现行引擎上确定性重放、重新导出富输入？抽样估计成功率和耗时（10-17 曾有旧轮次因引擎变化重放失败）。
2. **网络**：移植旧编码器（去掉 GRU、PPO critic、事件等待等 PPO 专用部分；学生是无记忆的，需要时可加上一帧差分），动作头沿用指针头，输出与现行训练器的三头（种 / 等、卡包、格子）对齐，接入 `pvz_recipe` 配方字段，默认不变，旧检查点逐位不变。宽度 / 层数 / 头数做成配置。
3. **对照**：用 Mac 能拿到的数据先做小规模对照（同 task153 的划分方法：见过 / 未见卡组），四臂：CNN64、等参宽 CNN、task153 实体分支（修订 1 之后的版本，如果已出）、v3。至少 3 个随机种子，报告完整动作准确率、种 / 等 AUC、等待行误判率、卡包 / 格子准确率，分地形。
   - 有条件时用回填出的富输入数据（含屋顶 / 雾 / 泳池），而不只白天。
4. **速度**：C++ 前向只需支持老师根节点（实测根节点 : 续走 = 1 : 132，续走仍用 CNN）；量每次根节点前向耗时。
5. **生产化方案**：写清进入生产 alternate-recipe 槽位需要的东西（采集写富输入、打包、训练、门禁），以及显卡占用。

## 约束

- AGENTS §6.5 第 8 条：只跑 `-k`，加 `timeout 300` 和 `ulimit -f 2000000`，一次一个 pytest；禁止整套跑那三个文件。不加自校验。
- 不部署，不连台式机生产进程；可以从台式机只读拷少量数据（≤ 2 GB，`ssh … wsl -d Ubuntu -e cat`，约 0.5 MB/s）。检查点和数据不进 git。
- 同一 Mac 上还有 codex-plus2（task155）和 qoder-t153（task153 修订 1）在跑，注意内存，训练对照串行。
- 结论逐阶段追加在本文件末尾。

## 主控补充（10-29）：C++ 前向做成通用模块

用户的要求是：生产里的前向，能用 C++ 就用 C++。第 4 步因此改为在 C++ 里写一套**按配置组装的通用前向**，要求如下：

- 支持的模块：卷积、带关系偏置的注意力（种类对、行差桶、列差桶、同格）、SwiGLU FFN、MoE FFN（共享专家 + 路由，包括按地形硬路由），以及指针头。
- 权重文件要自带结构描述。
- 用 Python 对拍。
- 根节点和续走都能调用。

这样以后换结构只需要改配置，不用再写 C++。

## 阶段 1：公开富输入（2026-10-29）

新增 `src/EnvironmentTokens.{h,cpp}`。`OBS_V3 1` 显式开启后，完整和精简观测均附带 `v3`，`_frame` 原样保存。默认关闭，旧帧字段与旧学生输入保持原样。采集入口为 `search_play.py --obs-v3`。C++ 内存中的 token 数组直接供后续原生根节点前向使用，Python 仅搬运数组。

### 输入契约

`v3 = {version: 3, tokens: [[kind, category, variant, row, col, target, f0..f31], ...]}`。类别与变体为原枚举 + 1，0 表示空值。无坐标与无目标为 −1，未使用特征补 0。没有截断实体数。前 66 项固定为 global、profile、54 个格子、10 个卡槽，后面为可变实体与行上下文。卡槽没有位置特征和行列坐标，空卡槽 category=0。保留旧编码器 12 种 kind 编号，当前没有额外构造 lane_enemy token。

| kind | 特征（顺序即 f0 起，未列部分为 0） |
|---|---|
| global(0) | 阳光/1000、当前波/总波、tick/60000、总波/30、倍率/10、夜间/泳池/雾/屋顶标志、原生传送带、保龄球、风暴黑屏、关卡/50；category=背景，variant=游戏模式 |
| profile(1) | 周目/2、卡槽数/10、包扎术/模仿者/泳池车/屋顶车所有权 |
| cell(2) | 行/5、列/8、雾量/255、该行可用；category=格地形，variant=行地形 |
| plant(3) | 行/5、种植列/8、血量与上限/3000、精确 x/900 与 y/700、状态/128、状态倒计时/1200、特殊倒计时/1200、保留位 −1、发射间隔/1200、射击计数/1200、唤醒计数/1200、睡眠、压扁、被蹦极抓取状态/3、动态信息可见标志、滚动标志、当前 vx/20 与 vy/20；category=植物，variant=模仿类型 |
| zombie(4) | 行/5、x/900、y/700、vx/5、身体/头盔/盾牌血量及各自上限/2000、当前动作相位/128、减速/黄油/冰冻计数/1200、有头/胳膊/持物/啃食/在场标志、缩放；蹦极改用公开落点标志(f20)、动作阶段/3(f21)、可见高度/700(f22) |
| projectile(5) | 行/5、x/900、y/700、z/500、三轴速度/20、伤害/500；category=种类，variant=运动方式 |
| defense(6) | 行/5、x/900、y/700；category=种类，variant=状态 |
| grid_item(7) | 行/5、列/8、x/900、y/700；category=可见物件类型 |
| seed_packet(8) | 标价/500、冷却计数/3000、冷却总长/3000、激活、可支付（原生传送带免费）；category=卡种，variant=模仿类型 |
| zombie_roster(9) | 1；category=本关僵尸类型 |
| lane(10) | 可见植物数/20、僵尸数/40、弹体数/50 |

### 公开信息取舍

复用引擎的 `EnvironmentZombieVisible`、`EnvironmentProjectileVisible`、`EnvironmentPlantFogHidden` 和 `EnvironmentBungees`。风暴黑屏不导出盘面实体，雾中静止植物仅保留种类与种植格，动态特征设 −1 并置可见位为 0。滚动坚果在雾中整体隐藏，真实位置不会由其出生格泄漏。保龄球速度来自当前动画 `_ground` 轨道和已发生的上下方向，不推演轨迹或随机拐弯。蹦极尚未公布落点时 col/target=−1，公布后 target 链接到固定格子 token。

植物计时保留已开始的特殊动作与动画进度。复核发现旧完整观测的 mLaunchCounter 包含未来随机产阳光时点，v3 将该槽固定为 −1。去掉旧编码里的阳光收入估算、经济火力比、下一波距离与隐藏出怪倒计时。僵尸不导出随机相位倒计时、召唤计数与内部目标格。植物和普通弹体不导出引擎内部 target ID。场地物件不导出未开花瓶的内容或内部计时。行上下文只汇总可见数量。

### 回填与实测

`scripts/backfill_observation_v3.py` 默认抽三局，读取存档实际卡组及任务配置，旧档缺配置时可指定原 manifest/task-id。按实际动作重放，比较紧凑状态及拒绝结果，整局成功才另存带 v3 的档案，保留源文件与标签。旧档缺后来新增的 sleep/boss/bungee/fog_amount/v2 字段时只跳过这些缺项。示范起点与 RESEED 档案暂明确跳过，需要完整原示范及其时序后再支持。

Mac 独立 Release 引擎下，`d7p1m_fast_search` 的 seed 1844096–1844098 三局均成功，共 3,193 帧、317,834 token，用时 4.33 / 2.47 / 0.78 秒（含写档压缩）。这是白天三局的结果，不能外推全语料。测试覆盖开关前后旧紧凑帧相等、完整/精简 token 一致、雾隐藏、滚动坚果实际位置变化，以及回填标签保留和失配整局跳过。

## 阶段 2：编码器与现行训练器（2026-10-29）

`python/pvz_token_student.py` 移植种类/类别/变体嵌入、32 维特征两层投影、行列嵌入，保留 pre-LN 多头关系注意力与 SwiGLU。每层包含按头的种类对、带符号行差桶、带符号列差桶、同格与公开目标链接。删除全部 GRU、PPO 价值头和事件等待头。`TokenStudent` 输出与现行训练器一致的 plant、packet、以指定卡槽为条件的 cell。卡包与格子均采用 query/key 指针头。

global token 作为无记忆状态表示。为继承 task153 修订 1 对卡组泄漏的修复，所有层的盘面与全局查询都不读取卡包 key，卡包查询只读取盘面，卡包没有槽号嵌入或空间关系项。卡包的种类通过独立 kind 区分植物。与旧完整注意力相比，这一项有意收紧，卡包进入卡头与格子条件，不进入种/等头。后续对照须检验这一信息边界的代价。

配方显式 `arch=token_v3, obs=v3, packet_head=identity`，可配置 width、token_layers、token_heads、token_ff_width。默认参考值 192/4/6/768，草稿为 `recipes/student_v3.json`。旧配方不新增默认架构字段，不改变旧模型构造或缓存指纹。`load_student` 按架构分派，支持 Python 出手与检查点读取。

现行 `train_rollout_policy.py` 复用原有三头损失、优化器、辅助损失与权重流程。富输入行从帧 i−1 监督帧 i，等待抽样沿用原 label-local RNG 及权重，实体只在小批次内补齐。当前 v3 直接读富帧，尚未接生产的不可变分段 pack，这项列入生产化阶段。

验证：卡槽置换只置换卡包分数，种/等及同卡格子分数逐位相同；局部补齐、反向传播、检查点往返通过。现行训练器用 823 行小样本完成一次优化和验证（24 宽/2 层/4 头，接口测试）。以改动前源码为参照，同种子构造 v1 和 v2timer identity 的全部参数及原生导出文件均逐位相同。

### 后续输入与回填修正

原始人类示范被拒绝的操作不自动等待 60 tick，回填据档案 schema 区分；实际铲除操作从 input_action 读取。其逐 tick 记录与合并等待会产生不同的 Python 滑窗收入事件时间戳，回填保留原收入字段，只比较引擎状态（v3 不使用该收入估计）。据此完整回填屋顶 L42、雾 L39、原生泳池传送带 L30 三局，分别 1,514 / 1,610 / 664 帧，用时 5.09 / 5.47 / 2.79 秒。它们的雾设置是原档案的 null，不代表 human_v2 遮挡验收。

白天比较用的 152 局全部完整回填成功，累计 348.79 秒。发现发射计数公开性问题后，只清除已生成数组中的该字段（53.48 秒），没有再次重放。后续采集由修正后的 C++ 直接输出 −1。Python 既有 render_episode 学生评估入口同步按架构启用 obs-v3，跳过卷积历史编码。

## 主控分工更新（10-29）：本分支收束到富输入与回填

按用户最新指令，C++ 通用前向由 task161（`cppnet`）负责，Python 网络与训练接入由 task163（`studentv3net`）负责。task157 负责 obs-v3、存档回填和输入接口。此前已提交的 Python 网络是初稿，后续由 task163 接续。本地未提交的通用前向试作已移出工作区，不合入本分支。

### 给 task161 / task163 的输入接口

线格式继续为 `v3.version=3`、`v3.tokens:[N,38]`，不截断实体。C++ 入口为 `PublicObservationTokens(app)`，其结构体包含整数成员，不能直接按 float 内存复制。`ObservationTokenArray(tokens, capacity=0)` 提供连续 float32 数组，capacity=0 时恰好 N 行，指定容量时在末尾补齐，容量不足报错。

Python 入口为 `pvz_token_observation.graph_inputs(frame, capacity=None)`，只做数组转换，返回 batch=1 的连续 float32：

| 图输入名 | 形状 | 内容 |
|---|---|---|
| tokens | `[1,T,38]` | 原始 38 列，前六列是整数值的浮点表示 |
| token_meta | `[1,T,3]` | kind、row、col，对接 task161 的 relation |
| coords | `[1,T,2]` | row、col，供 task163 决定哪些实体参与 scatter |
| valid | `[1,T]` | 实际项为 1，空卡槽与补齐项为 0 |

补齐项 kind/category/variant=0，row/col/target=−1，特征为 0。kind 使用有效的 0，确保 task161 的关系表能读取补齐位置，attention 用 valid 排除它。类别和变体已经加过 1，不得重复平移。目标列为绝对 token 下标，未知为 −1，末尾补齐不会改变它。固定索引为 global=0、profile=1、cells=`[2,56)`、cards=`[56,66)`；task161 示例的 cells=`[0,54)`、cards=`[54,64)` 不能直接套用。

task161 `cad071e9` 的 PVZGN001 图固定 T，网络导出方须确定容量或分桶方式。本任务不自行截断实体。其示例 tokens 输入已经是网络宽度的编码，task163 需要先把本接口的原始类别与 32 维特征编码为网络宽度，再送注意力。线格式 version=3 与旧原生 observation id=3（v2）分别命名，原生新编号由 task161 统一分配。

采集用 `env.set_token_observation(True)` 或 `search_play.py --obs-v3`，默认关闭。回填会在旧档缺 v2 时同时补上 v2，供同局卷积基线读取，原有字段和标签保持原样。输出每局最大 token 数与文件大小，便于导出方选择容量和估算存储。

### 回填交付与初稿证据

已有 155 局富输入：152 局 L7–L10 卡组扰动示范，以及 L42 屋顶、L39 雾、L30 原生泳池传送带各一局，共约 75 MB。白天 152/152 和多地形 3/3 均完整回填成功，原生动作拒绝及状态保持一致。人类原始示范的收入滑窗估计按前述口径保留。多地形三局的 fog_observation 沿用源档案，L39 为 null；训练与门禁采用 human_v2 时应使用相应口径的采集数据。

先前四臂初稿试跑已经结束，随机种子 1/2/3、每种子 12 轮，共 37,703 行（训练 24,313，见过 6,081，未见 7,309）。未见完整动作均值：CNN64 90.04%、等参 CNN90 90.32%、task153 修订 1 实体 90.68%、v3 初稿 90.26%。v3 初稿的种/等 AUC 只有 0.746，种植动作准确率 3.31%，格子准确率 42.72%。该初稿全局状态完全屏蔽卡包，task163 接续时需要留意卡包冷却等可见信息如何进入种/等判断。完整模型及分种子结果保存在本机 `artifacts/student/task157/deck_comparison/`，没有继续调参或重训。

### 输入对接验收与数据中转

使用 task161 的已发布 `origin/cppnet`（`cad071e9`）原始 GenericForward 和导出器做了一次真实帧接口对拍：将本适配器生成的 97 行输入（含 5 行补齐）直接送入其独立 C++ 驱动，格子切片、卡槽切片和关系表输出均与该库 Python 参考逐位相同，最大绝对误差 0。这验证原始输入与索引的连接，完整网络前向和原生 observation 编号接入仍由 task161 / task163 完成。相关输入开关、雾隐藏与实际动作回填测试 2 项、固定图数组接口测试 2 项均通过。

155 局已通过 HF 中转，NPZ 合计 76,875,762 字节，包含实际任务元数据及可移植的 `manifest.json`。Git 中只提交代码和说明。

| 项目 | 位置 |
|---|---|
| HF 仓库 | `realnewbiexvwu/pvz-agent-artifacts` |
| 远端前缀 | `transfer/task157_obs_v3_7e702892/` |
| 固定版本 | `d4b7149464321cc6f5d24d1b7a2a7d83b7640819` |
| 数据 | `<前缀>/frames/seed_<seed>/<label>/frames.npz`，旁附 meta.json |
| 划分 | `<前缀>/manifest.json`，记录实际卡组、关卡和 train/seen/unseen/terrain_sample |

task163 可按该前缀下载后直接读取 manifest；数据取用确认后删除该 HF 前缀，遵守仓库中转后清理的约定。没有部署或访问台式机生产进程。

## 台式机接手收尾：采集附件（2026-10-29，studentv3net）

已先 fetch：`origin/studentv3` 仍为 `e66d878a`，没有重复实现本节。`search_play.py --obs-v3` 默认关闭；开启时逐决策流式写 `tokens_v3.npz`，与 `frames.npz` 并列，紧凑帧不写入 v3。学生推理仍可读取内存中的富输入。附件采用 task162 的 ids/counts/columns/values/offsets/feature_offsets/packet_index/cell_index/terrain 稀疏 float32，另加公开 target_indices；不量化数值，保留负零。每局结束压缩，临时列文件随即删除。附件记录紧凑帧 ZIP 成员的 CRC 与长度，读取仅查目录，不重放。

实际采集开关小测试以及 float32/目标链接往返测试通过（2 项，限内存、单线程），开关两边紧凑帧所有数组逐位一致。未部署，未改生产 qc。按轮回填抽样和开销数字将在本节继续追加。

### 打包与训练路径（2026-10-29）

- 小型语料可直接 `pack_training_rows.py --obs v3 --packet-head identity --token-input obs_v3 ...`：优先读帧旁 `tokens_v3.npz`，兼容原有内联 v3；仍使用原帧 i−1 对动作 i 和既有行抽样。仅 obs_v3 配方把附件 ZIP 目录里的 CRC/长度纳入 pack 身份，旧配方指纹不变。
- 大语料优先沿用 task162 冻结的 pack 和 `sources.sqlite` 行计划：`prepare_decision_data.py tokens --sources <冻结源计划> --token-input obs_v3 --out <独立decision_tokens_v3>`。`split` 阶段硬链接紧凑帧与已有附件，token 阶段只读选中的 i−1 输入，不启动引擎。输出为 task162 同名 `.bin` 列与 `ready.json`，另加 `target_indices.bin`；按局记录进度，中断后截去未提交尾部继续。旧 compact decision_tokens 继续可读。不要把 rich 输出写进生产已有的 compact 目录。
- 训练：`train_rollout_policy.py --recipe <obs=v3,token_input=obs_v3> --pack <冻结pack> --merge-dir <独立缓存> --tokens <decision_tokens_v3>`。CPU 评估同样支持 `eval_window_policy.py --tokens ...`。直接 mmap 列，不再复制富 token 成大号密集 pack。稀疏附件行数/输入版本与配方不符会拒绝；续训只比较小型 ready 元数据身份，不重读语料。旧入口不传 --tokens 时行为不变。
- 混合 CNN 结构的 board 分支沿用 v1 网格通道，须使用对应的 v3/v1 board pack；不能直接把 v2timer board 通道交给混合结构。纯 token 结构不使用 board 张量，可沿用 task162 的 v2timer 冻结行 pack。

一个相关小测试串通：非连续监督行→附件 pack→稀疏列中断续写→训练 SIGUSR1 保存/续训→CPU 评估，且目标链接及 float32 字节保持一致。测试仅选该用例、ulimit -v 8000000、CPU 单线程完成；没有运行全量 pack 或 GPU 训练。
