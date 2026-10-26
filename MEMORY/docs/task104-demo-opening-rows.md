# task104：示范开局决策进学生训练

## 背景

task102 查明，屋顶冷启动 L42 失败的原因：

- 学生开局不停买花盆，前两分钟把 350–375 阳光花在盆上，攒不出 300 阳光种西瓜投手。
- 叶值在这个区域整体压到 0–0.007，最好的候选比学生动作只高 0.003，过不了 0.02 的换动作门槛，所以老师一次都没推翻学生。
- 用户示范的开局是：用预置盆种 10 株向日葵，约 86 秒种下第一株西瓜投手。

qc 现在用示范做“起点格”，也就是从示范第 N 步的局面开局，然后往前推。但示范自身的决策（tick 0 起的每个动作，连同它的观察与历史）是否作为监督行进入过学生训练，没有核实过。

## 要回答 / 要做的

1. **核实现状**：
   - 人类示范（`artifacts/human_demo`）和 LLM 示范（`artifacts/llm_demo`）的决策行，现在有没有进 qc 的训练包？
   - 列出证据：打包代码路径（`python/pvz_training_pack.py` 的 `training_rows` 等），以及最近一轮训练包里 demo 来源的行数。
2. **若没有进**：
   - 实现“示范决策行”进训练包，带完整历史，排除被引擎拒绝的操作；
   - 权重与配方走现有配方文件字段，默认关闭；
   - 只在本工作树和私有数据视图里做，不碰 tmux qc 与 cq。
3. **预登记评估**：
   - 用 r11 的同一份训练语料，加或不加示范行，各从零训练 2 个种子（task84b 证明单 draw 方差极大）；
   - 屋顶 L42 / L43 / L44 冷启动各取 32 个新种子，学生臂评估；
   - 再加一个非屋顶的回归检查，比如 task92b 的 v2 面板子集。
   - 判据要写在训练前并先提交。
4. 推到分支 `demorows`，不合并。部署由主控决定。

## 约束

- 台式机 GPU 训练要排队，不能和 qc 抢：先看 `nvidia-smi`，只在 qc 不训练时用。
- 也可以在 Mac 上用 MPS 训练，但必须先确认能和台式机逐位对齐，或者明确说明不对齐。
- 台式机每任务最多 2 个游戏进程。主控 2026-10-26 解冻 Mac 后，本机每任务最多 6 个游戏进程，可运行游戏与训练。
- 不写手写规则或打法，只用示范数据。

## 结果

### 代码核实（2026-10-26）

现有路径已经能消费示范，但不会自动把原始示范目录加入 qc。`scripts/curriculum_queue.py` 的 `pack_labels` 只合并采集日志里的标签与显式基础标签，`pack_command` 从配方读取行合同。`scripts/human_demo.py export` 原来仅复制完整帧与元数据，没有写 `training_rows`。普通行会按 `wait_keep` 抽样，配方缺少示范标签时对应行完全不入包。示范起点续局的 `label_source` 是 `teacher`，不能把这些续局计成人类或 LLM 自身决策。

`python/pvz_training_pack.py:training_rows` 已支持精确的决策索引，并把它写进行合同指纹。`scripts/pack_training_rows.py:build_from_archives` 用第 i−1 帧预测第 i 帧动作，历史更新先于行筛选，缓存与直接读取均排除 `action_rejected`。本次复用这条路径：显式导出的已验证示范写入所有成功 `plant` / `wait` 的索引，完整帧保持原样，等待行不再被抽样。铲子与收集操作保留在历史中，不生成当前动作头无法表达的监督标签。

同一任务和种子可能同时有人类与 LLM 示范，导出增加可选 `--label`，分别使用 `human_<任务名>` / `llm_<任务名>`。原始 `task.id` 与 `label_source` 不改，原始档案不改。配方继续用 `labels` 决定是否加入，用 `label_weight` 和 `source_weight` 乘权。`source_weight` 补认实际 LLM 来源值 `llm_gpt_pro`。所有生产配方保持原样，默认不会纳入新增示范标签。

已冻结示范输入清单 [task104-demo-inputs.json](task104-demo-inputs.json)：14 份人类胜局提供 15,419 条成功决策（1,220 次种植、14,199 次等待），排除 1,120 次拒绝操作与 189 次动作头无法表达的操作。12 份 LLM 胜局提供 16,623 条决策（1,258 次种植、15,365 次等待），没有拒绝操作。保留完整 33,377 帧历史。原始 fog31 大喷菇档案未验证，采用其同帧且已验证的重校验版本，只计一次。清单记录原始帧、元数据和监督索引的 SHA256。

26 份示范已用现有导出器写入本树私有 `artifacts/task104/demo_frames/`，逐档验证原始哈希、监督索引哈希及全部帧解码后相等。仅数值帧和元数据的 52 个文件（5,112,933 字节）经 HF 中转目录 `transfer/task104_demo_rows_02153c99` 传到台式机，固定版本 `c33d0ef1bb56b51e899dae4b7c962cb3775c7835`。台式机已逐项核对并建成训练包，中转目录已删除，清理提交 `1f07189f1f83915d9ab872f0bca18686a5337c7e`，远端复查为空。

实际段来源已核实：r11 的 16,982,931 行均按 `teacher` 来源消费，人类与 LLM 自身决策各为 0 行。最近已完成的 r19 有 18,598,230 行，同样全部为 `teacher`，其中示范起点续局占 92,103 行。15,360,388 条旧行缺省 `label_source`，训练器按现行合同归入 `teacher`。逐档来源与实际保留行数结合计算，证据见 [task104-training-inputs.json](task104-training-inputs.json)。

### 训练前预登记（2026-10-26，已提交 b7d119ba）

**输入冻结。** 基础语料严格采用 r11 的已解析配方和该轮实际训练包引用的不可变段，记录配方、索引与各段 SHA256，不重新按当前 qc 日志扩充。另审计最近一轮已完成训练包的来源，分别列 `human`、`llm_gpt_pro` 和 `teacher` 的实际保留行数。只从 `artifacts/human_demo/raw` 与 `artifacts/llm_demo/raw` 取已验证且通关的正式示范，跳过录制失败局、自动测试局和同局的重复校验副本。同一动作带的重复副本只取一份，人类与 LLM 同任务同种子的不同动作带各保留一份。导出前冻结档案清单与帧 SHA256，同时记录成功种植数、等待数和被拒绝数。

**两臂四次训练。** A 为 r11 原语料，B 为同一语料加上述示范全部可表达决策。两臂分别使用训练种子 1、2，从零初始化（`init=null`），完成 r11 的 12 个 epoch，其余模型结构和训练参数照 r11。示范 `label_weight=1.0`，`source_weight={"human":1.0,"llm_gpt_pro":1.0,"teacher":1.0}`，不按评估结果追加权重扫描。以最后一轮检查点评估，不选最佳种子、不选最佳 epoch。整局验证切分沿用原配方算法，报告两臂训练及验证行数与种子清单。新增示范种子会参与切分，验证损失仅记录，不作为效果判据。

四次训练统一使用台式机 CUDA，决定已写入冻结记录。训练前检查 `nvidia-smi` 和其他训练进程，可用内存准入沿用现有 `run_dw5_study.py` 的 14 GiB / 连续 30 秒，为约 9.5 GB 数据与运行缓冲留出余量。运行中复用现有进程树守卫，发现其他任务训练就暂停本任务，让 qc 优先。四次训练均采用最后一个 epoch，不把本次重训与旧 r11 权重之间的差异归因于示范。

**冷启动屋顶面板。** 直接复用 task101 / task102 的 L42、L43、L44 西瓜示范卡组与完整任务字段，保持倍率 2.0、二周目和预置盆语义。每关 32 个新游戏种子，预留 L42 `104000000–104000031`、L43 `104000100–104000131`、L44 `104000200–104000231`。运行前先对现行全机种子登记机械核对，若冲突则先提交修订再运行。A/B 同关同种子配对，四个模型全部直接执行学生动作（`search_play.py --arms fast`），无示范恢复、无老师代打。共 4×3×32=384 局，每任务最多两个游戏进程。

**非屋顶回归。** 冻结 task92b 选择面板 v2 的原始清单，取其中全部非屋顶任务，每项最多取原清单前 8 个种子，四个模型用完全相同的任务与种子。该面板只用于回归评价，不参与选模型。若原始 v2 清单取不到，先在本文件登记并提交替代面板及种子，再运行，不从已看到的本次成绩反向挑任务。记录白天、夜间、泳池与雾区实际覆盖，缺失区明确列出。

**判据。** 唯一效果指标是实际通关率。示范有效须同时满足：两个训练种子的屋顶三关合并胜率均提高，两种子平均净提高至少 10 个百分点，且任一屋顶关两种子平均净下降不超过 1/32。非屋顶回归两种子平均总体净下降不超过 3 个百分点，任一实际覆盖区净下降不超过 5 个百分点。逐关逐训练种子报告 A/B 胜数与 gain/harm，并报告两次训练间的散布。配对统计按游戏种子计算，两个训练种子不能当成大量独立训练重复。未达到上述全部条件则结论为本配方未证实收益，不调整既定阈值、不部署。

### 训练包核实与 A/B 语料（2026-10-26）

**示范决策行从未进过生产训练包。** r11（`round_000011`）的 `source_rows.json` 有 348 个来源、16,982,931 行，来源名全部是 `*_fast_search`，`human_` / `llm_` 前缀一个也没有；`round_000015`（415 来源 / 17,883,072 行）与 `round_000019`（497 来源 / 18,598,230 行）同样为 0。名字里带 `user` 的 `c7/c8foguserl31snowm100` 是老师在这些任务上的标注，不是人类决策。所以示范只以"起点格"进入采集，其自身决策没有作为监督行进过训练。

**A/B 语料已建成并逐项核对。** 在 `artifacts/task104/` 建私有语料视图：348 个 r11 段与档案以软链接镜像（不改生产数据），18 个示范标签（26 份示范归并为 18 个任务标签）作为新段加入。A 索引 `7d15b915a2192805.npz` 16,982,931 行、348 段，与 r11 生产包同名同键；B 索引 `304f0c00f0edf7ae.npz` 17,014,973 行、366 段。差值 32,042 行正好等于冻结清单里 15,419 条人类 + 16,623 条 LLM 成功决策，说明显式行没有被 `wait_keep=0.1` 抽样。348 个共有段逐字节相同，A/B 的选项只有 `labels` / `sources` / `label_weight` 三个字段不同。证据见 [task104-ab-verify.json](task104-ab-verify.json)。

### 测试设计

`python/test_human_demo.py:HumanDemoExportTest` 使用纯合成帧检查：显式保留所有成功等待与种植，拒绝行和铲子不作标签，但被筛掉的帧继续更新历史。直接打包与缓存打包均与完整历史逐行特征比对，同时验证等待权重不因 `wait_keep=0.01` 放大。还覆盖相同任务/种子的人类与 LLM 标签共存，以及原始元数据不被修改。`python/test_recipe.py` 补充 LLM 来源权重和默认不加入示范的配方测试。纯数据测试已通过。台式机换用已有 task82 修复版引擎后，三个真实引擎测试全部通过，详见下文最终核验。

### 真实引擎测试结果与 r11 来源拆分（2026-10-26 17:30）

**真实引擎测试：失败，且与 task104 改动无关。** 台式机 `demorows` 工作树已同步到 198e91b7，用 cq 的现行引擎（BuildNum 3069，`build/pvz-portable` 软链到 `cq/build5`）跑 `python.test_human_demo`：数据层 `HumanDemoExportTest` 通过，`HumanDemoTest` 三个真实引擎用例全部因 `demonstration initial state hash differs` 失败（`scripts/human_demo.py:173`，即录制时的初始状态哈希在新进程 / 新 env 里对不上）。为排除本分支引入，在同一台机器上对分支基点 144a7dce 另开临时工作树跑同一文件：`test_record_replay_export_and_restore` 以**同一错误**失败，所以这是既有问题，不是示范行改动带来的。本分支对 `human_demo.py` 的改动只在 `export`（新增 `--label` 与 `training_rows`），`replay` / `verify` 一行未动。

独立复现指向引擎侧偶发不确定性：同一进程里先录制 200 tick、关闭，再新建 env 执行同一 reset，初始哈希一次给出 `c3a398ae87fdcc6f`，而录制侧与新进程都是 `39d2656c1d3dea21`；随后四次重跑又全部一致。两次独立进程的 `STATE_BYTES` 全量转储逐字节相同（78,331 字节，board 75,772 + rand 2,500 + 9 个标量），说明 reset 路径本身确定，差异只出现在「同一进程内多次 reset」这一路。形态与 task63 记录的地址种子类问题（粒子 FIELD_SHAKE 用堆地址种 rand）一致：地址随分配变化，哈希就偶发漂移。**结论：示范行的监督标签不依赖引擎重放**——导出只读已录制帧、拒绝位取录制时的 `ok`，所以真实引擎测试的失败不阻塞本任务的训练与评估；但它意味着「用新进程重新 `verify` 一份旧示范」在这台机器上不可靠，已记入问题。临时工作树 `/tmp/t104base` 用后即删。

**r11 来源拆分（human / LLM / teacher 分列）。** `r11.source_rows.json` 的 348 个来源、16,982,931 行里：人类示范标签 0 行、LLM 示范标签 0 行（`human_` / `llm_` 前缀均为空）。名字里带 `user` 的只有两个来源 `c7foguserl31snowm100_fast_search` 与 `c8foguserl31snowm100_fast_search`，共 17,286 行（16,393 训练 / 893 验证，占 0.10%）；读其档案 `meta.json` 确认 `label_source=teacher`、`task.task_id=fog_user_l31_snow`、`arm=fast_search`、`policy=student`，是老师在这个用户示范任务上的标注，不是人类决策。按 `configs/human_demo_tasks.json` 里 31 个带 `demo_start` 的任务 id 机械匹配，r11 的来源里**一个都没有**——屋顶与雾区起点格在本轮语料里完全没有出现。因此「示范自身的决策行从未进过训练包」对 r11 成立，而且不是被抽样掉，是从来没被当作来源。
### 非屋顶回归面板冻结（2026-10-26 17:40）

按预登记从 task92b 的 v2 选择面板（`configs/student_selection_v2.json`，multiseed 分支提交 f78c328c，panel_version 20，38,938 行）机械取全部 19 个非屋顶任务，每项取清单里**前 8 个种子**，写成 [task104-regression-panel.json](task104-regression-panel.json)（源文件 SHA256 一并记录，19 任务 / 152 局）。覆盖：白天 7 个任务、夜间 9 个、泳池 3 个；**雾区与屋顶为 0**——v2 面板本身不含雾区任务，屋顶按预登记被排除。这四个模型的 152 局完全配对，用同一份任务与种子。

种子来源是 task92b 已冻结并做过全机核对的 v2 面板，所以这 152 个种子不需要重新领号。屋顶的 104000000–104000231 按主控 2026-10-26 17:58 的裁定处理：**不做全机重型扫描**（AGENTS.md §6.5 禁止这类审计），改为在 DESIGN.md §14.35 登记段号、只保证与已登记段不重叠——该块与门禁池 `[100000000, 101000000)`、LLM 全程块 81,000,000–81,030,039、稳健性块 83,000,000–83,000,307、未来种子块 46,000,000–47,000,255 都不相交。我此前启动的 `gate_seed_inventory.py --verify-lo/--verify-hi` 全量扫描在台式机上跑了 15 分钟未完成、已停用；恢复连接后第一件事是确认这个进程已经结束，还在就 kill 掉，然后再开训。

屋顶面板据此冻结在 [task104-roof-panel.json](task104-roof-panel.json)：三关的任务字段（卡组、`playthrough=2`、倍率 2.0、`preplanted=[]`）逐项复制自 `task104-demo-inputs.json` 里三份人类西瓜示范的 `task` 记录，只改 task_id 与种子；每关 32 个种子，共 96 局，冷启动、无 `demo_start`、只跑学生臂。
### 训练恢复并在跑（2026-10-26 22:28 台式机 / 10:28 Mac）

隧道在 Mac 10:19 恢复。自动接手脚本按设计没有抢：tmux 里已经有别人在 19:39 建的 `task104-train` 与 20:07 建的 `task104-eval`，脚本只打印状态就退出，没有重复开训。

现状（22:28）：`task104_train.py` 的准入循环在 21:45 等到窗口（当时没有他人训练、可用内存 17 GB），a_s1 的 `train_rollout_policy.py`（PID 998257）已载入 16,982,931 条决策 / 55,773 个种子 / 1,704,229 条验证行，**跑到第 10/12 个 epoch，step 73,501，19,578 行/s**。显存 7.1 GB / 利用率 7–40%，进程 RSS 9.8 GB，可用内存 8–10 GB。按 15.28 M 训练行算，一个 epoch 约 220 秒到 13 分钟不等，取决于当时的内存余量；早期 epoch 约 34 step/s（与 r11 的 200 秒一个 epoch 同量级），内存紧张时掉到 9.6 step/s。

`task104-eval` 的排队命令已核对：它 `while kill -0 970590` 等训练驱动退出，再逐个 `test -f` 四个 `model.pt`，然后依次跑屋顶 384 局、非屋顶 608 局、判定器，末尾打印 `TASK104_EVALUATION_COMPLETE`；评估环境 `CUDA_VISIBLE_DEVICES=""`、`--workers 2`、私有引擎 `build/pvz-portable` 与资源目录都写死为绝对路径。所以四个检查点一齐，评估与判定会自动接上，不需要人工再起。

台式机工作树在 `90e8622b`，比 `origin/demorows` 落后两个提交（都是本文档改动，不含代码）。训练期间不动它，避免在跑的过程中改文件。

### 交接（2026-10-26 19:12）

主控通知 Codex 额度恢复、要求提交后停止。当前状态：**代码、预登记、两套面板、种子登记、数据层验证全部完成并推送 `demorows`；四次训练未完成。**

- 已完成：DESIGN.md §14.35 登记屋顶种子块 104,000,000–104,000,231；`MEMORY/docs/task104-roof-panel.json`（3 关 × 32 种子 = 96 局）与 `MEMORY/docs/task104-regression-panel.json`（19 任务 × 8 种子 = 152 局）；`scripts/task104_panel_eval.py`（多模型 × 显式种子、纯学生臂、2 进程）；台式机数据层测试 10 项全过；B 语料 26 段示范、32,042 行、18 个标签与冻结清单逐项相同。
- 未完成：四次从零训练（a_s1/a_s2/b_s1/b_s2）。a_s1 两次启动都卡在第一个 epoch 之前——不是配方问题：9.5 GB 训练数据在 25 GB 机器上与 qc 采集（13.6 GB）共存时被换出，实测 600 行/s（r11 约 85,000 行/s，200 秒一个 epoch），`read_bytes` 累计 562 GB 而 `rchar` 仅 14.7 GB。已停掉 tmux `t104` 与训练进程，`artifacts/task104/a_s1/` 里只有 `recipe.resolved.json` 与 `source_rows.json`，**没有检查点**，不占显存也不留半成品。
- 接手要点：`~/pvz-logs/t104_train.sh` 是内存准入版启动器（avail≥11 GB 三采样开训、掉到 4 GB 持续 4 分钟自动中止重等），但实测 qc 采集会把窗口吃掉，靠自然空档不可行；要么由主控给一个 qc 暂停的 4 小时窗口，要么把 qc worker 降下来。384 局屋顶 + 608 局非屋顶回归只跑学生臂、2 个游戏进程，内存需求小，可以和采集并行，但必须等四个检查点存在。
### 交接前清理（19:20）

台式机 `demorows` 工作树同步到 `46c90c1a`（与 `origin/demorows` 一致，工作区干净）；我之前被 kill 的全机扫描在树里留了一份 `MEMORY/docs/gate-seed-inventory.json` 的改动（加了新工作树根），已 `git checkout --` 还原，不把这份半成品留给你。本任务在这台机器上没有遗留进程：`t104` 会话已删，`train_rollout_policy.py`、`gate_seed_inventory.py` 都没有残留；`artifacts/task104/a_s1/` 只有 `recipe.resolved.json` 与 `source_rows.json`，没有检查点、不占显存。
### 交接时机器状态快照（19:26 核对）

- `demorows` 工作树：`0a527ba7` 之后的 `46c90c1a` 基线，工作区干净；`artifacts/task104/a_s1/` 无检查点，`data_a` / `data_b` / 两套配方 / 两套面板齐全。
- 没有本任务的训练进程：`pgrep -af task104/recipes` 为空，`tmux t104` 已删。**Codex 尚未开跑**（`codex` 进程在，但没有 task104 的训练）。
- GPU 与内存现况：qc 自己的管线训练在跑（`round_000020/seeds/seed_000001`），显存 7.7 GB / 利用率 1%，可用内存 7 GB。也就是说，此刻即使启动本任务训练，也仍然是和 qc 训练叠在同一张卡、同一份内存上。
- 结论：本任务的四次训练要等一个 qc 不训练且可用内存 ≥ 11 GB 的窗口；`~/pvz-logs/t104_train.sh` 已按这个条件写好（三采样准入 + 挤压自动中止），直接 `tmux new-session -d -s t104 /home/newbiexvwu/pvz-logs/t104_train.sh` 即可。
### 评估链路实测跑通（2026-10-26 19:35）

在不动 GPU、不碰 qc 的前提下，用 r11 现成检查点把评估链路端到端跑了一遍（2 个游戏进程，L42 的 104000000 / 104000001 两个新种子）：`scripts/task104_panel_eval.py` 生成清单 → 调 `search_play.py --arms fast` → 汇总 `summary.json`，结果 r11 **0/2**、平均末波 3.5、平均丢车 0，与 task102 记的屋顶冷启动 0/16 同向，说明面板与驱动都按预期工作，四个新模型一出来就能直接上这一套。

顺带补了纯数据测试 `python/test_task104_panel_eval.py`（2 项，Mac 通过）：面板字段只挑 search_play 认的键、`terrain` / `copied_from` 这类记账字段不外泄；`--levels` 过滤生效；空选择与无种子任务报错。
### 判据可复算（2026-10-26 19:50）

把预登记判据写成代码 `scripts/task104_verdict.py`：读两套面板的 `summary.json`，输出每训练种子的屋顶 A→B 胜局、两种子平均净提高、逐关两种子平均净变化、非屋顶总体与分区净变化，并按四条判据给 `verdict`（`demonstration rows help` / `not proven; do not deploy`）。用 `Fraction` 算，避免浮点边界把 1/32 判错。`python/test_task104_verdict.py` 六项分支（通过、单种子持平、增益不足、单关受损、回归受损、缺模型）在 Mac 全过。这样四个检查点一出来，结论就是跑一条命令、把 `verdict` 抄进「结果」，不靠手算也不靠事后改阈值。
### 窗口抓到，a_s2 开训（2026-10-27 00:26 台式机）

deckvar 的 task93 expand 第二个种子在 00:25 前结束，可用内存从 11 GB 回到 13 GB、GPU 空出，`task104_train.py` 的准入循环连续 30 秒满足条件后自己起了 a_s2（PID 1040945，配方 `artifacts/task104/recipes/a_s2.json`）。第 1 个 epoch 实测 64,860 行/s，按 15.3 M 训练行算约 235 秒一个 epoch，12 个 epoch 约 47 分钟。这一步没有人工干预，也没有动 deckvar 或 qc 的任何进程。

### 进度（2026-10-27 00:23，主控叫停时）

a_s1 已训完：12 个 epoch、16,982,931 条决策、检查点 `artifacts/task104/a_s1/model.pt`（台式机 22:37:46），训练中实测 34 → 9.6 step/s、19.5k 行/s，显存 7.1 GB。a_s2 / b_s1 / b_s2 尚未开训：它们排在 `task104_train.py` 的准入循环里，条件是「无他人训练 + 可用内存 ≥ 14 GiB 连续 30 秒」。23:39 起 deckvar 的 task93 expand 连续训练（seed 1 于 23:37 结束，seed 2 随即在 23:39 起跑），同时 cq / task78 / llmimport / scoreboard2 的采集进程常驻，可用内存一直在 11–12 GB，14 GiB 窗口再没出现。按预登记「发现其他任务训练就让 qc 优先」，没有降阈值抢跑，也没有动别人的进程。

因此本任务**没有**跑出 A/B 判定：`task104-eval` 会话仍在等训练驱动退出，四个检查点里只有 a_s1 存在，屋顶 384 局、非屋顶 608 局、`verdict.json` 全部未产生。结论仍是「未证实」，不能报收益。

这一轮的另一件事是台式机隧道在 Mac 09:34–10:19 断了 45 分钟（TCP 通、无 banner），期间把「恢复即接手」脚本做好（`~/pvz-logs/t104_boot.sh` 经 `wsl -e bash -s` 从 stdin 执行，只读核对 + 建锁 + 起带准入的训练会话）；恢复后发现别的会话已经在 19:39 起了 `task104-train`，脚本按设计没有重复开训。

### 评估进行中：三模型屋顶全零（2026-10-27 02:22 台式机）

- 检查点：a_s1 22:37、a_s2 01:13、b_s1 02:16 各完成 12 个 epoch，守卫记录退出码均为 0；b_s2 仍缺。
- 屋顶面板 288/384 局（a_s1、a_s2、b_s1 各 96）：**三模型三关全部 0 胜**，即 A 臂 0/192、B 臂 0/96。
- 非屋顶回归 456/608 局：a_s1 116/152、a_s2 99/152、b_s1 102/152。
- b_s2 卡在 `task104_train.py` 的准入循环：要求「无他人训练 + 可用内存 ≥ 14 GiB 连续 30 秒 + 无 GPU 计算进程」，当前无他人训练、GPU 空闲，但 qc 采集常驻使可用内存在 12.9–14.2 GiB 之间摆动，峰值 14,205,484 KiB 仍低于 14,683,136 KiB 门槛，没有连续 30 秒达标。按预登记让路，未降阈值抢跑。
- 判定尚未产生：`artifacts/task104/verdict.json` 未生成。按预登记第一条判据「两个训练种子的屋顶合并胜率均提高」，A 臂两种子均为 0，B 臂已出的 b_s1 也是 0，结论方向已定，但仍等 b_s2 补齐四次训练后由判定器机械给出。

### 三模型评估完成，判据方向已定（2026-10-27 04:30 台式机）

已完成的三个检查点（a_s1 / a_s2 / b_s1）在冻结面板上全部跑完，b_s2 仍缺。

**屋顶冷启动（每模型 3 关 × 32 新种子）**

| 模型 | L42 | L43 | L44 | 合计 | 平均末波 |
|---|---|---|---|---|---|
| a_s1 | 0/32 | 0/32 | 0/32 | 0/96 | 4.51 |
| a_s2 | 0/32 | 0/32 | 0/32 | 0/96 | 5.00 |
| b_s1 | 0/32 | 0/32 | 0/32 | 0/96 | 4.75 |

示范行没有改变屋顶冷启动的任何一局：A 臂 0/192，B 臂已出的 b_s1 也是 0/96，与 task102 记的 r11 0/16、task104 链路实测的 r11 0/2 同向。学生依旧在第一株西瓜投手之前崩盘（平均末波 4.5–5.0）。

**非屋顶回归（19 任务 × 8 种子 = 152 局）**

| 模型 | 白天 | 夜间 | 泳池 | 合计 |
|---|---|---|---|---|
| a_s1 | 43/56 | 56/72 | 17/24 | 116/152 |
| a_s2 | 37/56 | 47/72 | 15/24 | 99/152 |
| b_s1 | 41/56 | 53/72 | 8/24 | 102/152 |

两个训练种子之间散布本来就大（A 臂 116 与 99），B 臂 b_s1 落在两者之间偏下；泳池 8/24 是三个模型里最低的，但单种子不足以定性。

**判据方向已定，与 b_s2 无关。** 预登记第一条要求「两个训练种子的屋顶三关合并胜率都提高（gain > 0）」。训练种子 1 的 A/B 都是 0/96，净提高为 0，不满足严格大于 0；b_s2 无论赢多少局都不能让这条通过，因此结论必然是「本配方未证实收益，不部署」。这只是把已得数据代进原判据，没有改阈值，也没有用未跑完的臂去凑结论。

**机械判定仍等 b_s2。** `task104_verdict.py` 对缺模型直接 KeyError（设计如此，防止半份数据出结论），所以 `artifacts/task104/verdict.json` 仍未生成，本文不手抄替代判定。

**b_s2 的阻塞原因（2026-10-27 04:30 实测）。** qc 第 21 轮从 02:35 起在同一台 26 GB 机器上并行训练两个种子：两个 `train_rollout_policy.py` 处于 D（disk sleep）状态，累计 `read_bytes` 4.09 / 4.10 TiB，swap 8 GB 全满，可用内存 10–11 GB。b_s1 起训时可用内存是 16.8 GB，现在 14 GiB 准入窗口不再出现，训练驱动（PID 970590）仍在每 10 秒轮询。按预登记让路：没有降阈值、没有动 qc 的进程、没有和它抢卡。

### 接手核对：三模型评估产物完整，b_s2 仍无窗口（2026-10-27 05:30 台式机 / 17:30 Mac）

主控恢复额度后接手，先只读核对现场，未动任何训练、评估或 qc 进程。

**评估产物逐种子核对通过。** 用面板 manifest 的冻结种子集逐任务比对 games.jsonl：屋顶 a_s1 / a_s2 / b_s1 各 0/96、种子集合与面板完全一致；非屋顶 116/152、99/152、102/152，同样逐任务种子一致，无缺局、无多余局。b_s2 两个目录都不存在。

**注意 summary.json 只装本次调用传入的模型。** 现存的 eval/roof/summary.json 与 eval/regression/summary.json 里只有 b_s1，因为 task104_panel_eval.py 每跑一个模型就把累计字典重写一次，而 task104-eval 是按模型分批调用的。所以最终那次汇总必须**一次传齐四个 --model**（已有结果会按种子跳过，不重跑），否则判定器仍会因缺模型 KeyError。这一条已写进下面的接手要点。

**b_s2 的阻塞没有缓解。** qc 第 21 轮 02:29 起两个种子各载入 19,649,496 条决策（board 6.50 GB / 进程），两个进程 05:32 时 state=Dl+、已运行 3 小时 01 分，train.log 停在 02:32 的预处理输出；swap 8 GB 用满，可用内存 10.7 GiB，仍低于 14 GiB 门槛（14,683,136 KiB）。训练驱动 PID 970590 存活并在每 10 秒轮询，task104-train 会话在等；task104-eval 会话（pane 1044797）也在 30 秒一轮的等待循环里存活，四个检查点一齐就会自动接上评估与判定，不需要人工再起。

**结论方向不变，仍不手抄判定。** 训练种子 1 的屋顶 A/B 都是 0/96，净提高 0，不满足预登记的严格大于 0，b_s2 补上也不能让第一条通过；但机械判定仍要求四模型齐全，verdict.json 未生成。

## 问题

**台式机被 qc 第 21 轮挤到换页，14 GiB 窗口短期内不会出现（2026-10-27 05:33 实测）。** 两个 qc 训练进程各载入 19,649,496 条决策、board 6.50 GB，60 秒采样显示 `read_bytes` 从 7.867 / 7.886 TB 涨到 7.904 / 7.924 TB，即合计每分钟读盘约 37.7 GB，CPU 各只前进约 11 秒（约单核 18%），`train.log` 停在 02:32 的预处理输出，进程已跑 3 小时 04 分。这是典型的内存不足换页：swap 8 GB 全满，可用内存 9.9–10.0 GiB。也就是说 qc 自己也在以不到两成的速度爬，本任务的 14 GiB 准入在它结束前不会满足。按预登记让路，没有降阈值、没有动 qc 进程。

**评估链路已逐条核实可用，缺的只有 b_s2 检查点。** 读 `artifacts/task104/eval_ready_models.sh` 确认：它对四个 run 逐个等 `model.pt` 与 `logs/guard/<run>.result.json`（并断言 returncode 为 0，a_s1 / a_s2 / b_s1 三份都在且为 0），再按面板跑该模型；四模型齐全后**再单独跑一遍同时传四个 `--model` 的汇总**（已完成的种子会被跳过），最后调判定器。所以只要 b_s2 一落盘，评估、汇总与 `verdict.json` 会自动接上，不需要人工干预，也不需要我改任何脚本。
**同一工作树上有另一个会话在并行写。** 2026-10-26 16:50–17:00（+08:00），台式机 `artifacts/task104/` 下出现本会话之外产出的 `data_a` / `data_b` / `recipes` / `training_inputs.json` / `r11.source_rows.json`，来源不是本会话的进程（本会话当时只建了 `corpus/`）。之后该会话不再写文件（截至 17:15 无新产物、无训练进程）。本会话按"不覆盖别人产物"处理：先只读核对，A/B 两份语料的行数与段字节都验过（见上节），再据此继续。若该会话恢复并开始训练，两个 1,700 万行的训练同时上卡会超出 16 GB 显存，需要先确认谁持有 GPU 再启动。

**新进程重验旧示范在这台机器上不可靠（既有问题，非本分支引入）。** `python.test_human_demo.HumanDemoTest` 的三个真实引擎用例在 HEAD 与基点 144a7dce 上都以 `demonstration initial state hash differs` 失败；独立复现显示同一进程内第二次 reset 偶发给出不同初始哈希，而两次独立进程的 `STATE_BYTES` 全量转储逐字节相同。因为 qc 的起点格要求档案 `verified`，这意味着「录制机之外、或同一进程里重跑 verify」可能失败；本次不修（修它要动引擎的哈希面，超出 task104 范围），示范行的导出与训练不受影响。
### 训练启动（2026-10-26 18:23）

**台式机隧道再次断线，剩余工作全部卡在这台机器上（2026-10-26 Mac 09:34 起）。** Mac 侧轮询 `ssh -p 22222 newbiexvwu@127.0.0.1` 从 09:34 到 10:02 连续失败：TCP 能连上（UU 远程在本机 `127.0.0.1:22222` 有监听）但等不到 SSH banner，说明台式机那端关机、睡眠或 UU 客户端掉线，不是本仓库或密钥的问题。局域网 192.168.0.112 / 114 的 22 端口能连但立即关闭，没有可用密钥通路。

影响面：四次训练、384 局屋顶、608 局回归、判定全部只能在台式机上跑。Mac 侧无 r11 语料——`artifacts/` 下 62,285 个 npz 全是评估分片，没有任何 `packs/segments` 目录；HF `realnewbiexvwu/pvz-agent-artifacts` 的 55 个文件里也没有 r11 段（只有 task69 的 transfer 目录）。所以「Mac 用 MPS 训练」这条路仍和之前记录的一样，需要先把约 10 GB 的 A/B 语料从台式机转出来，本身也依赖台式机先上线。

现场准备：本机 `~/pvz-logs/t104_watch.sh` 每 30 秒探测一次，一通就把 `hostname` / `date` / `nvidia-smi` / `free` / `tmux ls` / `pgrep task104` / `pgrep train_rollout_policy` 落到 `~/pvz-logs/t104_watch.log`，不写远端任何状态。台式机恢复后按「开训与评估的运行清单」逐条执行，先确认没有遗留 `gate_seed_inventory.py`，再按 GPU 与内存准入决定是否开训。

随后把探测器升级成恢复即接手：`~/pvz-logs/t104_boot.sh` 经 `ssh ... 'wsl -e bash -s'` 从标准输入执行，不做任何引号拼接。它只做三件事——打印 HEAD / 脏文件数 / 遗留进程 / GPU / 内存 / `tmux ls` / 四个检查点是否存在；若 `task104-train` 会话已存在或 `artifacts/task104/train.lock` 已被别的启动者占住就直接退出，不重复开训；否则建锁并 `tmux new-session -d -s task104-train` 起 `scripts/task104_train.py`。该驱动本身带 GPU 与内存准入（无他人训练、显存无计算进程、可用内存 ≥ 14 GiB 连续 30 秒才开训，运行中遇他人训练即暂停），所以提前起会话不会和 qc 抢卡，只是排队等待。

台式机 18:19 恢复连接。先按主控要求核对遗留扫描：`gate_seed_inventory.py` 已无进程；GPU 上 deckvar 的 task93 expand 训练仍在跑（7.0 GB / 37%），等到 18:23 它结束、显存回到 4.5 GB / 0% 才开训，没有与 qc 或别人的训练抢卡。

`tmux t104` 跑 `~/pvz-logs/t104_train.sh`，四个配方顺序训练。a_s1 已载入 **16,982,931 条决策 / 55,773 个种子 / 1,704,229 条验证行**，与 A 索引 `7d15b915a2192805` 的行数逐位相同；`data_memory` 报 board 5.56 GB、scalar 1.85 GB、packet 1.19 GB。训练前数据层测试在台式机通过（`HumanDemoExportTest` + `test_recipe` 共 10 项）。
### 训练受阻于内存：换页抖动，改为内存准入（2026-10-26 18:42）

18:23 启动的 a_s1 在 18:41 被停掉：`metrics` 显示 **0.07 step/s、135 行/s**，按这个速度一个 epoch 要 35 小时（r11 当年 200 秒一个 epoch）。同时 `read_bytes` 累计到 562 GB，而 `rchar` 只有 14.7 GB——进程 RSS 3.4 GB 却被换出，说明 9.5 GB 的训练数据在 swap 里来回换页，不是算得慢。

现场内存账（18:41）：`search_play` 7.1 GB + `pvz-portable` 6.5 GB（qc 采集 16 个 worker）= 13.6 GB，别人的 `task96_full_pack.py` 3.5 GB，node 2.7 GB，我的训练 3.2 GB，25 GB 机器 + 8 GB swap 全满。r11 当年训练时机器上没有 `task96_full_pack` 这类额外作业，所以同一配方能 200 秒一个 epoch。这不是配方问题，也不该靠调小 batch 掩盖（那会破坏与 r11 的可比性）。

处置：`~/pvz-logs/t104_train.sh` 改成**内存准入**版——`MemAvailable ≥ 11 GB` 连续 3 次采样才开训，训练中若可用内存连续 4 分钟低于 4 GB 就中止该次并重新等窗口（每次都会写 `ABORTED` 进对应 log）。这样只在自然空档里跑，不去 kill qc 的采集进程、也不改任何生产配置。tmux 会话 `t104` 已在 18:42 起来，当前 avail 8 GB，处于等待。
### 开训与评估的运行清单（台式机，连接恢复后按序执行）

全部路径写绝对路径，工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/demorows`（分支 demorows）。

1. **先清掉遗留扫描**：`pgrep -af gate_seed_inventory.py`，有就 `kill`；再确认没有本任务遗留进程。2026-10-26 18:19 已核对为空。
2. **GPU 准入**：`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` 加 `pgrep -af train_rollout_policy`。
   qc 或别的任务在训练（deckvar 的 task93 expand 在 18:19 仍在跑，占用 7.0 GB / 39%）就让路，不抢卡。
3. **四次训练**（顺序，不并行；12 epoch、`init=null`、seed 1/2）：

```bash
tmux new-session -d -s t104 /home/newbiexvwu/pvz-logs/t104_train.sh
tmux pipe-pane -t t104 -o 'cat >> /home/newbiexvwu/pvz-logs/t104_train.out'
```

启动脚本已在台式机 `~/pvz-logs/t104_train.sh`：四个配方顺序训练，各自写 `artifacts/task104/logs/train_<run>.log`，
末尾打印 `ALL DONE`。检查点落在配方 `out` 目录，即 `artifacts/task104/{a_s1,a_s2,b_s1,b_s2}/model.pt`。
配方自带 `data` / `out` / `epochs` / `seed` / `label_weight` / `source_weight`，命令行不再覆盖任何字段。

4. **屋顶评估**（4 模型 × 3 关 × 32 种子 = 384 局，`--workers 2`）：

```bash
cd /home/newbiexvwu/PvZ-search/.claude/worktrees/demorows
/home/newbiexvwu/.venvs/ml/bin/python3 scripts/task104_panel_eval.py \
  --panel MEMORY/docs/task104-roof-panel.json \
  --model a_s1=artifacts/task104/a_s1/model.pt --model a_s2=artifacts/task104/a_s2/model.pt \
  --model b_s1=artifacts/task104/b_s1/model.pt --model b_s2=artifacts/task104/b_s2/model.pt \
  --value /home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt \
  --workers 2 --out artifacts/task104/eval/roof
```

5. **非屋顶回归**（同四模型 × 19 任务 × 8 种子 = 608 局）：`--panel` 换 `MEMORY/docs/task104-regression-panel.json`，
   `--out` 换 `artifacts/task104/eval/regression`，其余同上。

6. **判定**（预登记判据的机械实现，读两份 summary）：

```bash
/home/newbiexvwu/.venvs/ml/bin/python3 scripts/task104_verdict.py \
  --roof artifacts/task104/eval/roof/summary.json \
  --regression artifacts/task104/eval/regression/summary.json \
  --out artifacts/task104/verdict.json
```

四条判据（两种子屋顶都提高、平均净提高 >= 10 个百分点、任一屋顶关两种子平均净降 <= 1/32、非屋顶总体 <= 3 且任一区 <= 5 个百分点）写在 `scripts/task104_verdict.py` 里，`python/test_task104_verdict.py` 用合成数据覆盖了通过、单种子持平、增益不足、单关受损、回归受损、缺模型六种分支（Mac 已过）。输出 JSON 的 `verdict` 直接抄进「结果」，不手算、不调阈值。

评估只跑学生臂（`--arms fast`），无 `--demo-archive`、无 `--verify-samples`，四模型同任务同种子配对；
`search_play` 会跳过已完成的种子，中断可续。
## 执行状态

2026-10-26 主控补充：冻结仅针对 Mac 本机，保持其游戏与训练进程原状。台式机经 SSH 的只读查看、代码工作及最多两个游戏进程的实验可以继续，GPU 训练仍须先确认 qc 未在训练。本分支代码与预登记已推送，台式机建立独立 `demorows` 工作树，未修改 qc/cq。已启动台式机数据测试与真实引擎测试，结果待返回。

剩余工作：

1. ~~等待台式机已启动的 `python.test_human_demo.HumanDemoTest` 真实引擎测试结果。~~ 已返回：失败，且基点 144a7dce 上同样失败，属既有的初始哈希偶发不一致，见「结果」。
2. ~~只读核实 r11 与最近完成轮的配方、实际包和 `source_rows.json`，把 human / LLM 自身决策行数及 teacher 起点续局行数分开填回。~~ 已填回：r11 的 human / LLM 行数均为 0，teacher 在 user 任务上的标注 17,286 行，`demo_start` 任务 0 个来源。
3. 已完成私有语料视图、不可变段及两臂配方冻结。A/B 共有段不变，26 份示范在实际包内的逐档索引、种植数和单位行权重均与冻结清单一致。
4. 获取并冻结 task92b v2 清单，核对预留新种子后提交输入身份，再按上述判据完成四次从零训练、384 局屋顶评估及非屋顶回归。训练前确认 GPU 准入或记录 MPS 后端决定，不与 qc 抢训练。
5. 填写结果与结论，追加当天项目记忆。检查暂存区不含模型、帧与日志，只提交代码和 KB 级证据，推送 `demorows`，不合并、不部署。

### 最终输入核验与运行入口（2026-10-26）

[task104-training-inputs.json](task104-training-inputs.json) 冻结 r11 配方与来源报告哈希、348 个共有段的哈希、A/B 索引和四份训练配方。原 r11 索引已被生产裁剪，按原配方与不变段重建，恢复相同索引键及来源行数。26 份示范的实际 `source_row` 与导出元数据逐项相同，索引哈希、种植数与全部行权重 1 均通过。A 为 15,278,702 训练行 / 1,704,229 验证行，55,773 个种子。B 为 15,305,552 训练行 / 1,709,421 验证行，55,787 个种子。两臂完整训练及验证种子列表在台式机 `artifacts/task104/split_a.json` / `split_b.json`，哈希记在冻结文件。

私有 `build/pvz-portable` 已复制并冻结台式机现成的 `hashdet/build/pvz-portable`，SHA256 为 `5e890b3a392c343478e0db0633bd47104e2e94a33b802f8cbd023e9b1536102f`。使用该引擎，`python.test_human_demo.HumanDemoTest` 三项在 19.236 秒内全部通过。复制后再次核对哈希相同，入口为独立文件，其他工作树后续重建不影响本实验。没有修改引擎源码或 qc/cq 的二进制，未放宽哈希校验。后续四模型评估统一使用此私有入口。

仅检查内存的旧 `t104` 等待器已停用，改用 `scripts/task104_train.py`，同时检查训练进程、GPU 与内存，运行期间复用 `dagger_experiment_runner` 的暂停/恢复守卫。启动命令为 `tmux new-session -d -s task104-train '/home/newbiexvwu/.venvs/ml/bin/python3 -u scripts/task104_train.py'`，工作目录为上述台式机私有树。四份配方与预登记不变。

评估摘要新增每个游戏种子的胜负，完成记录须与面板种子集合及数量相同。判定器复用 `value_widen_ab.paired`，同时输出屋顶及非屋顶逐关、逐训练种子的 gain/harm，以及两次训练结果的范围。11 项面板与判定测试通过，原四条数值判据保持不变。

`task104-eval` 已改为逐个接续已完成模型：等该模型的训练退出码为 0，就运行其屋顶 96 局与非屋顶 152 局，与其余模型训练交叠。四模型完成后复用现成断点续跑读取已完成游戏，汇总全部屋顶 384 局及非屋顶 608 局，再生成 `artifacts/task104/verdict.json`。台式机执行文件为 `artifacts/task104/eval_ready_models.sh`，仍只用原评估脚本。评估统一禁用 CUDA，同一时刻最多两个游戏进程，完整输出在 `artifacts/task104/logs/eval_pipeline.log`。任一步失败即停止后续命令。

已核验 a_s1、a_s2、b_s1 三个实际检查点：均为 CUDA、`init=null`、完整 12 轮，训练种子依次为 1、2、1，检查点内配方 SHA256 与冻结记录相同，训练退出码均为 0。模型哈希、实际来源报告哈希、行数与逐轮耗时记录在 [task104-training-results.json](task104-training-results.json)。A 两次来源报告与原 r11 报告逐字节相同，B 首次实际增加人类 15,419 行和 LLM 16,623 行，训练及验证行数与冻结切分一致。三者各完成屋顶 96 局与非屋顶 152 局，共 744 局，全部为 `fast` 臂。当前屋顶胜局均为 0/96，非屋顶依次为 116/152、99/152、102/152。b_s2 尚未开训，实际判定仍缺失，不能据此结束实验。台式机 2026-10-27 05:43 核对时，qc 第 21 轮两个训练进程仍在运行，本任务按原约束排队。

### Mac 解冻后的执行选择（2026-10-26）

主控允许改用 Mac，并确认台式机 qc 正在排空部署、训练第 20 轮第二个种子。本任务继续使用已冻结的台式机 CUDA 后端。该机第一种子后段实测每个 epoch 为 167–187 秒，四次训练共 48 个 epoch 的纯训练预算约 2.2–2.5 小时，另计当前等待和数据载入。Mac 的 MPS 可用、内存为 48 GiB，但本机没有已核实完整的 r11 冻结语料，也没有同规模 MPS 训练的实测耗时，迁移需增加语料传输与准备时间。现有 CUDA 队列已就绪，随 qc 排空接续，保留四次训练的统一后端及原配方。游戏进程限额仅计算本任务，其他任务的游戏进程不参与准入判断。
