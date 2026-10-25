# task78：叶值 v6——把雾区与屋顶的决策级数据加进来，并修"雾区估值随视界翻转"

> 主控 2026-10-25 写。执行者：台式机 Codex（Plus 号，新开 tmux `codex-v6`，`CODEX_HOME=/home/newbiexvwu/.codex-plus`）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/v6`（分支 `v6`，从 origin/pvz-env 拉出）。游戏进程最多 4 个；每次 GPU 训练 ≤1 小时，开始前 `MemAvailable` ≥10 GiB；**不碰 qc、cq、流水线**（流水线目录只读）。

## 为什么
- v5（task43，`MEMORY/docs/task43-value-v5-pool.md`）用泳池决策级标签修好了叶值压平，L21 配对胜率 14→28/32，之后被采纳。
- 但 v5 的训练语料里**没有雾区和屋顶**。task72b 发现：雾区 26 个分化态里有 10 个，偏好方向随视界 2500/4000/6000 翻转——估值噪声大。
- task46 在屋顶上测得 v4 叶值与终局的一致率只有 53.9%。
- 现在 qc 已经在雾 L31、屋顶 L41、三个示范起点格上积累老师帧；task72b 也有 94 个带终局真值的雾区态（`MEMORY/docs/task72b-probe-rows.jsonl`）。

## 要做的
1. **数据**：照 task43 的决策级标签管线，从 qc 帧（只读）里抽雾区和屋顶的决策点做标签，与 v5 原语料合并。配比与样本量写清依据。
2. **预登记门禁**（先写进本文件、提交，再训练）：照 v5 的采纳标准。
   - 白天、夜间、泳池非劣；
   - 雾区（L31 human 口径、L33 示范接手点）和屋顶（L41–L43）配对胜率；
   - 叶值与终局真值的一致率，用 task72b 的 94 态加屋顶新测的一批；
   - 雾区视界稳定性：方向翻转率。
   - 种子与训练段不相交，机械核对。
3. **训练与评估**：GPU 每次 ≤1 小时，可多轮。按预登记出结论。
4. 结论写 `MEMORY/docs/value-v6.md`，提交并 `git push origin v6`。是否采纳由主控定。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不加开关，不改生产默认。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 来源冻结完成：408份可用档案中按预登记选64份/128 root，源清单SHA `b3b86e2996e9e6b1173ac48264eb6858c00ee3a51470280b3b069b8976c2edf0`；3项身份/抽样测试通过。旧v5声明99项源路径均缺失；所选示范原始档案均可只读访问。详见 [来源盘点](task78-source-inventory.json)。尚未生成标签或训练。

- 2026-10-25：完成任务与本机来源初查；以下预登记写于任何 v6 训练、正式配对对局或候选标签生成之前。基线固定已采纳 v5，执行者不改生产默认。

- 2026-10-25：64/64 冻结档案在未改动引擎（SHA `49783a70…`）上逐帧物理重放通过（4 worker，642 s），每个 root 的引擎状态 SHA 与档案一致，无失败；`artifacts/task78/replay.json`（ignored）。采集脚本因此获得"root 可精确复原"的前置条件。
- 2026-10-25：预留评估段机械审计通过：本地 200 份面板/结论/配置 + 105,740 个档案 seed 身份 + v5 `trained_on` 3,661 项，0 碰撞 0 读取错误；未覆盖远端未下载的实验（`unseen_remote_experiments_covered=false`）。`artifacts/task78/seed_audit_summary.json`（ignored）。
- 2026-10-25：本机 torch 只在 `/home/newbiexvwu/.venvs/ml/bin/python`（3.14.4 / torch 2.14.0+cu132，RTX 5080 可用）；`/usr/bin/python3` 无 torch，采集与训练一律用前者。

- 2026-10-25：标签通道打通（`scripts/task78_collect.py`）。两处与 task43 不同的实现细节：生产 `ROLLOUT_VALUE` 只回值不回帧，采集器改用同一 root/seed/tick 的完整 `ROLLOUT` 并自行打叶值分（状态哈希与生产逐位相同，已用 `task78_leaf_check.py` 核对）；终局叶子的私有状态哈希与逐帧参考不同（公开帧与两项收入检查一致），这类行 v5 导出本来也丢弃，改为记录 `terminal_private_state` 而不中止，非终局叶仍要求逐位一致。
- 2026-10-25：**旧语料不可恢复，改用同源重建（偏离预登记，记录在此）**。v5 声明的 99 项来源全部缺失：`worktrees/v5/artifacts/task43/sparse64/data`、`artifacts/hard/branch/z13_384.pt`、`artifacts/hard/branch3/z15d_shards/`（96 分片）、`worktrees/vnet/artifacts/task30/full_audit_manifest.json`；HF 也无。替代：同源 qc 档案重建旧区决策标签（day_d7 / night_n9_n10 / pool_p9k 各 8 档案、1 视界 2500）与旧区整局 episode 语料（各 192 局 × 32 帧）。预登记的 branch 192 行份额并入 episode 份额（64→256），旧:新仍 384:128 = 75:25，其余配方不变。
- 2026-10-25：首批 6 个档案的标签分布（诚实记录）：L31 示范接手组 4/4 档案的 root 在 2500 tick 内全部终局且全胜（有效对 0，行被终局过滤丢弃）；屋顶样例 4 候选全负。也就是说雾区示范接手的 root 本身已经赢了，采不到可教信号。
- 2026-10-25：**视界采集的整组偏离（在任何 h4000/h6000 行被生成前记录）**：雾区示范接手档案（索引 0–31）在 h2500 的全部叶子都已终局，游戏既然在 2500 tick 内结束，h4000/h6000 的叶子必然同样终局、导出恒为 0 行。因此这 32 个档案只采 h2500；三个视界在 fog_l31_human（32–39）与 roof（40–63）上采齐。决定按整组统一施加，没有按结果挑档案，也没有删任何可能存在的行。
- 2026-10-25 14:50（主控通知）：台式机内存紧张，qc 只剩 5 个采集进程，本任务从此刻起**同时最多 2 个游戏进程**（此前为 4）。直接后果：屋顶 24 档案只采 h2500（原定三视界），配对对局的可跑规模随之受限。这是外部资源约束，不是按结果挑选。

- 2026-10-25 15:05（交接，主控要求停）：**标签采集完成 41/64 档案、68 个 (档案,视界) 任务**。已齐：雾区示范接手 0–31 的 h2500（32 档案）、fog_l31_human 32–38 的三视界、roof 48 的 h2500。**未完成**：fog_l31_human 39 的 h4000/h6000，以及 roof 40–47、49–63 的 h2500，共 **25 个任务**，清单已写在 ignored `artifacts/task78/jobs_new.txt`，用 `xargs -P 2 -n 2` 续跑即可（每局约 8–11 分钟，2 进程约 2 小时）。**导出、训练、94 态复评、配对对局都还没跑**。
- 2026-10-25 15:05：续跑命令（工作树内，绝对路径）：
  ```bash
  cd /home/newbiexvwu/PvZ-search/.claude/worktrees/v6
  xargs -P 2 -n 2 -a artifacts/task78/jobs_new.txt sh -c 'PVZ_NO_POPUP=1 PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources /home/newbiexvwu/.venvs/ml/bin/python scripts/task78_collect.py $0 $1 >> artifacts/task78/logs/batch_new.log 2>&1; echo "DONE $0 $1 $?" >> artifacts/task78/logs/batch_new.status'
  /home/newbiexvwu/.venvs/ml/bin/python scripts/task78_export.py            # 新区域 -> artifacts/task78/data
  # 旧区决策标签（另一份清单，需先跑 task78_replay.py artifacts/task78/old_sources.json）
  /home/newbiexvwu/.venvs/ml/bin/python scripts/task78_episode.py --games-per-cohort 192   # 已完成，18,432 行
  /home/newbiexvwu/.venvs/ml/bin/python scripts/task78_train.py --device cuda
  /home/newbiexvwu/.venvs/ml/bin/python scripts/task78_truth.py --horizon 2500 --workers 2
  ```
- 2026-10-25 15:05：已落盘的可用产物（全部 ignored，不在 git）：`artifacts/task78/{sources.json, old_sources.json, replay.json, seed_audit_summary.json, labels/, episodes/}`。旧区决策标签清单 `old_sources.json` 已冻结但**尚未重放验证**（`task78_replay.py artifacts/task78/old_sources.json`，24 档案、4 worker 约 4 分钟，现按 2 进程约 8 分钟）。


- 2026-10-25 续作：确认旧批次已停止后，以 `xargs -P 2` 续跑剩余25个标签任务（只启动本树采集器）；按修订后的计划重数总数为80个任务，交接的68个完成项混入了计划外视界，不能直接作完成分母。雾区39的h4000/h6000已补齐，屋顶继续采集。
- 2026-10-25 续作：修正导出器旧区误读新标签目录、强制要求已停采视界两处错误；导出前逐项核对source/模型/引擎SHA、root状态、全部候选、真实终局、历史检查及完整老师动作链。档案32的三个视界实测审计/导出30行成功，`--only`输出明确标为非完整导出。相关回归28项通过。
- 2026-10-25 续作：训练器不再静默丢弃缺失旧决策或无fit/val的新区域，修复episode构造参数错位，checkpoint的trained_on保留init见过的种子；GPU进程内限时3600秒。尚未训练，完整原语料仍是交付所需来源，同源重建只能作诊断，不能替代“与v5原语料合并”要求。
- 2026-10-25 续作：逐行重算task72b原94态wait/static-best终局差：88态相等，4态−0.125、2态+0.125，**没有任何对达到预登记abs(gap)≥0.25**。排序门目前必为证据不足，不能改阈值换通过。bootstrap改按(task,future seed)整条轨迹聚类，零有效对返回None区间；完整94身份/成功状态缺一即拒绝报告。


- 2026-10-25 续作：在任何正式配对局开跑前，修正代码面板漏掉的L33接手N=839/762/686三格以及L31缺少human观测字段；完整正式面板现为15格/512对（旧12格/416对清单不够覆盖预登记）。普通seed分配不变，三示范格未来分别78100000–78100031 / 78100032–78100063 / 78100064–78100095。示范按future配对，重复、非terminal、非布尔won均拒收；缺格不得通过分区门。冻结摘要见 [配对清单](task78-ab-plan.json)。
- 2026-10-25 续作：已完成的新标签中59个计划任务逐一通过原始标签身份、全部候选、完整动作链与root重放状态审计，403条非终局叶，失败0；另21个任务尚未完成。审计快照在ignored `artifacts/task78/label_audit_partial.json`，不冒充全量导出。
- 2026-10-25 续作：复核task72b说明，94行decision是接手后老师续走序号，不能当成人类原始示范的decision；原始/tmp/task72b探针源码、接手点和状态轨迹仍需取得。现有 `task78_truth.py` 的直接按decision重放流程尚不能定位这94态，正式复评尚未运行，须修复后再用。


- 2026-10-25 续作：正式训练器恢复原预登记五类配比：old episode 64 / old L7 branch 192 / old pool decision 128 / fog 64 / roof 64。此前生成的同源重建缓存仅留诊断，不充当v5原语料。原分片用mmap读取，新的fit采样对group/world/root/horizon等权；原分割行数、全部97 branch来源、64 pool分片、episode缓存与审计清单必须与v5检查点匹配。
- 2026-10-25 续作：从v5检查点机械取出原语料规格：branch fit/val=324559/29446、episode=75136/9600、pool decision=3404/456，共442601行（更正此前129992行的笼统描述）。完整恢复/路径映射规格见 [原语料要求](task78-v5-corpus-requirements.json)。训练前拒绝部分新数据导出、评估seed/future交叠和已有run目录；检查点记录初始化与新增future身份，写入采用原子替换。41项相关测试通过；尚未开始GPU训练。


- 2026-10-25 续作：完整配对执行入口已实现 `task78_ab.py freeze/v5/v6/report`：原候选训练必须完整且初始化/训练seed不碰评估段；两臂冻结模型、native引擎、核心Python代码、软件版本、示范三份原始文件与面板SHA。示范命令显式传原base seed、接手点、32个独立future。每格写运行回执、重新核对输入，先确认本树没有其他活游戏才启动2 worker；报告必须匹配回执。命令差异测试证明两臂只换叶值/输出路径。相关44项测试通过。
- 2026-10-25 续作：配对报告空跑前置核对完成，五地形均返回incomplete/passed=null，`complete=false`、`all_game_gates_pass=false`；未把尚未进行的正式对局记成通过。真实游戏仍只有续采批次。
- 2026-10-25 接手（CodeBuddy）：屋顶终局面板入口 `task78_roof_truth.py`（plan/trajectories/sources，24 世界 48 root × 3 共同未来，`training_eligible=false`）与导出器“雾区一律三视界”口径修正入库；采集器把独立面板的标签隔离到面板自己的目录，避免覆盖训练标签。恢复清单 `artifacts/task78/jobs_fog_restore.txt`（雾区 9–31 的 h4000/h6000，46 个任务）已生成、尚未运行。24 项相关测试通过。

## 预登记（2026-10-25，训练前冻结）

### 来源、抽样与训练

- qc 只读来源：`/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2`。按档案名区分 L31 human、L31/L32/L33 仙人掌/L33 香蒲示范接手、L41 r7；不把老师旧 targets 当终局真值。按 SHA256(档案绝对路径) 排序，五个雾组各取 8 份、屋顶取 24 份，共 64 份；每份取实际有 targets 的搜索序列中点分位两个 root（不足则全取），选择过程不读取胜负。固定源文件 SHA 后才生成候选标签。完整清单在本树 ignored `artifacts/task78/`，KB 级统计进 git。
- 标签复用 task43 `sample_root` 的真实搜索候选、原生学生叶、历史恢复核对及 `finish_teacher` 冻结老师终局续走；2500/4000/6000 三视界等量采集，完整候选与共同随机未来均记录。短推演已终局的叶保留审计，训练只用非终局叶，不把学生终局当老师标签。最多本任务 4 个游戏进程。
- 合并 v5 的完整旧语料（旧决策分片、L7 branch、跨区 episode 缓存），从 v5 权重热启动。每批 512 行：旧决策 128、雾区 64、屋顶 64、旧 branch 192、旧 episode 64。新区域按组/档案/root/视界等概率取样，避免 L41 长局或 L33 多候选独占；三个视界各占新数据 1/3。保留 75% 旧样本暴露以约束遗忘，新两区各 12.5% 有相同训练权重。64 档案/128 root 是与 v5 64 seed、768 root 相比的稀疏区域补充，不声称独立样本为候选行数。
- AdamW、lr=1e-4、weight_decay=1e-4、最多24 epoch、每epoch256步、训练随机种子0；不加 margin/手写价值公式。沿用 BCE、验证损失早停（4轮不改善、最小改善1e-4），只取最优验证模型。新数据按世界身份 SHA256 的前8位 mod10==0 作停止集，训练与停止身份不交叉；若任一区停止集不足2档案，取该区哈希最小2份作停止集并在训练前固定。旧数据沿用 v5 split。训练前 MemAvailable≥10 GiB；每次 GPU 训练用独立进程硬限≤3600秒，可恢复的多轮需另记开始/结束/设备/超参，不择优挑评估。

### 配对胜率（唯一改变叶值网）

- 基线 v5 SHA256 `8a636d337c03dcf3037f42ce41f79abc5e0b898eb5a2b39d9a00ad3b9d88c3b4`；学生 dw4i_s1、静态 branch1、beta=1、samples=3、verify_samples=9、student_candidates=2、horizon=2500、min_gap=300、per_card=1、halving、rollout_sim_skip=61。学生/静态/引擎/任务逐项 SHA 在开局前冻结。任务卡组沿用原任务族，不造新卡组，不修改 qc/cq。
- 白天 L7 d7、夜间 L14/L17/L20 n10、泳池 L21/L22/L23/L25 p9k，每格32对。各地形汇总 gain/harm 的配对差值区间：分别用 gain/n 的单侧97.5% Clopper–Pearson 下界减 harm/n 的单侧97.5%上界，构造保守单侧95%下界。白天仅32对可能证据不足，因此白天预设64对。三个地形分别下界 > -0.10，且任何格净胜不低于 -2/32（白天 -4/64）才通过“非劣”；不以 p>0.05 冒充非劣。若证据不足记未通过，不补挑种子。
- 雾区：L31 human snow m100 32对、L33 cactus snow m200 示范接手点 N=839/762/686 各32对（以 `configs/human_demo_tasks.json` 既有点为准）；另报 L31/L32/L33香蒲既有示范起点的诊断结果，不混入正式四格分母。接手点必须与原始示范逐帧核对，未来种子独立。
- 屋顶：L41/L42/L43 r7 原倍率各32对。先核对 terrain/deck/upgrade 所有权；卡组覆盖不通过则中止该格，不改卡组救分。
- 改善门：雾四格合并、屋顶三格合并分别要求 v6 净胜>0 且精确双侧 McNemar p<0.05；L31 human、L33合并、L41/L42/L43各自净胜均≥0。每格完整逐 seed 配对，不容缺局、重复或非terminal算失败。
- 新普通关评估种子预留 `[78000000,78002000)`，按上列任务顺序连续分配所需配对数；示范新共同未来预留 `[78100000,78102000)`，按接手任务与点顺序连续分配。训练来源 seed/future、旧模型 trained_on、任务面板训练段与两评估段必须机械零交集。预留不等于审计通过，发现历史冲突须在任何对局前另作修订并提交。

### 终局一致率与雾区视界稳定性

- 雾区固定 task72b 全94态，保留原 task/seed/decision/tick/候选及 truth 身份；不按真值筛训练，也不把它们混入新训练。task72b 只有 wait/static_best 两候选有终局标签时，只在这两者上检验，不给其余格伪造真值。屋顶用独立评估种子 L41–L43 各8局，每局实际搜索序列两中点 root、全部实际候选、每候选3共同未来新增终局标签；共最多48 root。
- 按同 root 非终局候选间终局胜率差≥0.25计算叶值排序一致率（正确1、预测平局0.5）；另报所有对、真值平局、终局旁路及有效 root/seed 数。主要视界2500；4000/6000同口径分报。分别以独立世界为簇10000次bootstrap（seed=780），报告配对改善95%区间。雾/屋顶各要求点估计改善且区间下界≥0；没有足够非平局对则证据不足，不删零信号root。
- 方向为同 root 同候选（task72b 的 static_best vs wait）的预测差符号，三视界有正也有负记一次翻转，0单列不当方向。全94态及原26个真值分化态分别报告分子/分母；固定原始共同随机未来，v5/v6只替换叶打分。要求 v6 两口径翻转率均≤v5一半，且按世界聚类bootstrap的翻转率差95%上界≤0。重算 v5 与旧汇总有协议差异时先解释，不能将旧四舍五入数值与新全精度混比。
- 四类门禁（旧三区非劣、新两区配对改善、终局排序改善、视界稳定性）取 AND。所有任务/种子/档案/标签/模型哈希与覆盖均通过才可给“满足预登记”；任何门未过或缺原始证据即不推荐替换，采纳由主控决定。最终完整结论写 `MEMORY/docs/value-v6.md`；权重与大分片不入 git，不自动上传或切换生产。

## 问题

- 数据量远小于预登记预期：每个 root 只有 4–10 个候选，64 档案 × 2 root × 3 视界 ≈ 1.2k 行；v5 的 129,992 行训练集在本机无法复现。v6 的可训练信号因此比预登记假设弱一个量级，结论必须按此说明。
- 雾区示范接手（fog_demo_l31/l32/l33）的档案从接手点开始就是终局前 2–3k tick，2500/4000/6000 三个视界的叶子基本都终局，导出会丢掉全部行。这些档案对"终局一致率"与配对胜率仍有用，但对标签训练几乎无贡献。
- 预登记的泳池第 4 格 L25 没有 p9k 任务族：L25 是原生传送带关（task75），面板里不存在对应卡组。改用同族面板里的 L24 p9k，并在结论里标注为偏离；不改卡组去"救"这个格。
- 本机 qc 与 cq 都在跑（cq 12 个 search_play 进程），标签采集只能用 4 个游戏进程，roof 档案每局续走到终局约 8 分钟，全量 64 档案 × 3 视界是数小时量级。

- 初查本机没有 `/home/newbiexvwu/PvZ-search/.claude/worktrees/v5/artifacts/task43/sparse64/`，v5 checkpoint 声明的旧 branch、episode 缓存也需逐项定位。2026-10-25 HF 文件列表只有 v5 权重与其他传输件，无 v5 原语料；完整合并训练依赖恢复原分片，不能用热启动替代“与v5原语料合并”。
- task72b 的94行有真值和已评分汇总，没有原生状态/叶特征；需恢复原始档案或精确重建协议后才能评分 v6。缺失时不宣布完成94态验收。
