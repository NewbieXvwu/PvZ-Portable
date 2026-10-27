# task78：叶值 v6——把雾区与屋顶的决策级数据加进来，并修"雾区估值随视界翻转"

> 主控 2026-10-25 写。执行者：台式机 Codex（Plus 号，新开 tmux `codex-v6`，`CODEX_HOME=/home/newbiexvwu/.codex-plus`）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/v6`（分支 `v6`，从 origin/pvz-env 拉出）。**现行上限：同时最多2个游戏进程，qc优先**（主控10-25修订，替代原4）；每次 GPU 训练 ≤1 小时，开始前 `MemAvailable` ≥10 GiB；**不碰 qc、cq、流水线**（流水线目录只读）。

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
- 2026-10-25 接手：**训练语料硬阻塞确认**：v5 原语料的 branch 分片（`z13_384.pt` + 96 个 `z15d_shards`）、64 个 pool 决策分片、episode 缓存在本机与 HF 都不存在（HF 只有 `transfer/` 三个文件），因此预登记要求的"与 v5 原语料合并"无法执行。新增 `--corpus reconstructed` 诊断通道：旧区 320/512 份额由同源重建 episode 承担（`old_episode 384 / fog 64 / roof 64`，75/12.5/12.5 不变），checkpoint 与 `paired.json` 都写 `preregistered_corpus=false`/`diagnostic=true`，不能冒充预登记候选。`task78_ab.py` 按 mix 与回执一致性双向校验。26 项测试通过。
- 2026-10-25 接手：离线门禁脚本 `task78_gates.py`：用已审计的雾/屋顶标签分片（同一 root 同一候选、三视界同共同未来）直接算终局排序一致率与视界方向翻转率，v5/v6 同态对比、按世界聚类 10000 次 bootstrap，无需再开对局。**偏离记录**：task72b 的 94 态不可重建（探针脚本只在已删除的 `/tmp/task72b`，接手点 N=229/306/382 未归档映射），改用手上冻结 64 档案语料，结论里标注。28 项测试通过。
- 2026-10-25 接手：雾区 9–31 的 h4000/h6000 补齐，完整导出通过审计：144 分片 / 615 行 / 436 胜 / 终局丢弃 1754 行 / 72 个分片全终局（export.json complete=true）。诊断训练（v5 热启动、同源重建旧区 384 + 雾 64 + 屋顶 64）跑完：验证损失 v5 ≈0.67 → 候选 0.548，但预登记配方在第 0 轮后即早停（旧区 episode 验证损失 0.617→0.916，过拟合），所以诊断候选≈v5+256 步。修掉检查点 encoder=2 与 load_value_net 期望的字符串不一致（否则任何训练产物都读不出来）。离线门禁：雾区终局一致率 v5 0.642→v6 0.715（差 −0.073，95%CI [−0.139,−0.017]）、屋顶 0.667→0.821（−0.154，[−0.253,−0.061]）；**雾区视界翻转率 v6 反而更高**（5/15 → 7/15），该门不通过。

- 2026-10-25 17:29（CodeBuddy 续作）：修正离线一致率门的一个口径错误——原实现要求同一 root 在三个视界下候选集完全一致才算，导致屋顶（只有 h2500）整区被排除在一致率之外；一致率本就按视界内配对计算，不需要跨视界同候选集，该过滤只对视界翻转率有效。修正后 §5 的雾 33 态/233 对、屋顶 47 态/84 对即为现行数字。已提交并推送。
- 2026-10-25 17:29（CodeBuddy 续作）：续跑配对面板。已完成并保留：fog_l31_snow_m100、fog_l33_cactus_d839、fog_l33_cactus_d762（两臂各 32 局）。上一代理在 fog_l33_cactus_d686 的 v5 臂刚启动即中断（0 局），故从该格重跑。剩余 12 格按"逐格先 v5 后 v6"顺序在后台执行，2 worker（符合同时最多 2 个游戏进程），进度写 ignored `artifacts/task78/ab/progress2.log`。

- 2026-10-25 18:17：配对面板首格完成（day_l7_d7，64 对）：两臂各 64 胜，gain 0 / harm 0，净胜 0，McNemar p=1.0；保守单侧下界 −0.056 > −0.10，白天地形非劣门**通过**。报告机制在真实数据上跑通（其余地形仍为 incomplete）。

- 2026-10-25 19:47（主控要求停，交接回 Codex）：配对面板进度 **2.5/15 格**。已完整：day_l7_d7（64 对，两臂全胜，非劣门通过）、night_l14_n10 的 v5 臂（32/32）；night_l14_n10 的 v6 臂 31/32（收尾那局被停机杀掉）。此前三格雾区（fog_l31_snow_m100、fog_l33_cactus_d839、fog_l33_cactus_d762）两臂各 32 局完整保留。其余 11 格 0 局。
  **续跑可直接重发同一条命令，不需要删目录**：`search_play.py` 启动时读已有 `games.jsonl`，按 `(arm, seed 或 future_seed)` 去重，只补缺的局（`scripts/search_play.py:906`）。因此上面 `START` 之后被打断的格会从断点继续，不会产生重复 seed（`task78_ab.py report` 对重复 seed 会直接报错）。
  续跑命令（工作树内，逐格先 v5 后 v6，2 worker）：
  ```bash
  cd /home/newbiexvwu/PvZ-search/.claude/worktrees/v6
  export PVZ_NO_POPUP=1
  for cell in night_l14_n10 night_l17_n10 night_l20_n10 pool_l21_p9k pool_l22_p9k pool_l23_p9k pool_l24_p9k roof_l41_r7 roof_l42_r7 roof_l43_r7 fog_l33_cactus_d686; do
    for arm in v5 v6; do
      echo "START $arm $cell $(date +%s)" >> artifacts/task78/ab/progress2.log
      /home/newbiexvwu/.venvs/ml/bin/python scripts/task78_ab.py $arm --cells $cell >> artifacts/task78/ab/run_all.log 2>&1
      echo "DONE $arm $cell $? $(date +%s)" >> artifacts/task78/ab/progress2.log
    done
  done
  ```
  实测速率：白天 36 s/局、夜间约 45–65 s/局（2 worker），屋顶预计更慢。跑完后 `task78_ab.py report` 写 `artifacts/task78/ab/paired.json`，§4 与门禁汇总即可落数。


- 2026-10-25 Codex续作核对：当前没有本任务遗留进程后，启动独立屋顶评估前缀（`task78_roof_truth.py trajectories`，24个预留评估世界、最多2游戏），不将已有训练档案上的离线数字当作独立终局门。原v5语料与task72b原94态尚未满足；诊断候选及其未完成配对不能替代这两项交付。
- 2026-10-25 Codex纠正范围：主控实际指令是“同时最多2游戏进程、qc优先”，并未授权取消屋顶h4000/h6000。撤销此前把并发限制推导成缩减视界的做法，恢复全部64档案三视界（192个任务）。已完成144个，屋顶尚缺48个高视界任务，清单ignored `artifacts/task78/jobs_roof_restore.txt`；原144导出与分片已保留快照，新完整导出在补齐前标false。
- 2026-10-25 Codex纠正证据范围：`task78_gates.py`评分的是训练/停止语料自身，现明确 `preregistered_gate_eligible=false`。空分母不再返回虚假的[0,0]区间。615行诊断数据的既有点估计保留为训练语料诊断，不能替代task72b94态或独立屋顶48root×3未来。相关50项测试通过。

- 2026-10-25 20:55（CodeBuddy 接手）：核对交接状态——屋顶 h4000/h6000 补齐批次在跑（48 任务，20:35 起，已完成 40/41 两档案共 4 个任务，剩余 44 个；本机只有这 2 个游戏进程，符合并发上限）。标签目录实测：148/192 分片已存在，缺的 44 个全部是 roof_l41 档案 42–63 的 h4000/h6000。旧区/雾区/屋顶 h2500 均齐。
- 2026-10-25 20:55：**配对面板触顶，记录为方法学问题**：已完成的格两臂几乎全胜——day 64/64 vs 64/64、fog 95/96 vs 96/96、night L14 30/31 vs 30/31。这种分布下"非劣"必然通过、"改善"没有可分辨空间（gain/harm 只在 0–1 之间），所以配对面板对"是否采纳 v6"的分辨力很低，剩余 12 格的边际信息量同样有限。结论必须写明：改善门在此面板上应记为"证据不足"，不能用 p=1.0 冒充通过或失败。真正有分辨力的是离线叶值排序一致率（雾 0.642→0.715、屋顶 0.667→0.821）与视界稳定性（v6 反而更差）。
- 2026-10-25 20:55：本机 MemAvailable 8 GiB（训练门要求 ≥10 GiB），当前不满足训练条件；本轮不重训，A/B 两臂继续用 16:16 冻结的诊断候选 `artifacts/task78/v6/model.pt`（训练于 144 任务快照），新补的 48 个屋顶高视界分片只进完整导出与离线门禁，不改 A/B 臂身份。

- 2026-10-25 21:10（CodeBuddy 续作）：接手核对——屋顶高视界补齐批次仍活着（xargs -P 2，21/48 任务完成，本树只有这 2 个游戏进程，符合并发上限），预计还需约 75 分钟。配对面板 2.5/15 格（day 完成、night L14 缺 1 局、其余 11 格 0 局）。**执行顺序**：屋顶补齐 → 配对面板剩余格（2 worker）→ 屋顶独立终局面板（3 份 k 清单各自 replay + 72 个 collect 任务）→ 完整导出与四类门禁 → 结论。44 项 task78 测试当前全绿。

- 2026-10-25 22:06（CodeBuddy 续作）：屋顶补齐批次 32/48（仍 2 进程）；44 项测试重跑全绿；屋顶独立面板 `plan` 重新校验通过（24 世界/48 root/3 共同未来，plan SHA `4afca60e…`，输入与 v5 基线 SHA 一致）。启动串行驱动 `artifacts/task78/driver.sh`（ignored）：等补齐批次结束 → 完整导出 192 任务 → 配对面板剩余 11 格 + L14 补 1 局（逐格 v5 后 v6，2 worker）→ `report` → 屋顶面板 24 局轨迹 → `sources` → 3 份 k 清单 replay + 72 个标签任务。整链任一阶段失败只记 rc 不跳过，日志在 `artifacts/task78/driver.log`。
- 2026-10-25 22:51：**屋顶 h4000/h6000 补齐完成 48/48**（xargs 退出，无失败 rc）。完整导出 192 分片 / 985 行 / 729 胜 / 终局丢弃 1832 行 / 72 个空分片，`export.json complete=true`（屋顶行数 213→583）。旧 144 任务快照仍保留在 `export_144_snapshot.json`。
- 2026-10-25 22:56：完整导出重跑离线门禁（`gates.json`，export SHA `1dc1cafd…`）：雾 33 态/233 对，v5 0.6416 → v6 0.7146（差 −0.0732，95%CI [−0.1390,−0.0170]）；屋顶 47 态/**209 对**（原 84 对），v5 0.6986 → v6 0.8230（差 −0.1242，[−0.2267,−0.0368]）。视界翻转率：雾 5/15 → 7/15（差 −0.132，[−0.357,+0.125]）；屋顶现在三视界齐，可测：v5 14/34 → v6 13/34（差 +0.027，[−0.100,+0.156]）。**稳定性门在两区都不通过**（要求 v6 ≤ v5 一半且差上界 ≤0）。`evaluation_scope` 仍是训练/停止语料诊断，`preregistered_gate_eligible=false`。
- 2026-10-25 23:03：配对面板 night_l14_n10 补齐（两臂 32/32），当前 **4.5/15 格**（day、fog L31、d839、d762、night L14）。**执行顺序调整（不是协议改动）**：实测单局 137–163 s（此前 45–65 s，机器 load 14–22，qc/cq/redeploy 同时在跑），11 格全跑约 14 h，所以把两个新区域的格提到最前：`fog_l33_cactus_d686 → roof L41/L42/L43 → night L17/L20 → pool L21–L24`；night L17 已跑的 6 局保留，续跑按 seed 去重。预登记的每格 32 对、格清单、种子段都不变。
- 2026-10-26 01:15（主控要求停，交接回 Codex）：driver2 串行驱动（PID 702083，独立后台进程，不随交接中断）推进顺利：**roof L41/L42/L43 三格两臂全部完成**（各 32/32），当前 night_l17_n10 v5 臂在跑（12/32，含此前 6 局按 seed 去重续跑）。剩余：night L17 v6、night L20、pool L21–L24（每格 v5 后 v6），之后 driver2 自动接 `report` → 屋顶独立面板 trajectories/sources → 3 份 k 清单 replay+标签（`xargs -P 2`）→ 面板导出合并（全部写在 `driver2.sh` 内，日志 `artifacts/task78/driver2.log`）。**无需人工干预，等 driver2 跑完即可**；接手者只需核对 `driver2.log` 尾部 rc 全 0，然后据 `paired.json`、`roof_truth/export.json`、`gates.json` 写 §4 门禁汇总与 `MEMORY/docs/value-v6.md` 结论。配对面板已完 7/15 格（day、fog×4、night L14、roof×3）。
- 2026-10-26 00:50（CodeBuddy 接手）：核对 driver2 串行驱动（PID 702083，23:03 起）推进正常：fog d686、roof L41/L42 两臂已完成（各 32/32），roof L43 v5 臂在跑（13/32）。剩余：roof L43 v6、night L17（已有 6 局续跑）/L20、pool L21-L24，之后 driver2 自动接 report 与屋顶独立面板链。本树 2 游戏进程，符合并发上限。
- 2026-10-25 23:50：雾区四格配对全部完成（L31 human 32 + L33 接手 839/762/686 各 32 = **128 对**）：v5 127 胜 / v6 128 胜，gain 1、harm 0、净胜 **+1**，精确双侧 McNemar **p=1.0**。方向条件（每格净胜 ≥0）满足，但**改善门要求净胜>0 且 p<0.05，因此不通过**——这是天花板效应（两臂都近乎全胜），不是 v6 变差。非劣下界 −0.028 > −0.10。报告见 `artifacts/task78/ab/paired.json`（ignored）。

- 2026-10-26 08:37 Codex续作：**诊断候选配对15/15格、512/512对全部完成**，重新逐格核对1024条终局记录、种子集合、命令回执与冻结runtime，0缺局0重复。L24为22→26胜、gain9/harm5；泳池合计104→103/128、gain21/harm22、非劣下界−0.14406且L22/L23低于逐格底线，未通过。夜间非劣、雾/屋顶改善亦未过，只有白天非劣通过；`all_game_gates_pass=false`，仍是`diagnostic=true`，不满足预登记训练。完整数字见[value-v6 §4](value-v6.md)与[回执摘要](task78-stage-status.json)。原supervisor6已自动接L31既有示范起点v5臂（2游戏、nice19）；屋顶及修正94态重建仍依次排队，未改模型、未重跑已完成局。

- 2026-10-26 09:03 Codex续作：补充L31 N777/L32 N1231/L33香蒲N867各8对全部完成，三组均v5 8/8胜、v6 8/8胜、gain/harm=0；核验48条终局记录、原future、接手点、模型及回执，诊断结果不混入正式四格分母。独立屋顶k0的24世界重放通过，两个首批标签均因MemAvailable<10GiB让位（0/216完成），原supervisor6及task72_resume7已停止。新`resume8.py`（句柄`artifacts/task78/resume8_handle.json`）实测存活等资源：≥10GiB且无本任务活游戏才续；核对SHA与24世界root后复用已有重放，只重试明确资源让位，其他错误停止，屋顶成功后再接修正94态重建。未降低资源门、未重做已完成对局。

- 2026-10-26 09:15 Codex续作：resume8已实测从资源让位恢复并正常采集；独立屋顶首个完整任务k0/archive00/h2500按`audit_labels`原始链核验通过（2 root、13个非终局候选、13份真实终局记录，来源/重放SHA与共同未来日程一致），没有另开游戏。当前仅1/216任务齐，未凑齐三未来，不评分、不进训练；首任务审计摘要写入[当前回执](task78-stage-status.json)，其余采集继续。

- 2026-10-26 09:24 Codex续作：首独立屋顶世界seed78000320的k0三视界已齐（3/216任务），按原始链核验每视界2 root/13个非终局候选与终局真值；跨视界root状态SHA、学生历史、上下文、候选动作及共同未来日程一致。39份候选记录仍只来自1世界、1/3共同未来，未评分门禁。调度已在资源恢复后自动跳过这3项并续下一档案，完成项未重跑；审计摘要见[当前回执](task78-stage-status.json)。

- 2026-10-26 Codex续作（后经元数据范围纠正）：原94态future101–108与v5、诊断v6、原Mac静态branch1三份有trained_on清单的检查点零交集；dw4i_s1只有labels，无种子清单，其交集必须记null。原94行、模型SHA及范围说明见[当前回执](task78-stage-status.json)，旧口径回执已保留快照。不证明原状态映射、探针噪声日程、学生完整历史或未披露训练输入，仍未评分原94态；屋顶原队列继续。

- 2026-10-26 Codex续作：现有A/B脚本硬编码诊断权重/训练回执，不能靠换输出目录完成正式候选评估；新增独立`task78_formal_ab.py`，旧冻结脚本未改。正式[15格/512对预登记](task78-formal-ab-plan.json)仅换独立种子/未来（普通78000512–78000927、示范78100512–78100607）及候选/输出路径，五类配比、完整192导出、原split与超参、GPU内存/时限均须匹配；真实原语料loader的descriptor/sha回执可接受，诊断替代配方明确拒绝。新广域审计226文件/118566档案身份0碰撞0读取错误；学生缺种子元数据单列，不虚构零交集。58项测试通过，正式候选缺失时拒绝freeze，空正式报告全incomplete/null；旧诊断/屋顶/94态冻结输入核验未变。未开GPU或正式游戏，屋顶原队列继续；命令见[value-v6 §8](value-v6.md)。

- 2026-10-26 10:26 Codex续作：独立屋顶k0/L41的8世界×三视界24任务全量审计通过，16 root/48个root×视界、225份候选终局记录（222非终局叶、3终局叶，终局旁路与真值相符）。跨视界状态SHA、学生历史、上下文和候选动作一致，审计摘要在[当前回执](task78-stage-status.json)，原始审计`roof_truth/k0/l41_collection_audit.json`保留全部证据SHA。仅k0一个共同未来，全面板未齐，不评分、不进训练；L42采集继续，未重跑已完成项。

- 2026-10-26 11:03 Codex资源调度修正：任务书10GiB要求针对GPU训练；当前两CPU采集任务的Python/游戏进程VmHWM之和实测2.24GiB。保留resume8整批恢复门10GiB，后续CPU工作项准入改为8GiB，低于8仍以原资源让位码停止排新任务；GPU训练仍10GiB/一小时，最多2游戏、nice19、qc优先。只改ignored的`roof_collect5.sh`一处资源比较，保存旧包装及前后SHA到`cpu_resource_policy.json`，没有重启现有采集或改模型/引擎/标签/统计阈值；[当前回执](task78-stage-status.json)记录测量与策略。

- 2026-10-26 15:10（CodeBuddy 接手）：**独立屋顶 k0 导出失败的真因已定位并修复**。task78_export.py 在 archive_15 root_1020 报 root engine state differs from replay。核对 72 个 k0 任务：18 个不符（6 档案 × 3 视界），**全部是每档案的第二个 root**。先排除引擎不确定性——同一档案用未改引擎连续重放两次，335/1020 两个 root 的私有状态 SHA 逐位相同（62864767…），与冻结 replay 一致。因此差异来自采集器本身：同一进程里第一个 root 的搜索留下的隐藏状态（SNAPSHOT 不覆盖）污染了之后的前缀重放。采集器改为**每个 root 开一个新引擎、从头重放前缀**（root 内的标签流程、决策种子、身份字段一律不变）；44 项 task78 测试通过，提交 c624ca51 并推送。受影响 6 档案（15/17/19/21/22/23）原标签整目录移到 ignored 的 roof_truth/labels_before_rootfix/，正在用新采集器重采 18 个任务；同时用未受影响的 archive_08 复采比对，验证旧标签在新代码下是否逐字节可复现（决定是否要整批重采）。
- 2026-10-26 15:10：task78_roof_truth.py sources 已生成 k1/k2 的冻结源清单（与 k0 同 24 档案、同 48 root，只换 seed_base 78410000/78420000 与 common_future_index），training_eligible=false 不变。k1/k2 各自仍需按协议先跑 replay 验证再采集 72 个任务。

- 2026-10-26 15:40：**修复验证通过**。archive_15 重采后两个 root 的私有状态 SHA 与冻结 replay 完全一致（1020：62864767…）；未受影响的 archive_08 两个 root 全部候选/摘要/root 记录与旧标签逐项相同（仅忽略计时字段），archive_15 root_335 亦相同 —— 新代码不改变有效数据，可与已采任务混用。k0 现 55 个任务审计通过、17 个待采、0 个失败。已启动 resume8.py 串行链（2 游戏、nice19、每任务 8 GiB 准入）：k0 补 17 → k1 replay+72 → k2 replay+72 → 三份导出合并 → roof_score → gates → task72 24 条轨迹重建 + bind。

- 2026-10-26 16:20：**更正上一条的归因**。按 root 隔离重采 17/19/21/22/23 后，第二个 root 仍与 replay 不符，说明“搜索污染后续前缀”不是原因。进一步实测：同一档案（17，frame 650）连续两次**冷重放**得到的私有状态 SHA 互不相同（d84c6cba… / 5ea3f48d…），而 replay.json 记录的是第三个值（93237f35…）；逐字段 diff 显示两者**只差 grid_items[1].state**（2052151040 vs −1613670656）。`grep mGridItemState src/` 只有 LawnApp.cpp 的序列化一处，**整个 src/ 从不写这个成员**，所以它是未初始化堆内容；而 `_frame` 只保留 grid item 的 row/col/type，这个值到不了任何策略或标签。因此 root 身份不能再用未掩码的私有状态 SHA。
- 2026-10-26 16:20：新增 `scripts/task78_state.py`：把 `grid_items[*].state` 掩掉后做 canonical 摘要。实测同一档案两次冷重放在该 root 上的掩码摘要完全一致（73686749…）。replay 现在每个 root 同时记 `engine_canonical_sha256`，采集器在 sample_root 结束、引擎仍停在 root 的同一时刻记 `root_state_canonical_sha256`；导出改为原始 SHA 相同即通过、否则要求掩码摘要相同。已通过的两份数据（54 个 k0 任务）不受影响，失败的 18 个任务整目录移到 ignored 的 `labels_before_rootfix`/`labels_before_rootmask` 并重采。**撤销**按 root 隔离的改动（未被证实，且会让新标签与 192 任务训练语料的生成路径不同）。62 项测试通过；提交 9502b8ff、51ddeba0。
- 2026-10-26 17:40（CodeBuddy 接手交回 Codex）：核对 resume8 链死因并交接。run2（17:07）失败于 k0 导出：`k0/replay.json` 是掩码修复前冻结的旧格式（root 只有 `engine_sha256`，无 `engine_canonical_sha256`），新导出的掩码回退比对不到、报 root engine state differs from replay（archive_17/h2500/root_650）；run3（17:18）启动即被静默杀掉（机器上 deckvar task93 训练占 7.7 GB，earlyoom 偏好杀 python3），链路现已无进程。盘上实测：k0 已有 19/24 档案标签（缺 15/17/19/21/22/23 共 18 任务待采），且全部是 RAW-ONLY（root.json 无 `root_state_canonical_sha256`，采于掩码修复前）。给 Codex 的关键提示：重启链后 ready8 会判 k0 replay missing 并重跑 replay（`task78_replay.py:77` 已写 canonical 字段，这步必需，6 个清空档案的新标签也只能对新 replay 比对）；但重跑后新 replay 的原始 SHA 对旧标签约 1/4 概率不再逐位一致（未初始化堆内容，实测 archive_17 两次冷重放即不同），而这些旧标签没有掩码摘要可回退，导出会在此类任务上硬失败（set -e 使整链 rc=1，resume8 只自动重试 rc=124+75 的资源让位）。需要给 roof_resume5 的导出步骤加『失败→把该任务标签目录移到 ignored 备份→重采→重试导出』的循环，预计需重采 54 个旧任务中约 1/4。交接时 MemAvailable 12.3 GB、无本任务游戏进程、无未提交代码改动（HEAD 8ce9beda = origin/v6）。
- 2026-10-26 17:55（CodeBuddy 续作）：按交接要点修复并重启链路。新增 ignored `artifacts/task78/roof_export_retry.py`：导出报 root engine state differs from replay 时，从 traceback 定位 (archive, horizon)，把该标签目录移到 ignored `labels_before_rootreplay/`、用现行采集器重采（复用 roof_collect5.sh 的 8GiB 准入，资源让位时输出 LABEL rc=75 并以 rc=124 退出，接入 resume8 既有的整链自动重试路径），同一任务二次不符则放弃不循环；roof_resume5.sh 三个 k 的导出步骤改走该包装器（merge 步骤不变）。语法检查通过后重启 resume8.py（PID 939078，nice 19，后台，日志 resume8_run4.log）：实测进入预期路径——ready8 判 k0 replay missing、REPLAY_START 重跑 replay（task78_replay.py:77 写 canonical 字段）。接下来预期：k0 补采 15/17/19/21/22/23 共 18 任务 → 导出（旧 RAW-ONLY 标签约 1/4 会被包装器重采）→ k1/k2 replay+72 任务 → 合并 → roof_score → gates → task72 24 轨迹重建。

- 2026-10-26 17:46 Codex续作：旧句柄827391及交接中的run4 PID939078均已不存在，当前无本任务游戏且两把锁空闲；run4仅留启动行，没有退出原因，不能据此归因earlyoom。核验屋顶/修正94态冻结输入后，以独立session恢复原resume8链（PID941146、nice19），保存旧句柄并更新`resume8_handle.json`。刷新后的k0 replay含48个掩码摘要，现有54个完整任务逐一通过原始链审计（0失败），其余18项补采中。另修复评分器仍用原始SHA比较跨视界/未来root的问题：导出与评分共用`verified_root_digest`，先证明原始或掩码身份匹配，再用重放的稳定摘要比较，学生历史/上下文仍严格相等；不放宽真实状态不符。50项相关测试通过。ignored导出重试器收紧为仅旧RAW-ONLY标签可移备份重采，新标签掩码不符立即停止；3种守卫情形验证通过。原始标签、模型、任务/种子、引擎及生产代码未改，仍不评分未齐面板。

- 2026-10-26 17:49 Codex续作：在前一轮54任务原始链审计之后，用新评分器的`root_signature`核对当前完整三视界的18世界/36 root/108视界记录；跨视界身份、学生历史、上下文及候选动作全部一致，记录所有root/complete文件SHA。此批108条均为旧RAW-ONLY格式，通过原始SHA匹配后采用replay的稳定摘要；不是新掩码标签的实际复采验证，也只覆盖k0一个共同未来，不评分、不进训练。完整证据在ignored的`roof_truth/k0/cross_horizon_identity_run5.json`，摘要见[当前回执](task78-stage-status.json)。采集进程仍存活，余下原任务范围不变。

- 2026-10-26 18:18 Codex续作：新archive15/h2500的root1020原始/掩码摘要均为918164e0…，仍不同于冷重放62864767…，不能以GridItem问题概括全部偏离。停止本树队列并保留失败数据；记录动作重放、仅复用标签的sample_root、单候选完整老师续走三项诊断均未复现，原因仍未定。采集器改为每root新引擎恢复归档前缀，打标签前作亚秒级身份比较，失败即留状态停止；不扩大掩码、不改老师/种子。失败任务及两份部分任务整份保留，首root已完成候选逐项复用。L42/h2500现完成，第二root恢复为62864767…，57项相关测试通过；PID949150已接续原链，当前55/216。下游已有的全量检查不在上游重复，正式训练、独立全量门禁和原94态范围不变。

- 2026-10-26 Codex续采：L42的k0/h6000完成，屋顶现57/216，k0的L41/L42三视界均收齐。原控制器PID949150已从资源让位自动恢复，以两游戏/nice19继续L43；不追加上游全量审计，下游导出/评分按原流程处理，任务范围不变。

- 2026-10-26 Codex续作：按主控校验成本规则，正式候选冻结复用已登记的种子证据，取消原计划的第二次全机扫描；实际检查点训练种子交集判断保留，快照范围与学生缺元数据限制仍明示。25项相关测试通过，原种子/面板/模型/统计口径和冻结登记证据均未改；屋顶采集持续运行。

- 2026-10-26 Codex续作：k1/k2与k0的24份归档、48个root和计划完全相同，仅标签未来种子不同，改为显式继承k0的前缀重放证明，不再各自重跑。两份派生回执绑定k0证明SHA `5a5017f6…`及各自清单SHA，注明新重放世界数0、继承世界数24；元数据操作0.058秒，省去当前机器约168秒重复重放。逐root实际冷恢复/标签前身份比较、全部216标签及下游导出评分仍执行；采集继续，当前59/216。执行修订见[屋顶协议](task78-roof-evaluation.json)，未改冻结清单/种子/模型。

- 2026-10-26 20:16 Codex续作：独立屋顶k0的24世界×3视界共72任务全部完成，正常导出首遍成功，无重采：72分片、672非终局叶/268胜，另3终局叶保留在原始标签。导出SHA `996c0d50fc829c79d48de2ec0981049e78a02947af7106dc973f6939fbaaf158`、`training_eligible=false`；这是一个共同未来，尚不评分。PID949150已自动接入k1，以两游戏/nice19继续；k1前缀证明按修订复用，未重放。只读取正常导出的结果更新记录，未追加原始链审计或测试；正式训练与原94态要求继续保留。

- 2026-10-26 20:41 Codex续作：资源恢复后原链重复导出完整k0（7秒/4秒），现用`task78_export_cache.py`复用正常导出证明：导出SHA及2064项依赖文件/目录的设备/inode/大小/mtime/ctime一致才命中，变化即回原流程；实际建回执与查询0.059秒，4项定向测试通过，末尾原始标签评分仍执行。与此同时采集到81/216（k0=72、k1=9），原PID949150因k1/archive02/h6000/root605候选0的叶恢复状态不符退出；2.9秒单叶诊断未重算老师终局，公有帧相同但现有掩码不等。差异为恢复叶额外38个类型98提示文字动画及列表顺序，尚不据此放行；失败数据保留，下一步检查原生TRACE的历史恢复路径。当前采集停止，原正式训练、216任务与94态要求保留。

- 2026-10-26 Codex续作：叶历史改从实际原生TRACE恢复，按学生实际读取的历史特征及上次种植标志/时间核对float32位；保留原生叶快照的状态、帧和收入检查，未扩大root或叶掩码。未记录的packet/cell动作不伪造；完整数值输入无损压缩，正常下游只绑定证明与文件哈希，不重复恢复。失败候选的短诊断到达老师入口后即停（4.3秒、零老师终局复算），117决策/351个历史输入匹配；原任务完整目录保留，首root四份已完成候选经精确叶与输入上下文核对后复用。原失败k1/archive02/h6000已在20.9秒完成（两root各4候选/4胜），当前84/216；41项相关测试通过，PID991266以两游戏/nice19继续原链。原共享老师/模型/引擎/任务种子/全部216任务及94态要求不变；执行修订与源SHA见[屋顶协议](task78-roof-evaluation.json)。

- 2026-10-27 00:05 Codex续采：独立屋顶完成121/216（k0=72、k1=49、k2=0）。k1的L41/L42各24项全部完成，L43已完成首项，两条游戏进程沿原链继续；原生历史修复后未再遇到新的采集失败。此进度只按正常complete标记统计，k1尚未导出，三共同未来面板未齐，不评分、不追加原始标签审计。k0既有导出与全部冻结输入不变，后续仍是补齐216项、正常导出/评分，再接修正的24条原94态重建；正式训练及原94态要求仍保留。

- 2026-10-27 01:54 Codex续采：独立屋顶k1全部72任务完成，正常导出成功，SHA `7a252df211e4192087359ccc341bdadc1cd461c3d54c8de542bf83bb8ef5f27c`。72分片、672非终局叶/297胜，另3终局叶保留在原始标签，0跳过；分区L41为222叶/201胜、L42为225叶/59胜、L43为225叶/37胜。`complete=true`、`training_eligible=false`。当前144/216（k0=72、k1=72、k2=0），原队列已自动接入k2，以两游戏/nice19继续；只读取正常导出结果，未追加标签审计或复跑。三共同未来尚未齐全，暂不评分；其后合并/评分、修正的24条原94态重建以及正式原配方训练要求均保留。

- 2026-10-27 02:49 Codex续采：k2的L41全部24项已完成，屋顶累计168/216（k0=72、k1=72、k2=24）。两条进程沿原链进入L42，保持nice19及原资源准入；此阶段只按正常complete标记记录进度，没有追加原始标签检查或复跑。k2尚未完整导出，三未来面板未齐，不评分；后续全216任务、正常导出/评分、24条原94态重建及正式原配方训练要求继续保留。

- 2026-10-27 03:45（CodeBuddy 接手）：核对交接状态——分支 `v6` 与 origin 同步、工作区干净；屋顶 k2 采集仍在跑（本树 2 个 `task78_collect`，k2 已完成 12 档案 ×3 视界，累计 214/216 个 `rc=0` 任务，剩 archive_12 的 h4000/h6000 与 archive_13–23 共 33 项），resume8/roof_resume5 链存活，符合并发上限。**新增证据（原语料阻塞复核）**：此前只查过 HF HEAD 57 文件与两个历史 revision；本轮改用公开只读 API 遍历该仓库**全部 72 个分支**（`list_repo_refs`），逐分支探测 `artifacts/hard/branch/z13_384.pt`、`artifacts/hard/branch3/z15d_shards`、`artifacts/task43/sparse64/data`、`artifacts/task30/full_audit_manifest.json`、`artifacts/value_net/v5/model.pt` 五条路径，**命中 0**（分支清单全是 PPO 期 evidence-* 分支）。因此“原 v5 442,601 行语料在本机与 HF 均不可得”不再只依赖 HEAD 快照，正式候选（`--corpus original`）在本轮仍无法训练；未改冻结输入、未开游戏。

- 2026-10-27 05:35（CodeBuddy 接手）：核对链存活——`resume8.py`（PID 991266）→ `roof_resume5.sh` → `xargs -P 2`（PID 1056871）仍在采 k2，本树恰 2 个游戏进程（archive_20 的 h4000/h6000），符合并发上限；工作区干净、HEAD `cac88d33` = origin/v6。k2 实测 61/72（archive_00–19 各 3 项齐、archive_20 仅 h2500），剩 11 项 = archive_20 的 h4000/h6000 + archive_21/22/23 各 3 项，按当前约 2 项/5–6 分钟估计约半小时。k0/k1 各 72 项已正常导出（SHA 996c0d50… / 7a252df2…）。`task72_rebuild_mac_branch1` 计划 24 个轨迹、0 个已完成，`resume8` 会在屋顶链（含 merge/roof_score/gates）rc=0 后串行接上（rebuild 自身要求本树无其他 task78 游戏，故不能并行提速）。本轮未开新游戏、未改冻结输入。

- 2026-10-27 06:20：**独立屋顶终局面板完成 216/216 并评分**（k2 于 06:08 收齐 72 任务）。k2 导出 72 分片/671 行/275 胜（SHA `e2188d4c…`），三未来合并 216 分片/2015 行/840 胜（SHA `a2f86231…`，`complete=true`、`training_eligible=false`）。`task78_roof_score.py` 报告 SHA `6d8d2d42…`：主视界 2500 为 162 对/48 root，v5 0.4691→v6 0.5185（+0.0494，95%CI [−0.0201,+0.1287]，下界<0，门未通过）；4000 为 0.5864→0.5741（−0.0123）；6000 为 154 对 0.4610→0.4805（+0.0195）。`resume8` 随后自动接修正后的原 94 态重建（24 条 Mac 静态网轨迹，N229/f101 06:12 起跑，单游戏/nice19）。

- 2026-10-27 06:45：**94 态重建（Mac 静态网批次）9/24 后卡在资源门**。resume8.py 的 rebuild 循环没有让位重试，306/102 在 Recorder.before 的 10 GiB 守卫处抛 “yield to qc”，check=True 直接把整条链打成 CalledProcessError 退出（PID 991266 已不在）。已完成 n229/f101–f108、n306/f101 共 9 条（每条 30–60 s）。新增 ignored 包装器 artifacts/task78/rebuild_resume9.py：跳过已完成、只对“资源让位”重试、把被中断的 partial signatures.jsonl 整目录移到 partial_before_resume9/ 保留（不删证据）、其他错误即停，最后调 bind()。启动门设为 11.2 GiB（冻结脚本内的 10 GiB 守卫 + 一局约 1.1 GiB 自身占用），因为 cq 两个训练进程占 6.5 GiB、MemAvailable 只在 10.4–10.8 GiB 徘徊，剩 15 条（n306/f102–f108、n382/f101–f108）现排在 qc 之后自动等待，未开新游戏。
- 2026-10-27 06:45：**不能放宽这个 10 GiB 守卫**：task78_task72_rebuild.py 自己的 SHA 被写进 plan.json 的 code()，frozen() 逐字节比对，改它会让已完成的 9 条轨迹与计划一起失效。因此只能等内存，不能改阈值。

- 2026-10-27 06:54：**原 94 态重建 24/24 完成，映射 26/94**。`bind()` 逐字段比对 94 个参考态：26 个唯一命中、68 个无命中、0 个多命中（分未来 f101 6/12、f102 6/12、f103 4/10、f104 3/12、f105 2/12、f106 4/12、f107 0/12、f108 1/12），`mapping_complete=false`、`original_probe_protocol_verified=false`、`rescore_eligible=false`。24 条轨迹全部到终局，输入签名逐决策落盘；报告 `task72_rebuild_mac_branch1/bindings.json`，摘要已写进 [当前回执](task78-stage-status.json)。因此原 94 态仍不能给 v6 评分。本轮可执行的采集/重建项到此全部结束；剩余只有原 v5 语料与正式候选训练（均未取得）。

- 2026-10-27 07:30：**94 态身份映射修正为 94/94（此前 26/94 是比对口径错误，不是状态不同）**。冻结的 bind 用 int(x) 截断比较 balloon_x（plan 里也这么写的），但用保留的全精度 frame 重算可见原探针用的是四舍五入：int(round(x)) 对上全部 94 个记录的 balloon_x，截断只对 26 个、ceil 只对 22 个，且两者**只差 balloon_x 一个字段**（tick/wave/sun/cells/static_best/动作逐项相同，94 个状态在同一 (seed,decision,tick) 上全部存在）。新增 tracked 模块 `scripts/task78_task72_bind2.py`：按全精度 frame 重算气球位置后比对，`unique_matches=94`、`unmatched=0`、`multiple=0`、`mapping_complete=true`；未改轨迹/种子/tick/动作/plan/模型，frozen() 仍通过。报告 `task72_rebuild_mac_branch1/bindings_v2.json`。
- 2026-10-27 07:30：**映射修好不等于 94 态门禁可评分**。预登记要的是 v6 在每个状态上的叶值：这些叶值是"对每个候选动作按探针协议 rollout 到三个视界后的叶子"，重建只重放了老师实际选的那一步，没有其它候选的叶子；原探针的噪声日程仍未验证，因此不能凭空补分。另外真值侧已核过：250 个已标注候选对里 236 对真值相同、14 对差 0.125，**没有一对达到 0.25 阈值**，所以终局排序一致率对任何模型都只能是"证据不足"。`original_probe_protocol_verified=false`、`rescore_eligible=false` 保持。

- 2026-10-27 07:40：**原语料不可得升级为全量枚举证据**。此前只逐分支探测 5 条固定路径；本轮用公开只读 API 把 `realnewbiexvwu/pvz-agent-artifacts` 全部 72 个分支的完整文件树拉平（114,688 条唯一路径），对 `z13`、`z15`、`sparse64`、`full_audit_manifest`、`branch3`、`hard/branch`、`task43`、`task30`、`value_net/v5` 九组关键词检索，**命中 0**；该用户下也只有一个 model 仓库、没有 dataset 仓库、没有 tag。本机 `/home` 与 `/mnt` 按精确文件名搜 `z13_384.pt`/`z15d_shards`/`sparse64`/`full_audit_manifest.json` 同样 0 命中，本仓库 git 历史里这 3 个名字的引用数也是 0。因此"与 v5 原语料合并"和"正式候选训练/正式 A/B"仍无法开始，且已不是"某个路径没找对"的问题。

- 2026-10-27 07:45：**"直接给 94 态打分"这条捷径实测不成立**：value net 的前向需要叶子上下文向量（实测输入 119 维 vs 权重期望 154 维），该向量来自候选动作 rollout 后的叶子记录，不是决策点 frame 本身；重建只重放了老师实际选的那一步，没有 wait/static_best 两条候选的叶子。所以 94 态要评分必须按原探针协议重放 2 候选 × 3 视界 × 3 未来的叶子，而原噪声日程仍未验证。这条留作后继入口，不拿凑出来的分数冒充门禁。

- 2026-10-27 07:55：**需要主控提供的输入（已写进 [语料要求](task78-v5-corpus-requirements.json)）**。①v5 原 442,601 行语料：`artifacts/hard/branch/z13_384.pt`、`artifacts/hard/branch3/z15d_shards/` 96 分片、`artifacts/task43/sparse64/data` 64 分片、episode 缓存、`artifacts/task30/full_audit_manifest.json`（任一可用副本即可，含 descriptor）；②原 task72b 探针：脚本或其精确 source revision 与参数、每个状态 8 个真值对应的 future 种子/噪声日程、`values` 里 3 个采样 future、三个视界相对决策 tick 的偏移定义。拿到 ①即可训练正式候选并按预登记跑正式 A/B；拿到 ②才能给 94 态打分。两样都没有时，本任务的可执行范围已经到顶，继续硬做只会产出不可信数字。

- 2026-10-27 08:10：**语料恢复改成机械判定**。新增 tracked `scripts/task78_corpus_check.py`（5 项测试）：`--paths` 校验 97 个原 branch 文件、64 个决策分片、审计清单；默认模式按冻结 schema 校验 `original_v5/corpus.json` 的 baseline/分支数/逐文件 SHA/决策分片数/审计清单/四个 episode 文件，任一不符即 rc=1。当前实测：`--paths` 97/97 缺失、0/64 分片、审计清单缺失；descriptor 不存在，rc=1、`ready=false`。所以"语料是否恢复"不再靠肉眼判断——主控或别的机器把文件放回来，跑这一条命令就能定。

- 2026-10-27 08:30：**正式路径收敛成一条可恢复命令**。新增 tracked `scripts/task78_formal_run.py`（5 项测试）：`corpus → train --corpus original → freeze → v5 → v6 → report`，逐步独立进程、断点续跑（train 见 `formal_v6/model.pt` 即跳过，面板工具本身按 seed 去重/写一次）。实测现在跑它会**在第一步语料门就停**（rc=1，不占 GPU、不开游戏），并指到 `artifacts/task78/formal_run.log`。另实测新区域数据可被正式加载器读出：雾 fit278/val126、屋顶 fit525/val58、49 个世界——即"新数据这一半"已就绪，只等旧语料。

- 2026-10-27 08:55：**实测了一次 94 态复现，定位到第二个协议差异（候选枚举，不只是 future 种子）**。用现行 search_play 在 takeover 306 / future 101 / h2500 上带 `--leaf-probe` 跑一局（1 游戏、44.7 s、nice 19）：轨迹本身忠实（decision 13 在 tick 15438，终局 385 决策，与参考同段），但**候选数 8 对参考的 14**——现行工具按"每卡最优格 + wait + 学生选择"建候选，task72b 记录的是**全部合法格 + wait**；同一决策附近候选叶值离散 std 0.0184 对参考 0.0040，说明两者的叶子集合根本不同。因此即使拿到 future 种子，用现行工具也复现不出那 14 个候选的 v5 值；原探针的**候选枚举方式**同样是缺失输入。`--leaf-probe` 只输出 mean/std/top1-top2，不输出逐候选值，而 `search_play.py` 的 SHA 冻结在 `plan.json` 的 `code()` 里，不能改。

- 2026-10-27 09:10：**把"等语料"变成自动续跑**。新增 tracked `scripts/task78_watch_corpus.py`（3 项测试）：每 10 分钟跑一次语料验收，**一旦通过就自动执行 `task78_formal_run.py`**（训练→冻结→两臂→报告），否则只记一行 `CORPUS_MISSING` 继续等；`touch artifacts/task78/corpus_watch.stop` 即退出。已用 nice 19 在后台启动（PID 1118300，句柄 `fVOuoI`，日志 `artifacts/task78/corpus_watch.log`），当前每轮只做一次只读校验，不开游戏、不占 GPU。这样主控把文件放回后无需再叫我，链路自己会走完并留下完整日志。

- 2026-10-27 09:25：**为什么连"单状态对拍"也做不成**：task72b 的 `values` 是"每个候选按 `--samples 3` 三个不同随机 future 各跑一次后的均值"，而现行 `_rollout_policy` 一次只接一个 seed。即使把候选枚举改成"全部合法格 + wait"、把 future 固定成该行的 seed，单次 rollout 与三未来均值也不可能逐位对上（候选间差本身只有 0.001–0.01），所以无法用记录值验证复现是否成功。要做 94 态评分，缺的仍是原探针的**三个采样 future + 噪声日程 + 候选枚举**三件套；`artifacts/task78/task72b-probe-rows.jsonl` 只留了聚合后的值。

- 2026-10-27 10:20：**94 态探针的种子与候选枚举已恢复，"完全不可复现"据此更正为"部分复现"**。三件事都是机械核对：①决策种子是 `future_seed*10000+decision`（`search_play.py:671`），三个采样 future 是 `decision_seed*1000+k`（`:407`），所以"三个采样 future 缺失"不成立；②枚举可从冻结记录器反推——`task78_task72_rebuild.py` 的 `Recorder.before` 只用合法 packet-8（仙人掌）格重建 `cells`/`static_best`，逐态与参考候选数完全对上（格数+1 个 wait）；③同一档案连续两次独立运行，全部候选值逐位相同（max diff 0），现行引擎在该路径上是确定性的。**复现程度**：n229/f101 的 4 个态里 3 个的候选值多数在 4 位小数内与 task72b 记录一致（12/15、11/15、14/15），但 (101,105) 的 h2500 全部 16 个值系统性偏离（wait 0.0606 vs 0.0631），而该态的 tick/wave/sun/balloons/cells/static_best 与重建逐项相同，差异来源未定位。因此 94 态可以**按同一套自洽协议**给 v5/v6 打分（同状态、同 future、只换叶值），但不能宣称复现了原探针的绝对数值。已用 `scripts/task78_fog94_probe.py` 启动 48 局（24 档案 × v5/v6，2 进程、nice 19）测全部 94 态的 wait/static_best 三视界方向。

- 2026-10-27 11:50：**94 态已按"重放冻结动作、只换叶值"的协议全部测完（v5r/v6 各 24 局，48/48 rc=0）**。v6 臂原先用搜索自选动作，换叶值网后动作就变、轨迹发散（实测在 n306/f105 的 decision 47 直接对不上格表），因此改成从重建的帧档案重放记录动作、只在参考态打分；控制实验：同一批 94 态 3441 个值，重放路径与搜索路径**逐位相同（max diff 0）**。另修掉 `compare`/`score` 按 (seed,decision) 匹配导致的错配——(102,113) 在两个接手点下同号不同 tick，按 tick 消歧后 94/94 态格表全对。
  **门禁结果（重建协议）**：94 态全部测到，93 态有方向（1 态恰为 0）；雾区视界翻转 v5 **57/93 = 0.613**、v6 **48/93 = 0.516**，差 −0.0968（按 future 聚类 8 簇、10000 次 bootstrap，95%CI [−0.2143, 0.0000]）。预登记要求"v6 ≤ v5 一半"→ 0.516 > 0.306 **不通过**；"差的上界 ≤ 0"成立。**稳健性**：把小于 4 位小数舍入误差（5e-5）的间隙当作无方向后，同一批 63 态 v5 0.460 / v6 0.476（v6 略差）；1e-3 死区下 31 态 v5 0.355 / v6 0.387。也就是说 5e-9 阈值下看到的"v6 翻转更少"完全来自数值可忽略的态，**没有证据表明 v6 更稳定**。
  **协议保真度（必须同时读）**：与 task72b 记录值相比，中位绝对差 9.6e-5，≤5e-5 占 36.6%、≤1e-3 占 72.5%；**方向一致率** 2500/4000/6000 分别为 75.5%/64.9%/69.1%；同一批记录值算出的 v5 翻转率是 **0.340**，而重建协议下是 0.613。所以重建协议**没有**忠实复现原探针的方向统计，预登记的"雾区视界稳定性"门仍然**不能算通过**，也不能用 0.516 这个数去宣称改善。

- 2026-10-27 12:20（主控要求停，交回 Codex）：本轮把原 94 态从“无法评分”推进到“可测但不通过”——探针种子与候选枚举已从冻结证据恢复（`scripts/task78_fog94_probe.py`），48 局重放测量完成，视界翻转 v5 0.613→v6 0.516、差 −0.097（95%CI [−0.214,0.000]），“≤ v5 一半”不满足，排除 5e-5 噪声后 v6 反而略差（0.460→0.476），保真度不足故 `original_probe_protocol_verified` 仍为 false。工作区干净，HEAD `b9c1c374` = origin/v6；语料监控 `task78_watch_corpus.py`（PID 1123053）在跑。**本轮又独立复核了一次语料缺失**：在 `/home/newbiexvwu/PvZ-search` 深度 6 内按 `z13_384*`/`z15d_shards`/`sparse64`/`full_audit_manifest.json` 精确名搜索 0 命中，且该树下没有任何 >20 MB 的 `.pt`/`.npz`（v5 的 324,559 行 branch 分片不可能这么小），与既有结论一致。剩余交付只有原 v5 语料 → `task78_corpus_check.py` → `task78_train.py --corpus original` → `task78_formal_ab.py freeze/v5/v6/report`，监控器会在语料落盘后自动执行。
- 2026-10-27 14:10（CodeBuddy 接手核验）：HEAD `5fa5346f` = origin/v6、工作区干净；159 项 task78 测试全绿；语料监控器（PID 1123053）存活、每 10 分钟只读校验（最近 `CORPUS_MISSING` 14:00），无本任务游戏进程。结论文档 `value-v6.md` §0/§7/§8 已含全部最新结果（诊断配对 15 格、屋顶独立面板 216/216、94 态重建协议），无需改动。可执行范围已到顶，等待主控提供原 v5 语料（清单 `task78-v5-corpus-requirements.json`）。

- 2026-10-27 17:45（CodeBuddy）：**语料恢复再加一轮机械枚举 + 两处文档一致性修正**。恢复侧（只读、未开游戏）：`/home/newbiexvwu` 全深度按 >20 MB 的模型/压缩文件排序（最大 40 个全是帧包 npz 与一个 LLM zip）、按 `z13*/z15*/sparse64*/full_audit_manifest*/branch3*` 名称匹配（只命中 `frames_v2` 的 z13dag3/z15dag* 帧目录）、HF 本地缓存（7.4 MB）、git 对象库最大 blob（45 MB 的 `.update_000073_*` CLI 自更新 zip）、`/mnt/c|d|e|f` 深度 6 名称搜索，**全部 0 命中**；结论与既有"全 72 分支 + 114,688 条路径全量检索"一致。文档侧：§5.1 里仍写 `original_probe_protocol_verified=false` 的 3 采样段落已改为指向 §6.2（旧数字标注作废）；补上预登记要求但一直没单列的"原 26 个真值分化态"——原掩码 `original_26_state_mask_restored=false`，按"任一已标注对真值差 >0"反推只有 7 态而非 26，两个口径有效对均为 0，故无论用哪个掩码都是 0/0、结论同为证据不足。

- 2026-10-27 17:52（CodeBuddy）：**查清"用本机原始帧重建 v5 语料"这条路的边界**（只读，未开游戏、未占 GPU）。v5 检查点 `trained_on` 3,661 个种子里 **3,242 个（88.6%）的原始行还在** `artifacts/student/frames_v2/seed_*/<task>_fast_search/frames.npz`（`boards (N,92,6,10)`、`scalars (N,115)`、`values (N,11)`、`rows/plant/packet/cell/hints` 全在，就是分支语料的载荷）；缺的 419 个集中在 1610000–1610355（Mac branch3，对应 96 个 `z15d_shards`）与 1886xxx/2000xxx/2296xxx 三小批。但重建仍不能替代原语料，三条独立理由：①缺 11.4% 种子源行，必然不完整；②写出该 schema 的脚本不在 v5 代码提交 `cbaa051c` 里（仓库只有读者 `train_value_widen.Branches`，期望 `boards/scalars/won/seed/position/candidate/packet/kind` + `config.task_id day_l7_*`）；③`frames_v2` 是 observation-backfill v2 重导出，每局行数已变（z13dag4 按 `held()` 切分 fit 341,548/val 29,694，v5 回执 324,559/29,446），连那 88.6% 也无法证明行级同一。清单落 [原始行覆盖](task78-v5-raw-coverage.json)，结论写进 `value-v6.md` §8：**原 97 个分片文件本身仍是必需输入**。

- 2026-10-27 17:55（CodeBuddy）：**把"语料到位后手工写 descriptor"这一步自动化**。`task78_corpus_check.py` 新增 `--write-descriptor --source-root DIR`：按冻结清单的文件名在目录树里逐一定位（重名或缺失即拒绝），逐文件算 SHA256，决策分片优先取名为 `data`/`sparse64` 的目录，episode 缓存必须四个名字齐全，写完后立刻用原 `check_descriptor` 按冻结 schema 复核；空目录实测 rc=1、不写文件（fail-closed）。新增 3 项测试（成功生成并通过校验、缺文件/重名拒绝、episode 缺名拒绝），task78 全量 **162 项**测试通过。用法已写进 `value-v6.md` §8 命令链第 0a 步。未改任何校验阈值、语料要求或训练配方。

- 2026-10-27 17:58（CodeBuddy）：**结论文档与回执之间的对账改成可重跑的工具**。新增 tracked `scripts/task78_reconcile_docs.py`：从 `ab/paired.json`（白天/夜间/泳池下界、雾与屋顶 gain/net/p、512 对）、`roof_truth/scored.json`（主视界 162 对/48 root、0.4691→0.5185、主门 false）、`task72_probe94/score_original_protocol.json`（47/93、0.505）、`score_full192_9sample.json`（46/93、0.495、差 −0.011、CI [−0.119,+0.083]）、`gates.json`（0.7146/0.8230）重算 16 项关键数字，逐项断言它们仍以原样出现在 `value-v6.md`；回执不在时 rc=3 拒绝运行（附 1 项测试）。实测 **RECONCILED**、0 漂移，task78 全量 **163 项**测试通过。此前 15:15 的人工对账结论因此可复现。

- 2026-10-27 18:35（CodeBuddy）：**语料验收补上行级证明**。`task78_corpus_check.py --verify-rows` 现在会逐分片 `torch.load(mmap)` 读出 `seed`，按 v5 训练器的口径（先用审计清单排除种子、再 `held(seed)=sha256('value-widen:<seed>')[:8]%10==0` 切验证集）重算三类行数，并与冻结的 branch 324,559/29,446、decision 3,404/456、episode 75,136/9,600 逐项比对——文件名对得上但内容不对的语料会在这里被挡住（此前 descriptor 的 splits 只是自报）。episode 走 `rows.npz` 的 `seeds`。新增 2 项测试（合成分片复现切分、缺审计清单即拒绝），corpus_check 共 10 项、task78 共 165 项测试通过。未改任何阈值或语料要求。

## 预登记（2026-10-25，训练前冻结）

执行状态（2026-10-27 02:49 当前回执）：69个历史错误任务已补齐，192任务完整导出987行/731胜；修正后192个示范root/视界历史检查全部通过。旧985行导出/分片与69个原始任务仍保留在ignored的`export_192_before_context_fix.json`、`data_192_before_context_fix/`、`labels_before_context_fix/`，映射与哈希见`context_label_preservation.json`。诊断配对15/15格、512/512对及补充示范24对均完成，只有白天非劣通过；夜间/泳池非劣及雾/屋顶改善均未过。屋顶已完成168/216，k0/k1各72任务已正常导出，分别672非终局叶/268胜及672非终局叶/297胜；k2的L41已收齐24项，L42继续采集，三共同未来未齐，尚不评分。屋顶成功后再接修正的24条原94态重建，旧错误静态网批次保留。整批恢复要求10GiB，后续CPU任务准入8GiB；GPU训练仍10GiB/一小时。最多2游戏、nice19；仅资源让位可自动重试，其他错误即停，原任务范围不变。

模型边界：诊断A/B保持原候选SHA不变；原v5语料恢复后，仍须按原配方在`artifacts/task78/formal_v6/`训练正式候选，并用独立正式入口与[正式种子预登记](task78-formal-ab-plan.json)评估。已有诊断A/B保留其证据价值，不能以保留它们为由取消正式训练或把正式训练移出本任务。学生无种子清单的元数据限制及未覆盖远端历史均单列，不把缺失清单当空集合。

独立屋顶评分（2026-10-26，分支标签开始前明确）：原`driver2`仅排72个h2500任务，不能直接续用；完整要求是24世界×3共同未来×3视界=216任务，主视界2500、4000/6000分报。用`task78_roof_score.py`读取原始审计链，按action JSON匹配三未来，不能合并候选编号后当成三未来均值。主统计只比三叶均非终局的候选对，同时单列含终局旁路的全部候选诊断、平局、排除数和全部48root覆盖；按世界聚类bootstrap，零分母不能伪造零一致率。协议细节与当前缺216任务见`task78-roof-evaluation.json`及ignored的`roof_truth/scored.json`。此步骤不启动游戏，也未改变预留世界、root、种子或模型。

### 来源、抽样与训练

- qc 只读来源：`/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2`。按档案名区分 L31 human、L31/L32/L33 仙人掌/L33 香蒲示范接手、L41 r7；不把老师旧 targets 当终局真值。按 SHA256(档案绝对路径) 排序，五个雾组各取 8 份、屋顶取 24 份，共 64 份；每份取实际有 targets 的搜索序列中点分位两个 root（不足则全取），选择过程不读取胜负。固定源文件 SHA 后才生成候选标签。完整清单在本树 ignored `artifacts/task78/`，KB 级统计进 git。
- 标签复用 task43 `sample_root` 的真实搜索候选、原生学生叶、历史恢复核对及 `finish_teacher` 冻结老师终局续走；2500/4000/6000 三视界等量采集，完整候选与共同随机未来均记录。短推演已终局的叶保留审计，训练只用非终局叶，不把学生终局当老师标签。现行资源上限为同时最多2游戏，qc优先。
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
- task72b 首条重建（N306、future101）已到终局，精确绑定0态。该轨迹d14 tick15438，而原同future/d14 tick15498；即使取同tick也有sun110/135、合法仙人掌格等差异，不允许移索引或用近似tick强行匹配。映射仍缺其余23轨迹，原探针协议未验证，v6复评未启动。原始输入位置已向主控请求；不把当前生产重跑视为原94态。
- HF当前HEAD 57文件中未找到原v5语料；记录的完整历史revision `8aff8e7bcdbf3c20c2a5657411e8129f2c5ee0b7` 与 `cb63056e6d133a56408c3ab27955995439a336e6` 当前返回RevisionNotFound，不能据此宣称所有历史数据均不存在。仍需其他机器/可访问版本的原始输入。
- 2026-10-26 02:05（CodeBuddy 接手）：提交 task72b 94态重建通道 `task78_task72_rebuild.py`（plan/bind 不启动游戏；TAKEOVER FUTURE 需单游戏，排队等采集预算）并禁用 `task78_truth.py` 的旧重放路径（reference decision 是续走序号不能直接重放），17+49 项测试通过，已推送。context-fix 补采批次在跑（13/69，单进程串行，符合 2 游戏上限），配对面板剩余格与屋顶独立评估排在采集之后。
- 2026-10-26 02:30（CodeBuddy）：context 补采曾因瞬态竞态（上一任务游戏未退净即计 2 游戏）让位一次，已重启续跑（30→32/69）。写好 `driver3.sh`（完整导出→配对剩余 6 格→report→屋顶面板 3 k×3 视界 216 任务→roof_score→gates→task72 重建 24 轨迹，失败即停、全程断点可续）与 `supervisor3.sh`（补采完成后自动接 driver3，仅对资源让位自动重启），已在后台运行。决策记录：诊断候选保持 16:16 冻结的 `v6/model.pt`（144 任务快照训练）——A/B 两臂身份不能中途更换，重训等于作废已完成 9 格配对；语料差异写进结论，不冒充预登记候选。
- 2026-10-26 03:25（CodeBuddy）：context 补采 42/69（supervisor 自动让位重启正常，本树 1 游戏、nice 19）；driver3 尚未启动，等补采完成自动接。屋顶 h4000/h6000 长局约 15 分钟/任务，预计上午补完。
- 2026-10-26 03:25（CodeBuddy）：**context 补采 69/69 完成，完整导出通过**（export.json complete=true, training_eligible=true，192 任务全量），driver3 自动接手：配对面板 night_l17 v5 起跑。维持 02:30 的候选冻结决策（重训会使已完成 9 格配对作废，192 任务重训列为后续工作）。
- 2026-10-26 04:30（CodeBuddy）：driver3 配对面板推进：night_l17 双臂完成（11/15 格），night_l20 v5 起跑；每臂约 20–36 分钟。
- 2026-10-26 05:25（CodeBuddy）：配对面板 night 区全部完成（13/15 格），pool L21–L24 开跑（8 臂约 4h），之后 report → 屋顶独立面板 → score → gates → task72 重建。
- 2026-10-26 06:25（主控要求停，交接回 Codex）：**配对面板 14/15 格完成**（day、night×3、fog×4、roof×3、pool L21；pool L22 v6 在跑），driver3 串行链（PID 764280，supervisor3 PID 748134）不随交接中断，剩余自动推进：pool L22 v6 → L23/L24 → report → 屋顶独立面板（trajectories 24 局 + sources + 3 k×3 视界 216 标签任务）→ panel 导出合并 → roof_score → gates → task72b 重建 24 轨迹 + bind。失败即停，日志 `artifacts/task78/driver3.log`。接手者只需：核对 driver3.log 尾部 rc 全 0；若中途让位/失败，重发 `bash artifacts/task78/supervisor3.sh`（全程断点可续）。据 `paired.json`、`roof_truth/export.json`、`gates.json`、`task72_rebuild/bindings.json` 写 §4 门禁汇总与 `MEMORY/docs/value-v6.md` 结论。context 补采已 69/69，完整导出 trainable；候选维持冻结决策（02:30 条目）。
- 2026-10-26 Codex实测续作：纠正交接格数，当前13/15完整格、448对（含已补齐pool L22）；夜区93→91/96、下界−0.11088且L20净胜−3，非劣门未过，剩L23/L24继续。69补采后的完整导出987行已核验，示范历史192/192通过。原94态首条轨迹已完成但0精确匹配，暂不评分v6。另把原预登记要求的L31/L32/L33香蒲既有示范起点各8对明确为补充诊断（N777/1231/867、future78101000–78101023，源/模型/正式面板零交集已审计），计划在开这些局前冻结到`task78-demo-start-plan.json`；这些结果不进入正式四格分母。后续屋顶216任务的worker错误会返回给调度器，不能让末尾echo吞掉失败。
- 2026-10-26 Codex原94态真值充分性核验：原记录16态有2个标签、78态有3个，不能只核对wait/static-best。`task78_truth_audit.py`已核对94个完整身份、2128条候选×未来数值记录和全部250个已标注候选对：236对真值相同、14对差0.125、按原0.25阈值有效对为0。结论层`task78-task72-truth-audit.json`只认证给定摘要的算术与候选覆盖，终局排序信号记insufficient_evidence，不造一致率或CI，不声称重放、模型评分或视界门完成。原26态掩码仍待确认，不能用这次终局信号集合替代；相关97测试通过。
- 2026-10-26 07:30（CodeBuddy 接手）：核对链路存活——paired_resume4.sh（PID 789111）已完 pool L22 双臂、L23 v5，L23 v6 在跑；supervisor6.py（PID 790915）待命将接示范起点诊断与屋顶 216 任务链（含 score + gates_corrected）。剩余：L23 v6 → L24 双臂 → report → 示范诊断 → 屋顶 replay/216 采集/导出/score/gates。接手者只监控不干预，跑完后据 paired.json、demo_start_diag/report.json、roof_truth/export.json、roof_score 输出与 gates_corrected.json 写 §4 门禁汇总与 MEMORY/docs/value-v6.md。
- 2026-10-26 08:05（CodeBuddy 交回）：配对面板 pool L23 双臂完成、L24 v5 完成（08:03）、L24 v6 在跑（07:32 起已核验 resume4/supervisor6 进程存活）；88 项 task78 测试全绿。此后 task78 交 Business 号 Codex，本代理停止监控，后台链路（paired_resume4 → supervisor6 → 示范诊断 → 屋顶 216 任务链）不打断、继续自动推进。
- 2026-10-26 Codex续作：发现原task72/72b指定Mac静态branch1（SHA `91e562a0…`），旧重建却冻结台式机branch1（`24858410…`）。本机`/home/newbiexvwu/pvz-logs/mac_branch1/model.pt`实测完整SHA与原记录一致；入口改为强制原静态网，新24轨迹计划另存`artifacts/task78/task72_rebuild_mac_branch1/`，旧`task72_rebuild/`计划、N306/f101终局轨迹与0匹配回执全部保留。19项定向测试通过。修订证据见[重建输入修订](task78-task72-reconstruction-amendment.json)；仅纠正已知输入差异，不声称因此恢复原探针协议或94态。
- 2026-10-26 Codex续作回执：完整14/15格、480对，L24另12/32，当前492/512对；新完成L23为v5 30/32、v6 26/32，gain1/harm5，逐格非劣底线未过。L24 v6及supervisor6均实测存活，未重启已有批次。修正重建的后继进程`task72_resume7.py`（PID810388，句柄`artifacts/task78/task72_resume7_handle.json`）已启动等待，严格核对前驱PID/启动时间与屋顶成功标记后才开游戏，失败即停。
- 2026-10-26 Codex独立屋顶输入核对（后经元数据范围纠正）：按完整导出的provenance实读并核验384份新区域root记录（128个不同标签未来）；屋顶24世界种子和144个root未来与训练来源世界/未来及v5/v6/静态网三份已有trained_on清单零交集。学生只有labels、无种子清单，其交集记null，完整历史未证；此前四检查点零交集的表述已纠正，原回执保留快照。完整SHA及限制见[屋顶评分协议](task78-roof-evaluation.json)的input_identity_preflight；未证明未披露历史或尚未齐的216终局标签。
- 2026-10-27 14:35（CodeBuddy）：**执行任务书 02:30 决策记录列明的后续工作——完整修正后 192 任务导出重训诊断候选**（诊断 A/B 已 512/512 完成，"重训作废配对"约束解除；MemAvailable 13.3 GiB ≥10 GiB 门、GPU 26 秒、早停于第 4 轮，best epoch 0 验证损失 0.5262 对 144 快照候选的 0.548）。新候选 `artifacts/task78/v6_full192/model.pt`（SHA `83551195…`，mix old_episode 384/fog 64/roof 64，`preregistered_corpus=false`），不触碰已冻结 A/B 的 `v6/`。离线重评：语料一致率雾 0.6416→**0.7232**、屋顶 0.6986→**0.8134**（旧诊断候选 0.7146/0.8230）；翻转率雾 5/15→**7/15**（与旧候选相同，稳定性门仍不过）、屋顶 14/34→**12/34**。屋顶独立面板（`roof_truth/scored_full192.json`，candidate SHA 已记录）：主视界 2500 **v5 0.4691 = v6 0.4691（162 对完全打平）**，4000 0.5864→0.5741，6000 0.4610→0.5000，`primary_gate_passed=false`。94 态探针（24 局重放、2 进程 nice19）已在跑，出数后补稳定性门结论。以上全部为诊断口径，不改变"正式候选仍被语料阻塞"的结论。
- 2026-10-27 14:43（CodeBuddy）：**full-192 后续工作收尾**。94 态探针 24/24 局 rc=0（重放冻结动作、只换叶值，nice19 两进程约 7 分钟），稳定性门：v5r 0.613 → v6f192 **0.527**（49/93，差 −0.086，95%CI [−0.208,+0.022]，8 簇 bootstrap）——"≤ v5 一半"与"差上界 ≤ 0"均不满足，与 144 快照候选（0.516）一致。报告 `task72_probe94/score_full192.json`。至此任务书 02:30 决策记录所列"192 任务重训"后续工作完整执行并评分完毕：新候选在全部三类诊断（语料门禁、屋顶独立面板、94 态稳定性）上与旧诊断候选同向，不改变"四类门禁无一通过、不推荐替换生产叶值"的结论；正式候选仍被原 v5 语料阻塞，监控器继续在跑。
- 2026-10-27 15:05（CodeBuddy）：full-192 候选的死区稳健性补齐（与旧候选同口径）：5e-5 死区下 63 个共同有方向态 v5 0.460 / v6f192 0.533、1e-3 下 31 态 0.355 / 0.522——v6 的表观优势同样完全来自 5e-9 阈值下数值可忽略的态，与 144 快照候选结论一致。报告 `task72_probe94/score_full192_deadzone.json`（ignored）。至此 full-192 后续工作全部收尾，无未完成的诊断项；剩余交付仍只有原 v5 语料与正式候选（监控器正常，15:00 轮 CORPUS_MISSING）。
- 2026-10-27 15:15（CodeBuddy）：**结论文档全量对账通过**。`value-v6.md` 的每类门禁数字逐一与底层回执机械核对一致：paired.json（15 格/512 对；白天过、夜间 −0.1109 与逐格底线未过、泳池 −0.1441 未过、雾/屋顶 net +1 且 p=1.0 未过、屋顶 direction_met=false）；roof_truth/scored.json（诊断候选 4df43aea，三视界 0.4691→0.5185 / 0.5864→0.5741 / 0.4610→0.4805）；task72_probe94/score.json（v5r 57/93=0.613、v6 48/93=0.516，CI [−0.214,0.000]，gate passed=false）；bindings_v2（unique_matches=94、unmatched=0、mapping_complete=true）；gates.json（雾 0.6416→0.7146 翻转 5/15→7/15、屋顶 0.6986→0.8230 翻转 14/34→13/34）。两份文档本地断链扫描 0。对账后未发现任何需要修正的数字；剩余交付不变（原 v5 语料，监控器正常）。
- 2026-10-27 15:30（CodeBuddy）：**(101,105) 类保真度异常的结构刻画完成**（任务书 10:20 条目留下的后继入口）。564 个 (态,视界,候选) 偏差：随接手深度增长 2–3 个量级（229≈舍入级→382 中位 1e-2 级）、382 内按决策序位无趋势（前缀长度效应）、非共同模式（仅 56/282 同向）——指向前缀冷重放私有 RNG 漂移，与 10-26 屋顶采集 archive15 root1020 的未解冷重放摘要分歧同因。229 上的对比基本舍入级可信，382 只能作方向参考；恢复原协议仍需原探针输入。报告 `task72_probe94/fidelity_profile.json`（ignored），刻画写进 value-v6.md §6.1。
- 2026-10-27 16:05（CodeBuddy）：**原 task72b 探针协议已精确复现——根因是采样数 3 vs 9**。从原提交 `202a1d51` 的同时代文档 `static-candidates-antiair.md` 找到被遗忘的关键事实：原探针对每个候选打 **9 个 rollout 未来**（"直接对「等待 + 全部合法仙人掌格」各打 9 个 rollout 未来 × horizon 的叶值"），而任务书 09:25 条目的"--samples 3"是错误回忆，本任务重建探针也只用了 3。给 `task78_fog94_probe.py` 加 `--samples`（默认 3 不变，既有标签不受影响，9 项测试通过）后用 9 采样重探两个组合：tk229/f101 全部 183 值、tk382/f108 全部 129 值与记录值之差 **max 4.99e-05**（即记录值 4 位小数的舍入半径）——协议逐位复现（判定实验 2 局）。此前 15:30 的"前缀 RNG 漂移"刻画作废：偏差随接手深度增长只是值量级随深度增大的假象，真因是均值 3 vs 9 的抽样差。**后果**：预登记的雾区 94 态稳定性门第一次可在原协议下评分；正在跑 v5/v6 两臂各 24 局（9 采样）出正式数字。94 态终局真值侧结论（无一对达 0.25 阈值→排序一致率证据不足）不受影响——那是真值属性，与探针协议无关。
- 2026-10-27 16:45（CodeBuddy）：**原协议 48 局全部完成（rc=0），雾区稳定性门首次按原协议出数**：v5 0.505（47/93）= v6 0.505（47/93），差 0.000（95%CI [−0.115,+0.104]，8 簇 bootstrap），翻转态重叠 34/13/13/33 对称；预登记门两条均不满足，但性质从"不可判/v6 更差"修正为"原协议下 v6 精确中性"。v5r9 与记录值 62/62 翻转一致、24 组合 max 偏差=记录舍入半径，`original_probe_protocol_verified=true`（此前所有"未复现"表述作废）。报告 `task72_probe94/score_original_protocol.json`，§0/§1/§6.2/§7 已更新。四类门禁总判定不变：无一通过，不推荐替换生产叶值；正式候选仍等原 v5 语料（监控器正常）。94 态的 v6f192（full-192 候选）9 采样评分未跑，留作后继稳健性检查（24 局）。

- 2026-10-27 17:25（CodeBuddy）：**收掉上一条留下的后继项——full-192 候选按原协议（9 采样）评分完成**。24 局 rc=0、两进程 nice19、约 13 分钟（`artifacts/task78/probe94_full192_9sample.sh`，标签目录 `task72_probe94/v6f192r9/`）。稳定性门：v5 0.505（47/93）→ v6f192 **0.495（46/93）**，差 −0.011（95%CI [−0.119,+0.083]，8 簇 bootstrap），"≤ v5 一半"与"差上界 ≤ 0"均不满足，**门不通过**；与 144 快照候选的 0.505（差 0.000）同量级，即两个独立训练的诊断候选在原协议下都与 v5 打平。死区稳健性同向：5e-5 下 58 个共同有方向态 v5 27 翻 / v6 28 翻（差 −0.017），1e-3 下 31 个共同态 12 / 13 翻（差 −0.032）——5e-9 下的"v6 略好"仍来自数值可忽略的态。报告 `task72_probe94/score_full192_9sample.json` 与 `..._deadzone.json`（ignored）；`value-v6.md` §0/§6.1/§7 已同步，[当前回执](task78-stage-status.json) 的 `task72_probe94` 改为原协议口径（旧的 3 采样块保留为 superseded），`unresolved_requirements` 现只剩原 v5 语料。3 采样口径下的 0.613→0.527 作废。四类门禁总判定不变（无一通过），正式候选仍等原 v5 语料（`task78_watch_corpus.py` PID 1123053 存活，17:40 轮 CORPUS_MISSING）。
