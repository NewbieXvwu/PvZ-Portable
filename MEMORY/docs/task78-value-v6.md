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

## 预登记（2026-10-25，训练前冻结）

执行状态（2026-10-26 当前回执）：69个历史错误任务已补齐，192任务完整导出987行/731胜；修正后192个示范root/视界历史检查全部通过。旧985行导出/分片与69个原始任务仍保留在ignored的`export_192_before_context_fix.json`、`data_192_before_context_fix/`、`labels_before_context_fix/`，映射与哈希见`context_label_preservation.json`。诊断配对15/15格、512/512对及补充示范24对均完成，只有白天非劣通过；夜间/泳池非劣及雾/屋顶改善均未过。屋顶k0重放通过但标签因10GiB内存门让位，原supervisor6及task72_resume7已停止；现由`resume8.py`等资源恢复后续216标签，屋顶成功后再接修正的24条原94态重建，旧错误静态网批次保留。各阶段最多2游戏、nice19；仅资源让位可自动重试，其他错误即停，原任务范围不变。

模型边界：当前A/B保持诊断候选SHA不变；原v5语料恢复后，仍须按原配方在新目录（例如`artifacts/task78/formal_v6/`）训练正式候选，并另立评估身份。已有诊断A/B保留其证据价值，不能以保留它们为由取消正式训练或把正式训练移出本任务。

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
- 2026-10-26 Codex独立屋顶输入核对：按现有完整导出的provenance实读并核验384份新区域root记录（128个不同标签未来）；独立屋顶24世界种子和144个root共同未来，与训练来源世界/未来及v5/v6/学生/静态网四检查点声明的trained_on均零交集，未启动游戏。证据与完整SHA补入[屋顶评分协议](task78-roof-evaluation.json)的input_identity_preflight；不证明未披露的历史输入或尚未完成的216终局标签。
