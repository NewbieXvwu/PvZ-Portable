# task42：老师候选改用“学生前 3”（B3），在现行设置上复验，过了就成为唯一实现

> 主控 2026-10-23 10:10（台式机时间）写。执行者：台式机 Qoder（DeepSeek-Flash），工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/b3`（分支 `b3`，从 pvz-env 拉出）。
> 同时最多 **3** 个你自己的游戏进程（qc 采集 16 进程在跑）。不碰 tmux `qc` 与别的代理会话，不重启 WSL。

## 为什么（做了没用上的典型）

- task13/task14（`MEMORY/docs/static-value-ablation.md`）在 10-20 已经证明：老师的候选从“静态价值网每卡 1 个 + 学生 2 个”（A）换成“**学生全局前 3 个种植**，不用静态网”（B3），
  与重老师的一致率只差 **−0.93pp**（95% 区间 [−1.67, −0.18]，预登记门槛 −3pp，**非劣通过**），同时**少 23.1% 推演、少 29.3% 搜索 CPU**。
- 当时“正式采集暂不切换”，之后没人回头；10-23 task33 清理时把 `--candidate-generator` 当成“从未设过的实验开关”删掉了（tag `retire-flags-20261023` 可取回）。
- 现在更该换：静态价值网 `value_branch1` 只在白天第 7 关的分叉数据上训过，课程已经覆盖白天 1–10、夜间、泳池、传送带，静态网在这些区都是分布外；B3 不依赖它。
- 用户 10-23 定的目标：纯 CPU 采集整局吞吐 10-27 前到旧构建的 2 倍，否则重新考虑 GPU 方案。B3 是 task35 之外最大的一块。

## 当时的设置与现在的差别（复验要覆盖）
- 当时：白天 L7 z17、学生 v30（槽位头）、价值 v3、32+256 局。
- 现在：学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`（按植物身份出手，PVZRN006）、价值 `~/PvZ-search/artifacts/value_net/v3/model.pt`、静态 `~/PvZ-search/artifacts/hard/value_branch1/model.pt`；
  课程任务族见 `scripts/make_multiplier_tasks.py` 的 `FAMILIES`。老师参数以 `scripts/curriculum_queue.py` 的 `command_for` 为准（`--student-candidates 2 --verify-samples 9` 等，自己读，不要猜）。

## 步骤
1. **取回实现**：从 tag `retire-flags-20261023` 的父提交找回 B3 的候选生成（`git show retire-flags-20261023^:scripts/search_play.py`，以及 task13/14 的比较与统计脚本 `static_value_noninferiority*.py` 等）。在现行 `search_play.py` 上重新实现 B3 的候选生成：**学生全局前 k 个种植（按身份头的概率）+ 等待 + 学生原动作**，k=3。先作为**比较臂**存在（只在比较脚本里用），不进正式路径。
2. **预登记**（先写进本文“结果”并提交，再跑）：
   - 种子：各族评估段 30000 起（与 task14 同口径，局内按 `Random(20261023+seed)` 抽 8 个搜索根）；
   - 族：白天 L7 `day_l7_train_d7`、夜间 L14 `night_l14_train_n10`、泳池 L24 `pool_l24_train_p9k`、原版传送带 L10 `day_l10_orig_train`，每族 64 局 × 8 根 = 512 点，共 2048 点；
   - 判据（与 task14 相同）：B3−A 与重老师一致率差的 95% 下界 > **−3pp**（整局为聚类单位，t 区间 + 10000 次整局 bootstrap 都要过），**四族分别报**，合并判定；同时报每点推演数与 CPU 秒（worker + 引擎）。
   - 重老师：照 task14 的定义（同一套未来、更多样本）；读 `static-value-ablation.md` 的“设施与预登记”，照抄口径。
3. **跑**：最多 3 个游戏进程。先每族 4 局冒烟，确认管线对，再跑满。
4. **结论**：
   - 通过：B3 成为老师候选的**唯一实现**——删掉静态价值网候选路径（`--static-value` 参数、`_static_frames`、`STATIC` 相关代码与测试），`curriculum_queue.py` 的 `DEFAULT_STATIC_VALUE` 与 `command_for` 里的 `--static-value` 一并删；`--student-candidates` 默认改成 3（按预登记）。旧路径打 tag `retire-static-candidates-20261023` 后删。
   - 不通过：报告里写清哪一族不过、差多少，代码只保留比较脚本，正式路径不动。
5. 报告：`MEMORY/docs/static-value-ablation.md` 末尾新节“task42：现行设置复验（2026-10-23）”，DESIGN 不用改（候选生成不是任务族）。

## 与别的任务的关系
- task37（codex2，分支 `vcpp`）正在把“静态候选打分”搬进引擎（`STATIC_SCORES`）。B3 若通过，这部分就不需要了——主控会告诉 codex2 先做 `LEAF_VALUES`。你不要改 vcpp 分支。
- task40（qoder-cut）在量推演“胜负已定就收手”，与本任务正交。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写规则。
- 不加开关：比较臂只在比较脚本里；通过后是唯一实现。
- 每完成一小步就提交、`git push origin b3`，在“结果”记一行；疑问写“问题”，按最佳判断继续。

## 结果

### 预登记（2026-10-23 09:58 台式机 CST；代码冻结 `5f5a1457`，写于任何对局之前）

在现行设置上复验 task14 的 B3 非劣性。样本、口径与判据在此一次定死，冒烟与正式跑都不改。

**实现**：`scripts/search_play.py` 的 `_search` 由私有 `_W["candidate_mode"]` 选候选生成，正式路径仍为 `static`（无 CLI 开关，`curriculum_queue.py` 未改）。
- **A（现行默认）**：静态价值网每卡前 1 + 等待 + 学生原动作 + 额外 2 个不同学生动作（`--student-candidates 2`，与 `command_for` 一致）。
- **B3（比较臂）**：学生全局前 3 个种植（按身份头概率 `ranked_plants`）+ 等待 + 学生原动作；不跑静态打分、无每卡下限。
- 两臂其余参数相同：samples 3、horizon 2500、halving、verify-samples 9、min-gap 300、MARGIN 0.02、rollout sim-skip 61。

**模型与二进制（SHA256，冻结）**

| 角色 | 路径 | SHA256 |
|---|---|---|
| 学生 | `~/PvZ-search/artifacts/student/dw4i_s1/model.pt` | `122703803804ab7e1e5aeda4ec382420d45acc349d74fe85344b859a8d760375` |
| 叶值 | `~/PvZ-search/artifacts/value_net/v3/model.pt` | `5171db54a31ab1795ce3cc136a25684a5bf6237cfd0e49a3b3380956dc26374a` |
| 静态价值（仅 A 与重老师池） | `~/PvZ-search/artifacts/hard/value_branch1/model.pt` | `2485841036cf3a8738174affb0dadc0201a925ca3ea0ebe6e3eb02e74621e314` |
| 模拟器 | `~/PvZ-search/.claude/worktrees/b3/artifacts/task42/pvz-portable`（qc 在役构建的冻结副本） | `7508e840ec19d0ef6f91d55f4052cf17846a82df7eb24e235fab2aa9b1aefe65` |

> 2026-10-23 10:00 冒烟前修正（样本与判据不变）：`~/PvZ-search/build/pvz-portable`（SHA `ed98…6034`）缺 `rollout_version >= 3`，当前 v2 学生 `dw4i_s1` 无法载入 rollout 网；改用 qc 队列在役的 `cq/build/pvz-portable` 同一份字节的冻结副本（复制到本工作树 `artifacts/task42/`，SHA 见上表）。单局管线已用该副本通过。

**样本**：四族 × 64 局 × 8 根 = 2048 点。种子 30000–30063（各族评估段 30000 起，与 task14 同口径）。

| 族 | 清单 | 任务 |
|---|---|---|
| 白天 L7 | `~/PvZ-search/artifacts/stage4/day_panel.json` | `day_l7_eval_d7` |
| 夜间 L14 | `~/PvZ-search/artifacts/stage4/night_panel.json` | `night_l14_eval_n10` |
| 泳池 L24 | `~/PvZ-search/artifacts/stage4/pool_panel.json` | `pool_l24_eval_p9k` |
| 原版传送带 L10 | `~/PvZ-search/.claude/worktrees/conveyor/artifacts/stage4/day_panel.json` | `day_l10_orig_eval` |

根抽样：`Random(20261023+seed).sample(eligible, 8)`，按决策序排序。eligible 与 task14 相同：有合法种植动作，且学生本步选择种植或距上一个 eligible 根 ≥300 tick。若某局 eligible <8，该局使用全部 eligible（不重采样、不补局），点数与原因如实记录。轨迹只执行学生、不执行任何老师标注，严格重放后才测。A/B3 的测量先后按 `(seed+decision)%2` 交替，共享 `D = seed×10000 + decision − 1` 的未来。

**重老师**（照 task14 定义）：仅当 A 与 B3 所选动作不同时运行；动作池 = 两臂全部候选 + 单模型静态每卡前 2 + 每卡一个随机剩余合法格，按动作键统一排序。每个池动作跑 64 个共同未来（`truthD = D + 33333333`，`(truthD×1000+k) mod 2^32`），v3 叶值；同一锚定验证（MARGIN 0.02，额外 18 个未来 × 2 个动作）。两臂同动作时标签精确抵消（差 0），不运行重老师。

**统计与判据**：
- 每个搜索点 `delta = 1[B3=重老师] − 1[A=重老师]`；同动作点 delta=0，按整局 8 点均值计入。
- 整局为聚类单位：对 64 个局均值做 t 区间（df=63）；另做 10000 次整局 bootstrap（seed 20261023，重采样的 delta 和 / 点数和），两种 95% 区间**下界都必须 > −3pp**。
- **四族分别判定，合并判定 = 四族全部通过**；任一族不过即 B3 不通过。合并 256 局的池化结果另报作参考。
- 只给配对差值，不给绝对一致率（同动作根不计标签），不外推胜率。
- 成本：每点实际续走数（halving + 验证）与 CPU 秒（Python worker user+system + 模拟器进程 user+system），按族报；重老师成本单列。

**资源与停止**：最多 3 个自有游戏进程；不碰 tmux `qc` 与别的代理会话。冒烟（每族 4 局）只验管线，数据放 `artifacts/task42/smoke`，不进最终统计；正式跑放 `artifacts/task42/full`。若冒烟暴露实现缺陷，修复后重新冻结并在本文件记一行，正式样本不变。固定 2048 点，不做期中功效判断、不按结果扩样。

**通过后的动作**：B3 成为老师候选唯一实现——删除静态候选路径（`--static-value`、`_static_frames`、`STATIC` 相关代码与测试），`curriculum_queue.py` 的 `DEFAULT_STATIC_VALUE` 与 `command_for` 的 `--static-value` 一并删，`--student-candidates` 默认改 3；旧路径打 tag `retire-static-candidates-20261023` 后删。不通过则只保留比较脚本，正式路径不动。

**进度**
- 10:00 单局管线（`day_l7` seed 30000）通过：8/8 点写出，1 个分歧触发重老师（池 20）；B3 每点候选 4、A 5–11，未发现实现缺陷。
- 10:08 冒烟四族各 4 局完成：16 局 128 点，全部 8/8 点；16 局全终局，严格重放 14783 帧；判分 15 点、同动作 113 点；B3 每点候选 4.0、A 5.4–9.8；B3 续走/CPU 均低于 A。统计链路通过（池化 `formal_final_analysis` 只在 256 局时为真，冒烟不再误标正式）。冒烟数据不进最终统计。首次后台启动因工具超时上限被停，改用 `setsid` 续跑；点名与轨迹均按原子写/已有文件续跑，未产生半截点。正式跑开始。
- 10:09 正式跑启动：四族 × 64 局 = 256 局，3 worker，输出 `artifacts/task42/full`，驱动 PID 204531（setsid 分离，日志 `logs/full.log`）；`plan.json` 已写出模型/二进制 SHA，与预登记一致。冒烟两族 0 分歧（白天/传送带）提示正式跑白天与传送带的判分点可能偏少，按预登记照跑不扩样。
- 12:45 巡检（WSL 12:00 / 12:40 两次重启后）：正式跑于 11:26 中断。轨迹完整 226/256 局（day_l7 / night_l14 / pool_l24 各 64，day_l10_orig 30000–30031）；有效点 1778 个（day_l10_orig 242）。发现 19 个 0 字节点文件（day_l10_orig seed 30031–30033）和 day_l10_orig seed_30032 / 30033 的 0 字节轨迹，均为文件系统 I/O 错误残留，已按明确路径删除作废；模型与二进制 SHA 复核与预登记一致。按**原 256 局种子**续跑补齐，不改种子、不换样本。
- 12:53 续跑启动：驱动 PID 3775（setsid，日志 `artifacts/task42/logs/full_resume.log`，pid `full_resume.pid`），3 worker；已确认自有模拟器进程恰为 3 个。已有轨迹与点名文件由脚本按“存在即跳过 + SHA 校验 + 严格重放”复用。
- 13:11 续跑完成 256/256 局（`compare_summary.json` 写出）。核对发现 16 点缺失（day_l10_orig seed 30031/30032 各 8 点）：续跑期间我删除这两个种子的旧点名文件，与比较进程的“存在即跳过”检查发生竞态，已写点未回补。13:13 按原种子第二次重跑，13:20 补齐；终稿 256 局、2048 点，轨迹 SHA256 全匹配、0 字节文件 0 个、严格重放 230214 帧，全部终局。所有族每局 eligible 根 76–463，无 <8 根局。
- 13:21 统计（`b3_noninferiority_stats.py`，预登记 t 区间 + 10000 次整局 bootstrap）：白天 L7 −0.98pp [−2.58,+0.63] 通过；夜间 L14 −3.32pp [−5.42,−1.22] **未过**；泳池 L24 −6.05pp [−8.54,−3.57] **未过**；传送带 L10 0.00pp [−0.56,+0.56] 通过；池化 −2.59pp [−3.53,−1.64]。四族全过才通过的合并判定：**B3 不通过**。成本仍省：池化续走 −34.7%、CPU −42.9%；损失集中在夜间/泳池（只 A 对 29/42，只 B3 对 12/11）。按预登记：正式路径不动（仍 `static`），只保留比较脚本；不删静态候选、不改 `--student-candidates` 默认、不打退役 tag。报告见 `MEMORY/docs/static-value-ablation.md` 的“task42：现行设置复验（2026-10-23）”。

### 第二轮预登记（2026-10-23 14:10 台式机 CST；写于任何第二轮对局之前）

主控 14:10 追加：B3 池化仍省 CPU 42.9%，但夜间/泳池未过；损失集中在“只 A 对”的点，值得看加大 k 能否补回。样本、口径与判据在此定死；实现代码在本预登记之后、任何 B4/B5 对局之前提交并冻结（hash 记入进度）。

**问题与样本**：只跑第一轮未过的夜间 L14（`night_l14_eval_n10`）与泳池 L24（`pool_l24_eval_p9k`）。两族 × 64 局 × 8 根 = 1024 点；**新种子 30064–30127**（第一轮 30000–30063 之外，不复用第一轮种子）。根抽样 `Random(20261023+seed).sample(eligible, 8)`，eligible 与第一轮相同（有合法种植动作，且学生本步选择种植或距上一个 eligible 根 ≥300 tick）；某局 <8 根则用全部 eligible，不重采样、不补局，如实记录。三臂共享 `D = seed×10000 + decision − 1` 的未来。测量先后按 `(seed+decision) mod 3` 轮换：0→A,B4,B5；1→B4,B5,A；2→B5,A,B4。

**臂**：
- A（现行默认）：静态价值网每卡前 1 + 等待 + 学生原动作 + 额外 2 个不同学生动作（`--student-candidates 2`）。
- B4：学生全局前 4 个种植（`ranked_plants`）+ 等待 + 学生原动作；不跑静态打分、无每卡下限。
- B5：同 B4，前 5 个。
三臂其余参数相同：samples 3、horizon 2500、halving、verify-samples 9、min-gap 300、override margin 0.02、rollout sim-skip 61。正式路径不动；`candidate_mode` 仍私有，只有比较脚本设置。

**重老师**（第一轮定义的直接推广）：仅当三臂所选动作不全相同时运行，每个分歧点只跑一次；动作池 = 三臂全部候选 + 单模型静态每卡前 2 + 每卡一个随机剩余合法格，按动作键统一排序。每个池动作跑 64 个共同未来（`truthD = D + 33333333`，`(truthD×1000+k) mod 2^32`），v3 叶值；同一锚定验证（margin 0.02，额外 18 个未来 × 2 个动作）。三臂同动作时两个 delta 均为 0，不运行重老师。每点 `delta_k = 1[Bk=重老师] − 1[A=重老师]`，k=4,5；同动作点 delta=0，按整局 8 点均值计入。

**统计与判据**：每族、每个 k 分别对 64 个整局均值做 t 区间（df=63），并做 10000 次整局 bootstrap（seed 20261023，重采样的 delta 和 / 点数和）；两种 95% 区间下界都必须 > **−3pp**。沿用第一轮的 95% 口径，按主控指示不做多重比较校正；两族池化另报参考，不参与判定。一个 k“通过”= 夜间与泳池两族都通过。只给配对差值，不给绝对一致率，不外推胜率。成本按族、按臂报每点实际续走（halving + 验证）与 CPU 秒（Python worker user+system + 模拟器 user+system）；重老师成本单列。

**第 2 步（必须先于 B4/B5 对局）**：在第一轮 `artifacts/task42/full` 里取“只 A 对”的点（重算 delta=−1；夜间 29 + 泳池 42 = 71），严格重放到根，用同一学生状态重算 `ranked_plants`，记录 A 选中动作的 1-based 排名；非种植动作（等待）单列。分桶：1、2–3、4、5、6–10、>10、等待、未找到。决策规则（按主控）：若 **>50% 的 71 点排名 >10**，则 B4/B5 无法覆盖，直接写结论、不跑 B4/B5。无论是否跑，都报告 rank≤4、rank≤5 的点数，作为 B4/B5 的理论覆盖上限。附加的明显无效判据（仅用于避免必败跑）：若某族“只 A 对且 rank≤k”的点数（最乐观地把它们全部补回、且不计任何新增损失）都不足以把该族点估计拉回 −3pp 以上，则该 k 在该族不可能通过；若所有 k 都不能同时在两族满足此必要条件，则不跑并写结论。

**进度（第二轮）**

- 15:02 第 2 步 rank 分析完成（`scripts/b3_round1_ranks.py`，对第一轮 `full` 数据严格重放到根、同一学生状态重算 `ranked_plants`；首轮该脚本用系统 python3 无 torch 启动失败，改用 `.venvs/ml` 后一次通过）：71 个"只 A 对"点中种植 70、等待 1、未找到 0。A 动作排名：夜间 rank≤4 有 9、≤5 有 10、≤10 有 18、>10 有 10；泳池 rank≤4 有 11、≤5 有 16、≤10 有 31、>10 有 11。合计 >10 占 21/71 = 29.6%（<50%，不触发"排名太靠后"跳过规则）。必要条件（乐观上界，MARGIN 0.03×512 点 = 需 >−15.36）：B4 在泳池族乐观上界 −20，**不可行**；B5 夜间 −7、泳池 −15，两族均可行 → 按预登记继续跑 B4/B5（B4 仍照跑，泳池族的不可行性如实报告）。数据 `artifacts/task42/round2/round1_ranks.json`。
- 15:14 第二轮实现冻结并提交（`8c3c841b`）：`scripts/b3_round2_arms.py`（git blob `946f7886a091bd57d1d7564eb844bc6ed2938a58`，三臂 A/B4/B5、种子 30064 起、测量顺序按 (seed+decision)%3 轮换、三臂不全同时动作才跑重老师、delta_k 相对 A）与 `scripts/b3_round2_stats.py`（git blob `1d4faa7f338fdfba93a599adf5932f8598f6e630`，每族每 k 的 t 区间 + 10000 次整局 bootstrap，池化只作参考）。冻结在冒烟/正式任何对局之前。
- 15:01 冒烟（每族 2 局，种子 30064/30065）通过：32/32 点写出，4 局全终局，严格重放 408–1546 帧；三臂齐全，7 点判分（重老师池 8–21），delta_B4 合计 0、delta_B5 合计 +1；统计链路通过（<64 局时 formal=null，不误标正式）。冒烟暴露一处统计脚本缺陷（漏导入 `correctness_difference`，驱动与样本无关），修复后重新冻结：`b3_round2_arms.py` 不变（仍 `946f7886`），`b3_round2_stats.py` 新 blob `c1aafbaa0b5e27b790755921221b38352cc0b500`。正式样本不变（种子 30064–30127，128 局 1024 点）。
- 15:25 GLM 交接（主控指示任务交回 Qoder）：rank 分析、实现冻结（`8c3c841b`/`56521c7b`）、冒烟均已完成并推送；正式跑**进行中**——托管后台任务 `wXhB6S`，驱动 PID 41747（`/home/newbiexvwu/.venvs/ml/bin/python3 scripts/b3_round2_arms.py --out artifacts/task42/round2/full --games 64 --workers 3`），自有游戏进程恰 3 个，输出 `artifacts/task42/round2/full/`（进度 240/1024 点，完成后写 `compare_summary.json`）。下一步（Qoder）：跑完后执行 `scripts/b3_round2_stats.py --data artifacts/task42/round2/full --out artifacts/task42/round2/full_stats.json` 按预登记判据写结论；期间不要杀上述进程、不要删 `points/` 下文件（见"问题"）。
- 15:25 Qoder 接手复核并续跑：4 个冻结 SHA256（学生/叶值/静态价值/模拟器）与 `plan.json` 一致；两个脚本 git blob 与冻结记录一致（`946f7886` / `c1aafbaa`）。GLM 的驱动 PID 41747 已不存在（随其会话结束），正式跑停在 278/1024 点（全部夜间 L14，泳池未开始），`compare_summary.json` 未写。按原命令、原样本续跑：驱动 PID 48975（setsid，3 worker，日志 `~/pvz-logs/b3_round2_resume_1525.out`，pid 文件 `artifacts/task42/round2/full_resume.pid`）；已有 278 点由 `dest.exists()` 跳过，未删 `points/`。
- 16:04 第二轮正式跑完成（续跑驱动 48975 已退出）：128 局 1024 点全齐（夜间/泳池各 64 局×8 点）、128/128 全终局、严格重放 122339 帧、0 字节与 `.tmp` 文件 0、点集与轨迹 meta 完全一致；本轮续写 746 点，先前 278 点复用。报告 `artifacts/task42/round2/full_stats.json`（冻结统计脚本 `c1aafbaa`）。
- 16:04 结论（按预登记）：**B4** 夜间 0.00pp [−2.15,+2.15] 通过、泳池 −2.34pp [−4.49,−0.20] 未过；**B5** 夜间 +2.34pp [−0.65,+5.33] 通过、泳池 −1.37pp [−3.92,+1.19] 未过（两种区间下界均 < −3pp，B5 差 0.92pp）。**没有 k 在两族同时通过**；按预登记不采用 B4/B5，正式路径仍 `static`，不删静态候选、不改 `--student-candidates` 默认，采用与否由主控决定。成本参考：B4 池化 CPU −13.3% / 续走 −8.5%（泳池 CPU −15.8%）；B5 池化 CPU −5.7% / 续走 −1.0%（夜间续走 +0.1%，CI 含 0）。只 A 对/只 B 对：B4 夜间 17/17、泳池 29/17；B5 夜间 18/30、泳池 30/23。判分 185、同动作 839；只报配对差值，不外推胜率。
- 16:05 Qoder 复核 GLM 的 rank 分析与冻结：`round1_ranks.json` 汇总与预登记记录一致（71 个只 A 对点中 >10 占 21/71=29.6%；夜间 rank≤4/≤5/≤10/>10 = 9/10/18/10，泳池 11/16/31/11；B4 泳池乐观上界 −20 不可行、B5 两族可行 → 不触发跳过规则）；`b3_round1_ranks.py` 与第二轮两脚本 blob 均与提交记录一致，正式跑使用的模型/二进制 SHA 与冻结表一致。

**资源与停止**：最多 3 个自有游戏进程；不动 tmux `qc` 与别的会话。B4/B5 数据放 `artifacts/task42/round2/full`；冒烟（每族 2 局）只验三臂与重老师管线，放 `artifacts/task42/round2/smoke`，不进最终统计。若冒烟暴露实现缺陷，修复后重新冻结并在本文件记一行，正式样本不变。固定 1024 点，不做期中功效判断、不按结果扩样。

## 问题

- 续跑期间不要删除正在运行的比较进程会检查/写入的点名文件：`compare()` 逐点先查 `dest.exists()`，删除已写点会导致该点被跳过且不回补。本次已用第二次重跑修复，终稿数据完整；后续续跑只删“已确认不会被本进程再次检查”的文件，或先停驱动。本次无未决问题。

## 主控追加（2026-10-23 14:10）：第二轮——加大 k 能不能把夜间 / 泳池拉回来
结论收到：B3 白天 / 传送带过、夜间 −3.32pp、泳池 −6.05pp 不过；但池化省 CPU 42.9%，太大，值得再试一轮。损失集中在“只 A 对”的点，说明学生前 3 漏掉了好动作。
1. **预登记**（先写进“结果”并提交）：只跑没过的两族（夜间 L14 n10、泳池 L24 p9k），**新种子** 30064 起每族 64 局（不得复用第一轮种子），每局 8 根；比较 A 对 B4（学生前 4）与 B5（学生前 5），判据同第一轮（每族 95% 下界 > −3pp，t 区间 + 整局 bootstrap 都要过），同时报每点推演数与 CPU。
2. 先看第一轮“只 A 对”的 71 个点：A 选中的动作在学生排序里排第几（第 4？第 6？还是根本不在前 10）。这个分布决定 k 取多少有意义，写进“结果”；如果大部分在前 10 之外，B4/B5 也救不回来，就直接写结论、不跑第 2 步。
3. 跑完写结论：哪个 k 在两族都过、省多少 CPU；都不过就如实写。正式路径仍不动，采用与否由主控决定（可能的方案：全部族用通过的最小 k）。
4. 最多 3 个游戏进程；续跑时不要删比较进程会检查的文件（见“问题”）。

## 主控收尾（2026-10-23 16:30）
主控认可第二轮结论：B3/B4/B5 都不采用，老师候选的正式路径仍是静态价值网，task42 关闭。
最后一步：在本工作树 `git merge --no-ff origin/pvz-env`，冲突按"两边都留"解：日志按时间排，任务书以本分支为准。
比较臂脚本（`b3_*`、统计脚本）随报告一起保留，作为证据链，正式路径不动。跑 `python -m pytest python -q` 不得多出失败，然后 `git push origin b3`，在"结果"写 `[close]` 一行（合并提交号、测试数字）。
