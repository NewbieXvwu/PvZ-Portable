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
| 模拟器 | `~/PvZ-search/build/pvz-portable` | `ed98a4536912512fa17a47b299e9d72f5175c6495961502439fa147711b26034` |

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

## 问题
