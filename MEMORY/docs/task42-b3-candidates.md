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

## 问题
