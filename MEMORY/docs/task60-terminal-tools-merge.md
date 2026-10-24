# task60：三条"接终局"诊断线合成一套，备好合入 pvz-env

> 主控 2026-10-24 10:05（台式机时间）写。执行者：台式机 Codex（plus 账号，`codex-merge` 会话）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge`（分支 `termmerge`，从 origin/pvz-env 拉出，主控已建好）。**以读代码、改代码、跑测试为主**，游戏进程最多 2 个，只用于复现核对；**不碰 tmux `qc`、不改别的工作树**。合入 pvz-env 由主控做。

## 为什么
- 三个分支各写了一套"在决策点对候选做续走到终局"的工具，都还没合入 pvz-env：
  - `fog`（task45 追加 E，16 个提交）：`fog_terminal_*`；
  - `roof`（task46 追加 B 与第三阶段，30 个提交）：`roof_terminal_*`、`terminal_candidate_audit.py`、`terminal_candidate_report.py`；
  - `forks`（task53，16 个提交）：`critical_forks.py` 及数据、评估脚本。task53 的任务书本来就要求"只有一种实现，三条线以后改用它，主控合并时删重复"。
- 这三个分支落后 pvz-env 49–161 个提交，越拖越难合。
- 它们还改了生产文件：
  - `python/pvz_env.py`、`python/pvz_frame_store.py`（fog）；
  - `python/pvz_training_pack.py`、`scripts/pack_training_rows.py`（forks）；
  - `scripts/search_play.py`、`scripts/episode_query.py`（roof）。

  其中 `pack_training_rows.py` 和 `curriculum_queue.py` 的同一条流水线，正在 `qcpack` 分支（task59）上改。

## 要做的
1. **以 `forks` 的 `critical_forks.py` 为唯一实现。**
   - 逐个读 fog 和 roof 的终局脚本，列出它们有、而 `critical_forks.py` 没有的能力（例如屋顶的候选审计、雾区的透视对照）。
   - 缺的补进 `critical_forks.py`。
   - 补齐后删掉重复脚本和它们的测试。
   - 各报告里的复现命令改成指向新实现，旧证据（json 与报告）原样保留。
2. **只想复现结论时**，用 2 个进程挑一两个点核对：屋顶取一个根，雾区取一个根，看新实现给出的终局结果与旧证据是否一致，写进结果。
3. **生产文件改动逐处审**，每处写一行：为什么需要、是否改变生产行为。
   - 改变生产行为、又不是诊断必需的，就撤掉。
   - 留下的必须由测试覆盖。
   - `pack_training_rows.py` / `pvz_training_pack.py` 的改动要与 origin/qcpack 对照，避免将来冲突；有冲突就写进"问题"，由主控定顺序。
4. **合并方式**：在 `termmerge` 上依次合入 forks、roof、fog，冲突逐个解决，保留双方的 MEMORY 日志条目。之后：
   - Mac 与台式机测试全部通过：`python -m pytest python/`，或至少相关的测试文件，写明跑了哪些；
   - 跑一次 `scripts/loose_ends.py`。
5. 推 `origin termmerge`，结果记在本文件。**主控审阅后合入 pvz-env。**

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`；引擎用 `/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3/build/pvz-portable`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写规则或公式。
- 不要 `git stash`；不要改写已推送的历史。
- 每完成一小步就提交、`git push origin termmerge`，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 合入 origin/fog；DESIGN 与两日日志的追加段全部保留，生产增量为 trace 同步与 NPZ 时间戳确定性，后续逐项验证。

- 合入 origin/roof；两日日志冲突保留双方条目，search_play 快路径条件同时保留 fog_diagnostics 与 observer，后续审查测试。

- 合入 origin/forks；仅当天 MEMORY 日志冲突，双方条目全部保留。后续统一实现与测试继续。

## 问题
