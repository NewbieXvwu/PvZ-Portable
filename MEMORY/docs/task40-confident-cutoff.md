# task40：老师推演“胜负已定就收手”——先量清能省多少，再按噪声底线验收

> 主控 2026-10-23 11:05 写。执行者：Qoder（DeepSeek-Flash），工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff`（分支 `cutoff`，从 pvz-env fcc1b039 拉出）。
> 同时最多 **2** 个你自己的游戏进程（qc 采集 16 进程在跑，task35/37 也在测）。不碰 tmux `qc` 与别的代理会话，不重启 WSL。

## 为什么做

- 体检（`MEMORY/docs/hotpath-audit.md` §1）：一局搜索局约 3962 条推演、980 万 tick，**平均每条推演 2470 tick，几乎都跑满视界 2500**。引擎仿真占 77%。
- 想省 tick 只有两条路：每 tick 更便宜（task35 在做），或者少跑 tick。
- “统一缩短视界”已经试过，**不过**：task8 把视界 2500→1800，老师决策一致率 66.54%，噪声底线 71.33%，差 −4.80pp，低于 −3pp 门槛（`MEMORY/docs/cuda-horizon1800-validation.json`、`cuda-rollout-report.md`）。
- 没试过的是**按局面收手**：推演中途在检查点（例如每 500 tick）看一眼价值网，**已经几乎肯定赢 / 输**就停，叶子直接用价值网的值（或 0/1）；拿不准的照旧跑满 2500。缩视界是一刀切，这是只砍“结局已定”的那部分。
- 用户 10-23 定的总目标：纯 CPU 采集整局吞吐到 10-27 达到旧构建的 2 倍，否则重新考虑 GPU 方案。本任务是 task35 之外少数能大幅省 tick 的路。

## 第 1 步：量上限（只读测量，不改搜索）

1. 局面：从 `/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2` 取 4 族真实对局的搜索点 root（白天 L7 d7、夜间 L14 n10、泳池 L24 p9k、传送带 L10 orig），每族 64 个 root，取法照 task35 的 `scripts/bench_engine.py manifest`（可直接复用它的清单格式；先读它）。
2. 每个 root 跑 9 条推演（学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`，`--rollout-sim-skip 61`，与正式参数一致），记录检查点 t ∈ {500, 1000, 1500, 2000} 与终点 2500 的叶观测。
   - 先验证：同 root 同 seed，视界 t 的推演是视界 2500 推演的**前缀**（叶状态与 2500 推演在第 t tick 的状态相同）。成立就每条推演只跑一次、中途取快照；不成立就分别跑，并把原因写进“问题”。
3. 每个叶子用正式老师的价值网打分（`search_play._leaf_values` / `_score_terminal_or_value` 的同一套，价值网 `~/PvZ-search/artifacts/value_net/v3/model.pt`，分叉网 `~/PvZ-search/artifacts/hard/value_branch1/model.pt`——参数照 qc 的 `command_for`，不要自己猜）。
4. 出表（每族一张）：检查点 t × 阈值 θ ∈ {0.90, 0.95, 0.98, 0.99}：
   - 在 t 时价值 ≥ θ 或 ≤ 1−θ 的推演占比；
   - 这些推演在 2500 时的值与 t 时值的差（均值、95 分位），以及“胜负判断翻转”的比例；
   - 若在 t 收手，能省的 tick 占全部 tick 的比例。
5. 写进本文“结果”，并给出你建议的 (t, θ) 组合（省得多、翻转少）。**这一步就算结论是“省不了多少”也是有效结果**，写清楚就收工，不要硬做第 2 步。

## 第 2 步：实现（只有第 1 步显示可省 ≥ 15% tick 才做）

- 改在引擎里做（Python 每 500 tick 往返一次会吃掉收益）：推演到检查点时调用引擎内的价值网前向。task37（codex2，分支 `vcpp`）正在把价值网前向搬进引擎（`src/EnvironmentValueNet.{h,cpp}`、`LOAD_VALUE_NET` / `LEAF_VALUES`）。**先看 vcpp 分支的进度**：已可用就基于它（`git merge origin/vcpp`，在“问题”里记下依赖）；还不可用就先在 Python 侧做一个只用于验收的慢版本（每个检查点一次往返），量质量，不量速度。
- 不加开关：通过验收就成为唯一实现；不通过就把数字写进报告，代码删掉。

## 第 3 步：验收（预登记，照 task8 的口径，先写进“结果”再跑）

- 质量：正式老师参数，4 族各 16 局（种子用 30000 起的评估段，与 task8 同口径），比较“收手版”与“原版”的搜索决策一致率；噪声底线 = 原版换随机未来的一致率（同局同点）。**收手版一致率 ≥ 噪声底线 − 3pp**，整局成组 bootstrap 95% 区间写出来。
- 速度：同种子配对，每局 CPU 秒（worker + 引擎），报比值。
- 强度：过了质量关才做——每族 64 局 beta=1 胜率配对，在噪声内。
- 结果、命令、种子、输入 SHA 写进本文与 `MEMORY/docs/engine-cpu.md` 的新节（如果该文件在你的分支不存在，就新建 `MEMORY/docs/rollout-cutoff.md`）。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥。
- 不手写规则：阈值是参数，按测量选；不按关卡或植物写特例。
- 每完成一小步就提交、`git push origin cutoff`，在“结果”记一行；疑问写“问题”，按最佳判断继续。

## 结果

## 问题
