# task98：qc 不再因为单点异常整体停机，并加看门狗

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-perf`）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/qcwatch`（分支 `qcwatch`，主控建）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
24 小时内 qc 停了三次，都是一个格或一个目录名触发的异常让整个控制器退出：
- 10-25 15:30：`int('f44300000')`；
- 17:01：同类目录名；
- 10-26 06:50：`demo_fog_32` 用完了 256 个未来种子，`plan_round` 抛了 SystemExit，qc 停了 1 小时 50 分，主控巡检也没发现。

B7 只处理了后台任务失败的情况，主循环里的异常还会直接杀掉 qc。

## 要做的
1. **审计**：列出 `scripts/curriculum_queue.py` 主循环会走到的所有 `raise` / `SystemExit` / `sys.exit` / 未捕获的异常路径，包括 plan、run_round、reconcile、load_panel、teacher_rates 读取、门禁调用等。逐条分类：
   - **配置错误**：启动时就应该失败，保持原样；
   - **单格 / 单组问题**：记日志、跳过或退休该格，qc 继续；
   - **可重试的瞬时错误**：例如 IO，重试一次。
2. **改**：按分类实现。每类至少一个测试，用伪造的坏格、坏目录名、种子用完来模拟。不加开关。
3. **看门狗** `~/pvz-logs/qc_watchdog.sh`，代码放仓库 `scripts/qc_watchdog.sh`：
   - 每 2 分钟检查 tmux `qc` 是否存活，以及 curriculum.log 是否超过 30 分钟没有新行；
   - qc 死了：用**最近一次部署脚本里的 `start_qc` 命令**重启（从 `~/pvz-logs/qc_deploy*.sh` 里取最新的），写 `qc_setup.log`，并把 qc.out 尾部 50 行存档；
   - 30 分钟内重启 2 次仍失败就停止重启，写一条醒目的日志，等主控处理；
   - 有 `~/pvz-logs/stop_curr` 或部署脚本在跑时不要重启，避免和部署冲突。
4. 结论写 `MEMORY/docs/qc-resilience.md`，`git push origin qcwatch`。上线由主控在组边界做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
