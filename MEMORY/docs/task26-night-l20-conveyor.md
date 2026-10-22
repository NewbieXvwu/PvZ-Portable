# task26（主控 2026-10-22 下发给台式机 CodeBuddy 第二条线）：2-10（夜间传送带关）为什么第 7 波就输

先读 `AGENTS.md`、DESIGN §14.8–§14.9、`MEMORY/docs/task20-conveyor-fidelity.md`、`MEMORY/docs/night-panel.md`、`MEMORY/docs/task25-night-n10.md`。

## 背景
原版开局的 2-10（`night_l20_eval_n9_orig` / `night_l20_*_n10_orig`，开局空、卡由引擎 `Challenge::UpdateConveyorBelt` 发）在统一面板上所有模型 0/128，冒烟在第 7 波丢完车。
白天原版 1-10 学生 256/256。只做分析，不训模型、不改环境。

## 要做的
工作区：`~/PvZ-search/.claude/worktrees/l20diag`（分支 `l20diag`，主控已建好），只在这里写脚本与报告。
1. **引擎事实**（读源码，写清行号）：2-10 的传送带发卡权重表与间隔（对照 1-10 的表）、波数、僵尸名单、坟头数与最后一波冒僵尸、有无别的特殊规则（mini-boss 关）。
2. **对局事实**：学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt` 跑 `night_l20_eval_n10_orig` 16 局（`--workers 2`，录帧 `--frames` 到本工作树），统计：每局发到的卡（种类与时刻）、学生实际种了什么 / 留着没种什么、
   车在第几波因哪路什么僵尸丢掉、阳光与坟头。再用老师（同学生 + 搜索，参数照 `~/pvz-logs/queue103.sh` 的 `run()`，但 `--dagger-beta 1`）跑 8 局对比。
3. **结论**：输在哪（发卡结构 / 学生不会用这些卡 / 老师也不会 / 别的），给出下一步建议（采集、卡组无关，因为传送带不用卡组）。
4. 报告写 `MEMORY/docs/night-l20.md`；提交、`git push origin l20diag`，在本文“结果”记一行，然后停下。

## 规则
不动任何 tmux 队列与别的代理会话；不改别的工作树；同时最多 2 个游戏进程；不改默认值；不写手工规则；不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。

## 结果

## 问题
