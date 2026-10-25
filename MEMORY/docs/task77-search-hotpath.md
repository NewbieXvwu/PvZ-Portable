# task77：老师搜索的热路径——每个决策的耗时花在哪，能不能逐位不变地再快一截

> 主控 2026-10-25 写。执行者：台式机 Codex（Plus 号，新开 tmux `codex-perf`，`CODEX_HOME=/home/newbiexvwu/.codex-plus`）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/perf2`（分支 `perf2`，从 origin/pvz-env 拉出）。游戏进程最多 4 个；**不碰 tmux `qc`、cq、build4、流水线**。

## 为什么
- qc 的产出（每小时多少局老师标签）直接决定学生学得多快。
- task35 优化过引擎 CPU，task51 测得采集吞吐约 1.8 倍。但老师搜索（fast_search：每卡最优格 + halving + 9 个未来复核 + rollout 2500 tick，学生前向）仍是每局耗时的大头。
- 之后又加了引擎内叶值（task37）、v5 叶值和 build4，热点可能已经变了。

## 要做的
1. **剖析**：用生产命令形态（`curriculum_queue.command_for`，deploy4 引擎，v5，台式机 branch1，现役学生），在白天 L7、夜间 L14、泳池 L24、雾 L31、屋顶 L41 各跑 4 局。用 perf / 采样剖析加 Python cProfile，给出每决策耗时的分解：rollout 引擎步、学生前向（C++ 内）、叶值前向、Python 调度与 JSON 往返、快照分支。
2. **改进**：挑占比最大的 2–3 处做优化，要求老师输出**逐位不变**——同种子同局动作序列、标签、胜负完全一致，照 task70 A2 的门禁脚本。不改搜索算法本身（候选数、未来数、视界、门槛都不动）。
3. **验收**：六族 × 8 种子，改前改后逐位一致；ABBA 配对计时与 qc 同负载，报每局耗时比与区间。
4. 结论写 `MEMORY/docs/search-hotpath.md`，提交并 `git push origin perf2`。部署由主控做（重建 PGO 引擎）。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不加开关，不改生产默认。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- [0] 已冻结生产 deploy4（16989f23）、在役 round11、v5、台式机 branch1 与六族清单；添加独立剖析工具，启动五地形各 4 局。细节见 search-hotpath.md；原始数据仅留 ignored artifacts/task77，不操作 qc/cq/流水线。

## 问题

- redeploy/build 已由 task70 B5 重建，不能当原 deploy4；正式基线只读复制生产 cq/build4（SHA 与 task70 deploy4 一致）。系统无 perf，采用现有 SIGPROF 采样器并结合 cProfile。
