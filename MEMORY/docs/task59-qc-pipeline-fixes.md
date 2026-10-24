# task59：qc 流水线三处修正——冷启动打包单核、全败格的 β 混合、组间崩溃重启

> 主控 2026-10-24 09:25（台式机时间）写。执行者：台式机 CodeBuddy（`cb-qcalloc` 会话，接着 task57 做）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/qcpack`（分支 `qcpack`，从 origin/qcalloc 拉出，主控已建好）。**只改代码与测试，不跑批量对局**，游戏进程最多 1 个；**不碰 tmux `qc`、不改 cq 工作树**。部署由主控做。

## 为什么
1. **冷启动打包单核。** 09:07 实测，round 2 的后台打包（第一次建 v2 段）已经跑了 4 小时 55 分，49k 局只缓存了 15.8k。原因在 `curriculum_queue.py` 的 `enqueue()`：
   ```python
   recipe['precompute_workers'] = min(recipe['precompute_workers'], max(1, self.estimate_peak() // (2 * GIB)))
   ```
   `estimate_peak()` 按**已有段**的体积估算；冷启动时段目录是空的，于是得 2 GiB，并发数降成 1。20 核的机器只用 1 核打包，剩下约 8.7 小时，期间后台预留一直压着 qc，前台只开得出 1–5 个 worker。主控已在旁边开了 8 进程的缓存预热（`~/pvz-logs/warm_cache.py`）救急，这次不用管。但根因要修。
2. **全败格的 β 混合**（task57 "问题"里你自己提的，主控批准）：
   - pool、roof、L20 这类格，学生开局就崩。beta=0 时学生轨迹只覆盖开局几十步，中后盘状态永远拿不到标签。
   - 标准做法是 Ross et al. 2011 的 β-mixing：部分步由老师执行，扩大状态覆盖，同时保留学生分布的成分。
3. **组间崩溃重启后本轮组不重排**（task57 "问题"第一条）：`allocate` 在 plan 时就写了 `last_round`，控制器若在组之间崩溃，本轮还没跑的组会被当作已经跑过。批准把 `last_round` 的写入挪到组完成时。

## 要做的
1. **打包并发按待建量估算，不按已建段估算。**
   - 每个段建造者的内存峰值，用待建标签的原始档案体积（或已有的 row/frame 计数）估算；`precompute_workers` = min(配方值, 可用内存 / 单建造者峰值, 待建段数)。
   - 后台预留（`reservation()`）用同一估算，保证不超内存。
   - 用 round 2 的真实规模（202 标签、约 5 万局）写一个测试或一段离线计算：冷启动时并发数应明显大于 1，且估算的内存峰值不超过当前 `MemAvailable − 6 GiB`。
   - 顺带看一下 `pack()` 第 361 行 `memory_limit` 的算法，它和上面的估算要一致。
2. **β 混合**：
   - 学生在该格最近的面板胜率（或最近一组的执行胜率）低于阈值时，`command_for` 给 `--dagger-beta` 一个较高的值。阈值和 β 写成配方或命令行参数（建议默认：胜率 < 0.1 时 β = 0.5），不写死在代码里。
   - 写进 `curriculum.log` 的组行：本组用的 β。
   - 先读 `search_play.py` 中 `--dagger-beta` 的现有语义，确认老师执行的步也照常产出标签、训练行与 beta0 的行能一起训练。不确定就写"问题"。
3. **`last_round` 在组完成时写**，同时保证防遗忘的语义不变。补一个"组间崩溃后重启"的测试。
4. 全部测试通过（Mac 与台式机的 `python/test_curriculum_*.py` 及相关）。报告写在本文件"结果"里，**主控合并并部署**。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则（β 与阈值是训练超参数，可以）。
- 一处修正一个提交，`git push origin qcpack`，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

## 问题
