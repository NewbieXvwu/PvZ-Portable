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

- [x] 修正 1（commit c70590d0）：打包并发按待建段估算。新增纯函数 build_plan
  （单建造者峰值 = PACK_BUILDER_BYTES 2 GiB + 待建标签最大原始档案体积；并发 =
  min(配方值, (MemAvailable − 6 GiB)/峰值, 待建段数)）与 TrainingPipeline.pending_builds
  （与 _prepare_corpus 同一合并视图、与 pack() 同一段契约，指纹只用相对路径+大小）；
  enqueue 写配方前调 _plan_build，estimate_peak = max(打包峰值, 3×已建段+2 GiB)，
  重启时从最新 queued job 的快照配方恢复估算。pack_training_rows.pack() 的 memory_limit
  抽成 pack_memory_limit(available, jobs)（2 GiB 底数 + 最大待建档案），与 build_plan 同一
  假设。测试：round 2 规模（202 标签×0.3 GiB、40 GiB 可用）冷启动并发 14 > 1 且峰值
  ≤ 可用−6 GiB；边界（空待建/极小内存/配方上限）；pack_memory_limit 一致性（稀疏文件）；
  pending_builds 合并覆盖与段契约。42 过。
- [x] 修正 2（β 混合）：student_rate() 先看当前倍率窗口、退回最近非空窗口、全无数据
  返回 None（首个组保持 β=0 探针）；run_round 对学生胜率 < --beta-threshold（默认 0.1）
  的格给 --dagger-beta = --starved-beta（默认 0.5），组行日志与 state log 的 groups 都记
  beta。search_play.py 语义已核实（search_play.py:571-661）：dagger_beta<1 时每个决策仍由
  搜索打标签（frame["label"]=搜索决策），只是以 1−β 概率执行学生自己的动作——老师执行的
  步照常产出训练行，与 beta0 的行同格式可混训，无阻碍。

- [x] 修正 3（组间崩溃重启）：allocate 不再写 last_round（plan 纯读，只在内存标 picked；
  scored 排序跳过 last_round == round_index 的格——本轮已完成的组不重采）；run_round 在
  每组成功完成后写 cell 级 last_round 并 save_state 落盘（原 bucket 级写入无人读，删除）。
  防遗忘语义不变：floor 轮转仍按 (last_round, base) 最久优先，FLOOR_ROUNDS 窗口判定未动。
  测试：allocate 纯度、12 轮无格间隔超 FLOOR_ROUNDS+1（按组完成语义模拟）、
  plan→前两组完成→重 plan：已完成组不重采、没跑的组照常排出且偏移接续。47 过。
## 问题
