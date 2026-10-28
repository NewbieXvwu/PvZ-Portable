# task141：训练积压时合并排队的训练作业，只训最新一轮

主控 2026-10-28 派给 Mac Qoder。分支 `coalesce`，以 `origin/deckcells`（生产控制器 = task137 版）为底。task138（`groupwidth`）也在改控制器，冲突由主控合并时处理，你只改训练队列这块。

## 现象

- 每轮训练 K=4 个候选、串行，加上挑选和门禁约 4 小时；采集一轮约 2 小时。10-28 11:20 时，第 28、29、30 轮三个训练作业在排队（`~/pvz-logs/curriculum_state.json` 的 `training_jobs`）。
- 每轮训练都是**全量数据从头训**（task135 核对过 `recipe.resolved.json`），所以第 29 轮的模型并不比第 30 轮多学到什么；排在中间的作业只会拖慢最新数据进入学生。

## 要做的

1. 训练执行器取下一个作业时，如果队列里有更新的已入队作业，就把较旧的标成 `superseded`：不训练、不过门禁，记录被谁取代。直接训最新那轮，它的数据已经包含旧轮。
2. 正在训练、挑选或门禁中的作业不打断。
3. 被跳过的轮次要留下清晰记录：状态文件和日志各一行，课程统计和种子账本不受影响。
4. 小测试：队列 [28 queued, 29 queued, 30 queued]，执行器空闲时只取 30，28、29 变为 superseded；重启恢复时语义一致。
5. 写 `scripts/qc_deploy14o.sh`（只替换相关文件；start_qc 照抄台式机最新部署脚本；stop_qc 不排空，直接杀 qc；部署前检查要便宜）。不执行。
6. 测试只在 Mac 上跑，严格遵守 AGENTS §6.5 第 8 条（只用 -k，timeout 300，ulimit -f 2000000，同时只开一个 pytest）。结论写在本文件末尾，推送分支。

## 结论（2026-10-27，Mac Qoder，分支 `coalesce`）

**实现**（`scripts/curriculum_queue.py`，commit `ce6dda32`）

- 执行器真正开始跑一个作业时（`_run` 开头、拿到 `training.lock` 之后）调用
  `_superseded_by(job)`：若存在**更新的**作业且它还会带来数据（`queued` / `running` /
  `done`，或失败但还会重试的 `failed`），就返回该轮次；`isolated` 与已
  `superseded` 的作业不使旧作业作废。
- 判定为被取代时不训练、不过门禁，`_run` 返回 `dict(superseded_by=N)`；
  由前台线程的 `_poll_job` 把状态写成 `status='superseded'`、`superseded_by=N`，
  存盘并写一行日志 `background round R superseded by round N; training and gate skipped`。
  课程统计（`log` / `cells` / `offsets`）和种子账本都不动。
- **不打断**：检查只在作业被取起时做一次。已在训练/挑选/门禁中的作业早已越过这一检查，
  新轮次入队不会影响它。已经写出合法 `complete.json` 的轮次也保留报告、不判 superseded。
- **重启一致**：`__init__` 重新排队时排除 `superseded`（否则重启会把旧作业复活）；
  `poll` 的完成处理也跳过 `superseded`，避免重复落盘/重复日志。重启后
  `[28 queued, 29 queued, 30 queued]` 仍只训 30。
- `set_gate_version` 的"流水线未排空"判定同样忽略 `superseded`。
- 顺带覆盖了一个后发情形：旧作业失败重试时若已有更新轮次，也会被取代，不会再拿旧语料
  重训一遍（新语料由之后的轮次全量包含）。

**测试**（`python/test_curriculum_queue.py`，`TrainingRestartTest`）

- `test_supersede_targets_the_newest_waiting_round`：28/29/30 → 28→30、29→30、30→None；
  `isolated` / 已 `superseded` 的不算，可重试的 `failed` 算。
- `test_backlog_trains_only_the_newest_round`：真实 `TrainingPipeline`（训练/评估打桩）
  跑队列 [28,29,30]：只有 30 执行 pack+train；28/29 变 `superseded`、`superseded_by=30`，
  日志恰两行；状态文件落盘一致；种子账本未被写入；用落盘状态再构造一次流水线（模拟重启）
  不会重新排队。
- 运行方式遵守 AGENTS §6.5-8：只用 `-k`、`timeout 300`、`ulimit -f 2000000`、一次一个 pytest。

**测试结果**

- 新增 2 个用例通过；`TrainingRestartTest` 9 个全过；
  `test_qc_resilience.py -k "bad_saved_jobs or orphaned_snapshot or twice_failed or gate_switch"` 4 个全过；
  `test_curriculum_pipeline.py -k "background_failure_is_saved or stale_comparator or evaluation_failure_is_closed_gate or all_seeds_failed"` 4 个全过。
- 三个失败是**基线既有问题**（在 `origin/deckcells` 的 worktree 上逐一复现）：
  `test_background_failure_retry.py::test_first_failure_is_recorded_and_retried_exactly_once`
  （task98 已把"第二次失败停 qc"改成"隔离并继续"，测试仍在找 `stopping qc`），
  以及 `test_multiseed_production.py` 的 `two_recipes_share_one_pack_*` 与
  `consecutive_rounds_score_*`（选卡面板任务数 48 vs 43、回执缺 `wins`）。本次未改这些。

**部署脚本**（`scripts/qc_deploy14o.sh`，不执行）

- 只替换 `scripts/curriculum_queue.py`；`REV=ce6dda32…`，blob sha
  `9d4829c0…3701`（脚本内有自校验）。
- `start_qc` 照抄台式机最新（deploy14m）版本：`--extra-manifest` 由 `qc_deploy14l.mode`
  的 `deckcells` 决定，其余分支参数不变；本次没有新增启动参数，也没有 mode 文件。
- `stop_qc` 用台式机版：直接杀 tmux 会话，不排空组；只回收本 worktree 的孤儿 `search_play`。
- 预检很便宜：单文件 sha256、git blob sha、`grep superseded_by`、`ast.parse`、
  （deckcells 模式下）144 任务面板 JSON 长度；不跑测试、不开游戏进程。
- 控制器基线接受 deploy14l（`a9d028fe…`）、deploy14m（`f98df24c…`）和本次（`9d4829c0…`）；
  若 task140 的 deploy14n 先上线，主控需把它的控制器 sha 加进 `BASELINE_CONTROLLER_SHAS`。
- 回滚：`qc_deploy14o.sh --rollback` 用最近一次备份恢复控制器并重启。

**给主控的说明**

- 冲突面：只改了 `scripts/curriculum_queue.py` 的训练队列部分；task138（groupwidth）改的是
  同一文件里的采集池/分组宽度，函数不重叠，但 merge 时 controller 文件需要人工合一次。
- 生产上首次启动（含部署重启）会在几秒内把队列里排在最新轮之前的作业标 `superseded`，
  状态文件可以直接看到 `superseded_by`；当前正在训练的那一轮不受影响。
