
---

## 结论（2026-10-28，Mac CodeBuddy）

### 1. 做法

`scripts/growable_pool.py` 取代 `scripts/search_play.py` 里的 `multiprocessing.Pool`。
每 worker 初始化（`_init_worker`）、一局一结果、`games.jsonl` 与断点续跑
（按 `(arm, seed/future_seed)` 跳过已完成）全部照旧；只有“进程怎么起来、活怎么分”变了：

- 父进程每 `--grow-interval`（默认 **30 s**）重读一次 `MemAvailable`，用控制器自己的准入公式
  `clamp((MemAvailable − reserve − background) // per_worker, 1, upper)` 算出现在能开几个；
  比当前多就 `Process.start()` 补上，**只增不减**，在跑的局不杀。
- `upper` = `--workers`（控制器传 `pipeline.foreground_limit()`，已扣掉可能开始的评估进程）；
  开工宽度走新参数 `--initial-workers`。
- `--worker-memory-bytes` / `--memory-headroom-bytes` 把控制器当时用的单 worker RSS 与后台预留
  原样带进去，扩容与开工用同一把尺子。
- 没实测 RSS（`--worker-memory-bytes 0`）或没有 `/proc`（macOS）时不扩容：按未知成本加进程
  正是 OOM 的来源。
- 作业分配改成“父进程派活”：每个 worker 一条私有 `SimpleQueue`，父进程记着
  `worker id -> 手上的那局`，直到结果回来才清空。**一个 worker 被 OOM 杀掉时既没有异常也没有
  消息**（第一版用共享队列 + 开工消息，消息还没来得及 flush 就死了），父进程因此分不清
  “这局还在跑”和“这局丢了”，剩下的 worker 只是堵在队列上 —— 会永久挂住。现在父进程在
  结果队列空转时检查“手上有活但进程已经没了”，直接报错退出；丢掉的那局按既有合同重采。
  `SimpleQueue` 没有 feeder 线程，派活也不会卡在未 flush 的缓冲区里。

控制器侧 `command_for()` 新增 `growth_args()`，把“开工宽度”和“上限”分开传。只给 `workers=`
的调用方（OOM 探针、一次性诊断、`task70_throughput_abba`）仍是固定宽度，参数与 task138 之前
逐字相同（有单测）。

### 2. 日志与既有解析

- 首行仍是 `N games on W workers`（W = 开工宽度），`scripts/qc_log_throughput.py` 的
  `START` / `PROGRESS` / `DONE` 正则都不改；宽度变化另打 `workers 1->4 (120s)`，不匹配任何
  既有正则，`python/test_qc_log_throughput.py` 全过。

### 3. 成本

- 每 30 s 一次 `/proc/meminfo`，可忽略。
- 新 worker 在 Linux 上 fork 父进程，继承已加载的 torch 与权重页（`_init_worker` 的
  `if "fast" not in _W` 分支），不再从磁盘加载；fork 的 COW 页只在真的写到时才复制。
- 没有额外的引擎进程、没有额外的局，种子集合与偏移完全不变。

### 4. 测试（全部在 Mac，≤ 2 个游戏进程）

- `python/test_growable_pool.py`（新增，8 项，1.6 s，真进程 + 模拟内存读数）：
  开工 1 个 worker、内存回升后扩到 4；内存再跌回 1 GiB 也不缩；未实测 RSS 保持开工宽度；
  后台预留 30 GiB 时上限压到 3；worker 抛异常能传回父进程；**worker 被硬杀（`os._exit`，
  模拟 OOM）时父进程报错而不是永久挂住**；准入公式的边界（未知内存、floor=1、`background`
  扣减）。
- `python/test_curriculum_queue.py`：87 过 1 挂 —— 挂的是
  `test_pack_memory_limit_matches_build_plan_per_builder`（6 != 7），在 `origin/selinc` 上同样挂，
  与本任务无关。新增两项：组命令同时带 `--initial-workers` 与 `--workers`；固定宽度路径不变。
- `python/test_curriculum_pipeline.py` 13 项全过；`python/test_qc_resilience.py` 里与 `command_for`
  相关的 10 项（`bad_group_is_saved_and_next_group_runs` / `restart_after_partial_group`）全过
  （整份文件要跑 15 分钟以上，与本改动无关，没有全跑）。
- 引擎冒烟（真进程、真对局，L7 `day_l7_z13_eval`，2 局）：

  | 运行 | 参数 | 结果 |
  |---|---|---|
  | A | `--workers 2 --initial-workers 1 --grow-interval 5` | `2 games on 1 workers`，2/2 完成，64 s，2 胜 |
  | B | `--workers 2`（固定宽度） | `2 games on 2 workers`，2/2 完成，40 s，2 胜 |
  | D | 同 A，最终版（私有队列派活） | `2 games on 1 workers`，2/2 完成，69 s，2 胜 |

  A / B / D 的 `summary.json` 逐项相同（2 局 2 胜、`mean_final_wave 30.0`、每局 22.0 次覆盖、
  221.5 次搜索），说明换池子没有改结果，只是并发形状不同（单 worker 串行 64–69 s，双 worker 40 s）。
  `games.jsonl` 恰好 2 行，`config.json` 记下了新参数。macOS 没有 `/proc`，所以 A / D 不会真的
  扩容 —— 扩容路径由 `test_growable_pool.py` 用真进程覆盖，引擎路径由 A / B / D 覆盖。

### 5. 部署 `scripts/qc_deploy14m.sh`（未执行，交主控）

- 只换 4 个文件：`scripts/search_play.py`、`scripts/curriculum_queue.py`、新增
  `scripts/growable_pool.py`，外加它的单测 `python/test_growable_pool.py`（REV `d1053e84`；
  单测进生产只为了预检能在本机跑一次真进程的池子，不参与采集）。
- `start_qc` 照抄台式机最新 deploy14l：`--extra-manifest` 由 `qc_deploy14l.mode` 决定
  （`deckcells` 才加），未激活 / 回滚分支与现网一致；`stop_qc` 按台式机版本直接
  `tmux kill-session -t =qc`，不排空，并额外清掉本工作树遗留的 `search_play` 子进程。
- **需要重启一次采集**：控制器把新参数写进组命令，重启后第一组生效；state / offsets / 账本
  全部保留，被打断的那一组按既有合同重采。后台训练 / 打包不排空。
- 预检（便宜、不跑引擎）：`search_play.py` 与控制器 sha256 必须属于 {deploy14l, deploy14m}；
  4 个新文件与 REV 的 blob 哈希一致（并检查 blob 里有 `--initial-workers` / `growth_args` /
  `target_workers`，因为预检发生在替换之前）；`ast.parse`；两个 `--help` 里出现新参数；
  task137 面板 144 条计数；替换后再跑一次 `python/test_growable_pool.py`（1.3 s，trivial 作业）。
- 回滚：脚本内失败自动恢复备份并重启；手动 `qc_deploy14m.sh --rollback` 取最新
  `~/pvz-logs/deploy14m_*` 备份恢复 4 个文件、删 `qc_deploy14m.mode`、重启。

### 6. 边界与遗留

- `pipeline.foreground_workers` 仍记开工宽度，所以评估的**进程数**上限按开工宽度算；内存上限
  由评估自己那次 `worker_count(MemAvailable)` 兜底 —— 扩容与评估都不会越过当时的可用内存，
  但两者同时进行时进程总数可能短暂超过 `--workers`。真出现再让控制器读 `search_play` 的实时
  宽度（需要在 out 目录留一个宽度文件），本任务不做。
- 扩容只加不减：如果一组开工时内存很宽、中途打包起来，池子不会缩，靠 `MemAvailable` 不再
  允许新增来兜底；已经在跑的 worker 不会被杀（这是任务书的要求）。
- macOS 上 `mem_available_bytes()` 返回 `None`，所以本机跑永远是固定宽度；生产是 Linux。


