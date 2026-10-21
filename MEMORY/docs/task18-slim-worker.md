# task18（主控 2026-10-21 下发给 Mac CodeBuddy / DeepSeek）：采集 worker 内存瘦身

先读仓库根目录 `AGENTS.md`（优先级最高），再读本文。

## 背景
台式机 WSL 只有 24 GB（宿主 32 GB），训练（约 11 GB）与采集同时跑时可用内存掉到 2 GB 以下、swap 满，已经两次差点 OOM。
实测（2026-10-21，台式机，`scripts/search_play.py` 16 进程采集）：**每个 worker 的 Python 进程约 650 MB**（`python3 -c from ...`），
外加每个 `pvz-portable` 约 210 MB；16 个 worker 合计约 14 GB——采集才是最大的内存用户，比训练还多。
目标：在**结果逐位不变**的前提下，把每个 worker 的 Python 常驻内存降到 250 MB 以下（16 个 worker 省出约 6 GB）。

## 要做的
工作区：`git -C /Users/newbiexvwu/PvZAgent worktree add .claude/worktrees/slimworker -b slimworker origin/pvz-env`，只在这里改代码、单独编译
（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。
主仓库工作区与 `build/` 不许碰（Mac 正式采集在用）。**同时最多 4 个游戏进程**。不要 `cd` 进别的工作树（用绝对路径 / `git -C`）。

1. **先测量**：用正式采集的参数（`artifacts/hard_mac/round15_flags` + `--student-candidates 2 --dagger-beta 0`，学生 v33（HF `transfer/v33_model.pt`，下到工作树的 `artifacts/student/v33/model.pt`；`$HF_TOKEN` / `$PVZ_HF_REPO` 在 `~/.zshrc`），
   清单 `artifacts/stage4/day_panel.json`，任务 `day_l9_train_d7`，`--seed-offset 60000 --seeds 8 --workers 2`，`--frames` 指向工作树里的临时目录）跑一次，
   记录每个 worker 进程的 RSS 随时间的峰值，并弄清 650 MB 由什么组成（torch 导入、学生 / 价值网络 / 静态价值网络各自、numpy 缓冲、每局档案缓存……）。
   可以用 `tracemalloc`、`psutil`、逐步注释导入的方式，结论写进报告。
2. **再瘦身**：常见手段（按实测挑，不要猜）：worker 里不需要的模块不导入；模型只在用到的进程里加载；用 `fork` 让只读权重在父子进程间共享页；
   局内累积的大数组及时释放；`torch.set_num_threads(1)`（如果还没设）。**不改搜索 / 标签 / 采集的任何逻辑与默认值**。
3. **逐位验证**：改前改后同一组种子（上面 8 局）的 `games.jsonl` 与每局 `frames.npz` / `meta.json` 内容必须完全一致（写个比对脚本，忽略纯计时字段）。
4. 跑现有测试 `python3 -m pytest python -q`，全部通过（Mac 上 `test_dagger_experiment_runner` 的内存暂停测试原本就因平台失败，注明即可）。

## 交付
- 分支 `slimworker` 本地提交，可以 `git push origin slimworker`；不推 pvz-env、不合并。
- 报告追加到本文末尾“结果”一节：650 MB 的组成、改了什么、改前改后每 worker RSS 峰值、逐位比对结果、测试数。
- 有疑问写在本文末尾“问题”一节，按最佳判断继续，不要停下来等。

## 不要做的
- 不改默认值、不改搜索与标签逻辑、不训模型、不写手工规则。
- 不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。

---

## 结果（2026-10-21，Mac，分支 `slimworker`）

### 0. 测量方法

三把尺子，数字全部来自本机实测（工作树 `slimworker`，学生 v33，`artifacts/hard_mac/round15_flags`，
`--student-candidates 2 --dagger-beta 0`，`artifacts/stage4/day_panel.json` / `day_l9_train_d7`）：

- **RSS**（`ps` / psutil 采样，0.2 s 一次，记每个 pid 的峰值）—— 任务书里 650 MB 用的就是这把。
- **private / dirty**（`vmmap` 区域表 DIRTY 列求和）—— 排除共享库映射，看进程真正拥有的内存。
- **tracemalloc / gc** —— 看 Python 对象层面到底留了什么。

一次"正式参数"的 8 局：种子 1826000–1826007（见"问题"第 1 条），2 worker，每局约 1281 个决策 / 91 s。

### 1. 650 MB 由什么组成

**启动阶段**（单进程，逐段 import，private 与 RSS 同时记）：

| 阶段 | private (dirty) | RSS |
|---|---|---|
| 解释器 | 10.4 MB | 22.7 MB |
| + numpy | 20.8 MB | 34.8 MB |
| **+ torch** | **156.7 MB** | **213.2 MB** |
| + 项目模块（pvz_env / event_env / render_episode / value_net / rollout_policy / tasks） | 157.2 MB | 213.3 MB |
| + 价值网络 | 158.0 MB | 215.2 MB |
| + 静态价值网络 | 158.7 MB | 215.9 MB |
| + 学生 | 158.8 MB | 216.3 MB |
| + 学生 C++ 导出 | 159.6 MB | 217.0 MB |

结论很干脆：**静态部分 160 MB private / 217 MB RSS，其中 136 MB 是 torch 本身**；
三个网络（学生 + 价值 + 静态价值）加导出**一共 2.4 MB**，numpy 10 MB，项目代码 0.5 MB。
把网络拆开看没有意义 —— 大头是 `import torch`，不是权重。

**一局之内**（同一进程，day_l9_train_d7 种子 1826000，1281 决策，91 s）：

| 时点 | RSS |
|---|---|
| reset 前（import + 三个网络） | 242 MB |
| 对局中 | 265 → 396 MB（随局面推进线性上升） |
| `write_frames` 写档案 | +50 MB（395.8 → 445.8） |

对局内那 150 MB 的组成：

- **档案帧列表 17.5 MB**（1281 帧，序列化后 2.76 MB JSON，驻留的是 dict/tuple 的 Python 开销）。
- **对局活对象约 57 MB**：终局时 tracemalloc 活着的 Python 对象 89.2 MB，其中 31.8 MB 是
  importlib 的模块状态（属静态部分），剩下是帧、目标值与搜索状态。
- **malloc 高水位约 70 MB**：终局时 MALLOC zone 保留 167.7 MB、实际分配 98.6 MB、碎片 0%。
  一局里 Python 层要解析约 460 MB 的 JSON（1281 次 `step` × 9–20 KB + 4371 个 rollout 叶子 +
  508 次 `branch_snapshot`，后者每次约 30 份完整观测），分配器把高水位留下了。
- **不是泄漏**：终局时整局只剩 **2** 个完整观测 dict 活着（用 `gc.get_referrers` 数过）；
  tracemalloc 把 10.7 MB 归到 `EventWaitEnv.step` 只是说明那批分配发生在那里。

**另一个进程**：每个 worker 还有一个 `pvz-portable`，本机实测 **237 MB**（台式机 210 MB），
不在这 650 MB 里，本任务没动它。

**所以 650 MB ≈ 213 torch + 3 网络 + 100 对局活对象 + 70 分配器高水位 + 50 写档案**。
Mac 实测峰值 454 MB、台式机 650 MB，差的约 200 MB 就是 glibc 比 macOS libmalloc 留得更多的那部分。

### 2. 改了什么（`scripts/search_play.py`）

1. **`_load_nets()` 从 `_init_worker()` 里拆出来**：只负责 import torch + 加载三个网络 + 导出学生。
   `_init_worker()` 发现 `_W["fast"]` 已存在就跳过加载（父进程预载过、子进程继承了页）。
2. **父进程预载 + `fork` 启动方式**（平台支持 fork 时；否则原样 `spawn`，行为与改动前一致）。
   父进程在 `main()` 里加载一次网络再建池，16 个 worker 共享这一份 torch 与权重页。
3. **`_freeze_startup()`：`gc.collect()` + `gc.freeze()`**，父进程 fork 前一次、每个 worker 初始化后一次。
   这条是本次的关键：只 fork 不 freeze 时，GC 每一轮都遍历继承来的整个堆，
   每个被摸到的对象头都是一次写时复制，共享页很快被子进程私有化（见第 3 节的两行对照）。
4. **`_release_freed_memory()`：每局写完档案后 `malloc_trim(0)`**（只有 glibc 有，Mac 上是 no-op）。

搜索 / 标签 / 采集逻辑与默认值一行未动；`--frames` 的编码函数（`encode_frames` / `features` /
`featurize_numpy` / `target_values`）一行未动，档案里的 `encoder` 哈希不变。

### 3. 改前改后每 worker RSS 峰值

8 局、2 worker、同一组种子、同一个采样器（0.2 s 一次，取每个 pid 的峰值）：

| 版本 | worker 峰值 (MB) | 两 worker 之和峰值 | 父进程 |
|---|---|---|---|
| 改前（spawn，各载一份） | **454.3 / 377.0** | 851.2 | 21 |
| 只 fork（未 freeze） | 392.4 / 322.8 | 715.2 | 221 |
| fork + freeze | 322.1 / 210.8 | 530.2 | 221 |
| **fork + freeze + trim（最终）** | **304.3 / 247.0** | 551.3 | 221 |

4 worker、最终代码（同样 8 局）：峰值 **288.2 / 210.8 / 199.0 / 191.0**，和 837.6 MB。
对局进行中的常态在 170–210 MB/worker，峰值出现在最长那局写档案的那一下。

父进程那 221 MB 是**一份**共享拷贝（fork 后子进程刚起来时 RSS 只有 14.6 MB，说明没有复制）。

按 16 worker 折算：改前 16 × 454 ≈ 7.3 GB；改后 221 + 16 × 304 ≈ 5.1 GB。
台式机的 650 MB 里 torch 更大、glibc 高水位更高，共享掉的那份也更大，
但**这是折算，不是台式机实测**，别当结论用。

### 4. 逐位比对

新增 `scripts/compare_run_bytes.py`：按 `(arm, seed)` 比对 `games.jsonl` 每一行，逐数组比对每局
`frames.npz`（含 `frames` 里那串 JSON 的逐帧比对）与 `meta.json`，只忽略纯计时字段
（`seconds` / `cuda_seconds`，含帧里 `verification.seconds`）。

    python scripts/compare_run_bytes.py /tmp/slim/baseline /tmp/slim/final_run
    games.jsonl: 8 rows compared
    archives: 8 compared
    identical

改前 vs 改后（2 worker）、改前 vs 4 worker，都是 `identical`。

### 5. 测试

    python3 -m pytest python -q
    1 failed, 228 passed, 12 skipped, 423 subtests passed

唯一的失败是 `test_dagger_experiment_runner.py::...test_low_memory_pauses_and_resumes_only_own_parent_and_child`
—— 任务书里点名的 Mac 平台既有失败（它只 import `dagger_experiment_runner`，与本次改动无关）。

### 6. "改前"的数字从哪来

改前是同一棵树上、未改动的工作区（工作树 commit `5b6b6dcd`，即 `origin/pvz-env`）跑出来的。
改后每个版本都在同一台机器、同一组种子、同一个采样器下重跑，没有复用旧数字。

---

## 问题

1. **任务书的 `--seed-offset 60000` 越界。** `day_l9_train_d7` 只有 8192 个种子（1820000–1828191），
   60000 切出来是空列表，`search_play` 直接打印 "0 games on 2 workers" 退出。
   按最佳判断改用 **6000**（种子 1826000–1826007），8 局。任务书里的 60000 应为笔误。
2. **250 MB 这个门槛没完全达到。** 常态 170–210 MB/worker 已低于它，但最长那局的峰值是 304 MB。
   峰值里约 50 MB 是 `write_frames`：它要为整局 1281 帧建 `boards` 列表（18 MB）再 stack（又 18 MB）再压缩。
   要再往下压只能改 `write_frames` 的写法（流式写），而档案里存的 `encoder` 哈希是
   `encode_frames` 的源码哈希 —— 动它等于让所有历史档案的 `encoder` 对不上，所以没动。
   要继续的话这是下一个目标，代价是"档案编码函数冻结"这条约束得重新谈。
3. **`malloc_trim` 在本机无法验证。** macOS 没有 glibc（`malloc_zone_pressure_relief` 实测不归还任何东西），
   所以第 4 条改动在这里是 no-op，对台式机 glibc 的效果**没有测过**。它只可能让 RSS 更低、不会影响结果；
   如果台式机上量出来没变化，说明那 200 MB 差额不是空闲堆，而是活对象/碎片，删掉它即可。
4. **墙钟时间这次没法比。** 本机同时跑着主仓库的正式采集（10 worker，load average 20），
   8 局耗时在 371–412 s 之间波动。改动本身只在 fork / 初始化 / 每局结束时各多一次调用，
   每决策路径没有增加任何工作。
5. **RSS 会把共享页重复计数。** 子进程的 RSS 里包含它已经摸过（因此被写时复制）的那部分父进程页，
   所以 fork 之后"每 worker RSS"不等于"这台机器多花了多少内存"。真正省下来的是
   **torch 那一份（213 MB）从 16 份变成 1 份**，加上每个 worker 少摸约 150 MB 的继承页。
   要看机器总量请用各进程 private 之和，不要用 RSS 之和。
