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

**一局之内**（同一进程，day_l9_train_d7 种子 1826000，1280 决策）。这一节在第二轮重测过：
第一轮用的是 fork 之前创建的 `psutil.Process()`，它在子进程里报的是**父进程**的 task，整局数字是平的；
每次新建 `Process()`（以及外部 0.2 s 采样器）才得到下面这条曲线。用 `vmmap -summary` 的 physical
footprint（不含共享库页）：

| 决策 | footprint | 其中 Malloc Large (empty) |
|---|---|---|
| 200 | 66.8 MB | 7.9 MB |
| 800 | 74.4 MB | 10.2 MB |
| 1000 | 118.8 MB | 52.2 MB |
| 1200 | 143.3 MB | 72.3 MB |
| 1280（终局） | 181.0 MB | 107.8 MB |
| 写完档案 | 215.6 MB | 132.0 MB |

对局内那段增长的组成（第一轮的归因是错的，这里是重测后的）：

- **不是活对象，也不是帧列表。** 终局时 tracemalloc 活对象只比第 200 决策多 2 MB；帧列表按 JSON 行
  存也只有 2.76 MB。涨的是分配器的**空区域**（Malloc Large (empty)：已经 free、但 macOS 不归还的大块）。
  所以 **worker 的 RSS ≈ 本局出现过的最大一次同时分配**，而不是"整局累计分配了多少"。
- **最尖的那一下是价值网络的前向。** 终局候选集约 240 帧 × 122 通道，`_values` 一次 `torch.stack`
  成全批再算，单次 12 MB 起步、候选数越大越贵，而这个高水位永不归还。批次固定成 32 行之后，
  同一局的终局 RSS 从 268 MB 掉到 **125 MB**（第 3 节）。
- **观测本身很小**：终局 `json.dumps(obs)` 只有 16 KB（3 个僵尸、15 株植物），1281 次 `step` 不是问题。
- **写档案**仍是单步最高的一个（+35 MB：`boards` 预分配 14.1 MB + 压缩），但已不在关键路径上。
- **不是泄漏**：终局时整局只剩 **2** 个完整观测 dict 活着（用 `gc.get_referrers` 数过）。

**另一个进程**：每个 worker 还有一个 `pvz-portable`，本机实测 **237 MB**（台式机 210 MB），
不在这 650 MB 里，本任务没动它。

**所以 650 MB ≈ 213 torch + 3 网络 + 一次价值网络全批前向的高水位 + 写档案的 35 MB**。
第一轮把对局内的 150 MB 记在"活对象 / 帧列表"上，第二轮重测证明那是错的：活对象是平的，
抬起来的是分配器的空区域，而制造那个高水位的是一次把整批候选喂给价值网络（见上表）。
Mac 实测峰值 454 MB、台式机 650 MB，差的约 200 MB 是 glibc 比 macOS libmalloc 留得更多的空堆 ——
同一段代码在 Mac 上的峰值是 125 MB（第 3 节），所以那 200 MB 的差额只能是分配器，不是代码。

### 2. 改了什么（`scripts/search_play.py`）

1. **`_load_nets()` 从 `_init_worker()` 里拆出来**：只负责 import torch + 加载三个网络 + 导出学生。
   `_init_worker()` 发现 `_W["fast"]` 已存在就跳过加载（父进程预载过、子进程继承了页）。
2. **父进程预载 + `fork` 启动方式**（平台支持 fork 时；否则原样 `spawn`，行为与改动前一致）。
   父进程在 `main()` 里加载一次网络再建池，16 个 worker 共享这一份 torch 与权重页。
3. **`_freeze_startup()`：`gc.collect()` + `gc.freeze()`**，父进程 fork 前一次、每个 worker 初始化后一次。
   这条是本次的关键：只 fork 不 freeze 时，GC 每一轮都遍历继承来的整个堆，
   每个被摸到的对象头都是一次写时复制，共享页很快被子进程私有化（见第 3 节的两行对照）。
4. **`_release_freed_memory()`：每局写完档案后 `malloc_trim(0)`**（只有 glibc 有，Mac 上是 no-op）。
5. **每局回收 worker 进程**（`Pool(..., maxtasksperchild=1)`，仅 fork 平台）。worker 的 RSS 只增不减，
   一个长期活着的 worker 要替它玩过的每一局都付一次高水位；每局重新 fork 之后，下一局从父进程的
   写时复制基线开始，峰值不再随局数叠加（551 → 384 MB 的和）。
6. **`VALUE_BATCH = 32`：价值网络按固定 32 行分批前向**（原来一次 `torch.stack` 全批）。
   网络逐行打分，批次只影响峰值：固定批次让每次前向形状一致，分配器复用同一块，
   而不是每遇到一个新的候选数就再占一块。**这是本次最大的一笔：277.3 → 124.8 MB（改前 304.3）。**
7. **`BRANCH_SNAPSHOT = 8`：`branch_snapshot` 每次 8 个分支**（原来 128）。模拟器为每个分支回一份
   完整观测，终局那 128 份是 worker 里第二高的同时分配；8 个分支是同样的工作在同样的顺序上，
   只是多几次往返（实测 301.9 → 277.3 MB，耗时不变）。

搜索 / 标签 / 采集逻辑与默认值一行未动；`--frames` 的编码函数（`encode_frames` / `features` /
`featurize_numpy` / `target_values`）一行未动，档案里的 `encoder` 哈希不变。第 5–7 条改的只有
**进程生命周期与批次大小**：同样的种子跑出逐位相同的 `games.jsonl` / `frames.npz` / `meta.json`（第 4 节）。

### 3. 改前改后每 worker RSS 峰值

8 局、2 worker、同一组种子、同一个采样器（0.2 s 一次，取每个 pid 的峰值）：

| 版本 | worker 峰值 (MB) | 两 worker 之和峰值 | 父进程 |
|---|---|---|---|
| 改前（spawn，各载一份） | **454.3 / 377.0** | 851.2 | 21 |
| 只 fork（未 freeze） | 392.4 / 322.8 | 715.2 | 221 |
| fork + freeze | 322.1 / 210.8 | 530.2 | 221 |
| fork + freeze + trim | 304.3 / 247.0 | 551.3 | 221 |
| + 每局回收进程 | 301.9 / 213.5 | 383.6 | 221 |
| + `BRANCH_SNAPSHOT=8` | 277.3 / 217.8 | 402.8 | 221 |
| **+ `VALUE_BATCH=32`（最终）** | **124.8 / 123.9** | **206.9** | 221 |

最终版 8 局各占一个进程（每局回收），所以是 8 个峰值：**124.8 / 123.9 / 122.9 / 112.1 / 109.6 /
98.9 / 94.6 / 92.4 MB** —— 每一个都在 250 MB 以下，最差的离门槛还有 125 MB。
4 worker、同样 8 局、最终代码：**124.7 / 122.2 / 120.8 / 120.5 / 108.9 / 106.6 / 99.7 / 97.0 MB**
（四个之和峰值 367.0 MB，耗时 213.9 s）。worker 数与峰值无关了 —— 每个 worker 只玩一局。

最后两行是阶梯式的：只改批次不动逻辑，峰值就掉 150 MB，说明高水位是"一次同时分配多少"，
不是"一局总共分配多少"。

父进程那 221 MB 是**一份**共享拷贝（fork 后子进程刚起来时 RSS 只有 14.6 MB，说明没有复制）。

按 16 worker 折算：改前 16 × 454 ≈ 7.3 GB；改后 221 + 16 × 125 ≈ 2.2 GB。
台式机的 650 MB 里 torch 更大、glibc 高水位更高，共享掉的那份也更大，
但**这是折算，不是台式机实测**，别当结论用。

> 台式机已经实测过了（见下面的"台式机复核"）：改前每 worker RSS 峰值 594.9 / 590.5 MB、
> 私有脏页 393.7 / 391.5 MB；改后每 worker 私有脏页 43–62 MB、PSS 143–181 MB，
> 逐位一致。折算的方向对，绝对数要看那一节。

### 4. 逐位比对

新增 `scripts/compare_run_bytes.py`：按 `(arm, seed)` 比对 `games.jsonl` 每一行，逐数组比对每局
`frames.npz`（含 `frames` 里那串 JSON 的逐帧比对）与 `meta.json`，只忽略纯计时字段
（`seconds` / `cuda_seconds`，含帧里 `verification.seconds`）。

    python scripts/compare_run_bytes.py /tmp/slim/baseline /tmp/slim/slim_run
    games.jsonl: 8 rows compared
    archives: 8 compared
    identical

比对的是**改前那棵树**（工作树 commit `5b6b6dcd`，即 `origin/pvz-env`）在同一组 8 个种子上跑出的结果。
中间每一版（只 fork / fork+freeze / 每局回收 / 分支批次 8 / 价值批次 32）都单独跑过同一个脚本，
全部 `identical` —— 包括只改 `VALUE_BATCH` 的那一版，说明按行分批前向对每一行的分数没有任何影响。

### 5. 测试

    python3 -m pytest python -q
    1 failed, 231 passed, 12 skipped, 423 subtests passed

唯一的失败是 `test_dagger_experiment_runner.py::...test_low_memory_pauses_and_resumes_only_own_parent_and_child`
—— 任务书里点名的 Mac 平台既有失败（它只 import `dagger_experiment_runner`，与本次改动无关）。

### 6. "改前"的数字从哪来

改前是同一棵树上、未改动的工作区（工作树 commit `5b6b6dcd`，即 `origin/pvz-env`）跑出来的。
改后每个版本都在同一台机器、同一组种子、同一个采样器下重跑，没有复用旧数字。

---

## 台式机复核（Linux，fork + glibc，2026-10-21）

### 1. 环境与跑法

| | |
|---|---|
| 机器 | 20 核 / 24 GB，load average 13–22（正式采集 16 worker + 训练同时在跑；没有动任何 tmux 队列） |
| 代码 | `origin/slimworker` @ `67c12e5` → 工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/slimworker` |
| 改前 | `/home/newbiexvwu/PvZ-search/.claude/worktrees/rowpool` @ `46e34cb`（= `origin/pvz-env`），只读，用它自己那份 build |
| 编译 | `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4`，GCC 15.2.0，产物 5 256 368 B（与 rowpool 那份同样大小；sha 不同只是因为编译路径写进了二进制） |
| Python | `/home/newbiexvwu/.venvs/ml/bin/python3` 3.14.4，torch 2.14.0+cu132，numpy 2.5.2 |
| 网络 | student v33、`artifacts/value_net/v3`、`artifacts/hard/value_branch1` |
| 任务 | `day_l9_train_d7`，种子 1826000–1826007（8 局），2 worker，`--verify-samples 9 --student-candidates 2 --dagger-beta 0`，带 `--frames` |

采样器是独立进程（不碰被测代码），0.25 s 一次读 `/proc/<pid>/status`（VmHWM）与 `smaps_rollup`
（Pss / Pss_Anon / Private_Dirty）。两条坑写在脚本注释里：

- 角色要按 `/proc/<pid>/comm` 判，不能按 cmdline —— fork 出来的 worker 继承了 argv，而 argv 里有
  `--executable .../pvz-portable`，第一版把每个 worker 都当成了模拟器（于是 `workers=[]`）；
- 子进程要从 `/proc/<pid>/task/*/children` 全线程找。只看主线程那份的话，第一局之后由
  multiprocessing 的 handler 线程 fork 出来的 worker 全都看不见（第二版量到的"峰值"只覆盖第一局）。

### 2. 逐位比对

    python scripts/compare_run_bytes.py /tmp/slim_out_base2 /tmp/slim_out_slim3
    games.jsonl: 8 rows compared
    archives: 8 compared
    identical

跑了两遍（第一遍 base/slim，第二遍 base2/slim3），两遍都 `identical`：8 行 `games.jsonl`、
8 个 `frames.npz`（逐数组 + 帧 JSON）+ `meta.json`。决策数与胜负也对上
（1235 / 1280 / 1170 / 1152 / 1099 / 1176 / 1169 / 1182，7 胜 1 负）。

### 3. 每 worker 内存

Linux 上"每 worker RSS"和 Mac 不是一回事：fork 出来的子进程 RSS 里包含它和父进程共享的那些页
（父进程带着 torch，RSS 552.6 MB），所以 RSS 会显得和父进程一样大。三个口径一起给：

| 口径 | 改前（spawn，2 worker） | 改后（fork，8 局 = 8 个 worker） |
|---|---|---|
| 每 worker RSS 峰值（VmHWM） | **594.9 / 590.5 MB** | **394.3 / 393.9 MB**（其余 392.4 / 391.9 / 380.9 / 380.2 / 374.8 / 373.9） |
| 每 worker PSS 峰值 | **401.7 / 399.1 MB** | **181.0 / 162.3 MB**（其余 143.6–162.3） |
| 每 worker 私有脏页峰值（Private_Dirty） | **393.7 / 391.5 MB** | **61.7 / 51.7 MB**（8 局 42.9–61.7） |
| 模拟器 RSS 峰值 | 208.8 MB | 208.9 MB（没动它，符合预期） |
| 父进程 | ~30 MB（spawn 不在父进程载网络） | 552.6 MB RSS / 134.9 MB PSS（全轮就这一份 torch） |
| 墙钟（8 局，2 worker） | 693.1 s | 609.2 s |
| 每局均值 | 168.3 s | 147.8 s |

怎么读这张表：

- **任务书的 650 MB 在台式机上复现了**：改前两个 worker 的 RSS 峰值 594.9 / 590.5 MB，
  第一遍（load 更低）是 650.5 / 640.5 MB —— 650 就是 spawn worker 各自 import 一遍 torch。
- **RSS 那一行不能直接拿去和 250 MB 比。** fork 之后 worker 的 RSS 是从父进程继承来的
  （父 552.6 MB），它量的是"这个进程地址空间里驻留了多少页"，不是"它自己花了多少"。
  一个 8 局全跑完的 worker 私有脏页只有 43–62 MB，PSS 143–181 MB。
- 能横向比的是 **PSS 402 → 181 MB（-55%）** 与 **私有脏页 394 → 62 MB（-84%）**。
  私有脏页才是"多开一个 worker 这台机器要多掏多少"：改前每个 worker 394 MB（torch 每人一份，
  PSS 里几乎全是私有），改后 torch 只剩父进程那一份（134.9 MB PSS），每 worker 43–62 MB。
- 整棵树同一时刻的 PSS 之和（对局中途实测快照）：父 134.9 + 两个 worker 121.3 / 122.1 +
  两个模拟器 196.2 × 2 = **781 MB**；改前光是两个 worker 的 PSS 就是 401.7 + 399.1 = 801 MB，
  还没算模拟器。
- 按 16 worker 折算：改前 torch 的私有部分 16 × 394 ≈ 6.3 GB；改后 135 MB + 16 × ~60 ≈ 1.1 GB。

### 4. 墙钟

两遍之间没有一致的差异：第一遍 640.3 s（改前）vs 669.0 s（改后），第二遍 693.1 vs 609.2。
这台机器 load 在 13–22 之间跳（正式采集 16 worker + 训练），同一份代码自己两遍就差 8%。
按每局均值看也是交叉的：改前 157.2 / 168.3，改后 163.0 / 147.8。
结论：**在这个负载下量不出改动的耗时影响**；能确定的是每决策路径的工作量没变（分批前向的总行数不变），
每局只多一次 fork 与一次模拟器启动。

### 5. 和 Mac 的数字为什么对不上

Mac 上每 worker 峰值 124.8 MB，这里 PSS 181 MB、RSS 394 MB —— 三个数不是同一件事：

- Mac 的 124.8 是 psutil 的 RSS，但 macOS 不把子进程没摸过的继承页算进它的 RSS（父 221 MB，子 124.8 < 父），
  所以它更接近"私有 + 摸过的"这个口径；
- Linux 的 394 是 VmHWM，把共享页算进去了（父进程那 552.6 MB 里的大部分）；
- Linux 的 181 是 PSS，把共享页按使用者摊开 —— 这个才和 Mac 那个数可以并列看。

两边都是"torch 一份 + 每 worker 几十 MB"，差别在分配器和记账方式，不在代码。

### 6. `malloc_trim` 在台式机上已经多余

glibc 的 `malloc_trim(0)` 在台式机上会真的归还（Mac 上那个 no-op 的坑见问题 §4），但它在每局结束时才调一次，
而 fork 平台每局结束本来就回收进程 —— 所以它对**对局内**的峰值没有任何影响，对最终数字也没贡献。
它只在把 `maxtasksperchild` 去掉时才有意义。这一条是读代码得出的，没有单独做对照实验。

---

## 问题

1. **任务书的 `--seed-offset 60000` 越界。** `day_l9_train_d7` 只有 8192 个种子（1820000–1828191），
   60000 切出来是空列表，`search_play` 直接打印 "0 games on 2 workers" 退出。
   按最佳判断改用 **6000**（种子 1826000–1826007），8 局。任务书里的 60000 应为笔误。
2. **250 MB 的门槛最后是靠"固定批次"过的，不是靠流式写档案。** 第一轮判断峰值在 `write_frames`，
   重测后不成立：写档案只占 +35 MB，真正的高水位是价值网络一次全批前向（终局 ~240 帧 × 122 通道）。
   所以档案编码函数一行没动，`encoder` 哈希不变。**代价**：`VALUE_BATCH` 与 `BRANCH_SNAPSHOT` 是
   显式的常量，将来如果候选集更大（更多卡、更大棋盘），这两个数要跟着复核 —— 它们现在约束的是
   峰值，不是正确性。
3. **每局回收进程会多花每局一次的模拟器启动（~0.7 s）。** 8 局实测 375.5 s vs 改前 366.6 s，
   差在噪声内，但 16 worker × 几百局时这笔钱要算进去；不想要的话把 `maxtasksperchild` 去掉，
   峰值会按改动前量到的叠加幅度回到 ~150–180 MB（仍在 250 以下，只是会随局数缓慢叠加，
   这一条是估计，没有在最终代码上重测过）。
4. **`malloc_trim` 在本机无法验证。** macOS 没有 glibc（`malloc_zone_pressure_relief` 对 100 MB 已释放的
   大块实测返回 0 字节，goal 传 0 / 1 MB / 2^62 都一样），所以第 4 条改动在这里是 no-op，
   对台式机 glibc 的效果**没有测过**。它只可能让 RSS 更低、不会影响结果。
5. **墙钟时间这次没法比。** 本机同时跑着主仓库的正式采集（10 worker，load average 20），
   8 局耗时在 366–412 s 之间波动。改动本身只在 fork / 初始化 / 每局结束时各多一次调用，
   每决策路径没有增加任何工作（分批前向的总行数不变）。
6. **RSS 会把共享页重复计数。** 子进程的 RSS 里包含它已经摸过（因此被写时复制）的那部分父进程页，
   所以 fork 之后"每 worker RSS"不等于"这台机器多花了多少内存"。要看机器总量请用各进程 PSS /
   私有脏页之和，不要用 RSS 之和（台式机实测把这笔账算清楚了，见问题 §8 与"台式机复核"§3）。
7. **`psutil.Process()` 在 fork 出来的子进程里会报父进程的内存。** 第一轮的"对局内数字全平"就是它：
   在 fork 之前建的 `Process` 对象带着父进程的 task 句柄，子进程里 `memory_info()` 一直返回父进程的值。
   要么每次新建 `psutil.Process()`，要么从外面采样。这一条差点让整轮的归因停在"帧列表"上。
8. **"每 worker RSS 250 MB"这条指标在 Linux + fork 上不成立，要换成 PSS / 私有脏页。**
   fork 出来的 worker 的 RSS 天生包含父进程那 552.6 MB 里的共享页，实测 394 MB —— 它反映的是
   "地址空间里驻留了多少"，不是"这个 worker 花了多少"。同一份代码在 Mac 上是 124.8 MB（macOS 不把
   没摸过的继承页算进子进程 RSS），两边不能直接比。台式机上真正可比的是 PSS 181 MB 与私有脏页 62 MB，
   改前分别是 402 / 394 MB。**下次派任务时，Linux 侧的门槛请写成"私有脏页 / PSS"，别写 RSS。**
9. **台式机上 `malloc_trim` 已经是多余的。** glibc 会真的归还，但它每局结束才调一次，而 fork 平台每局
   结束本来就回收进程；对局内峰值它一分钱没省。要留着它只在去掉 `maxtasksperchild` 时才有意义。
