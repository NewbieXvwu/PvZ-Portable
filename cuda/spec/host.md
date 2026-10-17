# CUDA 主机参考构建（`cuda/SimHost.cpp`）

阶段 1 验收测试的承重件。同一批双编译源（`SimImpl.inc` / `SimTick.inc` / `NetForward.inc` /
`Rollout.inc`）由 nvcc 编成设备侧（一线程一泳道），也由 g++ 编成这里的**单泳道 CPU 参考**：

- 主机构建与 GPU 构建不一致 ⇒ GPU 的 bug；
- 主机构建与原始模拟器不一致 ⇒ 机制的 bug；
- 两边都不一致 ⇒ 参考本身有问题，先看本文件的勘误清单。

它同时是 ASan 靶子：SoA 按**精确大小**分配（`IntSlotsTotal` = 2518 个 int32、
`FloatSlotsTotal` = 778 个 float，道距 1），越界的槽位下标会直接落进 red zone 并报出行号，
不需要一次 GPU 往返，也不受 nvcc 内联对归因的干扰。

---

## 1. 构建

```bash
g++ -fsanitize=address -g -O1 -std=c++17 -I. -o build/sim_host cuda/SimHost.cpp
```

从仓库根目录执行。不要用 `-j` 高于 4：本机在跑正式实验（`pgrep -af queue` 可见）。
峰值 RSS 实测 **125 MB**（其中 97.8 MB 是激活缓冲，见勘误 6），远低于 6 GB 上限。

必须带 `-I.`：`.inc` 之间用 `#include "SimImpl.inc"` 相对互引，而 `SimHost.cpp` 自身写的是
`"cuda/SimTick.inc"`，与 `cuda/RolloutCuda.cu` 的构建方式一致。

两个编译器差异（都不是本文件能修的，修在共享头里才对，见勘误 1）：
`cuda/NetCore.h` 裸写 `size_t` 却只包含 `<cstdint>`，nvcc 从自己的预包含里免费拿到全局
`size_t`，g++ 拿不到 —— 所以 `SimHost.cpp` 第一件事是 `#include <cstddef>`，把顺序问题解决在
自己的翻译单元里，不动共享头。

## 2. 运行与参数

```bash
./build/sim_host --roots artifacts/cuda/corpus_smoke/roots.cudaroot \
                 --net artifacts/cuda/student/v14.pvzrn --per-root 8 --horizon 2500
```

默认值与 `cuda/RolloutCuda.cu` 逐项对齐，因此两条命令跑的是**同一份工作负载**：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--roots` | `artifacts/cuda/corpus_smoke/roots.cudaroot` | PVC4 语料：一个 `LevelInfo` + N 个 `RootRecord`，每条记录前 12 字节标签 `{kMagic, kVersion, kind}`；`supported == 0` 的根在读取时就丢弃（与设备驱动同处过滤，两边根下标才一致） |
| `--net` | `artifacts/cuda/student/v14.pvzrn` | PVZRN002/003 学生权重转储，张量顺序 = `EnvironmentRollout::Load` 的顺序 |
| `--per-root` | 8 | 每个根几个未来，`futureSeed = seed_base + i` |
| `--horizon` | 2500 | 相对 tick；`S_UNTIL_TICK = root.tick + horizon`，与 `InitKernel` 同一式子 |
| `--max-actions` | 3000 | 回合数上限（`RoundTop` 用它判停） |
| `--repeat` | 1 | 整批重放，**不产生新样本**（勘误 7） |
| `--seed-base` | 1600000000 | 取的就是原始模拟器 `scripts/cuda_root_corpus.py` 的 `seed*1000 + future` |
| `--bench full\|sim` | full | `sim` = 完全不前向，只跑 tick 与等待回合，与设备驱动同名参数同义 |
| `--root-limit` | 0（全部） | 只用前 M 个根，做小样本排查用 |
| `--poison <float>` | 0 | 诊断开关：把**标量激活向量**预置成该值（board/平面 halo 保持 0），用来证明 `BuildFeatures` 没清零（勘误 3） |
| `--pairs FILE` | 空 | 显式工作负载，一行一条泳道，行序即泳道序：`<root> <future>`，可再接老师的候选着法 `<kind> <packet> <row> <col> [ticks]`（kind 0=学生立刻决定、1=种这张牌、2=等待 ticks）。只有两个数时语义与阶段 1 的语料文件完全相同，所以旧 pairs 照读。解析在 `cuda/PairList.h`（设备与主机同一份） |
| `--leaf-out FILE` | 空 | 每条泳道一个 `PVZLEAF1` 块（5668 float：board 92×60 + 113 标量 + 35 上下文），与 `RolloutCuda.cu --leaf-out` 同格式，因此两个文件可逐块对照 |
| `--root-features FILE` | 空 | 只导入不推演：把每个根**在第一个 tick 之前**的编码写出来就退出，`scripts/cuda_leaf_compare.py --root-features` 的靶子 |

退出码：跑完且没有根被 `LoadRoot` 拒收 = 0，否则 1。所有文件与参数错误都走 `Fail`，
打印 `sim-host: <原因>` 后退 1，不会崩。

数据流与设备侧一一对应：`ReadCorpus` ↔ 驱动的 `ReadCorpus`，`ReadNet` ↔ `ReadNet`
（主机不重排 trunk，因为主机跑的是 `NetForward.inc` 的逐泳道循环，不是 cuBLAS），
`pvzc::LoadRoot` ↔ `InitKernel` 里的同一次调用，`pvzc::RunRollout(..., useNet = 1)` ↔
`RoundPre` + 批量 trunk + `RunCellPre` + `RoundPost`，`pvzc::StoreResult` ↔ `StoreKernel`。

## 3. 叶子统计对照

### 3.1 主机参考 vs GPU（同一工作负载）

GPU 行引自 `~/pvz-logs/cuda_status.md` 阶段 0 第 4 轮（15253 条/秒那次），
主机行为 `build/sim_host --per-root 8 --horizon 2500` 的 `ALL` 行，12 根 × 8 未来 = 96 条：

| 每次续走均值 | GPU（21504 条） | 主机（96 条） |
|---|---|---|
| ticks | 2523.8 | **2523.8** |
| decisions | 49.5 | **49.5** |
| 前向次数 | 18.5 | **18.5** |
| 种植 | 5.1 | **5.1** |
| 被啃 | 5.8 | **5.8** |
| 击杀 | 3.4 | **3.4** |
| 错误位 | 0 | **0** |
| 叶子结局 | won 0 / lost 224 / ongoing 21280 | won 0 / lost **1** / ongoing **95** |
| 阳光 / 波次 / 剩余修剪机 | 驱动不打印 | 34.1 / 3.40 / 2.62 |

逐格相同（结局一行是同一分布：224 = 1 × 224、21280 = 95 × 224，见勘误 7）。
**这不等于"逐条相同"**：主机侧目前拿不到设备侧的 `RolloutResult` 数组来做逐条比对 ——
驱动有 `--out results.bin`，但阶段 0 的那几次测量没落盘。阶段 1 要做严格比对时，
GPU 侧加 `--out` 一次，主机侧本文件已具备同样的字段（`ticks / decisions / netCalls /
plantsPlanted / plantsEaten / zombiesKilled / result / sun / wave / mowersLeft /
plantedByType[] / rowMinZombieX[]`），逐条 diff 只差一个读取脚本。

主机的逐根表（`--per-root 8 --horizon 2500`，`-fsanitize=address -g -O1`）：

```
scope        rolls     ticks    decis      net    plant    eaten     kill       sun    wave  mower  palive  zalive        w/lost/ongoing   err
root0@3600       8    2505.2     46.9     28.0      4.8      3.4      2.6      40.6    2.00   4.00    5.88    0.88       0/    0/     8     0
root1@4200       8    2544.0     49.5     23.0      4.8      4.6      1.9      25.0    2.00   3.50    5.62    1.88       0/    0/     8     0
root2@5400       8    2545.6     49.0     14.2      4.0      7.4      2.8      15.6    3.00   2.88    4.62    2.12       0/    0/     8     0
root3@6649       8    2516.1     50.1     14.2      6.0      7.4      3.1      15.6    3.00   2.25    5.50    1.88       0/    0/     8     0
root4@10374      8    2454.5     49.1     15.2      5.2      6.2      4.0      37.5    5.00   0.62    6.75    1.62       0/    1/     7     0
root5@12774      8    2541.0     52.6      9.8      5.5      8.1      5.5      31.2    6.00   0.00    9.25    2.38       0/    0/     8     0
root6@3600       8    2538.0     47.4     19.9      4.4      2.6      2.5      37.5    2.00   4.00    6.38    1.50       0/    0/     8     0
root7@4200       8    2526.8     48.8     16.8      5.5      3.1      2.1      28.1    2.00   4.00    7.25    1.88       0/    0/     8     0
root8@5400       8    2541.5     50.0     24.9      5.1      4.2      2.9      59.4    3.00   3.12    6.88    3.12       0/    0/     8     0
root9@7247       8    2514.5     50.8     25.1      4.4      8.4      5.6      43.8    3.75   2.00    4.00    1.62       0/    0/     8     0
root10@7864      8    2531.1     50.4     14.1      5.9      6.9      3.2      25.0    4.00   2.38    8.00    2.00       0/    0/     8     0
root11@9691      8    2526.9     49.0     17.1      5.8      7.2      4.2      50.0    5.00   2.75   10.50    3.75       0/    0/     8     0
ALL             96    2523.8     49.5     18.5      5.1      5.8      3.4      34.1    3.40   2.62    6.72    2.05       0/    1/    95     0
```

没有 0 种植、没有 ticks 远离 horizon、没有负数、没有任何错误位：分布层面是干净的。

### 3.2 主机参考 vs 原始模拟器

`artifacts/cuda/corpus_smoke/corpus.jsonl` 是**原始模拟器自己**对这 12 个根 × 4 个未来
（同样的 `--seed-base`）跑到 2500 tick 的叶子。48 条均值对比：

| 指标 | 原始 | 主机 | 差 |
|---|---|---|---|
| ticks | 2525.6 | 2518.6 | −0.3% |
| decisions | 48.1 | 49.4 | +2.7% |
| 叶子阳光 | 39.1 | 33.9 | −13% |
| 起始波次 | 3.54 | 3.40 | −4% |
| 剩余修剪机 | 4.08 | 2.60 | **−36%** |
| 场上植物 | 10.02 | 6.60 | **−34%** |
| 场上僵尸 | 3.71 | 2.08 | **−44%** |
| 终局 | 0/48 | 1/48 败 | — |

前三项（ticks / decisions / 波次）在同量级，说明 tick 节拍、等待回合与波次调度大体对得上。
后四项是本机参考暴露出来的**机制线索**：主机的草坪比原始的空得更快。但**先别当 bug 记账** ——
两边的计数定义不同（dump 里 `plants` / `zombies` 是 `len(leaf[...])` 观测列表长度，
`StoreResult` 的 `zombiesAlive` 要求 `ZB_ON_BOARD && !(ZB_DEAD|ZB_DYING)`）。
见勘误 8。

## 4. 计时

单线程，本容器与正式实验共享 CPU，同一份 96 条工作负载两次跑出 0.78 与 0.91 条/秒，
所以量级即可：

| 构建 | 条/秒（1 线程） | 备注 |
|---|---|---|
| `-fsanitize=address -g -O1` | **0.78 – 1.04** | 交付构建；48 条 46.3 s，96 条 122.7 s |
| `-O2 -march=native` | 1.77 | 48 条工作负载的**整张逐根表**与 ASan 构建 `diff` 为空 |
| `-O2 -march=native -ffp-contract=off` | — | 同样整张表 `diff` 为空（本样本上 FMA 收缩不改变任何一次决策） |
| `--bench sim`，ASan 构建 | 12.11（每条 6080 tick ⇒ 73.7 ktick/s） | 只跑 tick |
| 已知 CPU 基线 | 90.18 | 原始模拟器 + OpenBLAS 学生前向 |

即参考路径比基线慢约 50–90 倍。原因在勘误 6，是布局代价，不是循环写得慢。
**它是正确性工具，不是快通道**；阶段 0 的吞吐结论只能来自 GPU 构建。

## 5. 勘误清单（本文件的存在暴露出来的东西）

按"会不会污染阶段 1 的判据"排序。1–5 是必须先处理的。

1. **ROLLOUT 协议的"首个动作"两边都没实现。**
   `RolloutRequest::firstActionKind / firstPacket / firstRow / firstCol / firstWaitTicks`
   （`cuda/CudaRollout.h:284`）没有任何调用方填过；为它准备的标量 `S_FIRST_ACTION`
   （`CudaRollout.h:363`）在 `cuda/` 全目录里**只出现在枚举定义那一行**，没人写、没人读；
   `RunRollout` 的签名里也没有它。而原始协议里这是核心一步：
   `src/main.cpp:397` 读 `first`，`:416` 解码，`:437` `isFirst = decisions == 0 && first != "A"`。
   教师评分候选着法要的正是"从指定着法出发续走"。现在的两条路径都是"学生立刻决定"，
   所以**当前所有叶子统计测的是学生的策略，不是候选着法的价值**。阶段 1 之前必须补，
   而且补进 `RolloutBegin` / `RoundTop`（设备与主机共用），不是补进某一个驱动。
   **已修（2026-10-18 第 8 轮）**：`Rollout.inc::RolloutBegin` 消费 `S_FIRST_KIND/PACKET/ROW/COL/TICKS`
   五个泳道标量，并且**绕不开 `RoundTop`** —— 首着的成本、决策计数、`S_PRIOR_*`、以及"本轮新产量"
   的基准都必须是观测过的时钟与已入库产量；跳过 `RoundTop` 会把整局累计阳光当成本轮新产记进
   6000-tick 窗口（实测 income 特征最多翻倍），这是补这条勘误时顺手抓到、也一并记在这里的教训。
   `--pairs` 的行因此扩成 `<root> <future> [kind packet row col [ticks]]`，两侧驱动都在 `LoadRoot`
   **之前**写这五个槽（kind=0 也显式写 0：标量块跨批复用）。
   验证工具 `scripts/cuda_first_action_check.py`：强制种植 + 0 tick 推进是确定性的，
   96 个候选逐 float 与原版 `BRANCH_OBSERVE` 子观测相同、device 与 host 逐字节相同。

2. **`LoadRoot` 漏掉 `F_PROJ_CHILL`，且完全忽略导出的 `RootProjectile::chill`。**
   弹清理循环（`cuda/SimImpl.inc:82-92`）只写了 `F_PROJ_TYPE / _ROW / _AGE / _DAMAGE` 四个，
   第五个 `F_PROJ_CHILL` 没清；导入循环（`:208-222`）也没写它 —— 而导出器是明确填了的
   （`src/CudaExport.cpp:284`：雪豌豆置 1，为了让内核不必认识 `ProjectileType`）。
   于是 `SimTick.inc:298` 那句 `if (lane.i(F_PROJ_CHILL, slot))` 读的是**该槽位的前一个主人**：
   设备上是 `cudaMalloc` 的未初始化内存（实践中是刚分配的零页，但这不是保证），
   主机上是我每次续走整体清零所以看不见。**后果**：(a) 从在飞的雪豌豆导入的局面丢掉减速标记，
   (b) 减速可能从上一局串到这一局。修法两行，且顺手把弹的清理改成和植物/僵尸一样的字段区间循环，
   再加一条"RootRecord 每个字段都必须被 `LoadRoot` 覆盖"的自检（这是本类漏洞的通用防线）。
   **已修**：清理在 `SimImpl.inc:88`，导入 `src.chill` 在 `:225`。

3. **`BuildFeatures` 清 board 不清 scalars，前向读 115 个标量里的 94 个从未被写过的值。**
   `cuda/SimImpl.inc:325-327` 把 92 个 board 通道清干净，随后 `:375-414` 只写：
   索引 0–4（5 个）、每张卡的 `5+kind` 与 `59+kind`（本批卡组 7 张 ⇒ 14 个）、
   `113`/`114`（2 个），共 21 个；而 `NetContext` 的标量支路读满 `net.scalarCount = 115`
   （PVZRN002 无任务上下文）。剩下 94 个是**别人的旧值**。
   主机侧用 `--poison 1.0` 实证了这条依赖：同一份 12 根 × 4 未来工作负载，只把标量向量预置成 1.0
   （halo 不动），叶子统计就变了 —— root1 前向 23.0→21.8、种植 4.8→4.5、被啃 4.8→3.5、
   叶子阳光 25.0→37.5、击杀 1.2→2.0。**决策确实依赖未初始化状态**。
   今天 GPU 与主机对得上，是因为两件事同时成立：驱动给了零页，且一批只有一个卡组（每道写的
   是同一批索引）。一旦混卡组、混任务族，或激活缓冲跨批复用，它就翻脸。
   （这条已在 `~/pvz-logs/cuda_status.md` 末尾列了；现在有了一个能复现它的开关和一组数字。）
   **已修**：`BuildFeatures` 现在先把 `kScalarFeatures` 个标量全清零再写（`SimImpl.inc:446`），
   `--poison` 保留作回归探针 —— 它清的是标量向量本身，与"没清"的那个版本对同一工作负载会给出的
   叶子统计不同，所以要用它验证"改了也没用"才说明清零真的生效了。

4. **`S_STOP_REASON` 是死字段。** 同样只在 `CudaRollout.h:372` 的枚举里出现一次，
   没有写入点，`RolloutResult` 里也没有对应列。于是"这条泳道是被 horizon 停的、
   被 `maxActions` 停的、还是终局停的"两边都说不出来。阶段 1 逐条比对时，
   horizon 截断和动作数上限两种叶子必须分开统计，否则分布会被静默混合。
   顺带：`S_INCOME_HEAD` 也基本是死的（`LoadRoot` 写 0，之后无人读）。

5. **设备构建"编译了"不等于"执行了" `NetForward.inc` 的 trunk 循环。**
   主机全量路径是 `RunRollout(useNet = 1)` → `NetDecide` → `NetEncode`(`Conv3x3`/`Conv1x5`)
   + `NetChoose`(`CellPre`)；设备全量路径是 `RoundPre` + cuBLAS 批量 trunk + `RunCellPre` +
   `RoundPost`，**从不调用 `NetTrunk`/`CellPre`**。这两点差别只在浮点次序（bias 折进累加器起点
   vs `ReluBiasKernel` 事后加；`CellPre` 循环 vs beta=0 的 SGEMM），是设计里说好的。
   但要写进判据：`Conv3x3`/`Conv1x5`/`CellPre` 里的**结构性**错误（下标、通道数、tap 次序）
   只有主机会犯，GPU 永远显示不出来；反过来 `ReluBiasKernel` 的 halo 越界只有 GPU 会犯。
   也就是说这两个方向需要**分别**建立信心，"两边逐格相同"只覆盖机制层，不覆盖批量层的布局。

6. **单泳道主机要为分块布局付 97.8 MB 激活，并且每次访存跨 4 KB。**
   `kNetChunk = 1024` 是 `PlaneIndex`/`VectorIndex` 的硬编码道距，主机也必须开一整块：
   board 42 MB + hidden/scratch 各 29 MB。于是主机前向是访存延迟受限，不是算术受限 ——
   每个 `(channel, cell)` 元素相隔 1024 个 float，一次前向约 320 万次加载落在 42 MB 上，
   空间局部性为零（连 4 KB 页都复用不上）。这就是 1 条/秒 vs 90 条/秒的来源。
   如果之后要让主机参考跑大规模一致性扫描，正解是**主机专用稠密切片**（或把 chunk 变成模板参数），
   不是把循环改写一遍；改循环不改本质。本文件保持设备布局是刻意的：换布局就换语义了。

7. **`--repeat` 是精确重放，不是新样本。**
   `RolloutOf` 用 `slot = (firstRollout + index) % perGame` 映射，`perGame = rootCount × perRoot`，
   所以驱动的 `--lanes 21504 --repeat 224` 跑的仍是那 **96 个**不同的 (根, 未来) 各 224 遍；
   续走对 (根, 未来, horizon) 完全确定，主机实测印证：`--repeat 1` 与 `--repeat 3` 的逐根均值
   逐格相同，只有计数翻倍。GPU 那行 `lost 224 / ongoing 21280` 本身就是证据（224 = 1×224）。
   **结论：阶段 0 的叶子统计有效样本量是 96，不是 21504。** 阶段 1 引用这些分布时必须写 n = 96；
   要真样本就得提高 `--per-root`，而 `--per-root` 提高会直接抬高显存与回合数。

8. **主机与原始的叶子计数定义不同，那 3 个百分比不能算机制 bug，只能算待查项。**
   `corpus.jsonl` 的 `plants` / `zombies` 是观测列表长度；`StoreResult` 的 `plantsAlive` 是
   `F_PLANT_TYPE >= 0` 的槽数，`zombiesAlive` 还额外要求 `ZB_ON_BOARD && !(ZB_DEAD|ZB_DYING)`。
   阶段 1 前要做的是**先把定义对齐，再重跑 §3.2 那张表**。如果对齐后"主机草坪更空"仍然成立，
   那才轮到 `SimTick.inc` 里那几个标着 PROVISIONAL 的常数（`cEatDamage`、`cDeathFadeTicks`、
   `cChewStepX`、`cMowerTriggerX`）和 `SimImpl.inc` 里 `StoreResult` 的存活判据。
   另一个硬缺口：**这份 dump 校验不了种植/被啃/击杀**——48 条记录的 `events` 全是 `{}`，
   因为 CPU 路径根本不发那个块（`scripts/cuda_root_corpus.py` 顶部自己写了原因：
   `src/main.cpp:403-405` 在扣除基线的那段共享尾部之前就读了计数器）。
   这三列现在只有"GPU vs 主机"的自洽，没有 vs 原始的正确性。

8a. **"这个僵尸还算活着吗"有四份互不相同的答案，而本文件写完之前 `CudaRollout.h` 又加了
   `ZB_MOWERED`（`1 << 11`，`PHASE_ZOMBIE_MOWERED`：被修剪机压过、在等压扁动画放完），
   目前四处一个都没提它。** 四处是：`SimTick.inc:96` 的 `ZombieLive()`（`!DEAD && !DYING`）、
   `SimTick.inc:146` 的伤害门控（同上）、`SimTick.inc:605` 的左侧区掩码（同上，但额外要
   `ZB_ON_BOARD`，见该函数注释里自己标的 PROVISIONAL）、`SimImpl.inc:289` 的 `zombiesAlive`
   （要求 `flags & (DEAD|DYING|ON_BOARD)` 恰好等于 `ON_BOARD`，即比前三处更严）。
   这四条不等价：一个 `DEAD|ON_BOARD` 的僵尸在前三处算死、在第四处也算死，但一个
   `DYING` 且没带 `ON_BOARD` 的、或将来带 `MOWERED` 的，各处结论不同。
   §3.2 里"场上僵尸 3.71 vs 2.08"这道 −44% 的缺口，很可能一部分就死在这里而不是死在伤害常数上。
   建议把存活判据收成 `SimCore.h` 里一个函数（设备与主机同一份），然后 §3.2 重跑一次再判常数。

9. **合法性判据只覆盖白天草地，按 AGENTS.md §8 不能直接摊到 50 关任务集。**
   `AnyLegalPlant`(`SimTick.inc:640`) / `CanPlantAt`(`:658`) / `NetPickPacket` 的"有无合法格"
   都是"该格为空"，把原始的 `Board::CanPlantAt(col, row, seed)` 换掉了。白天草地成立；
   泳池（水路要睡莲 16）、雾、屋顶（要花盆 33）、夜间墓地格、以及升级植物必须种在各自底座上
   （香蒲→睡莲）这些情形它全都不知道。它也没有 `SEED_IMITATER` 的替身卡处理
   （原始在 `EnvironmentRollout.cpp:283`）。冒烟语料无所谓，**扩语料之前必须先扩这个判据**。
   另外 `NetPickPacket` 的 `bestPacket` 起点是 `0.0f` 加 `havePacket` 标志，原始是 `-infinity`
   加 `<=` 比较（`:277-282`）：只有当某张牌的 logit 恰为 `-inf` 时两边才会不同，可忽略，
   但值得留一行注释免得后人以为这是抄错。

10. **共享源对 g++ 的可编译性没有守护。** 勘误 1 那类（缺 `<cstddef>`）不是机制问题，
    但它让"主机参考"在无人察觉时死掉 —— 事实上它已经死过一次，就是本任务存在的理由。
    建议在 CI/门禁里加一条**只编译不运行**的目标：
    `g++ -fsyntax-only -std=c++17 -I. cuda/SimHost.cpp`，
    以及一条真跑 1 条续走、断言 `ticks > 0 && errorFlags == 0` 的冒烟。
    这一步不需要 GPU，因此能在设备实验之前先挡住所有"参考已经烂了"的情况。

## 6. 已知不做的

- 主机不重排 trunk 权重（`RepackConv` 只服务 cuBLAS）；
- 主机不跑压缩/槽位分配（`netSlot` 恒为 0），因此**测不到压缩顺序相关的缺陷** ——
  那类缺陷属于勘误 3 与勘误 5 的交集，只能靠设备侧的 `PVZ_LANE_CHECK`（`cuda/SimCore.h:44`）
  或逐条 diff 抓；
- 主机不校验 float 次序与 GPU 一致（本来就不可能一致，见勘误 5）。
