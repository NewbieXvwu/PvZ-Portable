# task35 引擎 CPU 性能记录

本报告尚未完成；任务与停止条件见 [task35-engine-cpu.md](task35-engine-cpu.md)。工作树为 `/home/newbiexvwu/PvZ-search/.claude/worktrees/perf`，分支 `perf`。接手审计时原 CodeBuddy 仍在同树进行 C++ 实验，Codex 先补独立的基准与验收环节；其 7 个 C++ 文件随后由原会话提交为 `18a58d42`。未操作其他会话或队列。

## 基准与证据边界

- 固定清单 `artifacts/perf/engine_roots.json`：4 族 × 16 root，每 root 3 个固定未来种子，共 192 条推演。归档、模型与大型运行结果均不入 git。
- 重放用归档动作与 `EventWaitEnv`，逐决策帧比较公开状态。它不声称比较每个模拟 tick 的全部内部字段。
- 旧基准每族只保存前 4 条叶摘要，共 16/192；旧 A/B 只比较时间，不自动校验叶状态或返回统计。因此此前“64 root 全部相同”的记录不能仅凭这些 JSON 独立复核，须用增强夹具补验。
- 现夹具保存全部叶摘要及完整叶观测、estimated_result / decisions / mowers_left / planted / net_calls / ticks / substeps 的联合摘要；A/B 所有轮次均对比同一参考，差异与缺失证据直接失败。计时不进入正确性摘要。
- A/B 每轮原始证据单独保留，汇总写入 `--out`，交替 A/B 顺序。整局读 `games.jsonl`，拒绝局数不足与重复种子，默认并发 2、上限 2。
- 7 项无需游戏进程的回归测试通过：后段叶差异、统计差异、截断证据、空证据、重放失败、A/B 输出与整局文件/worker 限制。命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/newbiexvwu/PvZ-search/.claude/worktrees/perf/python /usr/bin/python3 -m unittest test_engine_bench`。

### 基准 v2：从快照的真实 tick 计算视界

`replay_episode` 在执行 `rows[index]` 的动作之前保存快照，而该行 `tick` 已是动作执行后的状态。旧 bench/replay 使用该行 tick 加 2500，实际视界多了这个动作经过的 tick。现改为使用 `env.restore(snapshot)['tick'] + horizon`，报告加入 `benchmark_version: 2`。新增回归用 restore=100、存档动作后=160，证明终点是 2600 而非 2660；全部 **8** 项基准回归测试通过。

历史 root 吞吐数字和此前 192/960 条证据对应 v1，仍可用于其自身同口径 A/B，不能直接与 v2 的绝对吞吐比较。v2 完整引擎复核待两组整局测试释放进程槽位后执行；整局 `search_play` 的 8 局对照不使用这个快照计时逻辑，不受影响。

## 已有优化记录（继承，待增强夹具补验）

以下数字来自任务书与本地旧结果，计时口径不同的行不可直接连乘；缺失的整局数字不做外推。

| 步骤 | 改前 → 改后 tick/CPU-s | 增益 | 每推演 | 验收记录 |
|---|---|---|---|---|
| Release 基线 | 82.0k | — | 28.5 ms，wall | 四族各 1 局逐帧 0 差异 |
| LTO | 82.0k → 87.5k | 6.7% | 未留独立记录 | 旧叶摘要记录 |
| march=native | 87.2k → 87.2k | 0%，已删 | 未留独立记录 | 未改变默认 |
| PGO | 87.5k → 103.2k | 18% | 未留独立记录 | 旧叶摘要记录 |
| 前向缓冲与核缓存 | 87.5k → 87.9k | 0.5% | 未留独立记录 | 旧叶摘要记录 |
| mimalloc | 103.2k → 114.8k | 11.2% | 22.1 ms，CPU | 旧叶摘要记录 |
| attacher 查找缓存 | 114.8k → 121.0k | 5.4% | 未留独立记录 | 旧摘要与整局重放 |
| packet dgemm | 配对 A/B | 10.2% | 比值中位数 0.908 | 旧叶摘要记录 |
| 稀疏轨道默认 | 配对 A/B | 6.0% | 比值中位数 0.944 | 旧摘要与整局重放 |
| 粒子视觉积分跳过 | 配对 A/B | 4.7%（保守） | 比值中位数 0.955 | 旧摘要与 16 局重放 |
| im2col 行 memcpy | 配对 A/B | 2.4% | 比值中位数 0.977 | 旧摘要与整局重放 |

整局基线 `artifacts/perf/search_base/games.jsonl` 只有种子 30000/30001 两局，分别 98.7/133.9 秒、966/1121 决策；合计 111.45 ms/决策、30.95 games/h（串行工作量口径）。这是 **2/8 局的部分结果**，不能作为完成的整局基线。旧脚本误读 `results.jsonl` 已修正。

## 增强夹具复核与模型重载修复

重启后审计另发现事件等待复用 bug：`PublicState::clear()` 未清 `plantCount`，真实读取器每次累加存活植物会在缓冲复用后漏掉植物减少条件。新增原地清空/累加场景，在第 2 tick 从 3 株减到 2 株：修复前断言失败，清空全部标量后 12 项 C++ 契约、20 项 Python 等待测试通过。现有 timeout 重放和 rollout 不覆盖条件 5，不能代替该回归；该修复不计性能收益。正式构建重放补验待当前 A2 计时结束后执行。

- `18a58d42` 构建相对 step9：192/192 条完整叶观测和逻辑统计联合摘要相同，16 个归档共 11,475 个决策帧重放 0 差异、16/16 同种子重复 rollout 一致。本地证据：`artifacts/perf/codex_full_gate_step9_current{,_rounds/}` 与 `codex_replay_16.json`。一轮测速比值 1.1298，前一轮为 1.0074，负载波动明显；这里用于补正确性证据，不替代原会话多轮收益结论。
- 修复前运行既有 24 配置 identity 测试：7/10 packet logits 超出 1e-5，最大绝对误差 0.1854288。新增“同布局换权重 → 与新进程比较”回归也失败：plant logit 为 0.0540556535 vs 0.0564118475。
- 根因：`EnvironmentRollout::Load` 换入新权重未清空 `kernelChannels` / `kernelBuffer` / `packetWeightDouble`。同布局与同活动通道无法感知权重已变，旧矩阵被复用。成功载入后统一清空衍生权重缓存、令 padding 计数失效、清除 packet logits 有效标记；非法载入仍保留旧模型。
- 修复后全部 4 项 `test_identity_cpp` 通过（含新增回归）。24 配置 × 8 真实决策最大 logit 误差 **1.0430813e-7**，所有 argmax/动作一致；传送带 80 决策（79 次空槽、13 次免费卡）最大误差 **7.4505806e-8**；重载与新进程的 plant/packet/cell/action 逐项完全相同。证据 `codex_reload_before.log` / `codex_identity_after.log`（不入库）。这是缓存正确性修复，不作为新的性能增益项。

## 独立锁步原型与给 task33 的接口草案

原型源码在 `scripts/prototypes/batch_rollout.inc`，构建入口 `build_batch_rollout.py`，验收/计时入口 `bench_batch_rollout.py`。已从提交 `08228252` 的源码导出独立 Release/LTO 构建，产物与源码副本只放 `artifacts/perf/batch_prototype/`，正式 CMake、源码与 build 均未注入实验命令。`build.json` 记录原提交及 helper/builder SHA256。此版本不带本机 PGO，故原生串行对照也在**同一个原型二进制**内运行，绝对值不能直接冒充最终生产构建。

原型的每个 lane 持有 `EnvironmentSnapshot`（含引擎 RNG/波表/计数器）、独立学生 History、6000 tick 收入窗口、前一动作上下文、子步和种植统计。轮流恢复状态、提取需要前向的输入；在一个决策边界将多个 lane 的三层卷积各合成一次 `W×K · K×(lanes×60)` SGEMM。首层使用该批输入的活动通道并集，零通道仍是零。然后将卷积输出交回现有 Encode 尾部、Heads 和 Predict 动作选择，保留原来的合法性、同分处理、等待和早期结果条件。行汇总与策略头目前仍逐 lane 执行。

原型用完整快照切换单个 Board，避免新开多个游戏进程；这是正确性与可行性实验，**不建议 task33 在最终热循环中每决策序列化/恢复两次快照**。计时分别列 network / simulation / snapshot / feature CPU 与进程总 CPU；基准对同一 root、同九个未来种子运行原生串行、锁步 batch=1、锁步 batch=9，既报告批处理内部收益，也报告对原生串行的实际总成本。全量验收计划为 16 归档整局逐帧重放和 64 root × 9 未来的完整叶观测/逻辑统计比较，尚未因两个整局测试占满槽位而开始运行，当前不能宣称原型加速或验收通过。

供 task33 讨论的 C++ 接口边界：

```cpp
struct RolloutLane;       // Board/effects + RNG + history + prior action + counters
struct StudentInputs;     // board/scalars/packet info, with lane id
struct StudentWorkspace;  // packed columns/weights/output, allocated once
void PrepareInputs(RolloutLane&, StudentInputs&);
void ConvolveBatch(std::span<const StudentInputs>, StudentWorkspace&);
void FinishPrediction(RolloutLane&, const HiddenView&); // existing heads and legality
void AdvanceToDecision(RolloutLane&);                  // existing tick/event-wait logic
```

最终 lane 上下文必须同时隔离 Board、全局 effects、共享 RNG 与静态缓存失效代数；只换 Board 指针不够。按决策轮次压紧仍存活且可种植的 lane，保留 seed/lane 顺序回填结果；不能让无合法动作的 lane 产生多余网络调用，也不能让 batch 的活动通道改变可见观测。工作区和固定权重应复用；首层并集扩大 K 的成本、尾部稀疏存活数以及细粒度快照成本都要由实测决定，不能用矩阵核上限代替整条推演收益。批量浮点结果若导致决策分叉，应按任务书统计验收，不能悄悄降低逐动作重放门槛。

## 降精度（int8 / AVX-VNNI）：核级快 3 倍，但 8 位激活误差太大，未进入生产

任务书 C 项。本机 i7-12700F 有 `avx_vnni`（`/proc/cpuinfo` 确认），但没有 AVX-512，所以 bf16 GEMM 不可用
（`vdpbf16ps` 属 AVX512-BF16），只试 int8。

`scripts/bench_int8_kernel.cpp` 是独立实验，不链接进游戏：读与 SSE 对照**同一份**固定夹具
（`scripts/bench_conv_kernels.py` 生成的 64 个真实 v2 root × 三层卷积，共 192 个输入），共享同一套稀疏通道选择与
im2col，只比矩阵核。激活按层标定（整张输入张量一个 u8 尺度），权重逐输出通道对称 s8；权重按
“8 输出通道 × 4 k 值 = 32 字节”预先打包（静态权重，一次打包，热路径零成本），激活按 cell 主序存放，
`vpdpbusd` 的操作数一个广播、一个整块加载。8 个累加器并行，避免 5 拍延迟串行化 k 循环。

| 卷积 | BLAS μs/次（中位数） | int8 μs/次（中位数） | 耗时比 |
|---|---:|---:|---:|
| 稀疏首层 3×3 | 15.8 | 8.4 | 0.53× |
| 稠密第二层 3×3 | 57.9 | 22.3 | 0.38× |
| 稠密第三层 1×5 | 32.5 | 14.3 | 0.44× |

（3 轮 × 30 次重复 × 交替先后顺序；本机 load 26，单轮抖动大，取中位数。证据 `artifacts/perf/int8_kernel_compare.json`。）

**核本身是对的**：与“把量化值反量化回 float 再算”的参考实现逐位相同（`kernel_vs_quant_ref = 0`），
所以剩下的差全是量化误差，不是实现 bug。

误差（相对 BLAS 的 fp32 结果）：首层 rms 0.030 / 最大 1.97；第二层 rms 0.012；第三层 rms 0.025。
把同样的量化套进整条 trunk（`scripts/int8_emulate.py`，64 个真实 root，torch 里先量化再反量化，
与核的数学完全一致）后：

| 指标 | 值 |
|---|---:|
| trunk 输出相对 rms | 8.9% |
| 全局嵌入 g（plant/packet 头的输入）相对 rms | 5.4% |
| g 的余弦相似度 | 0.995 – 0.999 |
| plant logit 最大绝对偏差 | 0.297 |
| plant 正负号（种/等）一致率 | 64/64 |

结论：**不进入生产**。理由是收益与风险不匹配，而不是“没测”：

- 收益上限是卷积栈的 3 倍。剖面里 BLAS 全家（sgemm 核 17.7% + 打包 4.4% + dgemm 1.8%）约 24%，
  打三折后整体约 **1.15–1.19 倍**。
- 代价是每层 5% 量级的嵌入误差，而策略头在 54 个格子里挑动作，logit 间距经常小于 0.3；
  验收要求“老师决策一致率不低于 72%”，这需要一次 64 局统计跑才能证明，
  而为了跑它必须先把它接进生产路径。
- 还有实现代价：本机有 AVX-VNNI，Mac/clang 没有，必须保留一条 BLAS 后备路径（CPU 派发），
  与“有效优化直接成为唯一实现”的约定冲突。

如果以后要继续这条路，正确的下一步是**逐输入通道的激活尺度**：把激活尺度折进权重尺度
（`q_w[c][k] = round(w[c][k] * s_a[c] / S)`），这样单个累加器仍然成立，而首层那种
“少数通道量级远大于其余”的输入不会再被一个全局尺度压扁。当前实现只做了任务书要求的“按层标定”。
## 合并 origin/pvz-env（2026-10-23，A1）

`perf` 落后主干 70 个提交。冲突只有两处，按任务书裁决：

- `CMakeLists.txt`：pvz-env 删掉了 `PVZ_CUDA_ROLLOUT` 选项与 `CMake/CudaRollout.cmake`，本分支新增的
  LTO / PGO 编译块紧挨着它，所以成了文本冲突。**保留 LTO/PGO/mimalloc，删除 CUDA 选项**；`cuda/`、
  `src/CudaExport.*`、`python/pvz_cuda_*.py` 等按对方删除。
- `MEMORY/2026-10-23.md`：两边都新建了当天日志，add/add 冲突，按 append-only 保留双方条目。

合并后重新走完 `scripts/build_pgo.sh`（源码变了，旧 profile 失效），再做验收：

| 验收 | 结果 |
|---|---|
| 64 root × 3 推演的完整叶观测/统计（对比 step9 二进制，4 轮配对） | 192/192 相同（`compare` 自带断言，差异会直接失败） |
| 整局逐帧重放（四族各 1 局） | 953 / 877 / 531 / 464 帧，**0 差异** |
| 同 root 同种子连跑两次 | 4/4 相同 |
| `python -m unittest discover python`（281 项） | 见下 |

`python/` 与 `scripts/train_rollout_policy.py` 相对 `origin/pvz-env` 只多了两个测试文件，所以下面这些不是合并引入的：

- `test_cpp_rollout`（5 项）：默认资源目录写死 Mac 路径 `/Users/...`，设 `PVZ_RESOURCE_DIR` 后 **5/5 通过**。
- `test_rowpool_aux`：模块级 `import pytest`，而 `~/.venvs/ml` 里没装 pytest（系统 python3 装了但没 torch）。
- `test_identity_training.test_cpu_training_checkpoint_is_loadable`：`train_rollout_policy.py` 的 cell gather
  越界（`index -9223372036854775808`）。这个文件在本分支与 pvz-env **逐字节相同**，属主干既有问题，
  归 task33/32 线处理，不在本任务范围。

合并后构建相对 step9 二进制的配对比值中位数 **0.92**（4 轮，单轮 1.175 / 0.785 / 0.933 / 0.909，噪声大），
即合并+第 11–13 项之后比 step9 快约 8%。
## 整局基准：新旧构建同种子对照（A2）

验收数字是**正式采集的整局吞吐**，不是 64 root 的 tick 率，所以这里用 `scripts/bench_engine.py games`：
同一个 `search_play`、同一份任务清单（`cq/artifacts/stage4/multiplier_panel.json`）、同一批种子（offset 0、每族 8 局），
先新构建后旧构建，逐族取比值。参数照 qc 的 `command_for`（`--verify-samples 9 --student-candidates 2`、
`--static-value value_branch1`、`--value v3`、`--dagger-beta 0`、`--workers 2`），但**不写 frames 归档**：
归档是采集端的成本（task33 负责），两边都不写，比的是引擎 CPU。

计时口径：`RUSAGE_CHILDREN` 的 CPU 秒（worker 池与其引擎子进程都算在内）。本机长期被采集队列占满（load 21–32），
wall 会被抢占放大，CPU 秒才是能跨构建比较的量。

| 任务族（m150） | 新构建 CPU s/局 | 旧构建 CPU s/局 | 旧/新 | 新 games/h | 旧 games/h | 逐种子结果 |
|---|---:|---:|---:|---:|---:|---|
| 白天 L7 `day_l7_train_d7_m150` | 149.3 | 236.7 | **1.586×** | 24.12 | 15.21 | 8/8 相同（全胜，1213 决策） |
| 夜间 L14 `night_l14_train_n10_m150` | 88.5 | 138.2 | **1.561×** | 40.68 | 26.05 | 8/8 相同（全负，两边一致） |
| 泳池 L24 `pool_l24_train_p9k_m150` | 待填 | 待填 | | | | |
| 传送带 L10 `day_l10_orig_train_m150` | 待填 | 待填 | | | | |

旧构建是 `cq` 工作树的二进制（`pvz-env` 合并前，只读、未重编）。逐种子结果（胜/负、末波、决策数）两边完全一致，
所以比值不是“新构建少算了什么”换来的。
## 锁步批量推演原型实测（B）

原型在 `scripts/prototypes/`，从已提交源码（`08228252`）导出到 `artifacts/perf/batch_prototype/` 的独立构建，
不修改正式源码与 build。`bench_batch_rollout.py` 对 64 个真实 root、每个 9 个未来种子跑三条路径：

- `native`：同一原型二进制里的原生串行（9 条推演逐条 `ROLLOUT`），
- `batch 1`：锁步路径但批大小为 1（量快照切换与打包的固定开销），
- `batch 9`：9 条推演锁步，前向合成一次 SGEMM。

三条路径的叶观测/逻辑统计逐条比对，任何一条不同即失败；CPU 分别记 network / simulation / snapshot / feature。

恢复检查发现旧日志在首个 batch 响应因缺少 `protocol_version` 中断；已修复成功/异常响应，原生臂改用与 batch 一致的进程 CPU 钟，补 network / simulation / snapshot 分项。报告显式区分完整覆盖与正确性通过，叶摘要差异直接非零退出；12 项无引擎回归通过，正重建当前提交的独立副本。

待填：三臂总 CPU 秒、`native/batch9` 与 `serial/batch9` 比值、前向加速比、逐 root 一致性计数。
## 部署（给主控：在 cq 工作树上一次做完）

本分支的引擎加速全部是**默认构建**的一部分，没有新增运行开关：LTO、PGO、mimalloc、
以及每一项引擎/前向优化都在源码里。部署只有“合并 + 重新生成 profile + 重编”三步。

```bash
# 1. 合并（perf 已 push 到 origin/perf）
git -C /home/newbiexvwu/PvZ-search/.claude/worktrees/cq merge origin/pvz-env   # 若落后先合主干
git -C /home/newbiexvwu/PvZ-search/.claude/worktrees/cq merge perf

# 2. 两阶段 PGO + 最终构建（同一个 build 目录，GCC 按目标文件路径找 .gcda）
cd /home/newbiexvwu/PvZ-search/.claude/worktrees/cq
JOBS=8 bash scripts/build_pgo.sh bench

# 3. 必须为 0：为 1 说明构建停在了 PGO 第一阶段（会慢约 1.8 倍）
grep -c fprofile-generate build/build.ninja
```

- **mimalloc 首次 configure 需要联网**（CMake `FetchContent` 从 github 拉取，`MI_OVERRIDE` 全局接管
  malloc）。之后检出留在 `build/_deps/`，断网也能重编。若首次 configure 失败，`cmake --build` 会直接报错，
  不会静默降级成系统 malloc。
- **`build_pgo.sh` 不会删 build 目录**（只清 `artifacts/perf/pgo`），因为删掉就要重新联网 clone mimalloc。
- **profile 用哪批 root 生成**：`artifacts/perf/engine_roots.json` 的 64 个真实决策点（白天 L7 / 原版传送带 L10 /
  夜间 L12 / 泳池 L24 四族各 16），每个 root 跑 3 条推演（horizon 2500、sim-skip 61、学生 `dw4i_s1`）。
  这就是正式采集的形状：每个决策 = 候选 × 验证样本条推演。
- **换学生模型后要重生成 profile**。profile 记录的是分支与调用频次，而首层卷积的稀疏通道集合、
  权重转置缓存是否命中都由权重决定；换权重后热点分布会变，重跑一次 `build_pgo.sh` 即可（本机负载 26 时约 15 分钟）。
  只换价值网（`--value`）不需要：价值网在 Python 侧，不进引擎进程。
- 换机器要各自重编：`-march=native` 已测过无收益并删除，但 PGO profile 是**本机生成、本机使用**的产物，
  不要跨机器拷贝 `artifacts/perf/pgo/`（目标文件路径不同，GCC 会静默找不到）。

## 仍需完成

### 公开坐标格式化候选（待整引擎验收）

`scripts/prototypes/bench_public_coordinate.cpp` 对比公开编码使用的 stream/stod 与 `to_chars(general, 6)` / `from_chars`。固定种子生成任意有限 float 位型（引擎坐标和速度都是 float，提升到 double），再覆盖整数附近与左侧边界，共 **1,017,346** 个样本，double 结果（含负零）逐位相同。三轮交替 CPU 秒分别为 stream 0.5862/0.5763/0.5701、chars 0.1289/0.1413/0.1340；中位数单函数加速 **4.30×**。这尚未进入生产，不能外推整引擎收益；v2 Features 每次提取坐标/速度均走该函数，是下一项整条推演候选。

### 分块手写 SIMD 对照：拒绝替换 BLAS

`scripts/bench_conv_kernels.py` / `.cpp` 是独立实验，不链接进游戏。读取固定清单 64 个真实 root（全部包含 v2 字段），从学生模型构造三层卷积各 64 个输入；第一层沿用稀疏通道选择，后两层为稠密。两条路径共享相同 im2col 与权重矩阵：显式 SSE 的 4 输出通道 × 4 单元分块累加，对照单线程 OpenBLAS SGEMM。`-O3 -ffp-contract=off`，交替先后顺序三轮，每个输入重复 30 次，按进程 CPU 时间测量。

| 卷积 | BLAS μs/次，中位数 | SSE μs/次，中位数 | SSE/BLAS 耗时比，中位数 | 最大绝对差 |
|---|---:|---:|---:|---:|
| 稀疏首层 3×3 | 19.22 | 63.71 | 3.50× | 5.72e-6 |
| 稠密第二层 3×3 | 81.39 | 267.16 | 3.28× | 1.53e-5 |
| 稠密第三层 1×5 | 42.94 | 145.85 | 3.40× | 1.72e-5 |

证据为 `artifacts/perf/conv_kernel_compare.json`。这是矩阵核比较，**不含共享的 im2col 与完整学生前向**，不能替代全引擎加速比或 batch 推演原型。手写路径各层都慢约 3 倍，且浮点累加顺序已与 BLAS 有差异，故不进入生产；生产继续唯一的 BLAS 路径（无 BLAS 的现有平台后备实现未改变）。当前 4de38495 的前 15 分布仍见下方采样，不因这个独立实验而改变。此次没有取得正收益，停止条件尚未满足。

1. 原会话已提交 `18a58d42` 的六项微优化（其报告 +8.3% 为合计，单项收益未独立证实）；继续在此基础上找下一项，避免重复实验。
2. 模型重载修复已补验：修复前后 192 条完整叶观测与逻辑统计全部相同（`codex_reload_compare.json`，单轮比值 1.0236，不计作性能收益），修复后另跑 16 局 11,475 帧零差异且同种子重复推演一致（`codex_replay_after_reload_16.json`）。
3. 完成基线与最终构建的同种子 8 局整局指标。缺失时不填写模拟推算值。
4. SIMD 与 BLAS 的卷积核对照已完成并拒绝 SSE；完整学生前向减半目标的实测证据仍需核实。
5. 锁步多推演 batch 原型、实测加速比与给 task33 的接口草案尚未提交。
6. 继续逐项优化和记录前 15 热点；尚未证明“连续 3 项收益 <2% 且前 15 无可证明冗余”的停止条件。

## 历史采样前 15

### 重启恢复后的工具与 libc 符号补全

已从 `archive.ubuntu.com` 下载与 apt 元数据校验一致的 perf/valgrind/libc debug 包，只解包到 `artifacts/perf/tools/`，未改系统安装。perf 7.0.14 的 `cpu-clock:u` 调用栈探针成功，valgrind 3.26.0 可运行；下一次游戏采样可用真正的 `perf record`。`prof_symbolize.py --debug-root <解包的 usr/lib/debug>` 按 ELF build ID 匹配符号，旧 5752 点数据重解析后前 15 如下。这仍是历史 4de38495 的采样，不冒充当前构建剖面。

| 函数或动态符号 | 占比 |
|---|---:|
| SGEMM kernel HASWELL | 17.70% |
| Reanimation::Update | 3.79% |
| SGEMM oncopy HASWELL | 2.96% |
| DataArray<Zombie>::Iterator::operator++ | 2.76% |
| operator（Reanimator.cpp 轨道 lambda） | 2.16% |
| PvzpParticleEmitter::UpdateParticle | 1.81% |
| DGEMM kernel HASWELL | 1.77% |
| Sexy::MTRand::NextNoAssert | 1.70% |
| EnvironmentRollout::Convolve | 1.46% |
| DataArray<Plant>::Iterator::operator++ | 1.44% |
| SGEMM itcopy HASWELL | 1.39% |
| libc __memset_avx2_unaligned_erms | 1.32% |
| DGEMM incopy HASWELL | 1.23% |
| _mi_page_malloc_zero | 1.18% |
| std::_Rb_tree_insert_and_rebalance | 1.16% |

原来的 libc `_dl_mcount_wrapper` 最近符号归因不准确，补符号后拆成上述 memset、strcmp（1.11%）等真实实现；不能据旧名称推导调用链。

### 4de38495 后采样（完整解析共享库）

`scripts/prof_symbolize.py` 现对共享库也运行 addr2line；保存的原始可执行文件可通过 `--binary` 映射。对稳定副本 `pvz-portable-codex-reload` 跑四族 64 root × 15 条推演，得到 **5752** 个已限定 rollout 阶段的 SIGPROF 样本（`codex_profile_reload.json`）。BLAS 热点从一个 28% 的库名拆成 SGEMM 内核、矩阵打包与 DGEMM。无调试信息的库仍可能只给最近动态符号，尤其 libc 的 `_dl_mcount_wrapper` 不能视为准确内部函数或直接优化依据。

`sudo -n apt-get install --no-install-recommends -y linux-perf` 和 valgrind 两次安装均因 BFSU 镜像站 **HTTP 403** 失败（并非无 sudo）；继续使用已有 SIGPROF 采样器。该采样有开销，计时值不用于对比非采样运行，也没有 perf 调用栈语义。

| 函数或动态符号 | 占比 |
|---|---:|
| [libopenblas.so.0] sgemm_kernel_HASWELL | 17.70% |
| Reanimation::Update | 3.79% |
| [libc.so.6] _dl_mcount_wrapper | 3.25% |
| [libopenblas.so.0] sgemm_oncopy_HASWELL | 2.96% |
| DataArray<Zombie>::Iterator::operator++ | 2.76% |
| operator | 2.16% |
| PvzpParticleEmitter::UpdateParticle | 1.81% |
| [libopenblas.so.0] dgemm_kernel_HASWELL | 1.77% |
| Sexy::MTRand::NextNoAssert | 1.70% |
| EnvironmentRollout::Convolve | 1.46% |
| DataArray<Plant>::Iterator::operator++ | 1.44% |
| [libopenblas.so.0] sgemm_itcopy_HASWELL | 1.39% |
| [libopenblas.so.0] dgemm_incopy_HASWELL | 1.23% |
| _mi_page_malloc_zero | 1.18% |
| [libstdc++.so.6] std::_Rb_tree_insert_and_rebalance | 1.16% |

SGEMM 内核约 17.7%，加输入打包约 4.4%，是接下来对照分块 SIMD / batch 原型的依据；RNG、DataArray 及粒子仍要保持语义。这里没有证明前 15 已无冗余，停止条件仍未满足。

以下从既有 SIGPROF 采样 JSON 原样提取（并非 perf 调用栈剖面）。库内样本被归并为库名，不能据此声称已定位其内部函数；较短的文件不足 15 项照实列出。每个阶段的二进制/编译与采样时点应结合原任务书核对，不能用 P7 采样代表当前未提交源码。

### prof_base2（1536 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 14.13% |
| Reanimation::Update | 12.04% |
| EnvironmentRollout::Heads | 10.09% |
| [libc.so.6] | 6.18% |
| EnvironmentRollout::Convolve | 4.30% |
| PvzpParticleEmitter::UpdateParticle | 4.04% |
| [libstdc++.so.6] | 2.54% |
| PvzpCurveEvaluate | 2.28% |
| PvzpParticleEmitter::UpdateParticleField | 2.15% |
| FloatTrackEvaluate | 2.08% |
| EnvironmentRollout::Encode | 2.02% |
| Zombie::IsTangleKelpTarget | 2.02% |
| Plant::UpdateAbilities | 1.76% |
| Board::UpdateGameObjects | 1.50% |
| Zombie::GetDrawPos | 1.43% |

### prof_pgo（1162 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 19.19% |
| Reanimation::Update | 12.99% |
| EnvironmentRollout::Heads | 11.45% |
| [libc.so.6] | 7.40% |
| EnvironmentRollout::Convolve | 5.08% |
| Board::UpdateGameObjects | 4.65% |
| PvzpParticleEmitter::Update | 4.48% |
| Plant::Update | 3.01% |
| PvzpParticleEmitter::UpdateParticleField | 2.50% |
| LawnApp::AdvanceLogicTick | 2.24% |
| Zombie::UpdateZombieWalking | 2.15% |
| Plant::FindTargetZombie | 2.15% |
| PvzpParticleEmitter::SpawnParticle | 1.98% |
| [libstdc++.so.6] | 1.89% |
| Board::Update | 1.81% |

### prof_mi（1103 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 17.41% |
| Reanimation::Update | 13.60% |
| EnvironmentRollout::Heads | 12.51% |
| EnvironmentRollout::Convolve | 5.89% |
| Board::UpdateGameObjects | 5.17% |
| PvzpParticleEmitter::Update | 4.35% |
| [libc.so.6] | 3.35% |
| PvzpParticleEmitter::UpdateParticleField | 2.81% |
| Plant::Update | 2.54% |
| Zombie::UpdateZombieWalking | 2.45% |
| Plant::FindTargetZombie | 2.36% |
| EnvironmentRollout::Predict | 2.09% |
| LawnApp::EnvironmentPublicWaitState | 1.99% |
| [libstdc++.so.6] | 1.99% |
| PvzpParticleEmitter::SpawnParticle | 1.81% |

### prof_a2（1059 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 17.85% |
| Reanimation::Update | 12.18% |
| EnvironmentRollout::Heads | 11.61% |
| EnvironmentRollout::Convolve | 6.33% |
| PvzpParticleEmitter::Update | 5.85% |
| Board::UpdateGameObjects | 5.38% |
| PvzpParticleEmitter::UpdateParticleField | 3.87% |
| [libc.so.6] | 3.68% |
| Plant::Update | 2.64% |
| [libstdc++.so.6] | 2.46% |
| Zombie::UpdateZombieWalking | 2.17% |
| LawnApp::AdvanceLogicTick | 2.17% |
| PvzpParticleEmitter::SpawnParticle | 1.98% |
| LawnApp::EnvironmentPublicWaitState | 1.79% |
| Plant::FindTargetZombie | 1.61% |

### prof_p4（885 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 27.12% |
| EnvironmentRollout::Convolve | 5.99% |
| Reanimation::Update | 5.76% |
| [libc.so.6] | 4.97% |
| PvzpParticleEmitter::UpdateParticle | 3.05% |
| [libstdc++.so.6] | 2.26% |
| operator | 1.92% |
| PvzpCurveEvaluate | 1.81% |
| Sexy::MTRand::NextNoAssert | 1.58% |
| FloatTrackEvaluate | 1.47% |
| Attachment::Update | 1.24% |
| PvzpParticleEmitter::UpdateParticleField | 1.24% |

### prof_p5（903 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 27.80% |
| [libc.so.6] | 5.54% |
| EnvironmentRollout::Convolve | 5.20% |
| [libstdc++.so.6] | 3.54% |
| Reanimation::Update | 3.32% |
| Sexy::MTRand::NextNoAssert | 2.44% |
| PvzpParticleEmitter::UpdateParticle | 2.10% |
| DataArray<Plant>::Iterator::operator++ | 1.88% |
| DataArray<Zombie>::Iterator::operator++ | 1.55% |
| operator | 1.44% |
| Zombie::IsTangleKelpTarget | 1.33% |
| Attachment::Update | 1.22% |
| Plant::Animate | 1.11% |
| Reanimation::GetTrackVelocity | 1.00% |

### prof_p6（2277 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 28.81% |
| [libc.so.6] | 5.67% |
| Reanimation::Update | 4.48% |
| memcpy | 3.12% |
| [libstdc++.so.6] | 2.46% |
| PvzpParticleEmitter::UpdateParticle | 2.33% |
| DataArray<Plant>::Iterator::operator++ | 1.89% |
| operator | 1.67% |
| DataArray<Zombie>::Iterator::operator++ | 1.67% |
| Sexy::MTRand::NextNoAssert | 1.58% |
| EnvironmentRollout::Convolve | 1.36% |
| Zombie::IsTangleKelpTarget | 1.36% |
| Linear | 0.92% |
| Zombie::GetDrawPos | 0.92% |
| DataArray<Reanimation>::DataArrayTryToGet | 0.83% |

### prof_p7（1885 样本）

| 函数或模块 | 占比 |
|---|---:|
| [libopenblas.so.0] | 28.65% |
| [libc.so.6] | 4.93% |
| Reanimation::Update | 3.87% |
| DataArray<Zombie>::Iterator::operator++ | 3.61% |
| [libstdc++.so.6] | 2.86% |
| EnvironmentRollout::Convolve | 2.71% |
| memcpy | 2.49% |
| operator | 1.86% |
| PvzpParticleEmitter::UpdateParticle | 1.75% |
| DataArray<Plant>::Iterator::operator++ | 1.64% |
| Sexy::MTRand::NextNoAssert | 1.33% |
| DataArray<Reanimation>::Iterator::operator++ | 1.17% |
| Plant::Animate | 1.11% |
| __find_if<float const*, __gnu_cxx::__ops::_Iter_pred<EnvironmentRollout::Convolve | 0.95% |
| TrackMemoHit | 0.95% |
