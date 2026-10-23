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

- `18a58d42` 构建相对 step9：192/192 条完整叶观测和逻辑统计联合摘要相同，16 个归档共 11,475 个决策帧重放 0 差异、16/16 同种子重复 rollout 一致。本地证据：`artifacts/perf/codex_full_gate_step9_current{,_rounds/}` 与 `codex_replay_16.json`。一轮测速比值 1.1298，前一轮为 1.0074，负载波动明显；这里用于补正确性证据，不替代原会话多轮收益结论。
- 修复前运行既有 24 配置 identity 测试：7/10 packet logits 超出 1e-5，最大绝对误差 0.1854288。新增“同布局换权重 → 与新进程比较”回归也失败：plant logit 为 0.0540556535 vs 0.0564118475。
- 根因：`EnvironmentRollout::Load` 换入新权重未清空 `kernelChannels` / `kernelBuffer` / `packetWeightDouble`。同布局与同活动通道无法感知权重已变，旧矩阵被复用。成功载入后统一清空衍生权重缓存、令 padding 计数失效、清除 packet logits 有效标记；非法载入仍保留旧模型。
- 修复后全部 4 项 `test_identity_cpp` 通过（含新增回归）。24 配置 × 8 真实决策最大 logit 误差 **1.0430813e-7**，所有 argmax/动作一致；传送带 80 决策（79 次空槽、13 次免费卡）最大误差 **7.4505806e-8**；重载与新进程的 plant/packet/cell/action 逐项完全相同。证据 `codex_reload_before.log` / `codex_identity_after.log`（不入库）。这是缓存正确性修复，不作为新的性能增益项。

## 仍需完成

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
