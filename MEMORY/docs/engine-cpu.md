# task35 引擎 CPU 性能记录

本报告尚未完成；任务与停止条件见 [task35-engine-cpu.md](task35-engine-cpu.md)。工作树为 `/home/newbiexvwu/PvZ-search/.claude/worktrees/perf`，分支 `perf`。原 CodeBuddy 仍在同树进行 C++ 实验；Codex 先补独立的基准与验收环节，不接管其未提交的 7 个 C++ 文件，也不操作会话或队列。

## 基准与证据边界

- 固定清单 `artifacts/perf/engine_roots.json`：4 族 × 16 root，每 root 3 个固定未来种子，共 192 条推演。归档、模型与大型运行结果均不入 git。
- 重放用归档动作与 `EventWaitEnv`，逐决策帧比较公开状态。它不声称比较每个模拟 tick 的全部内部字段。
- 旧基准每族只保存前 4 条叶摘要，共 16/192；旧 A/B 只比较时间，不自动校验叶状态或返回统计。因此此前“64 root 全部相同”的记录不能仅凭这些 JSON 独立复核，须用增强夹具补验。
- 现夹具保存全部叶摘要及完整叶观测、estimated_result / decisions / mowers_left / planted / net_calls / ticks / substeps 的联合摘要；A/B 所有轮次均对比同一参考，差异与缺失证据直接失败。计时不进入正确性摘要。
- A/B 每轮原始证据单独保留，汇总写入 `--out`，交替 A/B 顺序。整局读 `games.jsonl`，拒绝局数不足与重复种子，默认并发 2、上限 2。
- 7 项无需游戏进程的回归测试通过：后段叶差异、统计差异、截断证据、空证据、重放失败、A/B 输出与整局文件/worker 限制。命令：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/newbiexvwu/PvZ-search/.claude/worktrees/perf/python /usr/bin/python3 -m unittest test_engine_bench`。

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

## 正确性问题

现构建运行 `test_identity_cpp.IdentityCppTest.test_random_identity_heads_on_real_positions` 失败：7/10 packet logits 超出 1e-5，最大绝对误差 0.1854288。`EnvironmentRollout::Load` 换入新权重时未清空 `kernelChannels` / `kernelBuffer` / `packetWeightDouble` 等衍生缓存，是明确的代码疑点；需要专门复现模型重载并修复，再跑原 24 配置测试。此失败尚未归因于单项优化，也不能用同模型的 rollout 一致性替代该验收。

## 仍需完成

1. 协调原会话退出后接续 7 个未提交 C++ 文件，逐项 A/B 验证和提交；目前不能把它们合成一个已验收的优化。
2. 用完整观测/统计门禁补验现构建与上个提交，16 局动作逐帧重放；修复模型重载问题。
3. 完成基线与最终构建的同种子 8 局整局指标。缺失时不填写模拟推算值。
4. 学生前向 SIMD 与 BLAS 对照、前向减半目标的实测证据仍需核实。
5. 锁步多推演 batch 原型、实测加速比与给 task33 的接口草案尚未提交。
6. 继续逐项优化和记录前 15 热点；尚未证明“连续 3 项收益 <2% 且前 15 无可证明冗余”的停止条件。

## 历史采样前 15

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
