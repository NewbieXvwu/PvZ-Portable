# task77：老师搜索的热路径——每决策耗时花在哪，能不能逐位不变地再快一截

> 2026-10-25。工作树 `perf2`（分支 `perf2`，从 origin/pvz-env 拉出）。
> 原始数据只在 ignored 的 `artifacts/task77/`；本文件放结论与复核口径。
> 工具：`scripts/task77_search_hotpath.py`（prepare / profile / profile-summary / gate / verify / abba）、
> `scripts/task77_value_kernel_check.py`（离线逐位检查）、`python/test_search_hotpath.py`（证据回归）。

## 1. 结论

1. **每决策 74.8–161.0 ms。两块占 80–87%：rollout 引擎步（50.3–65.3%）与引擎内学生前向（19.7–32.0%）。**
   值网络前向（叶值 0.8–1.6% + 静态值 0.9–4.3%）、显式快照往返（0.3–0.7%）、IPC（3.6–8.1%）、
   Python 侧（5.7–10.0%）合计 < 20%。
2. 引擎进程 CPU 为整局墙钟的 57–78%（还包含 Python、IPC 和系统调度等成本，不能把差额全算成排队），
   其中 **89–95% 落在 `ROLLOUT_VALUE` 一条命令里**。
3. 采样集中在 BLAS、动画、粒子与对象数组遍历（§4）；平面 PC 采样不能把 BLAS 内部样本
   精确分给学生与值网络，二者分别耗时以 §3 的引擎计数器为准。
4. **本轮保留 BLAS 算术及动画/粒子生命周期。** sgemm 删零可能改变求和顺序；
   动画与粒子的稀疏跳过 2026-10-23 已经做过
   （`Reanimator.cpp` 的 sparse 路径、`ENV_LIGHT_ANIMATION`），剩下的 track 更新与粒子会消耗共享 RNG
   ——`PvzpCommon.h` 记着：跳过粒子让 16/64 局结果翻转。
5. 所以优化挑的是**占比最大且能证明逐位不变**的三处实现细节（§5）：
   值网络卷积的核打包缓存 + 无填充 memcpy、`DataArray` 取值改走 `get()`、`Reanimation::Update` 模式判断前置。
   离线核检查的 value/static 各 480 个 logit（各含 3 遍）输出 SHA256 完全相同；
   该工具验证数值，未持久化微基准计时，性能结论只采用完整 ABBA。

## 2. 冻结输入与复现

基线代码 `c0249d62`。命令全部经 `curriculum_queue.command_for` 生成，不改候选数、未来数、视界或门槛。

| 输入 | SHA256 |
|---|---|
| 生产 deploy4，引擎只读复制自 cq/build4 | `16989f23cdce3d67e862f4fb4279b423d22ec532d066eea8aa1565dbe76a1fd2` |
| 在役 round 11 学生 | `1cf82f5e6cf7994ca961d91e878d4ac65a4563eea2be50a1e96a9787c22dfd4c` |
| v5 叶值 | `8a636d337c03dcf3037f42ce41f79abc5e0b898eb5a2b39d9a00ad3b9d88c3b4` |
| 台式机 branch1 静态值 | `2485841036cf3a8738174affb0dadc0201a925ca3ea0ebe6e3eb02e74621e314` |
| 六族冻结清单 | `428760ee5be9afe289bd59c9794eae0ece7111bd57bef0c9dd95227bde8ad111` |
| 候选二进制（Release+LTO，无 PGO） | `b4999d6584571d99069625417fc70ebdc12eefdbbc0e6395c1f99975dace9c4a` |
| 同 flags 的改前二进制（ABBA 的 A 臂） | `51949eff3a832b0bdccbdbfc5e16af095e5438df85e768e9573363606d34f8f4` |

白天/夜间/泳池/传送带沿 task70 A2 的 m125 任务；屋顶用 L41 r7 原版；
雾用在役 `fog_user_l31_snow`（human）。human_v2 尚待主控部署，不在本任务换族。
模型、引擎副本、完整基线 Python/C++ 源码与每族前 8 个既有种子冻结在 `artifacts/task77/`，全部保持 gitignore。
预检中发现 redeploy/build 已被 task70 B5 重建为另一引擎，故正式实验取生产 cq/build4 的 deploy4 副本；未修改 cq 或 build4。

系统无 perf，使用仓库 SIGPROF 采样器（`scripts/prof_sample.c`，setitimer + SIGPROF，只记 PC）
经 `LD_PRELOAD` 挂引擎，worker 内跑 `cProfile`，另记每种协议命令的往返耗时与引擎自带的
`simulation_seconds` / `network_seconds` / 值网络 features-forward-calls 计数器。
采样和 cProfile 有开销，其整局耗时不作为 ABBA 性能结论。

```bash
P=/home/newbiexvwu/PvZ-search/.claude/worktrees/perf2
PY=/home/newbiexvwu/.venvs/ml/bin/python3
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py prepare --out $P/artifacts/task77
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py profile --out $P/artifacts/task77
$PY $P/scripts/task77_search_hotpath.py profile-summary --out $P/artifacts/task77
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py gate --out $P/artifacts/task77 --candidate $P/artifacts/task77/frozen/pvz-portable-candidate
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_final --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
```

`prepare` 拒绝覆盖既有冻结；`profile` 可续跑已落盘的局（首批 day/night 缺引擎计数器，已按同一任务与种子
另存 `profile_complete/`，不覆盖旧证据）。
`abba_final/` 是本轮全新计时目录，含冻结清单/身份文件的副本与指向冻结 `baseline/` 的软链，
避免 `search_play` 自动跳过旧结果破坏 ABBA 时间顺序。旧的未完成尝试单独保留，不拼接进本轮。

组件定义（`task77_search_hotpath.py:profile_summary`）：`rollout_simulation` = 引擎回报的
`simulation_seconds`；`cpp_student` = `network_seconds`；`leaf/static_features_forward` = 引擎计数器增量；
`*_ipc` = 对应命令的 wall 减去其中被引擎自己计时的部分；`python_outside_commands` = 整局 wall 减所有命令 wall。
采集时 qc 正在跑（loadavg 23–29 / 20 核），所以绝对值偏保守，**比值**才是可比量。

## 3. 分解（ms / 决策，五地形各 4 局）

| 组件 | day | night | pool | fog | roof |
|---|---|---|---|---|---|
| rollout 引擎步 | 80.93 | 65.75 | 91.35 | 72.29 | 41.79 |
| 引擎内学生前向 | 47.79 | 40.14 | 29.59 | 25.86 | 14.76 |
| 叶值网络（features+forward） | 1.45 | 1.37 | 1.09 | 1.40 | 1.18 |
| 静态值网络（features+forward） | 6.85 | 2.03 | 1.70 | 1.13 | 1.06 |
| 显式快照往返 | 0.57 | 0.50 | 0.47 | 0.76 | 0.55 |
| 静态分支恢复/动作/IPC（残差） | 6.62 | 1.51 | 1.79 | 0.78 | 0.79 |
| rollout 恢复/控制/IPC（残差） | 4.51 | 3.64 | 3.26 | 5.51 | 4.16 |
| 其他命令（真实对局推进等） | 3.02 | 2.13 | 2.73 | 5.79 | 3.01 |
| Python 侧（命令之外） | 9.29 | 8.45 | 7.93 | 8.53 | 7.49 |
| **合计** | **161.0** | **125.5** | **139.9** | **122.1** | **74.8** |

占比：rollout 引擎步 50.3 / 52.4 / 65.3 / 59.2 / 55.9%；学生前向 29.7 / 32.0 / 21.1 / 21.2 / 19.7%。

两点值得记：

- **day 的静态值明显偏重**（6.85 + 6.62 = 13.5 ms/决策，8.4%）：白天族的 `STATIC_SCORES` 调用最多
  （引擎 CPU 42.1 s，其余族 5–22 s）。也就是说值网络这条路的优化只在白天族有可观收益。
- **屋顶与雾的 Python 侧更高**（10.0% / 7.0%）：对局短、决策少，固定开销摊不薄。

## 4. 采样热点（下表待采样格式审计后重算）

审计发现泳池 `pcs.522801.bin` 的两段数组长度不一致（多 4 字节）。旧读法丢尾后可能错认数组边界，
不能当成有效证据。已修复 SIGPROF 导出期间计数仍增长的问题，读取器严格拒绝坏长度/模块号；
原文件保留、明确排除，待用修复后的采样器补采 1 局。下表是旧汇总，最终版将替换。

| 占比 | 累计 | 符号 |
|---|---|---|
| 11.40% | 11.40% | libopenblas `sgemm_kernel_HASWELL` |
| 4.99% | 16.38% | `Reanimation::Update` |
| 3.53% | 19.92% | `PvzpParticleEmitter::UpdateParticle` |
| 3.29% | 23.21% | `Reanimation::Update` 的 track lambda（`operator()`，nm 显示落在 `Reanimation::Update` 内） |
| 2.90% | 26.11% | libc 内部函数（addr2line 只能落到最近的导出符号 `_dl_mcount_wrapper`） |
| 2.86% | 28.97% | `DataArray<Zombie>::Iterator::operator++` |
| 2.60% | 31.57% | `Sexy::MTRand::NextNoAssert` |
| 2.04% | 33.61% | `DataArray<Plant>::Iterator::operator++` |
| 2.00% | 35.61% | libopenblas `sgemm_oncopy_HASWELL`（打包开销） |
| 1.71% | 37.32% | `Zombie::IsTangleKelpTarget` |
| 1.48% | 38.80% | `PvzpParticleEmitter::Update` |
| 1.44% | 40.24% | `Plant::Animate` |

- openblas 包含学生与值网络的矩阵运算；平面采样不能区分共享内核的调用者。
- 动画合计约 12.5%：`Reanimation::Update` + lambda + `NextEnvTrack` + `FloatTrackEvaluate` +
  `PvzpCurveEvaluate` + `GetTrackVelocity` + `TrackMemoHit`。
- 粒子合计约 7.5%：**不能跳**（共享 RNG）。
- `DataArray` 迭代器与查找合计 8.2%：2026-10-23 已做过四槽一判的扫描优化，这里是剩余成本。
- `_dl_mcount_wrapper` 不是 `-pg`：二进制里 `mcount` 符号计数为 0，这是 libc 无 debug 符号时的误标。

## 5. 改了什么（都只改实现，不改搜索）

1. **`src/EnvironmentValueNet.cpp` 的 `Convolve`（叶值与静态值共用）**
   - 首层核打包按「活动通道集」缓存：候选分支通常只改通道内的数值、不增删通道，原来每次前向都重排
     `width × count` 个权重，现在只在通道集变化时重排（`Load()` 清空缓存，跨模型不会复用）。
   - im2col 从「逐格判越界再写」改成「先清零、再按偏移整段 memcpy」：`dc ∈ {-1,0,1}` 时有效列数恒为
  9 或 10，行方向只遍历落在 `[0,6)` 的部分，把逐格边界判断换成逐行连续拷贝。
   - 逐位不变：两种写法写出的 `columns` 与 `kernel` 布局相同，sgemm 参数不变、累加顺序不变；
     越界格仍写 `+0.0f`（先清零、不覆盖），与原来的逐格写 0 一致。
2. **`src/PvzpLib/DataArray.h`**：`operator*` 与 `DataArrayTryToGet` 从 `mItems[i]`（`unique_ptr<T[]>::operator[]`）
   改成 `mItems.get()[i]`，`Iterator::operator++` 标 `always_inline`。去掉的正是采样里
   0.8% 的 `unique_ptr<...>::operator[]` 与 `DataArrayTryToGet` 中的非内联取址。
3. **`src/PvzpLib/Reanimator.cpp` 的 `updateTrack`**：把 `gEnvSimSkip & ENV_LIGHT_ANIMATION` 的判断
   放到读 `mShakeOverride` 之前。rollout 里 shake 恒不更新，原来每 tick 每条活动 track 都要先冷读那个
   float 才发现不用做。

## 6. 逐位一致性

- **离线核检查**（`scripts/task77_value_kernel_check.py`，最终证据 `artifacts/task77/value_kernel_check_final/evidence.json`）：
  用 `build/compile_commands.json` 里的生产编译参数分别编 before（冻结源码）与 after（当前源码），
  同一批输入跑 3 遍（第 2 遍换一份改过首层权重的模型，验证缓存不跨模型复用）：
  value 与 static 各 480 个 logit，before/after 输出字节 SHA256 完全相同
  （value `9762c586…`、static `f4f8f4a0…`）。输入覆盖全零、`-0.0`、稠密与 1–23 个稀疏通道。
  最后一条和重新 Load 后第一条均为全部通道活动，确保覆盖“换模型但活动通道集相同”的缓存失效场景。
  原先首条为全零的核检查仍保留在 `value_kernel_check/`，不据它单独宣称已覆盖此场景。
- **端到端门禁**：见 §7。

## 7. 端到端门禁与计时

### 7.1 逐位一致（六族 × 8 种子）

每族前 4 个种子走老师（`--dagger-beta 1`），后 4 个走学生（`--dagger-beta 0`），两条臂都带搜索打标签，
所以标签、软目标、复核次数、override 决策都在比较范围内。

- 结果：`gate_evidence.json` `pass=true`，**48/48 对局一致**（6 族 × 2 臂 × 4 种子）。
- 比较项：task70 `COMPARE_FIELDS`（胜负、末波、末 tick、决策数、种植、终局僵尸等逻辑结果）
  + `rollout_batches` / `rollout_lanes` / `halving_candidates` / `halving_rollouts` /
  `halving_selected_lanes` / `student_candidate_added` / `student_candidate_best` /
  `student_candidate_override` / `override_kinds`，再加**完整决策帧归档**（帧数 = decisions + 1，
  逐帧按规范化 JSON 比较，`-0.0` 与 `+0.0` 区分，只有复核耗时 `verification.seconds` 排除）。
- `state_hash` 按 task70 A2 的口径单独报告，不作为策略门禁：35/48 相同、13/48 不同（27.1%）。
  本轮未把这 13 处差异归因于某个存档块；不能据完整帧一致就声称内部状态哈希也全同。
- 旧臂的老师那 20 局（5 族 × 4）直接复用 §3 的剖析局：同一冻结输入、同一任务与同一批种子，
  重打只会得到同一局。传送带没有剖析局，旧臂老师现场打。

### 7.2 ABBA 配对计时（正在跑）

按主控 2026-10-25 14:57 的指令（Codex 额度恢复、交回执行权）在 3 个 block 后停止，**没有产出配对比值**：
旧轮次没有完整证据。当前已在全新 `artifacts/task77/abba_final/` 启动完整一轮：
六族各 2 种子 × A-B-B-A，共 48 个完整老师局，`--workers 2 --count 2`。

- 已落盘的只是**未完成的 block**：`abba_0_A`（day，A 臂 3 局）、`abba_1_B`（day，B 臂 3 局）、
  `abba_2_B` 进行到 2/3 局被中断。同一 block 必须 A-B-B-A 四臂齐全才能配对，所以这些数字不能当证据用。
- 复跑命令见 §2，两条硬约束：`--workers 2`（主控 2026-10-25 起本任务同时最多 2 个游戏进程）；
  A 臂必须显式传 `--baseline …/pvz-portable-release-before`，工具现已拒绝缺省回落到生产 PGO 二进制。
- 本轮结束后将报告每局 B/A 比值及配对 bootstrap 95% 区间，并核验四臂结果/标签与后台 qc 负载。

## 8. 局限与后续

- **候选二进制不带 PGO。** 生产 deploy4 带本机 PGO（历史实测 18%）。ABBA 的 A 臂因此用同 flags 重编的
  `pvz-portable-release-before`，只量代码差异；真正上线要由主控按 PGO 两阶段重建，PGO 与补丁的交互收益尚未测量。
  **不要把本文的比值直接当成上线后的墙钟比。**
- 后续可研究共享 BLAS 内核和动画 track 更新；本轮没有实现新的 GEMM，也没有跳过额外动画生命周期。
  任何后续方案都需独立验证浮点求和顺序与 track 重新激活行为，不能把移动零块或稀疏跳过直接当成保序优化。
- 采样器的 libc 符号是误标的（无 debug 信息）。要看清 libc 那 2.9% 得装 `libc6-dbg` 或给 `addr2line` 指定 debug 路径。
