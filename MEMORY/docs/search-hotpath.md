# task77：老师搜索的热路径——每决策耗时花在哪，能不能逐位不变地再快一截

> 2026-10-25。工作树 `perf2`（分支 `perf2`，从 origin/pvz-env 拉出）。
> 原始数据只在 ignored 的 `artifacts/task77/`；本文件放结论与复核口径。
> 工具：`scripts/task77_search_hotpath.py`（prepare / profile / profile-summary / hotspots / gate / verify / abba）、
> `scripts/task77_value_kernel_check.py`（离线逐位检查 + 交错计时）、`python/test_search_hotpath.py`（证据回归）。

## 1. 结论

1. **每决策 74.8–161.0 ms。两块合计占 76–86%：rollout 引擎步（50.3–65.3%）与引擎内学生前向（19.7–32.0%）。**
   值网络前向（叶值 0.8–1.6% + 静态值 0.9–4.3%）、显式快照往返（0.3–0.7%）、IPC（3.6–8.1%）、
   Python 侧（5.7–10.0%）及真实对局推进等，其余组件合计占 14–24%。
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
   离线核检查的 value/static 各 480 个 logit（各含 3 遍）输出 SHA256 完全相同。
   同一工具还做**交错计时**（同核 pin、before/after 交替 9 轮 × 400 次、取中位数）：
   value/static 前向 **164.1 → 136.1 µs，快 17%**（§7.3）。整局结论仍只采用 ABBA，微基准只作机制核对。
6. **逐位不变成立；有真实但很小的加速，整局量级测不出来。**
   - 组件级：value/static 前向快 **17%**（§7.3）。
   - 整局：两轮 ABBA（各 12 对、48 局）每局耗时比 geomean **0.993**（95% [0.948, 1.035]，qc 负载下的首轮）
     与 **0.990**（[0.969, 1.012]，负载较轻的第二轮）；第二轮同时记引擎进程 CPU 时间，比值 **0.985**（[0.959, 1.011]）。
     第三轮在**六族全程同一个 qc 控制器**下同时记墙钟与引擎 CPU：**1.0045**（[0.987, 1.022]）、CPU **1.0050**（[0.985, 1.029]）——三轮的点估计在 0.99–1.005 之间、区间都覆盖 1。
   - 三者互相印证：值网络前向只占每决策耗时 2.0–5.2%（§3），快 17% 只能换来整局 **0.3–0.9%**，
     低于两轮 ABBA 的分辨率（±2–4%）。`DataArray` 取值 7.25%、动画 8.52% 上的改动每次只省一条间接寻址或一个分支。
   - **结论：不要为提速单独上线这个补丁**（收益 <1%，还要重建 PGO 引擎）。补丁本身逐位不变、无风险，
     可以随下一次引擎重建顺带带上。要真正提速只能动 rollout 引擎步与学生前向（合计 76–86%），
     那需要新的浮点保序方案，不在本任务范围（§8）。

## 2. 冻结输入与复现

基线代码 `c0249d62`。命令全部经 `curriculum_queue.command_for` 生成，不改候选数、未来数、视界或门槛。
主控 2026-10-25 明确：台式机内存紧张，本任务同时最多 2 个游戏进程，qc 优先。
实验工具默认 workers=2，命令生成器拒绝超过 2；补采与计时串行执行。

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
`prepare` 从固定提交 `c0249d62` 提取基线源码，避免在优化后重跑工具时把候选代码误当成改前版本；
本轮冻结副本及新提取副本的四个改动引擎文件、搜索驱动均已逐字节核对原提交。
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
$PY $P/scripts/task77_search_hotpath.py hotspots --out $P/artifacts/task77
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py gate --out $P/artifacts/task77 --candidate $P/artifacts/task77/frozen/pvz-portable-candidate
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_final --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
# 第二轮：同设计，另建目录，并记引擎进程 CPU（§7.3a）
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_cpu --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
$PY $P/scripts/task77_search_hotpath.py abba-verify --out $P/artifacts/task77/abba_cpu --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
# 第三轮：qc 负载下的同设计 ABBA（六族全程同一控制器，§7.4）
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_qc --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
$PY $P/scripts/task77_search_hotpath.py abba-verify --out $P/artifacts/task77/abba_qc --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
# 组件级：逐位核检查 + 交错计时（§7.3b）
$PY $P/scripts/task77_value_kernel_check.py --time
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

## 4. 采样热点（最终版，`hotspots` 阶段重算）

审计发现泳池 `pcs.522801.bin` 的两段数组长度不一致（多 4 字节），旧读法丢尾后可能错认数组边界。
已修复 SIGPROF 导出期间计数仍增长的问题，读取器严格拒绝坏长度/模块号；原文件保留、明确排除
（`invalid_sample_files`），并用修复后的采样器按同一任务与种子补采 1 局（`profile_recovery/`）。
最终样本 = day 4 + night 4（`profile_complete/`）+ pool 3 + 补采 1 + fog 4 + roof 4 = 20 局，
**566,659 个样本**，由 `hotspots` 阶段从 `profile_summary.json` 汇总成 `artifacts/task77/hotspots.json`。
类别互斥、按首次匹配归类；每格只保留最重的 45 个符号，其余记为 `unlisted`（31.4%），
所以下表类别份额是**下界**。

| 占比 | 样本 | 符号 |
|---|---|---|
| 14.64% | 82969 | libopenblas `sgemm_kernel_HASWELL` |
| 4.81% | 27284 | `Reanimation::Update` |
| 3.37% | 19074 | libc 内部函数（addr2line 只能落到最近的导出符号 `_dl_mcount_wrapper`） |
| 3.02% | 17138 | `Reanimation::Update` 的 track lambda（`operator()`，nm 显示落在 `Reanimation::Update` 内） |
| 2.37% | 13415 | libopenblas `sgemm_oncopy_HASWELL`（打包开销） |
| 2.21% | 12539 | `DataArray<Zombie>::Iterator::operator++` |
| 2.08% | 11786 | `DataArray<Plant>::Iterator::operator++` |
| 1.95% | 11046 | `PvzpParticleEmitter::UpdateParticle` |
| 1.88% | 10642 | `Sexy::MTRand::NextNoAssert` |
| 1.56% | 8833 | `Plant::Animate` |
| 1.34% | 7572 | libopenblas `dgemm_kernel_HASWELL` |
| 1.32% | 7501 | `EnvironmentRollout::Convolve` |
| 1.25% | 7096 | `PvzpParticleEmitter::Update` |
| 1.25% | 7078 | `Reanimation::NextEnvTrack` |
| 1.17% | 6647 | libopenblas `sgemm_itcopy_HASWELL` |
| 1.15% | 6509 | `EnvironmentValueNet::Convolve` |
| 1.13% | 6395 | `Plant::UpdateAbilities` |
| 1.12% | 6331 | `DataArray<Reanimation>::DataArrayTryToGet` |
| 1.05% | 5976 | `DataArray<Reanimation>::Iterator::operator++` |
| 1.03% | 5861 | libopenblas `dgemm_incopy_HASWELL` |

按类别（互斥，占全部样本）：

| 类别 | 合计 | day | night | pool | fog | roof |
|---|---|---|---|---|---|---|
| openblas | 20.55% | 25.6% | 24.6% | 16.0% | 16.0% | 15.3% |
| 动画 | 8.52% | 7.5% | 7.6% | 9.3% | 9.7% | 10.4% |
| `DataArray` | 7.25% | 6.0% | 7.0% | 7.8% | 8.3% | 8.3% |
| 粒子 | 4.62% | 2.7% | 3.7% | 7.7% | 3.3% | 1.2% |
| 值网络卷积 | 2.47% | 3.3% | 3.0% | 1.7% | 1.9% | 2.2% |
| libc | 4.17% | 5.3% | 4.4% | 3.4% | 3.6% | 4.8% |

- openblas 包含学生与值网络的矩阵运算；平面采样不能区分共享内核的调用者，二者分别耗时以 §3 的计数器为准。
- 动画合计 8.52%（`Reanimation::Update` + lambda + `NextEnvTrack` + `FloatTrackEvaluate` +
  `PvzpCurveEvaluate` + `GetTrackVelocity` + `TrackMemoHit`）：**不能跳**（共享 RNG，见 §1.4）。
- 粒子合计 4.62%：**不能跳**（共享 RNG）。
- `DataArray` 迭代器与查找合计 7.25%：2026-10-23 已做过四槽一判的扫描优化，这里是剩余成本。
- 两处 `Convolve` 合计 2.47%，但卷积只是值网络前向的一部分（矩阵乘法记在 openblas 里）：
  值网络前向整体占每决策 2.0–5.2%（§3），**那才是 §5 那两处改动的收益天花板**，与 §7.3 的计时一致。
- `_dl_mcount_wrapper` 不是 `-pg`：二进制里 `mcount` 符号计数为 0，这是 libc 无 debug 符号时的误标。
- 上一版表格（11.40% sgemm、动画 12.5% 等）是采样审计前只含 day/night 的汇总，已作废。

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
- **组件级计时**（`--time`，证据 `artifacts/task77/value_kernel_timing/timing.json`）：
  同一批冻结模型与输入，before/after 交替在同一核上跑，value/static 前向 **164.1 → 136.1 µs（0.829）**，
  逐轮比值 0.79–0.85 无重叠；同一次运行里 480+480 个 logit 仍逐字节相同（§7.3b）。
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

### 7.2 ABBA 配对计时（六族 × 2 种子 × A-B-B-A）

`artifacts/task77/abba_final/`：48 个完整老师局（6 族 × 2 种子 × 4 臂），`--workers 2 --count 2`。
A 臂 = 同 flags 的改前二进制 `51949eff…`，B 臂 = 候选 `b4999d65…`；四臂 config 除可执行文件外逐项相同。
`abba-verify` 审计通过：12 对全部逐位一致（36 次完整帧比较、40,686 帧；只排除 `verification.seconds`），
`state_hash` 12/36 不同（按 §7.1 的口径单独报告，不作策略门禁）。

| 族 | 每局比 B/A（种子 1 / 2） | geomean | 计时窗口 loadavg | 窗口内 qc 引擎数 |
|---|---|---|---|---|
| day | 1.123 / 1.086 | 1.104 | 11.9 – 34.3 | 13 – 21 |
| night | 0.865 / 0.838 | 0.851 | 14.4 – 29.2 | 8 – 19 |
| pool | 1.009 / 1.005 | 1.007 | 8.4 – 14.4 | 2 – 8 |
| conveyor | 1.018 / 1.016 | 1.017 | 6.6 – 8.5 | 0 – 2 |
| roof | 1.000 / 1.000 | 1.000 | 6.4 – 6.6 | 0 |
| fog | 0.996 / 1.000 | 0.998 | 6.1 – 6.6 | 0 |

- **六族合一**：geomean **0.993**，中位 1.003，配对 bootstrap 95% **[0.948, 1.035]**；
  按族整体重采样（同族两粒种子共享负载窗口）的 95% 区间 [0.926, 1.052]。区间覆盖 1。
- **qc 正常运行子集**（day + night，同一控制器 `508104` 在 8 个 block 边界都在场）：4 对，
  geomean 0.970，95% [0.851, 1.104]。
- **稳定负载子集**（conveyor + roof + fog，窗口内 loadavg 6.1–8.5、qc 引擎 0–2）：6 对，
  geomean **1.005**，95% **[0.998, 1.012]**，按族 [0.998, 1.017]。
  这一层不是按结果挑的，是按"ABBA 的恒定负载假设成立"挑的；它把"能快多少"的上界压到 **< 1.7%**。
- roof 的两个 ABBA 种子是短局（2.4–2.6 s、199/207 决策、第 4 波就输），`seconds` 只有 0.1 s 分辨率，
  所以 roof 的 1.000 是量化产物、不携带信息；它留在表里是为了不事后剔除。

**day/night 的 ±10–15% 不是补丁效应。** 每臂原始耗时（秒）：

| 种子 | A1 | B1 | B2 | A2 |
|---|---|---|---|---|
| day 5012288 | 111.4 | 135.5 | 138.6 | 132.7 |
| day 5012289 | 138.6 | 173.0 | 154.2 | 162.8 |
| night 5157696 | 221.2 | 147.8 | 173.7 | 150.6 |
| night 5157697 | 256.2 | 167.7 | 186.7 | 166.5 |

A 臂占 block 两端，两端的负载并不对称：day 的第一个 A 落在整块最轻的时刻（loadavg 11.9，其余 25–34），
night 的第一个 A 落在最重的时刻（29.2，随后降到 14.4）。"两个 A 取均值"因此继承了这个偏置。
night 更麻烦：B1 与 A1 同在 loadavg 29.2，却快 33%（147.8 vs 221.2），端点负载解释不了它；
不把它归因于补丁，只采用稳定负载子集与六族合一区间。
墙钟这条路到此为止——后来补的第二轮换了观测量，见 §7.3。

**首轮判定：逐位不变成立，提速不成立。** 稳定负载下 +0.5%（[0.998, 1.012]），
与 §4 的"两处 `Convolve` 只占引擎样本 2.47%"一致——即使这三处改动全部命中，整局上限也在 1–2%。
12 对的设计本身只有 ±4.5% 的分辨率，**小于 4% 的效应在这套设计下不可判定**；
要判定 1% 量级需要 40 对以上（约 3–4 小时独占机器）。
所以另做了一轮换观测量的实验 + 组件级计时，见 §7.3——它们把结论从"测不出"变成"测得准，且确实只有 0.3–0.9%"。

### 7.3 第二轮 ABBA（引擎 CPU 观测）与组件级计时

首轮（§7.2）的问题很具体：墙钟在别人占机器时会被排队污染，区间 ±4.5%，判不了 1%。所以补了两件事。

**(a) 换观测量：引擎进程 CPU 时间。** `abba_cpu/` 是全新一轮同设计实验（六族 × 2 种子 × A-B-B-A，
`--workers 2 --count 2`，48 局），每臂同时记墙钟与**引擎进程 utime+stime**
（`EngineCpu`：每 0.5 s 扫 `/proc/<pid>/exe` 认二进制，A/B 是两个不同文件，不会串）。
审计同样通过：12 对、36 次完整帧比较、40,686 帧一致，`state_hash` 13/36 不同。

| 族 | 墙钟比（种子 1 / 2） | 墙钟 geomean | 引擎 CPU 秒（A 均值 → B 均值） | CPU 比 |
|---|---|---|---|---|
| day | 1.038 / 1.035 | 1.037 | 132.8 → 137.6 | 1.036 |
| night | 0.993 / 0.998 | 0.995 | 223.0 → 221.7 | 0.994 |
| pool | 0.939 / 0.938 | 0.938 | 279.1 → 261.9 | 0.938 |
| conveyor | 0.967 / 0.971 | 0.969 | 179.7 → 174.0 | 0.968 |
| roof | 1.039 / 1.043 | 1.041 | 4.4 → 4.4 | 1.011 |
| fog | 0.965 / 0.962 | 0.964 | 88.5 → 85.3 | 0.964 |

- 六族合一：墙钟 geomean **0.990**（95% [0.969, 1.012]，按族 [0.961, 1.020]）；
  引擎 CPU geomean **0.985**（[0.959, 1.011]）。
- 这一轮负载轻（别的 agent 大多收工），墙钟与 CPU 在 pool/conveyor/fog 上几乎相等（0.938/0.938、0.969/0.968、
  0.964/0.964）——说明这三个族是 CPU 受限的，两个观测量一致；day/night/roof 仍不一致，是残余排队。
- **但逐族符号并不稳定**：同一补丁在 day/roof 上 CPU 略升（+3.6%/+1.1%），在 pool/conveyor/fog 上降 3–6%。
  这不是补丁会有的形状，而是 ±4% 的族级噪声；只有六族合一才有意义。

**(b) 组件级交错计时。** `task77_value_kernel_check.py --time`：用生产编译参数分别编 before（冻结源码）
与 after（当前源码），两者**交替**在同一个核（`taskset -c 19`）上跑，9 轮 × 400 次前向 × 2 个模型（value/static），
取中位数。交替是为了让漂移在两个版本上均摊。

| 版本 | value/static 前向中位数 | 相对 |
|---|---|---|
| before（`c0249d62` 源码） | 164.1 µs | 1.000 |
| after（本轮改动） | 136.1 µs | **0.829** |

18 个中位数里 before 163–167 µs、after 135–140 µs，逐轮比值 0.79–0.85，没有重叠。
同一次运行里 value/static 各 480 个 logit 仍逐字节相同（`9762c586…` / `f4f8f4a0…`）——
**快 17% 且逐位不变**。

**三者对得上。** 值网络前向（features+forward）占每决策耗时：day 5.2%、night 2.7%、pool 2.0%、fog 2.1%、roof 3.0%
（§3）。快 17% 只能换来整局 **0.3–0.9%**，正好落在两轮 ABBA 的点估计（0.985–0.993）里、
落在它们的区间宽度（±2–4%）之下。这就是"改得对、但整局看不出来"的完整解释。

### 7.4 第三轮 ABBA：qc 负载下的六族（墙钟 + 引擎 CPU）

前两轮各占一半：§7.2 有 qc 负载但只有墙钟、且只有 day/night 在同一个控制器下；§7.3 记了引擎 CPU 但机器基本空闲。
生产永远跑在 qc 负载里，所以补了这一轮：同设计（六族 × 2 种子 × A-B-B-A，`--workers 2 --count 2`，48 局），
每臂同时记墙钟与引擎进程 CPU，**24 个 block 边界上都是同一个 qc 控制器**（pid 626824，独立只读监控 535 个样本，
6–9 个 qc 引擎全程在场，loadavg 13–16）。

审计通过：12 对、36 次完整帧比较、40,686 帧一致；`state_hash` 18/36 不同（按 §7.1 的口径单独报告）。

| 族 | 墙钟比（种子 1 / 2） | 墙钟 geomean | 引擎 CPU 秒（A → B） | CPU 比 |
|---|---|---|---|---|
| day | 0.987 / 0.990 | 0.988 | 184.7 → 182.6 | 0.989 |
| night | 1.018 / 1.013 | 1.016 | 295.2 → 299.3 | 1.014 |
| pool | 1.016 / 1.018 | 1.017 | 376.8 → 383.1 | 1.017 |
| conveyor | 0.987 / 0.982 | 0.984 | 259.9 → 255.8 | 0.984 |
| roof | 0.988 / 0.947 | 0.967 | 7.0 → 6.8 | 0.973 |
| fog | 1.056 / 1.058 | 1.057 | 120.8 → 127.5 | 1.056 |

- 六族合一：墙钟 geomean **1.0045**（95% [0.9874, 1.0221]，按族 [0.9823, 1.0298]），
  引擎 CPU **1.0050**（[0.9846, 1.0286]）。这是三轮里最窄的区间（±1.8%），仍然覆盖 1。
- 族级符号依旧不稳定：day / conveyor / roof < 1，night / pool / fog > 1。fog 的 +5.6% 在两粒种子、
  墙钟与 CPU 上都一致，但同一族在 §7.3 是 −3.6%，方向翻转；每族只有 2 粒种子、族级噪声约 ±4%，
  因此不把它归因于补丁，只采用六族合一的区间（与 §7.2 的口径相同）。
- 三轮的点估计：0.993（qc 负载、墙钟）/ 0.990（轻载、墙钟）· 0.985（轻载、CPU）/ 1.0045（qc 负载、墙钟）· 1.0050（qc 负载、CPU），
  区间全部覆盖 1。**结论不变：逐位不变成立，整局没有可测的提速；补丁的价值仍在组件级（§7.3b）。**

## 8. 局限与后续

- **候选二进制不带 PGO。** 生产 deploy4 带本机 PGO（历史实测 18%）。ABBA 的 A 臂因此用同 flags 重编的
  `pvz-portable-release-before`，只量代码差异；真正上线要由主控按 PGO 两阶段重建，PGO 与补丁的交互收益尚未测量。
  **不要把本文的比值直接当成上线后的墙钟比。**
- **整局计时只能判到 1–2% 量级。** 三轮各 12 对 ABBA 的区间分别是 ±4.5%（qc 负载、只有墙钟）、
  ±2%（负载轻）、±1.8%（qc 负载、墙钟与引擎 CPU 同时记，§7.4），引擎 CPU 观测是 ±2.2–2.6%。族级噪声约 ±4%，符号都不稳定（§7.3）。要在生产负载下判定 0.5% 需要 40 对以上、
  机器独占 3–4 小时；本轮在"同时最多 2 个游戏进程、qc 优先"的约束下没有做。
  另外 `search_play` 记录的每局耗时是 `time.time()` 差、保留 0.1 s（`scripts/search_play.py:790`），
  40–250 s 的对局对应 0.04–0.25% 的量化，短局（roof 2.4 s）则是 4%。
  **能判到 1% 以下的是组件级交错计时（§7.3b），但它只覆盖值网络前向，不能外推到整局。**
- **外部负载无法控制。** 计时窗口内有别的 agent 在跑（`all_engines` 2–23），qc 引擎数也在变；
  按负载分层只能缓解、不能消除。引擎进程 CPU 时间已作为第二观测量实现（`EngineCpu`，§7.3a），
  它在 pool/conveyor/fog 上与墙钟一致、在 day/night/roof 上仍受残余排队影响。要彻底干净只能独占机器重跑。
- 后续可研究共享 BLAS 内核和动画 track 更新；本轮没有实现新的 GEMM，也没有跳过额外动画生命周期。
  任何后续方案都需独立验证浮点求和顺序与 track 重新激活行为，不能把移动零块或稀疏跳过直接当成保序优化。
  按 §4 的占比，真正值得投入的仍是 rollout 引擎步（50–65%）与学生前向（20–32%）。
- 采样器的 libc 符号是误标的（无 debug 信息）。要看清 libc 那 4.17% 得装 `libc6-dbg` 或给 `addr2line` 指定 debug 路径。

- **试过但回退：学生网 im2col 缓冲按层拆分。** `EnvironmentRollout::Convolve` 的三层（0/2/4）
  共用一块 im2col 缓冲，而三层的 feature 数不同，零边因此在每个前向被重建三次。改成每层一块
  （零边只在通道集变化或缓冲增长时重建）后，用 `scripts/rollout_bench.py` 做 9 轮 A/B
  （单核 pin、同任务同种子、每轮 126 局 rollout、A/B 交替）：引擎自报
  `network_seconds / net_calls` 的比值为 **0.988（0.967–1.008）**，`sim_μs/tick` 对照 1.005，
  即无可测收益，已回退、未进候选。学生前向受 sgemm 主导（§4 的 openblas 20.55%），
  缓冲清零不是可测份额；同一条路上的其他微优化（迭代器、分支前置）收益更小，不必再试。
