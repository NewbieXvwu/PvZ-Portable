# task77：老师搜索的热路径——每决策耗时花在哪，能不能逐位不变地再快一截

> 2026-10-25。工作树 `perf2`（分支 `perf2`，从 origin/pvz-env 拉出）。
> 原始数据只在 ignored 的 `artifacts/task77/`；本文件放结论与复核口径。
> 工具：`scripts/task77_search_hotpath.py`（prepare / profile / profile-summary / hotspots / gate / verify / abba）、
> `scripts/task77_value_kernel_check.py`（离线逐位检查 + 交错计时）、`python/test_search_hotpath.py`（证据回归）。

## 1. 结论

1. **每决策 74.8–161.0 ms（五地形；传送带补测 151.9 ms，§3.1）。两块合计占 76–86%：
   rollout 引擎步（50.3–65.3%）与引擎内学生前向（19.7–32.0%）。**
   值网络前向（叶值 0.8–1.6% + 静态值 0.9–4.3%）、显式快照往返（0.3–0.7%）、IPC（3.6–8.1%）、
   Python 侧（5.7–10.0%）及真实对局推进等，其余组件合计占 14–24%。传送带是例外：静态值合计 10.6%
   （每决策 37.3 次 `STATIC_SCORES`，§3.1），所以那张表不能直接套到第六族。
2. 引擎进程 CPU 为整局墙钟的 57–78%（还包含 Python、IPC 和系统调度等成本，不能把差额全算成排队），
   其中 **89–95% 落在 `ROLLOUT_VALUE` 一条命令里**。
3. 采样集中在模拟对象自身（26.4%）、BLAS（21.6%）、`DataArray` 遍历（10.7%）、动画（10.4%）
   与粒子（7.2%）（§4）；平面 PC 采样不能把 BLAS 内部样本精确分给学生与值网络，
   二者分别耗时以 §3 的引擎计数器为准。
4. **本轮保留 BLAS 算术及动画/粒子生命周期。** sgemm 删零可能改变求和顺序；
   动画与粒子的稀疏跳过 2026-10-23 已经做过
   （`Reanimator.cpp` 的 sparse 路径、`ENV_LIGHT_ANIMATION`），剩下的 track 更新与粒子会消耗共享 RNG
   ——`PvzpCommon.h` 记着：跳过粒子让 16/64 局结果翻转。
5. 所以优化挑的是**占比最大且能证明逐位不变**的三处实现细节（§5）：
   值网络卷积的核打包缓存 + 无填充 memcpy、`DataArray` 取值改走 `get()`、`Reanimation::Update` 模式判断前置。
   离线核检查的 value/static 各 480 个 logit（各含 3 遍）输出 SHA256 完全相同。
   同一工具还做**交错计时**（同核 pin、before/after 交替 9 轮 × 400 次、取中位数）：
   value/static 前向 **164.1 → 136.1 µs，快 17%**（§7.3）。整局结论仍只采用 ABBA，微基准只作机制核对。
6. **测试面板上逐位不变；组件级有加速，尚未测出整局加速。**
   - 组件级：value/static 前向快 **17%**（§7.3）。
   - 整局：两轮 ABBA（各 12 对、48 局）每局耗时比 geomean **0.993**（95% [0.948, 1.035]，qc 负载下的首轮）
     与 **0.990**（[0.969, 1.012]，负载较轻的第二轮）；第二轮同时记引擎进程 CPU 时间，比值 **0.985**（[0.959, 1.011]）。
     第三轮在**六族全程同一个 qc 控制器**下同时记墙钟与引擎 CPU：**1.0045**（[0.987, 1.022]）、CPU **1.0050**（[0.985, 1.029]）——三轮的点估计在 0.99–1.005 之间、区间都覆盖 1。
     第四轮改用**两侧同负载重训的 PGO 二进制**：墙钟 **1.0144**（[0.989, 1.045]）、CPU **1.0161**（[0.983, 1.064]），
     点估计仍覆盖 1；其中 day 的 +11.5% 来自唯一没有负载监控的 A 臂（同二进制两次差 26–29%），
     day-only 复测为 **0.979**，去掉 day 的其余五族为 0.995（§7.5）。
   - 值网络前向占每决策耗时 2.0–5.2%（§3）。假定微基准收益能原样转移、调用分布不变，
     简单外推这一部分可节省整局 **约 0.3–0.9%**；这是估算，不是整局实测。`DataArray` 与动画两处改动的独立收益没有分离测量。
   - **目前不建议仅为提速单独部署。** 已测面板的行为一致性支持随下一次引擎重建评估带上，部署由主控决定。
     后续优先研究 rollout 引擎步与学生前向（合计 76–86%），仍需验证浮点求和顺序和生命周期语义（§8）。

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
| 同负载训练的 PGO 改前二进制（§7.5 的 A 臂） | `ca7d9283ad42117c1a184cb655e729496332f74cccb416e522d55aa8f73d3704` |
| 同负载训练的 PGO 候选二进制（§7.5 的 B 臂） | `a5098255f7fff4b3e602860c35c76a468bef30766bfbcc96899710c626d4c71d` |

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
# 传送带补测（§3.1）：默认面板仍是五地形，选到非默认面板时汇总另存，不覆盖 profile_summary.json
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py profile --out $P/artifacts/task77 --cells conveyor
$PY $P/scripts/task77_search_hotpath.py profile-summary --out $P/artifacts/task77 --cells conveyor
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py gate --out $P/artifacts/task77 --candidate $P/artifacts/task77/frozen/pvz-portable-candidate
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_final --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
# 第二轮：同设计，另建目录，并记引擎进程 CPU（§7.3a）
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_cpu --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
$PY $P/scripts/task77_search_hotpath.py abba-verify --out $P/artifacts/task77/abba_cpu --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
# 第三轮：qc 负载下的同设计 ABBA（六族全程同一控制器，§7.4）
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_qc --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
$PY $P/scripts/task77_search_hotpath.py abba-verify --out $P/artifacts/task77/abba_qc --candidate $P/artifacts/task77/frozen/pvz-portable-candidate --baseline $P/artifacts/task77/frozen/pvz-portable-release-before --workers 2 --count 2
# 第四轮：两份 PGO 的 ABBA（§7.5）
# 两份 PGO 用同一流程、同一训练负载（day L7、种子 5012288/5012289、2 进程）各自两阶段构建：
#   基线 = c0249d62 的临时 worktree；补丁 = 本工作树；均 -O3 -flto=auto -ffp-contract=off -g1
#   阶段1 -fprofile-generate=<data> → 跑上面的 search_play 训练负载 → 阶段2 -DPVZ_PGO_DATA=<data>
# 产物：artifacts/task77/pvz-pgo-base（ca7d9283…）、artifacts/task77/pvz-pgo-cand（a5098255…）
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_pgo --candidate $P/artifacts/task77/pvz-pgo-cand --baseline $P/artifacts/task77/pvz-pgo-base --workers 2 --count 2
$PY $P/scripts/task77_search_hotpath.py abba-verify --out $P/artifacts/task77/abba_pgo --candidate $P/artifacts/task77/pvz-pgo-cand --baseline $P/artifacts/task77/pvz-pgo-base --workers 2 --count 2
# day 单元复测：主轮 day 的 A0 臂跑在负载监控启动之前（§7.5），另建目录整轮重跑该族
PVZ_NO_POPUP=1 $PY $P/scripts/task77_search_hotpath.py abba --out $P/artifacts/task77/abba_pgo_day --candidate $P/artifacts/task77/pvz-pgo-cand --baseline $P/artifacts/task77/pvz-pgo-base --workers 2 --count 2 --cells day
$PY $P/scripts/task77_search_hotpath.py abba-verify --out $P/artifacts/task77/abba_pgo_day --candidate $P/artifacts/task77/pvz-pgo-cand --baseline $P/artifacts/task77/pvz-pgo-base --workers 2 --count 2 --cells day
# 组件级：逐位核检查 + 交错计时（§7.3b）
$PY $P/scripts/task77_value_kernel_check.py --time
# 结论层复核：§3 / §3.1 / §4 与 §7.5 的每个数字都从证据复算，不匹配就抛错
$PY $P/scripts/task77_audit_report.py --out $P/artifacts/task77
$PY $P/scripts/task77_audit_pgo.py --out $P/artifacts/task77
```

两个审计脚本放在 `scripts/`（随 git 走），默认 `--out` 指向本仓库的 `artifacts/task77`；
它们把报告里的期望值写成断言，所以结论层不依赖 ignored 目录里的临时脚本。

`--cells` 只收窄面板（用于补测单族），不改变四臂配对、`--count` 或统计口径；`abba`、`abba-verify`、`profile`、
`profile-summary` 都接受。`profile` 默认仍是五地形面板；`profile-summary` 只在选择等于五地形时写
`profile_summary.json`，否则写 `profile_summary_<cells>.json`（显式列出那五族仍算默认）。
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

- **day 与传送带的静态值明显偏重**（day 6.85 + 6.62 = 13.5 ms/决策、8.4%；传送带 16.1 ms、10.6%，§3.1）：
  这两族的 `STATIC_SCORES` 调用最多（17.5 / 37.3 次每决策，其余四族 2.7–6.3），引擎 CPU 也最高
  （day 42.1 s，其余族 5–22 s）。也就是说值网络这条路的优化只在这两族有可观收益。
- **屋顶与雾的 Python 侧更高**（10.0% / 7.0%）：对局短、决策少，固定开销摊不薄。
- **Python 侧没有单一可优化点。** 8 局的 `cProfile` 里 90.8% 的 tottime 落在
  `pvz_env.py:376 _read_message`——那是阻塞读、在等引擎算完，属于命令内时间而不是 Python 计算。
  去掉这一项后剩下的 143 s / 8 局（cProfile 放大约 1.8 倍，对应 worker CPU 9.9 s/局）分散在
  `torch.conv2d` 0.42%、json 解码 0.39%、`_io.open` 0.39%、`v2_extra` 0.18% 等，最大一项不到半成；
  worker CPU 只占整局墙钟 5.8%（9.87 s / 170 s）。所以这一层没有“改一处省一截”的空间。

### 3.1 传送带（补测，2026-10-25 22:44）

验收面板有六族，但 §3 的主面板只覆盖五地形——`search-hotpath.md` 早先记着"传送带没有剖析局，旧臂老师现场打"。
按同一冻结输入、同一 `command_for` 形态补 4 局（种子 5083968–5083971，`profile --cells conveyor`，
另存 `artifacts/task77/profile_summary_conveyor.json`，不覆盖 §4 的 20 局样本集）：

| 组件 | conveyor |
|---|---|
| rollout 引擎步 | 80.44 |
| 引擎内学生前向 | 40.97 |
| 叶值网络 | 1.08 |
| 静态值网络 | 8.22 |
| 显式快照往返 | 0.33 |
| 静态分支恢复/动作/IPC（残差） | 7.89 |
| rollout 恢复/控制/IPC（残差） | 3.04 |
| 其他命令 | 2.18 |
| Python 侧 | 7.73 |
| **合计** | **151.9** |

rollout+学生 = 79.9%（rollout 53.0% / 学生 27.0%），与五地形同量级。**但静态值这条路是六族里最重的**：
8.22 + 7.89 = 16.1 ms/决策（10.6%），高于 day 的 13.5 ms（8.4%）。原因是候选动作数：
传送带每决策 **37.3 次** `STATIC_SCORES`，day 17.5、night 6.3、pool 5.8、fog 3.1、roof 2.7。
也就是说"优化值网络/静态评分"只在传送带与白天族有可观收益，其余四族天花板不到 4%。

## 4. 采样热点（最终版，`hotspots` 阶段重算）

审计发现泳池 `pcs.522801.bin` 的两段数组长度不一致（多 4 字节），旧读法丢尾后可能错认数组边界。
已修复 SIGPROF 导出期间计数仍增长的问题，读取器严格拒绝坏长度/模块号；原文件保留、明确排除
（`invalid_sample_files`），并用修复后的采样器按同一任务与种子补采 1 局（`profile_recovery/`）。
最终样本 = day 4 + night 4（`profile_complete/`）+ pool 3 + 补采 1 + fog 4 + roof 4 = 20 局，
**566,659 个样本**，由 `hotspots` 阶段从 `profile_summary.json` 汇总成 `artifacts/task77/hotspots.json`。
每格保存**完整**符号直方图，不再截断 top-45。上一版每格只留 45 个符号，`unlisted` 占 31.4%，
而截断位次会随符号解析粒度变化（同一批 libc 偏移一次落到 `_dl_mcount_wrapper`、另一次落到
`__memset_avx2_unaligned_erms`），类别份额既只是下界、又不可复现。现在 `unlisted` 为 0，
类别份额是精确值，且只取决于模块名与符号前缀，不随解析粒度改变。类别互斥、按首次匹配归类。

| 占比 | 样本 | 符号 |
|---|---|---|
| 14.64% | 82969 | libopenblas `sgemm_kernel_HASWELL` |
| 4.81% | 27284 | `Reanimation::Update` |
| 3.02% | 17138 | `operator()` 汇总（含 `Reanimation::Update` 的 track lambda；同名符号不能全归给这一调用者） |
| 2.37% | 13415 | libopenblas `sgemm_oncopy_HASWELL`（打包开销） |
| 2.21% | 12539 | `DataArray<Zombie>::Iterator::operator++` |
| 2.08% | 11786 | `DataArray<Plant>::Iterator::operator++` |
| 1.96% | 11132 | `PvzpParticleEmitter::UpdateParticle` |
| 1.88% | 10642 | `Sexy::MTRand::NextNoAssert` |
| 1.60% | 9042 | libc `__memset_avx2_unaligned_erms` |
| 1.56% | 8833 | `Plant::Animate` |
| 1.34% | 7572 | libopenblas `dgemm_kernel_HASWELL` |
| 1.32% | 7501 | `EnvironmentRollout::Convolve` |
| 1.25% | 7096 | `PvzpParticleEmitter::Update` |
| 1.25% | 7078 | `Reanimation::NextEnvTrack` |
| 1.17% | 6647 | libopenblas `sgemm_itcopy_HASWELL` |
| 1.15% | 6509 | `EnvironmentValueNet::Convolve` |
| 1.13% | 6395 | `Plant::UpdateAbilities` |
| 1.12% | 6331 | `DataArray<Reanimation>::DataArrayTryToGet` |

按类别（互斥，按首次匹配归类，占全部样本）：

| 类别 | 合计 | day | night | pool | fog | roof |
|---|---|---|---|---|---|---|
| openblas | 21.60% | 26.9% | 25.8% | 16.9% | 17.0% | 16.2% |
| 模拟对象（`sim`） | 26.44% | 22.6% | 24.7% | 28.5% | 29.9% | 31.1% |
| 动画 | 10.37% | 9.6% | 9.2% | 11.1% | 11.6% | 13.0% |
| `DataArray` | 10.72% | 8.8% | 10.2% | 11.7% | 12.5% | 11.2% |
| 粒子 | 7.23% | 5.3% | 6.2% | 10.6% | 5.6% | 3.0% |
| 学生/值网络卷积前处理 | 2.59% | 3.5% | 3.1% | 1.8% | 2.0% | 2.3% |
| libc | 4.71% | 5.8% | 4.9% | 3.8% | 4.2% | 5.8% |
| 其他（未归类） | 16.35% | 17.6% | 15.8% | 15.7% | 17.1% | 17.2% |
| `unlisted` | 0.00% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

- **`sim` 是新增的一类**（26.44%）：`Board::` / `Zombie::` / `Plant::` / `Projectile` / `Coin::` /
  `GridItem` / `LawnMower` / `EffectSystem` / `Attachment::` / `CursorPreview` / `LawnApp::` /
  `MTRand` 前缀，即推进对象本身的函数。上一版没有这一类，它整个藏在 `unlisted` 里。
  它和 §3 的 rollout 引擎步（50–65%）不是同一口径：动画、粒子、`DataArray` 遍历里也有相当部分
  是在模拟循环里被调用的，所以 §3 的引擎步样本分散在这四个类别中，类别之间也不能直接相加。
- openblas 21.60% 包含学生与值网络的矩阵运算；平面采样不能区分共享内核的调用者，
  二者分别耗时以 §3 的计数器为准。
- 动画合计 10.37%（`Reanimation::Update` + lambda + `NextEnvTrack` + `FloatTrackEvaluate` +
  `PvzpCurveEvaluate` + `GetTrackVelocity` + `TrackMemoHit`）：**不能跳**（共享 RNG，见 §1.4）。
- 粒子合计 7.23%：**不能跳**（共享 RNG）。
- `DataArray` 迭代器与查找合计 10.72%：2026-10-23 已做过四槽一判的扫描优化，这里是剩余成本。
- 两处 `Convolve` 合计 2.59%（证据内沿用键名 `value_net`），其中学生网 1.32%、值网络 1.15%；
  矩阵乘法另记在 openblas 里。值网络整体占每决策 2.0–5.2%（§3），可用于估算值网络优化的量级，
  但 **2.59% 不是三处补丁的收益上限**，因为补丁也改了 `DataArray` 和动画。
- **libc 4.71% 现在与按模块统计的 26,664 / 566,659 完全一致**：完整直方图下没有符号被截掉。
  最大的一项是 `__memset_avx2_unaligned_erms` 1.60%，与 `scripts/task77_libc_classify.py` 按采样地址
  的指令窗口分类（memset `rep stos` 26.9% = 占全部 1.26%）同向——分类器只认指令特征，范围更窄。
  同一工具还给出 AVX 向量拷/写 0.93%、`rep movs` 0.22%、其余 2.29%（未分类）。
  `libc6-dbg` 在这台机上装不上（apt 走代理返回 403），所以只能到这个粒度；
  它既不能确定所有未分类样本的函数，也不能确定清零/拷贝的调用者。
- **其他 16.35%** 是长尾：`operator()` 3.02%（lambda 聚合，不能归给单一调用者）、
  `Linear`/`IdentityLinear` 1.01%（值网络非 BLAS 路径）、`libz crc32_z` 0.63%、
  `memcpy` 0.62% 等；不再往下归因。
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
  重打只会得到同一局。传送带当时没有剖析局，旧臂老师现场打（§3.1 之后才补了 4 局，
  但门禁已在 15:14 跑完，不改用它、也不重跑）。

- 门禁仍然描述当前树：四个改动文件最后一次提交是 `740671ed`（门禁在 15:14 跑，早于其后的任何提交），
  之后 `src/` 无新提交、工作区干净，冻结候选 `b4999d65…` 与改前 `51949eff…` 的 SHA256 未变。
  剖析集五地形各 4 局共 20 局（`profile_summary.json` 的 `profiles` 列表，泳池早先那份坏样本已由 `profile_recovery/` 补上）。

### 7.2 ABBA 配对计时（六族 × 2 种子 × A-B-B-A）

`artifacts/task77/abba_final/`：48 个完整老师局（6 族 × 2 种子 × 4 臂），`--workers 2 --count 2`。
A 臂 = 同 flags 的改前二进制 `51949eff…`，B 臂 = 候选 `b4999d65…`；四臂 config 除可执行文件外逐项相同。
`abba-verify` 审计通过：12 对全部逐位一致（36 次完整帧比较、40,686 帧；只排除 `verification.seconds`），
`state_hash` 12/36 不同（按 §7.1 的口径单独报告，不作策略门禁）。

| 族 | 每局比 B/A（种子 1 / 2） | geomean | 计时窗口 loadavg | 旧监测器引擎树计数* |
|---|---|---|---|---|
| day | 1.123 / 1.086 | 1.104 | 11.9 – 34.3 | 13 – 21 |
| night | 0.865 / 0.838 | 0.851 | 14.4 – 29.2 | 8 – 19 |
| pool | 1.009 / 1.005 | 1.007 | 8.4 – 14.4 | 2 – 8 |
| conveyor | 1.018 / 1.016 | 1.017 | 6.6 – 8.5 | 0 – 2 |
| roof | 1.000 / 1.000 | 1.000 | 6.4 – 6.6 | 0 |
| fog | 0.996 / 1.000 | 0.998 | 6.1 – 6.6 | 0 |

注：此列来自修复前的宽泛进程树匹配，会误计 tmux 祖先树内其他引擎，不能视为真实 qc 引擎数。
qc 在场判定使用另行精确匹配控制器 argv 的只读监控；全程生产负载的完整六族结果见 §7.4。

- **六族合一**：geomean **0.993**，中位 1.003，配对 bootstrap 95% **[0.948, 1.035]**；
  按族整体重采样（同族两粒种子共享负载窗口）的 95% 区间 [0.926, 1.052]。区间覆盖 1。
- **qc 正常运行子集**（day + night，同一控制器 `508104` 在 8 个 block 边界都在场）：4 对，
  geomean 0.970，95% [0.851, 1.104]。
- **较稳定负载子集**（conveyor + roof + fog，窗口内 loadavg 6.1–8.5、精确监控未见 qc 控制器）：6 对，
  geomean **1.005**，95% **[0.998, 1.012]**，按族 [0.998, 1.017]。
  此分层依据负载记录，仍不能证明负载恒定；不将这个子集区间推广为生产吞吐的上界。
- roof 的两个 ABBA 种子是短局（2.4–2.6 s、199/207 决策、第 4 波就输），`seconds` 只有 0.1 s 分辨率，
  所以 roof 的 1.000 是量化产物、不携带信息；它留在表里是为了不事后剔除。

**day/night 的大幅差异无法可靠归因补丁。** 每臂原始耗时（秒）：

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

**首轮判定：测试面板行为一致，未测出整局提速。** 较稳定负载子集比值 1.005（[0.998, 1.012]），
完整 12 对区间约 ±4.5%，不足以判断 1% 量级的收益；这个区间宽度是本轮观测，不能当成设计的固定分辨率。
因此另做引擎 CPU 观测与组件级计时（§7.3）。组件计时只能证明值网络部分加快，不能证明整局净收益是 0.3–0.9%。

### 7.3 第二轮 ABBA（引擎 CPU 观测）与组件级计时

首轮（§7.2）的问题很具体：墙钟在别人占机器时会被排队污染，区间 ±4.5%，判不了 1%。所以补了两件事。

**(a) 换观测量：引擎进程 CPU 时间。** `abba_cpu/` 是全新一轮同设计实验（六族 × 2 种子 × A-B-B-A，
`--workers 2 --count 2`，48 局），每臂同时记墙钟与**引擎进程 utime+stime**
（`EngineCpu`：每 0.5 s 扫 `/proc/<pid>/exe` 认二进制，A/B 是两个不同文件，不会串）。
记录值是每个进程最后一次观测到的累计 CPU 时间，退出前尾段可能漏采；名义上可达约 0.5 s/进程，
实际轮询延迟还受调度影响。因此短屋顶局的 CPU 比尤其不宜精读。
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
- 这一轮负载较轻，墙钟与 CPU 在 pool/conveyor/fog 上的比值接近（0.938/0.938、0.969/0.968、
  0.964/0.964）。day/night/roof 的差异可能含调度、Python/IPC 与 CPU 尾段漏采，不能只归因排队。
- **但逐族符号并不稳定**：同一补丁在 day/roof 上 CPU 略升（+3.6%/+1.1%），在 pool/conveyor/fog 上降 3–6%。
  每族只有两粒种子且共用负载窗口，这些差异的来源无法可靠分离；跨轮方向变化也不能单独证明补丁没有影响。

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

值网络前向（features+forward）占每决策耗时：day 5.2%、night 2.7%、pool 2.0%、fog 2.1%、roof 3.0%
（§3）。将微基准的 17% 直接乘以这些份额，得到整局节省 **约 0.3–0.9%** 的简单估算。
它小于现有整局区间的宽度，但调用分布、缓存与 PGO 都可能改变可转移收益；整局净效应仍未确定。

### 7.4 第三轮 ABBA：qc 负载下的六族（墙钟 + 引擎 CPU）

前两轮各占一半：§7.2 有 qc 负载但只有墙钟、且只有 day/night 在同一个控制器下；§7.3 记了引擎 CPU 但机器基本空闲。
为覆盖 qc 持续运行的生产场景，补了这一轮：同设计（六族 × 2 种子 × A-B-B-A，`--workers 2 --count 2`，48 局），
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
  墙钟与 CPU 上都一致，但同一族在 §7.3 是 −3.6%，方向翻转；每族只有 2 粒种子，跨轮存在百分数级漂移。
  这些证据不足以把变化可靠归因于补丁或纯噪声，报告保留逐族结果，并以预定六族汇总描述本面板。
- 三轮的点估计：0.993（qc 负载、墙钟）/ 0.990（轻载、墙钟）· 0.985（轻载、CPU）/ 1.0045（qc 负载、墙钟）· 1.0050（qc 负载、CPU），
  区间全部覆盖 1。**结论不变：逐位不变成立，整局没有可测的提速；补丁的价值仍在组件级（§7.3b）。**

### 7.5 两份匹配 PGO 的 ABBA

按主控后续指示自编两份 PGO：A 的源码取 `c0249d62` 临时工作树，B 取本树的四文件补丁。
两侧都先用 `-fprofile-generate` 构建，运行相同训练负载（`day_l7_train_d7_m125`，
种子 5012288/5012289，老师 β=1、2 workers），各产生 141 个 `.gcda`，再用
`-O3 -flto=auto -ffp-contract=off -g1` 和 `-fprofile-use` 重建。二进制不含 gcov 训练期运行时。
基线临时工作树已清理；两个产物与构建脚本保留在 ignored 的 `artifacts/task77/`，完整 SHA256 见 §2。

本轮用全新 `abba_pgo/` 目录，按六族 × 2 种子 × A-B-B-A、`--workers 2 --count 2`（48 局），
同时记墙钟、引擎 CPU 与只读负载监控。24 个 block 边界都是同一个 qc 控制器（pid 626824，552 个监控样本）；
边界快照 loadavg 12.3–25.9、qc 引擎 7–10 个，样本里另有 2 个采样点落在 qc 换轮的空档（0 引擎）。

| 族 | 墙钟 geomean | 95% 区间 | 中位 | 引擎 CPU 比 | CPU 秒（A→B） |
|---|---|---|---|---|---|
| day | 1.1152 | [1.0919, 1.1391] | 1.1155 | 1.1342 | 181.4 → 205.7 |
| night | 0.9642 | [0.9588, 0.9696] | 0.9642 | 0.9733 | 275.1 → 267.8 |
| pool | 0.9737 | [0.9692, 0.9781] | 0.9737 | 0.9855 | 393.9 → 388.2 |
| conveyor | 1.0270 | [1.0246, 1.0295] | 1.0270 | 1.0295 | 249.2 → 256.5 |
| roof | 1.0200 | [1.0141, 1.0260] | 1.0200 | 0.9915 | 6.5 → 6.4 |
| fog | 0.9932 | [0.9930, 0.9933] | 0.9932 | 0.9912 | 127.6 → 126.5 |

- **六族合一**：墙钟 geomean **1.0144**（95% [0.9886, 1.0446]，按族重采样 [0.9808, 1.0578]），中位 1.0037；
  引擎 CPU **1.0161**（[0.9834, 1.0638]），中位 0.9913。
- 完整帧审计 `pass`：12 对、36 次比较、40,686 帧逐位一致；36 次比较里 state_hash 13 处不同、23 处相同（与前三轮同样的口径：hash 只作观察项，逐位判据是完整帧）。
  两侧二进制在测量后被复核 SHA256 未变。

**day 的 +11.5% 归因于该臂的窗口，不是二进制。** day 的 A0 臂（20:18:11–20:19:46）跑在负载监控启动
（20:19:47）之前，是 24 个臂里唯一没有监控样本的；同一个 A 二进制跑两次，引擎 CPU 158.3 s 与 204.5 s
（+29%）、墙钟 96.9 s 与 122.2 s（+26%），这个离散度比 A/B 之差（181.4 vs 205.7 s）还大。
因此在 `abba_pgo_day/` 单独补了一轮 day-only（同设计、全程监控：85 个样本，4 个边界同为 626824，
qc 引擎 1–8 个，loadavg 15.3–22.9）：2 对、6 次完整帧、6,411 帧一致，墙钟 **0.9790** [0.9760, 0.9820]、
CPU **0.9780**；同一臂内 A 的两次为 176.6 / 181.6 s（差 3%），面板比主轮紧得多。**与主轮的 +11.5% 反号。**

- 去掉 day 后其余五族 geomean **0.9953**；把 day 换成复测的敏感性估计（12 对）为 **0.9926** [0.9796, 1.0064]。
  这只是敏感性分析，主轮结果原样保留、不覆盖、不拼接。

训练负载一致有助于比较补丁在 PGO 下的表现，但这套新训练 profile 与历史生产 deploy4 的 profile 不同，
因此不能直接当成上线吞吐比。结论与 §7.2–§7.4 一致：**逐位不变成立，PGO 下同样没有可测的整局提速。**

## 8. 局限与后续

- **前三轮使用无 PGO 的匹配构建，第四轮用同负载重训的 PGO。** 生产 deploy4 带本机 PGO（历史实测 18%），
  所以前三轮 A 臂用同 flags 重编的 `pvz-portable-release-before`；§7.5 另用相同负载重新训练两份 PGO，
  墙钟 1.0144 [0.989, 1.045]、CPU 1.0161 [0.983, 1.064]，同样没有可测收益。两类实验分别报告，
  都不直接代表历史生产引擎上线后的墙钟比。部署由主控执行。
- **当前整局区间仍有百分数级宽度。** 三轮各 12 对 ABBA 的区间分别约 ±4.5%（部分 qc 负载、只有墙钟）、
  ±2%（负载轻）、±1.8%（qc 负载、墙钟与引擎 CPU 同时记，§7.4），引擎 CPU 观测约 ±2.2–2.6%。
  同族种子共享负载窗口，所以同时给出按族重采样区间。若未来要判断亚百分数收益，需更多独立 block 与功效设计；
  本文不声称“40 对、3–4 小时”即可保证检出，本轮也未开展这种扩样。
  另外 `search_play` 记录的每局耗时是 `time.time()` 差、保留 0.1 s（`scripts/search_play.py:790`），
  40–250 s 的对局对应 0.04–0.25% 的量化，短局（roof 2.4 s）则是 4%。
  **能判到 1% 以下的是组件级交错计时（§7.3b），但它只覆盖值网络前向，不能外推到整局。**
- **外部负载无法控制。** 计时窗口内有别的 agent 在跑（`all_engines` 2–23），qc 引擎数也在变；
  按负载分层只能缓解、不能消除。引擎进程 CPU 时间已作为第二观测量实现（`EngineCpu`，§7.3a），
  CPU 时间不计等待调度的时间，但仍随频率、缓存/内存竞争和执行路径改变，且本工具有尾段漏采（§7.3a）。
  负载监控不能证明 A/B 运行条件完全相同。
- 后续可研究共享 BLAS 内核和动画 track 更新；本轮没有实现新的 GEMM，也没有跳过额外动画生命周期。
  任何后续方案都需独立验证浮点求和顺序与 track 重新激活行为，不能把移动零块或稀疏跳过直接当成保序优化。
  按 §4 的占比，真正值得投入的仍是 rollout 引擎步（50–65%）与学生前向（20–32%）。
- 采样器的 libc 符号是误标的（无 debug 信息），`libc6-dbg` 在本机装不上（代理 403）。已用采样地址的
  指令窗口分类代替（`scripts/task77_libc_classify.py`，§4）：libc 4.71% 里 memset 1.26%、AVX 拷贝 0.93%、
  `rep movs` 0.22%，其余 2.29%；这个口径只能回答“是不是 memset/拷贝”，不能回答谁调用的。

- **§4 的类别份额现在可复现，逐符号表不可。** 每格保存完整直方图后，类别份额只取决于模块名
  与符号前缀，`unlisted` 为 0；同一批 `pcs.*.bin` 重跑 `profile-summary` + `hotspots` 得到相同类别值。
  逐符号的 top 表仍会随 addr2line 的解析粒度变化（同一批 libc 偏移可能落到
  `_dl_mcount_wrapper`，也可能落到 `__memset_avx2_unaligned_erms`），所以表中 libc 行的具体符号名
  不能当稳定证据，libc 的**份额**可以（按模块统计）。§3 用引擎计数器，与符号解析无关。

- **试过但回退：学生网 im2col 缓冲按层拆分。** `EnvironmentRollout::Convolve` 的三层（0/2/4）
  共用一块 im2col 缓冲，而三层的 feature 数不同，零边因此在每个前向被重建三次。改成每层一块
  （零边只在通道集变化或缓冲增长时重建）后，用 `scripts/rollout_bench.py` 做 9 轮 A/B
  （单核 pin、同任务同种子、每轮 126 局 rollout、A/B 交替）：引擎自报
  `network_seconds / net_calls` 的比值为 **0.988（0.967–1.008）**，`sim_μs/tick` 对照 1.005，
  即这次实验未测出收益，已回退、未进候选。学生前向总耗时以 §3 的独立计数器为准，
  不能把 §4 的共享 openblas 21.60% 全归给学生，也不能由这次否决推断所有缓冲或遍历优化都无效。
