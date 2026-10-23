# task35（主控 2026-10-23 00:45 下发给台式机 CodeBuddy 账号 0，tmux cb-perf）：引擎与 C++ 推演的 CPU 性能——榨到没有油水为止

先读 `AGENTS.md`、`MEMORY/docs/task32-memory.md` 与 `MEMORY/docs/task33-hotpath-audit.md`（两个 Mac 会话在做内存与主干体检，分工见下）、`src/EnvironmentRollout.cpp`、`src/main.cpp`、`LawnApp::Environment*`、`CMakeLists.txt`。

## 背景
采集的算力几乎都花在引擎里：每个决策，搜索要从当前局面分叉出“候选动作 × 验证样本（`--verify-samples 9`）”条推演，每条推演由 C++ 学生网络（`EnvironmentRollout`，PVZRN006 identity 头）逐步出招、引擎逐 tick 模拟，最长 2500 tick。
用户的判断：此前几个代理的优化远没有榨干——例如学生网络前向是**一次一个局面**（`EnvironmentRollout::Convolve` 是手写循环，只有行汇总用了一个 6×width 的小 `cblas_sgemm`），编译也没开 LTO / PGO / 针对本机。
本任务只管**引擎进程内部的 CPU 时间**；分工：引擎启动时的资源加载（贴图 / 字体 / 特效）归 Mac 上的 task32（cb-mem），数据流与进程架构（C++ 单进程采集、批量推演的总体设计）归 task33（cb-audit）——你可以在“问题”里给它们提建议，但不改它们负责的代码。

## 要做的（工作树 `git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/perf -b perf origin/pvz-env`，单独编译，不碰主工作区与 cq 的 build）
0. **基准与正确性夹具（先做，后面每一步都靠它）**：
   - 速度：`scripts/bench_engine.py`——固定一组真实局面（从 `~/PvZ-search/artifacts/student/frames_v2` 取白天 L7、原版 1-10、夜间 2-4、泳池 L24 各 16 个决策点的快照，写成清单文件），
     对每个局面跑固定条数的 C++ 推演（`env.rollout`，学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`），报告：每秒模拟 tick 数、每条推演耗时、学生前向占比（用计时器或 perf）。再报一个整局指标：`search_play` beta=0 同参数（照 `~/pvz-logs/queue103.sh` 的 `run()`）8 局的每决策耗时。
   - 正确性：同一动作序列重放，逐帧状态必须相同（硬门槛）；推演结果（终局波数 / 胜负 / 返回的统计）在同种子下必须相同——若某项优化改变了浮点求和顺序导致学生前向的 logits 有 1e-6 级差异、进而推演分叉，
     则改为统计验收：老师决策与改前的一致率不低于老师自身噪声底线（同一老师换随机未来约 75%），64 局同种子胜率在噪声内。写清每项用的是哪种验收。
1. **编译**：Release 下开 `-O3`、LTO（`CMAKE_INTERPROCEDURAL_OPTIMIZATION`）、`-march=native`（各机器各自编，产物不跨机器拷贝）；再做 PGO（`-fprofile-generate` 跑基准 → `-fprofile-use`）。
   试换分配器 mimalloc（能联网就用 CMake FetchContent，不能就记进“问题”）。每一项单独测，留下有效的，**直接改成默认构建**（不加 CMake 开关；PGO 的 profile 生成步骤写成 `scripts/build_pgo.sh`）。
2. **学生前向**：`Convolve` 改成 im2col + `cblas_sgemm`（或分块手写 SIMD，两者都试，取快的）；权重布局预先转置好；去掉每步的临时 `std::vector` 分配（预分配缓冲）；稀疏输入（大部分通道为 0）若手写路径更快就保留并说明。目标：单次前向耗时减半以上。
3. **引擎每 tick 的无用功**：用 `perf record -g`（WSL 里没有就装 `linux-tools`；装不了用 `valgrind --tool=callgrind` 小规模跑）找热点。无界面推演里凡是不影响胜负与状态的更新——粒子、拖尾、屏幕震动、音效 / foley、纯渲染用的动画插值、`PoolEffect`、`ZenGarden` 等——
   逐项确认后跳过（每项都要过“逐帧状态相同”）。动画定义里决定攻击 / 判定时机的部分不能丢。
4. **推演的批量化（只做设计与原型，不并入）**：同一决策下的多条推演改为“锁步”推进，每一步把各条推演的学生前向攒成一个 batch 做一次 sgemm。写一个原型量出加速比，结论与接口草案写进报告交给 task33 的 C++ 单进程采集设计。
5. **持续压榨直到没有油水**：每做完一项就重跑基准、把 perf 前 15 名的函数与占比贴进报告；**停止条件**：连续 3 项尝试的收益都 < 2%，且 perf 前 15 名里没有可以证明“不影响逻辑”的无用功。没达到停止条件就继续找下一项，不要自己宣布完成。
6. 报告 `MEMORY/docs/engine-cpu.md`：每一项的改前 / 改后数字（tick/s、每推演耗时、每决策耗时、整局 games/h）、验收方式与结果、perf 前 15 名的变化。

## 规则
- 同时最多 2 个你自己的游戏进程（qc 采集与训练在跑）；不动任何 tmux 队列（qc）与别的代理会话（cb-cuda、cb-vnet、qoder-oom、codex-cuda）；不改别的工作树与主工作区。不重启 WSL。
- 分支 `perf` 提交，可以 push；不推 pvz-env、不合并。**不加新开关**：有效的优化直接成为唯一实现，旧路径在同一系列提交里删除。不写手工规则；不写字面密钥；不 `git add -f`；大文件不进 git。只用绝对路径。
- 每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。额度用完由接力链接手，进度一定要落在提交和“结果”里。

## 结果

0. **基准与正确性夹具**（`scripts/bench_engine.py`、`scripts/prof_sample.c`、`scripts/prof_symbolize.py`）。
   - 局面清单：`bench_engine.py manifest` 从 `~/PvZ-search/artifacts/student/frames_v2` 里挑 4 族各 4 个真实对局、每局 4 个决策点（共 64 个 root）：白天 L7（d7dag）、原版传送带 L10（c1d10orig）、夜间 L12（n12dag）、泳池 L24（p24dag）。任务参数从存档 `meta.json` 还原，卡组从首帧 `packets` 的类型读回（`context.roster` 不是卡组）。
   - 每个 root 跑 3 条 C++ 推演（`env.rollout`，`--sim-skip 61`、`--horizon 2500`、学生 `dw4i_s1/model.pt`），计时取 C++ 的 `simulation_seconds`/`network_seconds` 加进程 CPU 时间（`/proc/<pid>/stat`，机器上还有采集队列在跑，wall 会被抢占污染）。
   - 正确性（硬门槛）：`bench_engine.py replay` 用存档里录下的动作序列**整局**重放，逐帧比较 tick/wave/sun/plants/zombies/mowers/packets/graves/lanes/row_terrain/scene，四族各 1 局（953 / 877 / 531 / 464 帧）**0 差异**；同 root 同 seed 连跑两次 rollout，叶状态摘要相同。等待必须走 `EventWaitEnv` 的 `until:timeout`（存档里有 2/3/4/28/33/46/49/53 tick 的提前结束等待，压成固定 60 tick 会分叉）。
   - 基线（Release `-O3`，无 LTO）：**82.0k tick/CPU-s**，28.5 ms/推演（wall 口径），学生前向 34%（openblas 14.1% + Heads 10.1% + Convolve 4.3% + Encode 2.0%）。
1. **编译**：LTO（`CMAKE_INTERPROCEDURAL_OPTIMIZATION`）**+6.7%**（82.0k → 87.5k tick/CPU-s，各 3 次，64 root 叶摘要全部相同）。`-march=native` 与 LTO 叠加 **0 收益**（87.2k vs 87.2k），且会让二进制与机器绑定，已删。PGO、mimalloc 见后续条目。

2. **PGO**（`scripts/build_pgo.sh`，两阶段同一个 build 目录——GCC 按目标文件路径找 `.gcda`，换目录会静默失效）：**+18%**（87.5k → 103.2k tick/CPU-s，各 3 次，叶摘要全部相同）。profile 放在 `artifacts/perf/pgo/`（不入库，机器各自生成）；CMakeLists 检测到 `.gcda` 就自动用 `-fprofile-use`，没有 profile 的机器照常编译。
3. 学生前向：缓存转置卷积核与 im2col 零边界、前向缓冲全部预分配、每决策只算一次 packet 头。**+0.5%**（87.9k vs 87.5k，叶摘要相同）；瓶颈不在这些分配上，见后面 openblas/Heads 的占比。

4. **分配器 mimalloc**（FetchContent 静态链接，`MI_OVERRIDE` 全局接管 malloc）：**+11.2%**（103.2k → 114.8k tick/CPU-s，各 3 次，叶摘要相同）。对照：系统 jemalloc（LD_PRELOAD）只有 +3.6% 且波动大（103.1k–110.4k）。首次 configure 需要联网下载，之后用 `build/_deps` 缓存。
   - 累计：基线 82.0k → **114.8k tick/CPU-s（+40%）**，每推演 28.5 ms → 22.1 ms（CPU 口径）。

5. **引擎每 tick 的无用功 ①**：`Reanimation::UpdateAttacherTrack` 每 tick、每条 attacher 轨道都把 `reanim/<名字>.reanim` 用 `std::format` 拼出来，再对整张 `gReanimationParamArray` 做 `strcasecmp` 线性扫描；`AttacherInfo` 还是每次新建（两次 string 分配）。改成按名字查一次就记住（表在启动后不变，命中前仍走原来的顺序扫描，语义一致），`AttacherInfo` 复用。**+5.4%**（114.8k → 121.0k tick/CPU-s，3 次中位数），叶摘要与整局重放逐帧全部不变。

6. **学生前向 ②：packet 头改成一次 dgemm**。剖面里 `IdentityLinear`（10 个卡槽各做一遍 259 维 double 点积）占 **16.7%**，是当时单项最大。改成把有效卡槽拼成 10×259 矩阵、一次 `cblas_dgemm`（权重在 Load 后转 double 缓存一次）。**+10.2%**（配对 A/B，比值中位数 0.908，3 轮）。验收：叶摘要仍然**逐位相同**——dgemm 与顺序累加在 double 里差 ~1e-16，转 float32 后没有翻位，所以没走到统计验收那条路。
   - 方法说明：机器上同时跑着采集队列（13 个模拟器、load ~13），绝对数抖动 ±5%，所以从这里开始一律用 `bench_engine.py compare` 做**配对 A/B**（同一轮里两个二进制背靠背跑，取比值中位数）。
   - 累计（配对口径）：当前构建 / 基线 = **1.41×**（82.2k → 114.2k tick/CPU-s）。

7. **引擎每 tick 的无用功 ②：轨道稀疏遍历成为唯一实现**。`Reanimation::Update` 的 `updateTrack` 里，非 attacher、无挂件、无 blend 的轨道每 tick 只做三次 load+比较，而一个 reanim 有 50–200 条这样的轨道；剖面上这个 lambda 占 ~13%。稀疏遍历（位图 + 缓存）原来藏在 `ENV_SPARSE_TRACKS` 开关后面，现在只要 `ENV_LIGHT_ANIMATION` 打开就用（跳过的条件正是函数体里检查的那三个，按构造精确；`scripts/sim_skip_agreement.py` 已在 224 局上逐动作验证过）。**+6.0%**（配对 A/B，比值中位数 0.944），叶摘要与整局逐帧重放仍然全部相同。

8. **引擎每 tick 的无用功 ③：粒子的每 tick 视觉积分**。`PvzpParticleEmitter::UpdateParticle` 里 field 曲线求值 + 位置/自旋积分只喂渲染；寿命（`mParticleAge`）必须照旧推进，因为粒子死亡可能 spawn 一个 cross-fade emitter、从共享 RNG 里取数。改成 `ENV_LIGHT_ANIMATION` 下只做寿命记账（年龄、time value、最后 time value），跳过其余数学。**+4.7%**（配对 A/B 4 轮比值中位数 0.955；同一改动另一次 3 轮测得 0.854，机器负载 13 时抖动很大，报告取保守值）。验收：64 个 root 叶摘要**逐位相同**，16 个存档整局逐帧重放 0 差异。

9. **im2col 按行整段拷贝**：3×3/1×5 卷积的 im2col 原来逐格判断边界再赋值，改成先算出行内连续区间、`memcpy` 整段（值完全相同）。**+2.4%**（配对 A/B 4 轮中位数 0.977），叶摘要与逐帧重放不变。

10. **Codex 补齐独立验收夹具**：`bench_engine.py` 保存全部叶观测/逻辑统计摘要，跨 A/B 与轮次逐条校验、差异立即失败，各轮证据和汇总分别落盘；整局读 `games.jsonl`，默认/最多 2 worker，拒绝不足局数。`test_engine_bench` 7 项回归测试通过。已建立 `engine-cpu.md`，汇集既有测速与采样前 15，明确旧摘要只有 16/192 条、整局基线只有 2/8，尚不能宣称全部验收完成。原 CodeBuddy 在同树运行，未接管其 7 个 C++ 实验文件。

11. **六项热点微优化（合计 +8.3%）**：① `DataArray<T>::Iterator::operator++` 按 64 位字一次跳 4 个空槽（剖面里 projectile/coin 的三个实例合计 6.1%）；② `Zombie::IsTangleKelpTarget` 记住海藻槽位，不再每 tick 遍历全部植物（3.3%）；③ `Reanimation::Update` 的 `SECONDS_PER_UPDATE*mAnimRate/mFrameCount` 按两个输入缓存（2.1%），轨道循环加预取；④ `Convolve` 的零通道判定改成 64 位按字或（0.7%）；⑤ im2col 的连续区间按 10/9/8 固定长度内联拷贝（变长 memcpy 的 PLT 往返 3.1%）；⑥ `Plant::Update` 把普通对局分支提到最前，粒子归一化时间值不再计算。**+8.3%**（配对 A/B 3 轮比值中位数 0.924，189.2k vs 177.0k tick/CPU-s；单项都在 1–3% 量级，低于本机 ±5% 噪声，故按批测）。验收：64 个 root 叶摘要**逐位相同**（a/b 两份 compare 记录逐族比对）、4 个存档整局逐帧重放 0 差异、同种子 rollout 可复现。
   - 构建注记：`build/` 曾被留在 PGO 第一阶段（`-fprofile-generate`），此时测出来比 step9 慢 1.77×；重新走完 `scripts/build_pgo.sh` 第二阶段才得到上面的数字。以后换手前先确认 `build/build.ninja` 里 `fprofile-generate` 计数为 0。

12. **Codex 修复模型重载缓存**：新增同布局换权重与新进程比较的回归，修复前 plant logit 0.0540556535 vs 0.0564118475；Load 成功后清空转置卷积核/packet double 权重/padding 衍生缓存。修复后全部 4 项 identity C++ 测试通过，24 配置 192 决策最大误差 1.04e-7、argmax/动作全部一致。另用增强门禁补验 `18a58d42` 对 step9：192 条完整叶观测/统计一致，16 局 11,475 帧重放零差异。基准夹具与报告提交 `aca743ab`。

13. **Codex 补共享库热点解析**：`prof_symbolize.py` 解析共享库动态符号，并支持给原始模块指定保存的二进制副本。稳定副本四族 960 条 rollout 采样 5752 点，SGEMM 内核 17.7%、其打包约 4.4%，前 15 已写 `engine-cpu.md`。perf/valgrind 两次安装被镜像 HTTP 403 拦住，继续 SIGPROF；libc 最近动态符号不能当精确函数归因。缓存修复提交 `4de38495`。

14. **Codex 完成 SIMD/BLAS 卷积核对照，拒绝 SSE**：独立 `bench_conv_kernels.py/.cpp` 从清单 64 个真实 v2 root 生成三层各 64 输入，稀疏首层/稠密后两层共享 im2col。三轮中 SSE 4×4 分块耗时分别是 BLAS 的 3.50/3.28/3.40 倍，最大误差 1.72e-5；不进入生产，也不新增运行开关。报告附全部数字与计时边界，不能把矩阵核结果算作整次前向或 batch 推演收益；停止条件尚未满足。

15. **Codex 修正快照视界口径（基准 v2）**：快照在存档动作执行前保存，旧脚本却用动作后的存档 tick 加视界，会多算一个动作的 tick。bench/replay 现从 restore 返回的真实 tick 计算终点，报告标记 `benchmark_version: 2`；新增 restore=100、存档=160 的回归证明终点为 2600，全部 8 项夹具测试通过。旧 v1 数字保留同口径 A/B 的效力，不能与 v2 绝对吞吐直接比较；整局 8 局任务不受影响，完整 v2 复核待进程槽位空出。

16. **Codex 独立 C++ 锁步原型已编译**：`scripts/prototypes/` 从已提交源码导出专用构建，不修改正式源码/build。每 lane 保留快照/RNG/历史/收入，三层卷积按 batch SGEMM，继续复用正式 Encode 尾部、Heads 与动作选择；报告已写 task33 的接口草案和全局状态隔离要求。原型计时将区分快照/前向/模拟成本，比较同二进制原生串行与 batch=1/9，并全量检查 64×9 条叶观测/统计；当前两个整局任务都仍活着，等槽位空出后运行，尚未声称加速或验收完成。基准 v2 修复提交 `e6dc25c9`。

11. **事件等待的公开状态不再每 tick 建 set**。`WAIT_UNTIL_LEGAL` 每决策最多重复 60 次 `EnvironmentWaitUntil(60,0)`，而 `EnvironmentPublicWaitState` 每个 tick 都重建三个 `std::set`（就绪卡 / 割草机 / 左侧僵尸）再加一个去重 set：每个元素一次 malloc + 一次红黑树插入。改成可复用的有序 vector（`SortedInsert`/`Contains`，迭代顺序、去重、成员判定与 set 完全一致），`Run` 用两块 `PublicState` 缓冲交替原地填充，去重表改成 thread_local vector，`clear()` 保留容量。剖面：`_Rb_tree_insert_and_rebalance` 1.9% → **0**，`LawnApp::EnvironmentPublicWaitState` 1.4% → 掉出前 20。整机吞吐在本机 load 23 下测不出（7 轮配对中位数 0.938，对照同二进制自比 5 轮中位数 1.015、单轮 ±9%，见“问题”）。验收：4 个存档整局逐帧重放 **0 差异**，64 个 root 叶摘要与 step9 **逐位相同**，原生 wait 契约测试 11 条全过。
   - `scripts/build_pgo.sh` 不再 `rm -rf build`（只清 profile 目录）：build 目录里有 mimalloc 的 FetchContent 检出，删掉就要重新联网 clone。

12. **火炬树桩判定记住槽位**：`Projectile::PeaAboutToHitTorchwood` 每颗豌豆每 tick 都要遍历全部植物找火炬树桩（1.5%）。改成与海藻判定同样的备忘：按 `mPlants.mNextKey` / `gBoardContentEpoch` 失效，缓存持有火炬树桩的槽位，之后只遍历这几个槽位（没有就立即返回）。**配对 A/B 9 轮中位数 0.988（≈+1.3%，单轮 0.758–1.043，见“问题”里的噪声下限）**；剖面里该函数 1.5% → 掉出前 22，`DataArray<Plant>::Iterator::operator++` 1.7% → 0.8%。验收：4 个存档整局逐帧重放 0 差异。

13. **边界坐标只在真正需要时做十进制渲染**：`EnvironmentPublicWaitState` 对 `|x-160| < 1` 的僵尸调 `RenderedPublicCoordinate`（`std::ostringstream` + `std::stod`），剖面里 stream/locale/printf 相关合计 **2.6%**（`parse_printf_format` 1.0% + libstdc++ locale 1.7% 的一部分 + `strtof64`）。六位有效数字在 [100,1000) 上只保留到千分位，四舍五入最多移动 0.0005，所以窗口从 1.0 收窄到 0.001 之后判定结果不变（窗口外的值不可能被舍入跨过 160）。**剖面：stream/locale/printf 相关条目全部掉出前 22**；配对 A/B 9 轮中位数 1.010（噪声内，见“问题”）。验收：4 个存档整局逐帧重放 0 差异。


17. **A1 合并 origin/pvz-env（70 个提交）并重新走完 PGO**。冲突只有 `CMakeLists.txt`（CUDA 选项删除 vs 本分支的 LTO/PGO 块相邻）与当天记忆 add/add；按任务书以 pvz-env 的删除为准，LTO/PGO/mimalloc 保留。合并后 `scripts/build_pgo.sh bench` 重生成 profile（本机 load 26，约 15 分钟），`grep -c fprofile-generate build/build.ninja` = 0。验收：192/192 条完整叶观测与逻辑统计与 step9 二进制相同（`compare` 内置断言）；四族整局逐帧重放 953/877/531/464 帧 **0 差异**；4/4 同种子重复推演一致；`unittest discover` 281 项里 5 项 `test_cpp_rollout` 需要 `PVZ_RESOURCE_DIR`（设置后 5/5 通过）、`test_rowpool_aux` 需要 pytest（venv 未装）、`test_identity_training.test_cpu_training_checkpoint_is_loadable` 在 pvz-env 侧就失败（相关文件与 pvz-env 逐字节相同，属主干既有问题）。构建相对 step9 的配对比值中位数 0.92（4 轮）。
18. **A3 部署说明**写进 `engine-cpu.md`：cq 工作树上 `merge` → `JOBS=8 bash scripts/build_pgo.sh bench` → 检查 `fprofile-generate` 计数为 0；mimalloc 首次 configure 需联网；profile 由 `artifacts/perf/engine_roots.json` 的 64 个真实 root 生成；换学生模型必须重生成，只换价值网不必。
19. **C 项 int8（AVX-VNNI）核级实测：不进入生产**。独立夹具上与 SSE 实验同一批 192 个真实输入：int8 核耗时是 BLAS 的 0.38–0.53 倍（首层 15.8→8.4 µs、第二层 57.9→22.3、第三层 32.5→14.3），且与“量化后反量化再算”的参考实现**逐位相同**（`kernel_vs_quant_ref = 0`），说明核本身正确。但按层标定的 u8 激活误差把 trunk 输出推到 8.9% 相对 rms、全局嵌入 5.4%，而收益上限只有整体 1.15–1.19 倍，且需要为无 AVX-VNNI 的机器保留 BLAS 后备路径。数字与判断写进 `engine-cpu.md`，代码只留独立实验脚本。
20. **A2 整局基准 2/4 族**（`bench_engine.py games`，CPU 秒/局，新旧构建同种子各 8 局）：白天 L7 149.3 vs 236.7 = **1.586×**、夜间 L14 88.5 vs 138.2 = **1.561×**，逐种子胜/负、末波、决策数 8/8 相同；泳池 L24 与传送带 L10 仍待填（表在 `engine-cpu.md` 的 A2 节）。
21. **进度与下一步**：A1（合并 pvz-env + 重生成 PGO）、A3（部署说明）、C（int8 判定不入生产）已完成入库，B 原型已编译未实测；下一步先补完 A2 剩余两族整局对照，再跑 B 的 native / batch 1 / batch 9 三臂与 64×9 全量叶摘要比对。停止条件未满足，本任务交回 Codex。

22. **重启后恢复与 B 夹具修复**：核对 status/log 后保留 A2 已完成两族；泳池只有新臂 8 局、旧臂 2 局，缺完整配对 CPU 计时，整对作废并在独立 recovery 目录从头重跑，随后接传送带。旧 B 日志实际为响应缺 `protocol_version` 而中止；已补成功/异常响应的版本、原生臂 CPU 分项和差异失败门禁，12 项无引擎回归通过。原型在当前提交源码副本上重建，尚未声称 B 验收通过。

23. **修复事件等待复用计数**：`PublicState::clear()` 只清容器、未清 `plantCount`，两块缓冲复用后植物数量被重复累加，可能漏掉条件 5 的植物减少事件。新增与真实读取器相同的原地清空/累加回归（第 2 tick 从 3 株减到 2 株），修复前失败、修复后通过；12 项 C++ 契约与 20 项 Python 等待测试通过。清空全部标量并保留 vector 容量；当前 A2 只用 timeout，不受条件 5 影响，正式引擎补验安排在该配对运行结束后。

24. **下一个热点候选：公开坐标的十进制往返**。独立 `scripts/prototypes/bench_public_coordinate.cpp` 用固定随机浮点位型与边界邻点共 1,017,346 个有限坐标，`to_chars/general/6 + from_chars` 与旧 stream/stod 的 double 结果逐位相同。三轮交替进程 CPU 秒中位数 0.5763 → 0.1340（单函数 4.30×）；尚未接入正式引擎，不计整体收益，待 A2 结束后做整条推演配对与逐帧验收。另已从可用的 Ubuntu 官方源下载并在 perf ignored 目录解包 valgrind（未改系统或其他工作树），版本 3.26.0 可执行。

25. **恢复真正的调用栈剖析能力**：从 Ubuntu 官方源取包并仅解包到 perf 的 ignored 工具目录，`perf version` 与 `perf record -e cpu-clock:u -g -- /usr/bin/true` 均成功（未改系统安装/权限）。valgrind 3.26.0 同样可运行。`prof_symbolize.py --debug-root` 按 ELF build ID 匹配解包的 libc 调试符号，旧 5752 点剖面里误显示为 `_dl_mcount_wrapper` 的份额已拆为 memset/strcmp 等，报告保留新前 15；这些是历史样本重解析，正式当前构建调用栈采样将在 A2 结束后跑。

26. **A2 泳池恢复重跑完成（3/4 族）**：L24 p9k m150 新旧各 8 局，完整 `RUSAGE_CHILDREN` CPU 220.988 / 402.321 秒，即 27.624 / 50.290 秒/局、旧/新 **1.8206×**，CPU 工作量口径 130.32 / 71.58 games/h。8/8 种子胜负、末波、决策数相同（全负，平均 425.875 决策）。已从 recovery 日志保存完整单族条目；传送带两臂继续运行，未重用重启前缺失配对的旧计时。

27. **后续运行检查点（不代表完成）**：传送带恢复任务仍活着（`artifacts/perf/games_recovery.log`，父进程 `bench_engine.py games`，输出 `engine_games_recovery/`）；必须先核实活进程和完整 `engine_games.json`，不能因日志暂时不增而重启。B 的协议/CPU 修复已在 adcf9b3f 源码副本编译成功，build.json 的 helper/builder SHA 与当前脚本相同，待 A2 后运行 `scripts/prototypes/bench_batch_rollout.py`。完整 pytest 已用 ml Python + 追加 `/usr/lib/python3/dist-packages` 成功收集 290 项（未执行），可避开原来缺 pytest 的环境问题。坐标候选已编译到 `artifacts/perf/coordinate-candidate/pvz-portable`，正式 `build/pvz-portable` 的 SHA256 保持不变，A/B 旧臂副本为 `artifacts/perf/pvz-portable-before-coordinate`；CMake 输出目录已恢复默认（只 configure、未覆盖正式旧二进制）。两处候选源码改动（EnvironmentWaitEvents.h、bench_public_coordinate.cpp）仍未提交，百万坐标和原生契约通过，但须在槽位释放后跑候选 compare / 四族重放 / 真 perf 调用栈，再按整引擎收益决定保留；现 profile 对该函数存在 coverage-mismatch，最终保留时须重生成 PGO。

28. **首层格子稀疏卷积独立实测**：`scripts/prototypes/bench_sparse_conv.cpp` 使用既有 64 个真实 root 的首层输入（仅 1.788% 格子非零），比较完整 im2col+BLAS 与直接非零输入累加（含输出布局转换，静态权重/活动通道准备均不计时）。同 CPU0、逐 root 三臂轮换九轮：BLAS / AVX2 / 普通 C++ 中位数 18.240 / 10.422 / 14.854 µs，配对耗时比中位数 0.580 / 0.824；稀疏两臂逐位相同，相对 BLAS 最大误差 5.72e-6、相对 RMS 6.62e-8。只留独立实验，未并入正式引擎，不将首层收益外推整体；可与坐标候选之后的真实剖面结合决定是否接入。传送带 A2 活进程仍在继续，未新增游戏进程。

29. **A1 无引擎训练侧补验**：ml Python 追加系统 pytest 包后，`test_identity_training.py` + `test_rowpool_aux.py` 实测 **8 passed / 1 failed**（12.65 秒）。rowpool 不再因缺 pytest 无法执行；identity CPU 训练仍复现 `train_rollout_policy.py:549` 的 cell gather `-9223372036854775808` 越界。这四个训练/打包/测试相关文件与当前已合入的 origin/pvz-env 无差异，未修改 task32/33 代码；A1“完整 pytest 全过”仍有既有训练失败，不能标成全绿。传送带新臂继续推进，未占额外游戏槽位。

## 问题

- **A1 测试门槛仍有非 Mac 既有失败**：`test_identity_training.test_cpu_training_checkpoint_is_loadable` 在配备 torch+pytest 的环境仍失败，`train_rollout_policy.py:549` 把 cell 候选索引转 long 后出现 `-9223372036854775808`。训练/打包/测试文件与合入的 origin/pvz-env 相同；按 task35 分工不修改 task32/33 的训练代码。8 项相关测试通过，但 A1 全套 pytest 尚未满足，不能以“环境缺包”概括这个失败。

- 已修复（结果 12）：原构建 identity C++ 多配置测试 packet logits 最大误差 0.1854288，根因是 Load 沿用上一模型的衍生权重缓存；旧摘要与单模型重放不能替代模型重载验收。

- **吞吐基准的分辨率**（2026-10-23 实测）：采集队列让 load 长期在 22–27（12 核 20 线程）。同一份二进制自比（`compare --rollouts 3 --rounds 5`）比值 1.005 / 0.943 / 1.015 / 1.090 / 1.093，中位数 1.015；`--rollouts 9` 3 轮为 1.038 / 1.011 / 1.000。也就是说单轮 ±9%、3–5 轮中位数约 ±2%。**2% 以下的改动不能靠吞吐判定**，改用剖面份额变化（`prof_symbolize`）与机制计数，并把吞吐数字如实标为“噪声内”。
- `build/` 曾被留在 PGO 第一阶段（`-fprofile-generate`）而没人发现，量出来比 step9 慢 1.77×。换手/重测前先查 `grep -c fprofile-generate build/build.ninja` 应为 0。

## 主控追加（2026-10-23 11:00）：上线优先，然后三条没做透的路

背景：用户决定 GPU（CUDA）方案**暂缓决策、不复活也不放弃**，决策日 **2026-10-27**——到那天看纯 CPU 能不能达到目标：
**正式课程采集的整局吞吐（局/小时，同任务族同倍率对比）≥ 10-22 旧构建的 2.0 倍**。所以本任务的成绩最终按**正式采集里的整局吞吐**算，
不是按 64 root 的 tick/CPU-s 算。另：本分支此前从未 push 过，主控 11:00 已代为 push 到 `origin/perf`；以后每次提交都 push（本地 `refs/remotes/origin/perf` 因与 `perf/attention-fusion` 重名建不出来，push 用 `git push origin perf:perf`）。

按顺序做：

A. **上线准备（最优先）**
   1. `git merge origin/pvz-env`（落后 69 个提交；pvz-env 已退役整条 CUDA 线，`CMake/CudaRollout.cmake`、`cuda/`、`src/CudaExport.*` 已删，冲突一律以 pvz-env 的删除为准）。合并后重跑：64 root 叶摘要与合并前逐位相同、4 个存档整局逐帧重放 0 差异、`python -m pytest python -q` 全过（Mac 专属的除外，写清）。
   2. 整局基准（这是验收用的数字）：`bench_engine.py` 的整局模式，旧构建（pvz-env 合并前的 cq 二进制 `/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build/pvz-portable`，只读，不要重编它）对新构建，同种子、同参数（照 qc 的 `command_for`），每族 8 局：白天 L7 d7、夜间 L14 n10、泳池 L24 p9k、传送带 L10 orig，报每局 CPU 秒与比值。机器负载高时按“配对 + 多轮中位数”。
   3. 部署说明写进 `engine-cpu.md`：主控要在 cq 工作树上一次做完的命令（`scripts/build_pgo.sh` 的两阶段、mimalloc 首次联网、检查 `fprofile-generate` 计数为 0），以及“PGO profile 用哪批 root 生成、换学生模型后要不要重生成”。
B. **锁步批量推演的原型实测**（第 16 条只编译了，没量）：batch=1/9 与原生串行比，分开报快照 / 前向 / 模拟；最新剖面里 libopenblas 已占 **28.65%**，是现在单项最大，这条路的上限比你当初估的高。
C. **推演里的学生前向降精度（没派过的路）**：本机 i7-12700F 支持 AVX-VNNI。试 int8（逐通道对称量化权重、激活按层标定）或 bf16 的 GEMM，取快的。这是近似，按**统计验收**：老师决策与 fp32 的一致率不低于老师自身噪声底线（同老师换随机未来约 75%）减 3pp，64 局同种子胜率在噪声内；过了就成为唯一实现（不加开关），不过就把数字写进报告后删掉。
D. 停止条件不变；A 做完立刻在“结果”里记一行，主控要拿它去上线。
