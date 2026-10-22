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

## 问题
