# task47：Mac 上的引擎构建跟上台式机——clang 的 LTO / PGO / mimalloc，并把 Mac 采集换到新构建

> 主控 2026-10-23 20:40（台式机时间）写。执行者：Mac 上的 Codex（Business 账号）。工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/macbuild`（分支 `macbuild`，从 pvz-env 拉出，主控已建好）。
> Mac 是 Apple M5 Pro（15 核），上面正在跑泳池采集 `artifacts/hard_mac/mac_dagger_p5b.sh`，10 个 worker，用的是旧工作树 `view` 的构建（5b5fc489，task35 之前）。**编译用 `-j4`，基准最多 2 个游戏进程**，不要停采集。

## 为什么
- task35 在台式机把整局 CPU 降到旧构建的约 1/1.67（LTO + PGO + mimalloc + 引擎无用功清理；报告 `MEMORY/docs/engine-cpu.md`，任务书 `task35-engine-cpu.md`）。
- 引擎代码的改动对 Mac 同样生效，但 PGO 部分是 **GCC 专用**：`CMakeLists.txt` 第 35–44 行用的是 `.gcda` 和 `-fprofile-use=<dir>`。Mac 是 AppleClang，要用 `-fprofile-instr-generate` / `llvm-profdata merge` / `-fprofile-instr-use=<file>.profdata`。
- mimalloc 在 macOS 上的接入方式也要核实（FetchContent 编静态库）。
- Mac 采集现在还跑在 task35 之前的旧二进制上，等于白白慢一截。

## 步骤
1. 在 macbuild 工作树普通 Release 构建，确认能编、能跑。
   - 先量基线：`scripts/bench_engine.py` 的整局模式与 64 root 叶摘要模式，旧 `view` 二进制对新 Release，同种子；
   - 机器有采集负载，按"配对 + 多轮中位数"。
2. **clang 的 PGO**：让 `CMakeLists.txt` 按编译器选 PGO 方式。
   - GCC 保持现状，逐字不变，台式机构建不能受影响；
   - Clang：instr-generate → 跑与 `scripts/build_pgo.sh` 相同的 bench 工作负载 → `llvm-profdata merge`（`xcrun llvm-profdata`）→ instr-use；
   - `scripts/build_pgo.sh` 改成同时支持两种编译器，**一个脚本**，按编译器分支，不加开关；
   - LTO 在 AppleClang 上用 `-flto=thin` 还是 full，取量出来快的。
3. **一致性门禁**（与台式机同一把尺子）：
   - 4 个存档整局逐帧重放 0 差异；
   - 64 root 叶摘要与 Mac 普通 Release 逐位相同；
   - 特别注意：Mac 是 ARM，浮点与 x86 可能不同。门禁比的是**同一台 Mac 上**新旧构建，不跨机器比。
4. **吞吐**：报整局 CPU 秒 / 局，旧 view 二进制 vs 新 PGO 构建。四族各 8 局：白天 L7 d7、夜间 L14 n10、泳池 L24 p9k、传送带 L10 orig；最多 2 个进程。
5. **切换 Mac 采集**：门禁过了，在"结果"写新二进制路径。**不要自己改正在跑的 `mac_dagger_p5b.sh`**，主控会在两个标签之间换。
6. 测试：`python3 -m pytest python -q -p no:cacheprovider` 与合并前比不得多出失败。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树。
- 禁止 `git add -f`；`.pt` `.npz` `.log` `.profdata` 不进 git；不写字面密钥。
- 台式机的 GCC 构建行为必须逐字不变：`git diff` 里 GCC 分支的 flags 不能变。
- 每完成一小步就提交、`git push origin macbuild`，在"结果"记一行。疑问写"问题"，按最佳判断继续。

## 结果

## 问题
