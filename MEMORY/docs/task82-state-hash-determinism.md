# task82：同一二进制两次 reset 的 state_hash 不同——快照里混进了什么？

> 主控 2026-10-25 写。执行者：台式机 Codex（plus 号，tmux `codex-llmplay`，task81 收尾后接）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/hashdet`（分支 `hashdet`，从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq、别的工作树**。

## 现象
task81 在台式机上用全静态链接的引擎（`~/pvz-logs/llmplay_pkg/stage` 里的 `pvz-portable`）观察到：同一个二进制、同一个任务，每次 reset 得到的开局 `state_hash` 都不一样（`d3ec36d5…` / `83328bcc…` / `cefd2775…`）。另一方面，Mac 录制的人类示范存档在录制和 verify 时都做了逐操作哈希校验，并且通过了（`hash_schema=portable_snapshot_v1_without_draw_statistics`）。

## 为什么要查
- 如果哈希里混进了指针、未初始化内存或时钟，那它只是“哈希不稳”，模拟本身没问题；
- 如果是模拟状态本身有未初始化字段，同一操作序列在不同进程里就可能走出不同的局。这会影响老师续走、示范起点格、门禁配对的可复现性。
- 必须分清是哪一种。

## 要做的
1. **复现**：
   - 用 deploy4 正式引擎（`/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable`）和 task81 的静态引擎，各做 N=8 次独立进程 reset，记录开局哈希；
   - 再各跑一局固定操作序列（取一局人类示范的前 2000 个操作），逐操作记录哈希，找出第一个分歧点。
   - Mac 正式构建上也做同样的 8 次 reset（Mac 引擎路径见 `build/`；你在台式机，Mac 这一步写成脚本，交给主控去跑）。
2. **定位**：把快照序列化按字段分段哈希，找出是哪些字段不一致。分清：
   - 地址 / 指针；
   - 未初始化的内存（可以用 MSan，或者 valgrind memcheck 的 `--track-origins`）；
   - 时间 / 随机种子来源；
   - 静态链接或 malloc（mimalloc）引起的布局差异。
3. **判定是否影响模拟**：对不一致的字段，说明它会不会进入游戏逻辑。用“同序列、多进程、整局终局与每个决策点的阳光 / 植物 / 僵尸是否全同”来证明，至少 16 局。
4. **修**：
   - 如果只是哈希口径的问题，就把这些字段排除出哈希，新开 `hash_schema` 版本，旧存档仍按旧口径校验；
   - 如果是模拟状态的问题，就修引擎（初始化该字段），并证明修复后旧存档重放结果不变，或者把会变的局列出来。
   - 两种情况都不加开关、不改生产默认；部署由主控做。
5. 结论写 `MEMORY/docs/state-hash-determinism.md`，提交并 `git push origin hashdet`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- 2026-10-25：两种指定引擎各 8 次 reset + 前 2000 操作已复现；首次分歧为 reset，定位到 Challenge 的未初始化 `mCloudsCounter[6]`，同引擎观察/事件一致；工具与初步证据见 `state-hash-determinism.md`。
- 2026-10-25：整局又定位到 `Projectile::mLastPortalX`；两处初始化已修。指定引擎各 8 次整局的 827 个决策点逻辑状态/事件一致（合并等待与逐操作基准逐决策核对通过）；修复版 8 次整局哈希也全部一致，新增构造/弹丸/快照回归通过。继续做四份旧示范的修复前后配对与 Mac 交接说明。

- 2026-10-25：`Coin::mCollectX/mCollectY` 初始化修复与探针 chunk 5 支持提交为 `e8ba7716` 并推送；四份旧示范修复前后逐决策配对完成（827/1310/893/923 个边界点，privileged 观测与事件全同，终局全同；哈希首差分别在 3995/4191/3599/3874，属未初始化字段被保存，不是模拟分歧）。

- 2026-10-25：Mac 交接脚本用覆盖路径在本机修复版上完整跑通（8 进程 / 2000 操作 / 2001 点全同 / 1 种 reset 哈希），并修正交接清单里的比较文件名（脚本写 `$OUT/comparison.json`，不是 `mac-comparison.json`）；memcheck 交叉核对：修复前 24 条 SaveGame 未初始化报告，修复后 0 条。

- 2026-10-25：结论层 `state-hash-determinism.md` 与 `state-hash-determinism-results.json` 提交并推送。交接复核：最终构建 8 份整局 trace 的 SHA256 全同（`5fb2c460…`）；四份示范修复后各 4 进程 trace 全同、修复前各 2 进程全异；回归测试从仓库根目录运行通过（测试自补 `python/` 与 `scripts/` 路径），协议 35 项 OK。Mac 探针脚本 `scripts/state_hash_mac_probe.sh` 交主控，台式机未执行。
- 2026-10-25：接手复核（Claude）：工作树干净、HEAD 与 `origin/hashdet` 一致；最终二进制 sha256 `5e890b3a…` 与证据一致；`test_state_hash_determinism` 8 个独立进程回归通过（3.6 s），协议测试 35 项 OK；`artifacts/`、`logs/` 的 `check-ignore --no-index` 为空。Mac 侧仍待主控执行 `scripts/state_hash_mac_probe.sh`。

## 问题

- 原始存档来自不同引擎，原有跨二进制回放本就采用 tick/ok/outcome 校验。本任务另外比较完整 privileged 状态（含隐藏僵尸、RNG）和事件；不把旧存档的跨版本哈希变化误报为模拟变化。
