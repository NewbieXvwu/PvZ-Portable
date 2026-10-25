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

## 问题
