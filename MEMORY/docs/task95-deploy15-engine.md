# task95：deploy15 引擎——把 task82 / task85 / task75 之后的 C++ 改动过门禁

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-mem`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy15`（分支 `deploy15`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
deploy14 的引擎 `76ff36dc` 是用 a8d00858 的源码编的。在那之后，pvz-env 又合进了几处 C++ 改动：
- **task82**：三处未初始化字段在构造时初始化（`Challenge::mCloudsCounter`、`Projectile::mLastPortalX`、`Coin::mCollectX/Y`）；
- **task85**：L50 Boss 的可见观测（`src/LawnApp.cpp`、`src/main.cpp`、`src/EnvironmentRollout.*`）；
- 其他合入后的改动：用 `git diff a8d00858 origin/pvz-env -- src` 列全。

它们要进生产：L50 的学生观测需要 Boss 字段；task82 让跨进程哈希稳定，以后门禁不必再容忍粒子线差异。

## 要做的
1. 列出 `a8d00858..origin/pvz-env` 的全部 `src/` 改动，逐项说明对游戏逻辑的影响，预期都是“无”。
2. 照 task70 A1 的 PGO 流程构建，**构建完成立即把二进制拷到 `artifacts/task95/` 留档**（B8 吃过亏：PGO 重建不能逐位复现）。
3. **逐位门禁**：对 `76ff36dc` 做 A2 的做法，六族 × 老师 / 学生 × 8 种子，要求逻辑 0 差异。
   - 另外核对：新引擎同进程、多进程的 state_hash 应当全同（task82 的效果），旧引擎的“粒子线”差异应当消失。
4. **吞吐 ABBA**：照 A4，中位配对比要求 ≥ 0.98。
5. **L50**：新引擎在 `boss_l50_orig_eval` 上跑 4 局。确认 Boss 观测字段存在，且终局与不读这些字段时的旧引擎一致。
6. 写 `~/pvz-logs/qc_deploy15.sh`，结构照 `qc_deploy14.sh`，**只换引擎**，代码与面板沿用 deploy14。只写，不执行。
7. 结论写 `MEMORY/docs/task70-engine-redeploy.md` 新的 B9 节，`git push origin deploy15`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认，不加开关。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 和二进制不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
