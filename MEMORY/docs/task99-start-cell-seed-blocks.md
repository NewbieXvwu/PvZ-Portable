# task99：示范起点格的未来种子块自动续期

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-v6`）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/startseeds`（分支 `startseeds`，主控建）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
起点格每格只登记了 256 个未来种子（`configs/human_demo_tasks.json` 的 `demo_start.future_seed_start/count`）。qc 每组用 24–48 个，几轮就用完了：L32 已经在 10-26 06:50 用完。deploy14 之后用完的格会被退休，不再让 qc 停机，但这些格本身是课程里最有价值的部分，不该因为种子用完就停采。现在已有 24 个起点格，GPT Pro 交回后还会更多。

## 要做的
1. **账本**：起点格的未来种子改为按需领块。每次 128 或 256 个，从一个专用池里领，记进加锁账本（可复用 task88 / task89 的账本机制）。
   - 新池与全部现有段不相交，用 `scripts/gate_seed_inventory.py` 机械核对；
   - `human_demo_tasks.json` 里原有的块作为第一段保留，旧数据的可比性不变。
2. **qc 侧**：`plan_round` 发现当前块不够时，先续领新块，领不到才退休。续领要写进 state 和日志。原有偏移语义不变，只是把种子表从一段扩成多段。
3. **测试**：
   - 续领之后种子与旧块不相交、与其他格不相交；
   - 重启后 state 能恢复多段种子表；
   - 已退休的格在续领后能重新激活（deploy14 已退休 L32）。
4. 写清上线步骤，与 task98 一起在组边界上线。结论写 `MEMORY/docs/start-cell-seed-blocks.md`，`git push origin startseeds`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
