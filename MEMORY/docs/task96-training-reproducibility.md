# task96：同配方同种子重训为什么不能复现 r11？

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-clean`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/repro`（分支 `repro`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**GPU 训练让 qc 先**；同一时间 GPU 上最多 1 个本任务的训练。

## 现象
task84 的变体 A：r11 的 348 个标签，配方与 r11 逐字段相同，seed 1，重训一次。结果 heldout 124/192，而 r11 是 166；对照 d7 是 172。task92（每轮训多个种子再挑选）的前提是“训练随机性是主要来源”。所以必须先分清：
- (a) **训练本身不确定**：cuDNN 非确定性算子、数据加载顺序、多进程预计算；
- (b) **标签内容变了**：同名标签目录在 r11 之后被追加、重标或回填，例如 `backfill_observation_v2`、`relabel_verify`；
- (c) **代码变了**：打包器、编码器或训练脚本在 r11 之后有改动，`encoder_signature` 能否发现？

## 要做的
1. **数据**：对比 r11 训练时的 `source_rows.json` / pack 指纹，与变体 A 打包时的同名标签，逐标签比较行数、内容哈希和 mtime。
2. **代码**：列出 r11 训练时的提交与 A 所用提交之间，`python/` `scripts/` 中训练链的差异。
3. **确定性**：在同一份 pack 上用同一 seed 训 2 次，各跑 1 个 epoch（≤ 15 分钟），比较权重哈希与验证损失。不同就找出非确定性的来源：
   - `torch.use_deterministic_algorithms` 打开后的报错点；
   - DataLoader 的 worker 与 shuffle 种子。
4. **结论与修法**：
   - 如果是 (b) 或 (c)，给出“冻结标签快照、pack 指纹进轮目录”的方案；
   - 如果是 (a)，评估确定性训练的成本（耗时比），并给出建议：是用确定性训练，还是接受随机性、交给 task92 去挑。
   不改生产。
5. 结论写 `MEMORY/docs/training-reproducibility.md`，`git push origin repro`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
