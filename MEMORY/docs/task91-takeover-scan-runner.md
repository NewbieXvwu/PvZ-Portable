# task91：接手扫描与起点格登记自动化——GPT Pro 交回几十份通关后，一条命令接到课程里

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-perf`；额度用完由 agent_pool 交 CodeBuddy）。工作树沿用 `/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport`（分支 `llmimport`，先 `git merge origin/pvz-env`）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
现在的流程是：示范或 LLM 通关局 → task71b 接手扫描（12 点 × 8 未来 × 老师 / 学生两臂，192 局）→ 找出老师胜率过 50% 的起点 → 在 `configs/human_demo_tasks.json` 登记 `demo_start` 格。
- 每份都由代理手工完成，L46–L49 的 12 份花了好几个小时。
- 用户即将交回 GPT Pro 的结果，预计几十份通关。照 192 局一份算，要上万局。
- task89 已经能入库，并生成扫描作业清单（`scripts/import_llm_run.py`，见 `MEMORY/docs/llm-run-import.md`）。缺的是**运行、判读、登记**这三段。

## 要做的
1. **运行器**：`scripts/takeover_scan.py run <清单>`。
   - 用生产老师参数：学生 r11、叶值 v5、静态网 branch1、`--dagger-beta 1 --verify-samples 9 --student-candidates 2`。学生臂用 β=0。
   - 断点续跑：已完成的 (存档, 点, 未来, 臂) 不重跑。
   - `--workers` 默认 2。
   - 结果按存档写 json，含每局的胜负、末波、破防行与僵尸类型。
2. **自适应取点**：先在 4 个等距点上跑老师臂，再在胜率跨过 50% 的区间里二分，最后在 50% 线附近补齐阶梯档位。目标是每份存档的局数降到 192 局的一半以下。
   - 用 task71b 已有的 L46 / L48 / L49 人类示范和 task87 的 LLM 局做回测：同样的存档，自适应得出的 50% 线与 12 点结果相差不超过一档。
3. **判读与登记**：`scripts/takeover_scan.py register <结果>`。
   - 老师 ≥50% 的存档按 task71b 第 2 步的规则生成阶梯：档位全部 ≥50%，顶档取最晚的稳定点。
   - 写入 `demo_start` 格；未来种子块用 task89 的加锁账本分配，与全部段不相交。
   - 同步 DESIGN 登记，测试期望集合自动更新。
   - 老师接不住的存档（全程 <50%）只记入报告，不登记。
4. **端到端测试**：用 task89 的测试包（`/home/newbiexvwu/pvz-logs/task89/llm_free_out_numeric.zip`）里任取 2 份，走完 import → run → register 全程。为节省时间，可以只用 2 个未来，但要走完整路径。
5. 结论写 `MEMORY/docs/takeover-scan-runner.md`，提交并 `git push origin llmimport`。上生产由主控做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 和存档本身不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
