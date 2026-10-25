# task89：GPT Pro 交回的存档一键入库

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-perf`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport`（分支 `llmimport`，主控已从 origin/pvz-env 拉出；先 `git merge origin/demo origin/llmplay`）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
用户回来后会把 task81c/d 的包交给 GPT Pro 跑 3 小时，预计交回几十到上百份存档：33 个任务 × 多个种子，有的赢有的输，meta 里带 `robustness` 和 `manual_patch`。上次免费模型的 12 份由 Mac `cb-demo` 手工入库（task87，见 `MEMORY/docs/task87-llm-demo-import.md`），花了几个小时。下次要一条命令完成。

## 要做的
1. **入口**：`scripts/import_llm_run.py <out.zip 或解压目录> --dest artifacts/llm_demo/raw`。
   - 只读输入，拒绝 zip 里的绝对路径和 `..`。
   - 按 llmplay 的导出格式找全部存档。
2. **逐份处理**：
   - 用本机引擎 `human_demo.py verify` 校验。不要求逐位哈希，task82 已证明哈希差异来自未初始化字段；要求操作、tick、终局一致。
   - 读 meta：`task_id`、种子、倍率、结果、`robustness`、`manual_patch`。
   - 质量分档照 task81d：
     - **full**：通关，且第 10 波检查点 ≥5/8；
     - **half**：通关但不稳健；
     - **loss**：没通关，只入报告。
3. **任务族**：种子不是对应 demo 族自己的种子时，按 task87 的命名规则自动生成同卡组、不同种子的 task_id，并写进 `configs/human_demo_tasks.json`，DESIGN 同步登记。
   - 卡组从存档 meta 读，**不信任文件名**。
   - 已存在的族不重复登记。
4. **扫描计划**：为 full 与 half 的局生成 task71b 接手扫描的作业清单（12 点 × 8 未来 × 两臂）。
   - 未来种子块自动分配，与现有全部段不相交（可复用 task88 的盘点脚本；没合并之前就自己做机械核对）。
   - 只生成清单，不运行。运行交主控排队。
5. **报告**：输出一张表，含每份存档的任务、种子、波数、操作数、verify、分档、稳健性、是否新族。另附汇总：各层（A–E）的通关数、full 数、half 数。
6. **测试**：
   - 用 task87 的 12 份免费模型存档做端到端测试。文件在 Mac，主控经 HF `transfer/` 拷过来，取用后删除；测试用到的只是数值存档，不含游戏资源。
   - 再加两种坏输入：zip 路径穿越、meta 与存档不一致。
7. 结论写 `MEMORY/docs/llm-run-import.md`，提交并 `git push origin llmimport`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 和存档本身不进 git；不写字面密钥；不手写打法规则或公式。LLM 的打法脚本只读，不进训练。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- 2026-10-26：入口 `scripts/import_llm_run.py`（安全解压、发现、校验、分档、新族登记、扫描清单、报告）+ `python/test_import_llm_run.py` 17 项测试全过（zip 穿越、meta 自相矛盾、分档、新族命名、种子块、幂等）。
- 2026-10-26：补 frames.jsonl 行数与 meta.frame_count 一致性校验（CodeBuddy 接手，ec86a598）。
- 2026-10-26：真实 zip 管道测试（--no-verify）通过：发现 102/102 份存档、12 通关全档 half（免费模型存档无 robustness 检查点记录）、288 条扫描作业、种子块 84.0M 起与现有段不相交；task87 已登记的 8 个 `_s2026` 族幂等跳过（registered=[]）。修 `resolve_family`：族与种子均已登记时不再标为新族（01a11c33，18 项测试）。
- 2026-10-26：端到端（带引擎校验）全过：102/102 verify 通过，12 份通关入库且名单与包内 report.md / task87 表一致，坏输入两种（zip 穿越、meta 不一致）均正确拒绝（exit 1）。结论写 `MEMORY/docs/llm-run-import.md`。

## 问题

## 主控补充（2026-10-26 00:35）
测试输入已到台式机：`/home/newbiexvwu/pvz-logs/task89/llm_free_out_numeric.zip`（SHA256 前缀 `d199f511…`）。它是免费模型交回的原样目录，只保留了 json / jsonl / md（402 个文件），去掉了 LLM 写的 py 与运行日志。12 份通关存档的清单见其中的 `report.md`；Mac 上 task87 的入库结果可作对照（`origin/demo` 的 `artifacts/llm_demo` 不在 git 里，比较 task87 文档的表即可）。HF 上的中转文件已删。
