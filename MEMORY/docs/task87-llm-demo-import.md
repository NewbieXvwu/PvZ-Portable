# task87：导入 LLM 打出的屋顶通关存档，做成示范起点格

> 主控 2026-10-26 写。执行者：Mac CodeBuddy（`cb-demo`）。工作树沿用 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo`（先 `git merge origin/pvz-env`）。Mac 上每个任务最多 6 个游戏进程。

## 来源
用户拿 task81 正式包让免费模型跑了 20 分钟（`label_source` 写的是 `llm_gpt_pro`，实际是免费模型），交回 12 份通关存档：屋顶 L46 / L47 / L48 / L49 的 doom 族，每关 3 个种子，全部 2.0 倍。
- 解压目录（只读）：`/Users/newbiexvwu/.claude/jobs/861880de/tmp/out1/out/`。通关的 12 份见 `report.md` 的“最佳验证结果”表；其余目录是中途或失败的分段存档，不要用。
- 主控已在 Mac 引擎上抽验 `L49_seed20261049_h151_092_wave40_tick91915`：`human_demo.py verify` 通过（1658 个操作，tick 91915，第 40 波，通关）。
- 它的做法是一个约 12 个参数的打法脚本加爬山搜索，代码在 `out/scripts/`，只作分析用。

## 要做的
1. **入库**：把 12 份通关存档拷进 `artifacts/llm_demo/raw/seed_<种子>/<task_id>/`（`artifacts/` 下照现有 `.gitignore` 规则，不进 git）。在 meta 里补 `label_source_detail: "free-model probe of task81 package, 2026-10-25"`，原字段不改。12 份逐一 `verify`，全部要通过，不通过的列出来并剔除。
2. **任务族**：其中 7 份用的种子不是对应 demo 族自己的种子（例如 L46 用了 20261047 / 20261048）。按 AGENTS §8，为这些“同卡组、不同种子”的局登记新的 task_id（例如 `demo_roof_46_doom_m200_s20261047`），DESIGN 同步登记，种子本身照原样。
3. **接手扫描**：照 task71b 的流程，每份存档 12 个点 × 8 个未来 × 老师 / 学生两臂，报老师过 50% 与 8/8 的最早点。L46–L49 的人类示范已有起点格（N=418 / — / 474 / 1348），把 LLM 局与人类局的 50% 线放在一张表里对比。
4. **登记起点格**：老师过 50% 的局照 task71b 第 2 步登记 `demo_start`。未来种子块要独立，与现有全部段不相交（机械核对）。
5. 结果写 `MEMORY/docs/demo-start-states.md` 新的一节，`git push origin demo`。部署由主控做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式（LLM 的脚本只读、只作分析）。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题

## 主控追加（2026-10-26 00:20）
用户担心 LLM 打的局“只求通关、不够稳”。第 3 步的对比表是对这个担心的直接检验，要做细：同一关的人类局与 LLM 局，分别报老师 50% 线 N 占全局动作数的比例，以及从各自 50% 线到终局之间老师续走的平均末波。结论里明确回答：“LLM 局的中盘局面是否比人类局更难接手？”
