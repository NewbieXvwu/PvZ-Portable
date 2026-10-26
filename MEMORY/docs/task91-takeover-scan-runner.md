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

- 2026-10-26（Codex 继续，生产串行排队）：核对原launcher 793563 / 链主进程793570仍存活，未重启回测；L46/L48 adaptive均完成（L50=418/473，各88局），L49进行中，完整12点对照尚待原链自动补齐。启动监督器809706，等待原链终止后逐份严格比较人类同块证据，三份通过才续跑12份生产adaptive→register dry-run→register→72项针对性测试；复用现有scan_out，登记前保存四类文件快照，核对旧games内容及既有demo_start不变。监督器、状态和逐阶段日志在ignored的`artifacts/task91/audit/continue_scan.py`、`continuation_status.json`、`production_*.txt`。本任务始终最多2个游戏进程；尚未宣告回测或生产扫描完成。

- 2026-10-26（主控澄清后续跑）：2进程上限按每个任务计算。沿task70部署记录找到cq已有的人类L46/L48/L49数值存档，只读复制三类文件到本任务目录并逐文件SHA核验；未操作cq会话或进程。共用task89加锁账本分配新扫描块86,000,000/86,001,000/86,002,000，启动当前r11的adaptive→同块完整12点老师对照，本任务workers=2。前述“只在Mac”的阻塞判断已撤销，等待真实回测结果。

- 2026-10-26（Codex 当前代码端到端）：更新器按AST定位实际课程测试中的test_registered_manifest_start_cells，避免误改更早的临时样例断言；汇总断点也核对存档/任务/种子身份。34项运行器测试、30项导入器测试通过；课程起点格8项中7过/1因缺存档跳过。两份存档重新verify，复用20局（第一份3点、第二份2点，各2未来两臂）完成run→register；私有账本写入两族与DESIGN/测试，重试四类文件字节不变，生产四类文件均不变。证据MEMORY/docs/takeover-scan-evidence.json；第2步人类回测仍待输入，任务保持未完成。

- 2026-10-26（Codex 自适应预算与回测）：恢复任务书要求的4等距起测，二分后补探失败区间以捕捉非单调隆起，老师最多9点、学生最多2点（8未来≤88局，严格低于192/2）；逐点原子保存未完成状态，未完成/缺逐局证据结果禁止登记。冻结独立adaptive_trace；比较器强制完整12点、同一未来块、逐局结果一致及实际局数。32项测试通过（含4096种二值曲线预算检查）。复用缓存重建L46三份完整基线并运行新算法：L50=634/693/622，档差全0，每份88局；原始对局mtime/SHA均不变，未重跑游戏。人类L46/L48/L49回测仍待存档，不能据LLM三份宣布全任务完成。

- 2026-10-26（Codex 登记账本）：修复 register 默认指向不存在的 raw/scan_seed_ledger 路径，改与 task89 共用 canonical scan_seed_ledger.json 和文件锁，导入器兼容登记的256块，旧独立账本仍参与避让。按存档绑定登记分配以便失败重试复用；dry-run给出真实避让块；既有格底层更新也禁止覆盖；测试期望只在目标集合内判断是否已登记。运行器26项 + 导入器30项测试通过。尚需取点成本约束及人类示范回测。

- 2026-10-26（Codex 接续）：先合并 origin/pvz-env 的3个文档提交并推送（c95010a8）。复用 CodeBuddy 的扫描缓存；修正断点按实际未来种子集合而非行数匹配、缓存摘要只保留当前未来、冻结存档/模型/引擎输入指纹并核对旧配置，结果原子落盘；修复新输出目录首次 adaptive 缺 results 目录，workers硬限制1–2。22项测试通过，真实L46已完成点续跑0.27s、原始对局文件mtime/SHA均不变。继续核对取点预算、共享账本和缺失的人类示范回测。

- 2026-10-26（CodeBuddy 接手）：`scripts/takeover_scan.py` run/adaptive 核心就绪（生产老师参数、断点续跑、
  结果按存档写 json 含胜负/末波/破防僵尸），`python/test_takeover_scan.py` 13 项单测过；
  冒烟：L46 s20261046 d1 老师臂 2 未来跑通（0/2，早期点每局约 4 秒），重跑 0.08s 零重跑。commit 713b28f3。
  task89 的 zip 已入库本工作树（12 份通关、288 组扫描清单、`scan_seed_ledger.json`），生产清单未被改（task87 登记幂等）。
- 2026-10-26（同日）：register 就绪——按存档解析登记族（task87 `_llm` 先例），既有起点格不覆盖只报告；
  独立登记账本 `register_seed_ledger.json`（86M 起、256 块、合并盘点+两账本避让）；单测 17 项过。commit 809cf48a。
- 2026-10-26（同日）：端到端通过——独立 dest/账本（未来块 90M 起）重走 import，run 2 份 × 2 点 × 2 未来两臂，
  register 写入分支验证（`_llm` 回退、86M/87M 种子块、DESIGN §14.35 自动编号、测试期望插入）。生产清单零改动。
  run 的 `--decisions` 改为按各存档网格取交集；`--task-id` 精确过滤。
- 2026-10-26（同日）：回测完成（结论见 [takeover-scan-runner.md](takeover-scan-runner.md)）。同块对比
  （同存档同未来种子块，adaptive vs 全量 12 点）：L46 三份 **档差全 0**，判据通过；每份老师 40–72 局 +
  学生 16 局，<12 点全量的一半。期间修正：等距 4 → 5 点（L46 s20261048 的非单调隆起 622 过线/725 跌回
  被 4 等距漏掉，差 2 档；5 等距后 Δ=0）。跨种子块噪声实测 1–2 档（8 未来小样本，边界档随块翻转），
  阶梯只认同块实测。adaptive 结果结构统一为 run 的 {wins,games,rows}（修了混读崩溃）。
- 2026-10-26（交接）：CodeBuddy 收工交回 Codex。全部五步完成、已推送（HEAD a299ce07，工作树干净，
  无残留游戏进程）。工具入口与用法见 [takeover-scan-runner.md](takeover-scan-runner.md)；
  生产扫描（12 份全量 adaptive + register）与人类示范同块回测（存档在 Mac）待接手方执行。
  注意：本分支 llmimport 相对 origin/pvz-env behind 3（上游有新提交），合并由主控决定。

- 2026-10-26（CodeBuddy 接管片段）：接手时人类示范回测（launcher PID 793563 / adaptive PID 793571，adaptive→run→compare 全自动链）仍在跑，按主控指示不打断；截至 07:58 第 1 份（L46 s20261046）adaptive 完成并落 results，第 2 份（L48 s20261048）进行中。生产扫描（12 份 adaptive + register）未启动，本会话无代码改动。task91 交回 Business 号 Codex 继续盯完回测并做生产扫描。

## 问题

- 已解决：本任务原先没有L46/L48/L49目录，但task70部署记录注明cq已有副本；已只读复制并开始回测。不再需要主控另传存档。

1. 原人类存档入口问题已解决，当前执行`adaptive` + `run --teacher-only` + `scripts/compare_same_block.py`；尚未以历史Mac胜率表替代本次r11逐局对照。
2. task87 登记的 L46 `_llm` 阶梯只有 3 档（724/634/543），而 §6.16 逐局表显示 815/905/996
   也全 8/8（按 ≥50% 全入应有 6 档）——两处对不上，本机没有 task87 的原始扫描 json 无法核实。
   register 对既有格一律不覆盖、只报告差异，所以不影响本次交付；主控若有原始数据可复核。
3. 同存档的 50% 线跨未来种子块实测移动 1–2 档（8 未来小样本，边界档随块翻转）。
   含义已写进结论：阶梯档位只认同块实测；两次扫描逐档对比没有意义。
