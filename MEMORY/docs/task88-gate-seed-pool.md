# task88：门禁种子池扩容——两天内会用完

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-hashdet`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/seedpool`（分支 `seedpool`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq、`~/pvz-logs/gate_seed_ledger.json` 的现有内容**（只读，改动由主控在组边界做）。

## 现状
- 门禁 v2 / v3 每轮从 `~/pvz-logs/gate_seed_ledger.json` 领一段全新的种子，从不复用（`python/pvz_promotion.py` 的 `SEED_POOL_V2`，池是 [31024, 65536)）。
- 现在已分配到 36272，剩 29264 个。v3 每轮约 1500–1800 个，一轮约 3 小时，所以大约 2 天后用完。用完之后门禁会直接报错，课程会卡在“候选无法评估”。

## 要做的
1. **盘点种子段**。把仓库里和两台机器上所有已登记的种子段列成一张表，并机械地求出空闲区间。来源包括：
   - DESIGN.md 的种子段登记；
   - `configs/` 下各清单；
   - `~/.pvz-dist/ledger.json`；
   - `gate_seed_ledger.json`；
   - 课程控制器的 offset 规则；
   - 示范与 llmplay 的种子块（2026xxxx、81,000,000+）。
   脚本进 `scripts/`，输出 json，**不要手抄**。
2. **选新池**。从空闲区间里选一段，至少够 200 轮，例如 1,000,000 个种子。
   - 要求：与上面全部段不相交；离训练用的种子足够远。
   - 先确认引擎与清单对大种子没有截断：种子是怎么传进引擎的，有没有 16 位或 32 位的边界？
3. **改代码**。让 `pvz_promotion` 支持多段池，即旧池用完后接新池。
   - 旧轮次的冻结面板逐位不变，用测试锁住；
   - 新增 `GATE_VERSION` 只在必要时做。能不升版就不升，理由写清楚；
   - ledger 格式向后兼容：旧文件不改写也能读。
   - 测试：耗尽旧池时切到新池；任意两轮的区块不相交；与第 1 步的全部段不相交。
4. **dry-run**。在一份 ledger 的副本上模拟领 250 轮，报告每轮区块，并确认不相交。
5. 结论写 `MEMORY/docs/gate-seed-pool.md`，提交并 `git push origin seedpool`。合并与上生产由主控做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果
- 2026-10-26 第 1 步（脚本，扫描仍在跑）：新增 `scripts/gate_seed_inventory.py`，从 DESIGN.md 正则、各工作树/主检出/~/pvz-logs 的 JSON 种子字段、门禁账本机械汇总区间并求空闲区间；输出 json，不手抄。
- 2026-10-26 第 3 步（代码）：`pvz_promotion.SEED_POOLS_V2 = ((31024, 65536), (100_000_000, 101_000_000))`，`SEED_POOL_V2` 保留为首段别名；`allocate_seed_block` 按池序领取、旧账本（只有 `pool` 键）可读并在首次写入时原地升级为 `pools`；`_freeze_ledger_panel` 接受任一段内的块。**不升 GATE_VERSION**：panel / policy / 契约字段一字未改，v2/v3 旧证书逐位不变（两条 SHA256 断言锁住）。
- 2026-10-26 第 2 步（引擎种子宽度实测）+ 第 4 步（dry-run）产物入库：`scripts/gate_seed_width_check.py` + json 实测 9 位种子逐位保留（非 16/8 位截断）；`scripts/gate_seed_dryrun.py` 在账本副本上模拟 250 轮，逐对不相交、全部落在已登记池内、窗口单调；盘点脚本加紧凑汇总模式。真实账本未动（仍止于 36272）。（commit ce7951db）
- 2026-10-26 盘点脚本两处修正：verify 正则排除浮点小数/十六进制串（`0.100845682` 的小数部分、state_hash 数字串曾被误报为 9 位种子命中，实测 1688 命中全为误报）；扫描排除本任务自身工件（gate-seed-*.json 的测试/演示值不是种子声明，曾把占用区间污染成 [0, 100000002]）。重跑全量扫描中。
- 2026-10-26 交接 CodeBuddy→Codex：代码交接时全仓 24 测试绿（test_promotion 14 + test_promotion_evaluation 10）。修正后的全量盘点重扫仍在后台跑（pid 731438，输出到 /tmp/gate-seed-inventory-full.json，含 --candidate 100000000 101000000 与 --verify-lo/-hi 字面量复核），预计十几分钟内结束；接手者跑完后用 `--from-inventory /tmp/gate-seed-inventory-full.json` 生成紧凑汇总入库，并甄别剩余命中（第一轮 1688 命中已证实全是浮点小数/十六进制串误报，新正则下 task23-results.json 命中清零）。第 5 步结论文档未写。

## 问题
