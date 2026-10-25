# 门禁种子池扩容（task88）结论

> 2026-10-26。执行：Codex（第 1/3 步大部分）+ CodeBuddy（第 1 步修正、第 2 步、第 4 步、本结论）。合并与上生产由主控做。

## 结论

1. **旧池确实要用完**：门禁 v2/v3 账本 `~/pvz-logs/gate_seed_ledger.json` 已发到 36272，池 [31024, 65536) 剩约 29k 种子；v3 每轮 1920 个，两天内耗尽，耗尽即 `SeedPoolExhausted`、课程卡死。
2. **新池选 [100000000, 101000000)**，即第二段 100 万个种子（约 520 轮 v3）：
   - 与全机全部已登记段机械盘点后**逐位不相交**（`scripts/gate_seed_inventory.py` 全量扫描：61 个占用区间、约 496 万种子，最高段为 llmplay 的 [83000000, 83000308)；candidate 报告 `disjoint: true, overlaps: []`，紧凑汇总入库为 `MEMORY/docs/gate-seed-inventory.json`）；
   - 对 [1亿, 1.01亿) 做了全机 JSON/JSONL 字面量复核：251774 个文件中 108 个命中**全部甄别为误报**——`rss_bytes` 进程内存值与十六进制 state_hash（如 `"100387332e90fcf1"`），零真实种子占用；
   - 引擎种子宽度实测（`scripts/gate_seed_width_check.py`，结果 `MEMORY/docs/gate-seed-width-check.json`）：9 位种子逐位保留，同一种子两次确定，与 uint16/uint8 残差及相邻种子观测链均不同——无 16/8 位截断。
3. **实现**（commit 06d3a3d2）：`pvz_promotion.SEED_POOLS_V2 = ((31024, 65536), (100000000, 101000000))`，按池序领取；`SEED_POOL_V2` 保留为首段别名；旧账本（只有 `pool` 键）可读，首次写入原地升级为 `pools`（旧代码遇新账本 fail-closed，不会错领）；`_freeze_ledger_panel` 接受任一段内的块。
4. **不升 GATE_VERSION**：panel / policy / 契约字段一字未改（policy 里的 `seed_pool` 保持首段字面值，账本记录每块实际来自哪个窗口），v2/v3 旧证书逐位不变，两条 SHA256 断言锁死。
5. **dry-run**（`scripts/gate_seed_dryrun.py`，结果 `MEMORY/docs/gate-seed-dryrun.json`）：在账本副本上按 v3 轮量连领 250 轮（48 万种子），逐对不相交、全部落在已登记池内、窗口单调：前 15 轮用完旧池尾部，从第 16 轮起进入新池，最后一块 [100449280, 100451200)。真实账本未被触碰。

## 交接注意

- 账本格式升级发生在下一次真实领块时（门禁下一轮自动完成），无需人工迁移；**主控在组边界改账本，执行者只读**。
- 盘点脚本两次修正要随本分支合并：verify 正则排除浮点小数/十六进制串（否则每个 9 位小数 mantissa 都是假命中）；扫描排除 `MEMORY/docs/gate-seed-*.json` 自身工件（测试/演示值不是种子声明，否则占用区间被污染成 [0, 1.00000002亿]）。
- 新池登记进 DESIGN.md 与主 ledger 由主控在合并时做。

## 产物索引

- `python/pvz_promotion.py`：SEED_POOLS_V2 / 向后兼容 / 测试
- `scripts/promotion_gate.py`：allocate_seed_block 多池
- `scripts/gate_seed_inventory.py`：种子段盘点（本文件数字的来源，可重跑复现）
- `scripts/gate_seed_width_check.py` + `MEMORY/docs/gate-seed-width-check.json`
- `scripts/gate_seed_dryrun.py` + `MEMORY/docs/gate-seed-dryrun.json`
- `MEMORY/docs/gate-seed-inventory.json`：紧凑盘点汇总（全量明细是临时文件，不进 git）
- `python/test_promotion.py`、`python/test_promotion_evaluation.py`：24 条测试（含旧证书 SHA256 锁、跨池续领、旧账本升级、250 轮不相交）
