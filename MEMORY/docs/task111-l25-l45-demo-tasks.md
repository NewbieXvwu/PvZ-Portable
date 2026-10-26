# task111：登记 L25 / L45 原生示范录制任务（让用户能录）

依据 task110 的调查：
- L25 已作为原生传送带族 `native_l25` 接入，只是 `configs/human_demo_tasks.json` 没有它的录制任务；
- L45（蹦极闪电战）走原生传送带，点牌、种植的录制路径与 L30 / L50 相同。

要做的：
1. 在 `configs/human_demo_tasks.json` 登记两个录制任务，命名和字段照 `demo_pool_30_orig_m100`：
   - `demo_pool_25_orig_m100`；
   - L45 的原生任务。
   具体要求：
   - 空牌组，`conveyor_start: "original"`，倍率 1.0；
   - 种子从 DESIGN 新登记的段里取；
   - DESIGN 同步登记（§8：新任务族要登记）。
2. 冒烟验证：每关各 1 个进程、各跑一次，几百 tick 就停。
   - 用 `human_demo.py` 的同一入口，以非交互方式 reset，确认原生发牌正确、能点牌种下；
   - 再做一次短录制，并 verify 回放。
   Mac 冻结只针对采集和训练，这种冒烟可以做。不做别的审计。
3. 写清给用户的录制命令（`python3 scripts/human_demo.py record --task ...`），以及这两关的操作要点（L45 蹦极落点有红靶提示等），放在本文件结果节。
4. 推到 pvz-env（`git pull --rebase` 后只改 configs / DESIGN / 本文件 / 当天日志 / 测试期望）。

遵守 AGENTS.md §6.5。
