# task107：用户新录的 L30 / L40 / L50 原生传送带示范入课程

用户 2026-10-26 录完三局，全部通关：

| 任务 | 种子 | 操作数 | 被吃植物 |
|---|---|---|---|
| `demo_pool_30_orig_m100` | 20261030 | 32 600 | 5 |
| `demo_boss_50_orig_m100` | 20261050 | 33 294 | 4 |
| `demo_storm_40_orig_m100` | 81029000 | 46 457 | 24 |

存档已放到台式机 `/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/human_demo/raw/seed_{20261030,20261050,81029000}/`，与其他示范起点格同一个目录。录制用的是 Mac 引擎。

## 要做的

1. 每局用 deploy14 引擎（cq `build5/pvz-portable`）`human_demo.py verify` **一次**，三局并行。这是入库关口，不要再重复做别的复验。
2. 按已有示范起点格的做法（task70 / task87 的起点格与倒推课程，参考 `demo_fog_*_hv2`、`demo_roof_*` 格的登记方式），给三局建起点格：
   - 原生传送带：`deck: []`、`conveyor_start: "original"`；
   - L40 暴风雨注意可见性口径与 task75 一致。
   - 起点从示范后段开始，让 qc 自动往前推。
3. 在 DESIGN 和配置里登记新格，写好部署补丁或脚本（结构照前几次示范格上线），**不要执行**，由主控在组边界部署（会与 task108 / deploy14d 一起上）。
4. 分支 `convdemo`（从 origin/deploy14c 开），推送。

## 约束

- 每个任务最多 2 个游戏进程，qc 优先。
- 遵守 AGENTS.md §6.5：不加任务书以外的审计。

## 结果

（待填）

- 2026-10-26 verify（deploy14 引擎 = cq build5/pvz-portable，每局一次）：demo_pool_30_orig_m100 32600 ops / tick 32520 / wave 30 / result 1；demo_boss_50_orig_m100 33294 ops / tick 33200 / result 1（boss 层 obs wave=0，与存档终局一致）；demo_storm_40_orig_m100 46457 ops / tick 46319 / wave 30 / result 1。三局 verified=true。
- 2026-10-26 起点格登记：configs/human_demo_tasks.json 三条目加 demo_start（pool 624..264 / boss 656..256 / storm 900..336，未来种子 45300000/45400000/45500000 各 256，index=0 由 qc 自动往前推）；storm 条目补 fog_observation=storm（task75 口径，纯表现层）；测试期望 31→34；DESIGN.md §14.36 登记 + 主控部署清单（不执行，随 task108/deploy14d 上线）。分支 convdemo 已推送（78ca7023）。
