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

## 结果（2026-10-26）

主检出 `pvz-env` 已执行 `git pull --rebase --autostash`。清单新增 `demo_pool_25_orig_m100` 与 `demo_roof_45_orig_m100`，均为空牌组、原生传送带、倍率 1.0。DESIGN §14.23 登记示范种子 20261025 / 20261045，其他字段与 L30 原生示范采用相同默认值，旧任务保持原样。

### 录制命令与操作

在仓库根目录分别运行：

```bash
python3 scripts/human_demo.py record --task demo_pool_25_orig_m100
python3 scripts/human_demo.py record --task demo_roof_45_orig_m100
```

L25 是小僵尸泳池关。开局传送带为空，等待引擎发牌后，先点牌再点目标格。原生牌表为睡莲、坚果、豌豆射手和樱桃炸弹，种植免费，牌用掉后会从传送带消耗。水路先放睡莲，再种陆生植物。小僵尸数量多、体型小，注意各行推进。

L45 是屋顶蹦极闪电战，开局保留每行前三列共 15 个花盆。原生发花盆、食人花、南瓜和樱桃炸弹，种植同样免费。屋顶空格先放花盆，再放植物。蹦极下降时留意地面的红靶，它提示对应落点，普通偷取与空投运送都由游戏控制目标，玩家操作仍是点牌种植。

两关均可按 Esc 或空格暂停/继续，暂停时仍可种植和铲除，右键取消选卡。关闭窗口结束录制后会自动验证回放，存档分别写入 `artifacts/human_demo/raw/seed_20261025/demo_pool_25_orig_m100/` 与 `artifacts/human_demo/raw/seed_20261045/demo_roof_45_orig_m100/`。

### 短录制实测

使用默认 `build/pvz-portable` 与录制命令的同一任务装载路径（`load_training_task` → `training_task_kwargs` → `TaskSpec` → `Recorder`），代码驱动窗口点击，全程无需人工输入。每关只录一局，串行运行，录完关闭游戏再调用 `human_demo.verify()`，两关及回放共耗时 4.57 秒。

| 任务 | 原生开局 / 300 tick 发牌 | 实际点牌种植 | 录制与回放 |
|---|---|---|---|
| L25 / 20261025 | 10 空槽、无植物 / 樱桃炸弹(2) | 槽 0 → 第 1 行第 1 列，成功 | 321 tick、322 操作，verify 通过 |
| L45 / 20261045 | 10 空槽、15 花盆 / 南瓜(30) | 槽 0 → 第 1 行第 1 列，成功 | 321 tick、322 操作，verify 通过 |

回放使用同一引擎，每次操作的状态哈希、时刻与动作成功标记全部匹配。冒烟存档留在本机 `artifacts/task111/raw/`，来源标记为 `smoke_test`，不混入人类示范目录、不进入版本库。现有测试的任务数量和起点格期望无需调整。本次未修改录制器、模型或生产课程。
