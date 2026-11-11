# task269：修模仿者卡组导出与原生派题，合并 238j，打 238k

分支 `pkg238k`，基于生产 `pkg238` aab98068（含 task268 数据瘦身）。合并 `origin/pkg238j` 后，功能提交 `6c6de850`（合并与两处修复）与 `d2093225`（包内 `workflow` 导入顺序），`26f4ce31` / `914b6b11` 为报告与 `deck_rules` 加固，分支最终 HEAD `914b6b11`。未部署，未跑整局对局、未开窗口；台式机只跑一次 `accept_package.py`，目录用后删除。

## 1. 模仿者卡组导出崩溃

### 根因

L47 用户卡组是 `[1,33,41,[48,1],39,44,37,4,17,30]`，模仿者槽是一个嵌套列表。`export_archive` → `delivery.record_archive` → `workflow.derive` → `decks.derive` 在把卡组转成集合时直接调用 `set(entry.get("deck"))` / `set(cards)`，Python 对列表元素求哈希，抛 `TypeError: unhashable type: 'list'`。导出在写盘后、发布进度前失败，所以打赢的 L47 局也入不了库，后续接回题被整批跳过。

同一错误不只在 `decks.py` 一处。`git grep` 后确认以下位置都会对卡组做 set / hash / sort / 比较：

- `scripts/llmplay/decks.py`：胜组过滤、地形成员判断、前沿/主线的 legal_cards 子集、升级植物交集；
- `scripts/llmplay/workflow.py`：`card_diff`、组内卡组排序、C 层指定卡组比较；
- `scripts/llmplay/delivery.py`：E 诊断卡组排序；
- `scripts/llmplay/game.py`：玩家选卡的未解锁卡检查；
- `scripts/llmplay/build_package.py`：carry 诊断键的卡组排序；
- `scripts/pvz_tasks.py`：地形覆盖、随机卡组排除/告警签名；
- `scripts/make_random_deck_cells.py`：读取已登记卡组时的签名（这里会读到 L47 用户卡组，已实际复现 `sorted` 比较 list 与 int 的崩溃）；
- `scripts/pvz_live_curriculum.py`、`scripts/unified_panel.py`：卡组签名。

### 改动

卡组 → 卡集合只保留一个实现，放在 `python/pvz_env.py`：

- `deck_cards(deck)`：按现有约定，`[48, x]` 同时贡献 48 和被模仿的卡 `x`；普通卡贡献自身。
- `deck_sort_key(card)` / `sorted_deck(deck)`：把嵌套对规范成 `(48, x)` 后稳定排序，结果可哈希，混合 int / 列表不再抛错。

所有上述位置改为调用这三个函数。`scripts/deck_rules.py` 的 v5 生成器只接受 0–47 的普通整数卡（模仿者不在其输入域），这里也把非整数卡提前判为非法，不再先 `set(deck)` 而抛 `TypeError`。两处语义细节：

- 合法卡池只含植物 0–47，所以“卡组是否都在合法池”用 `deck_cards(deck) - {48}`；模仿者载体 48 不参与植物黑名单。
- 地形覆盖、升级植物配额和覆盖计数用完整 `deck_cards`，所以模仿者复制睡莲/花盆/升级植物时，被复制的卡单独计数。

定向用例 `test_task269_imitater_deck_export_derive_carry_and_import` 用真实嵌套卡组走完：写存档 → `record_archive` → `derive` → `progress` → `export_bundle` → `carry` → `resume` → 导入器的 `_deck_cards` / `family_entry` / `discover`。另在台式机验收中实际构造含 `[48,1]` 的 L47 对局并导出，`derive` 正常，`victory_decks` 为 25。

## 2. 原生派题与开局检查不一致

### 根因

`next_task()` 在认领时用一份进度快照排定 native 项并写入 `claims.json`；`Game()` 构造时 `decks.check()` 又重新读一遍进度，检查当前队列顺序、是否达到目标胜数、种子是否还在 `available_seeds`。两个子代理并发时，认领后其他代理的胜局会让 `wins` 从 31 变成 32，或让更早队列出现空闲项，于是 `next_task` 已经发出的 native 项在开局时被 `decks.check` 拒绝（“学生败局中盘接回优先”或“本关达到 32 胜后不再分配”）。

### 改动

- 新增 `decks._issued(task_id, seed)`：只要该 `task_id|seed` 仍在有效认领表中，就认为它是 `next_task` 已经发出的项。`decks.check` 对 frontier、native、main 的**已认领项**跳过实时队列顺序、目标完成状态和种子窗口检查；native 仍要求卡组为空，frontier/main 仍做卡组合法性检查。
- `decks.native_items()` 按“已胜 + 仍被认领”计算剩余名额：`remaining = target - wins - outstanding_claims`。目标胜数在发题时由认领计数控制，不再等开局时拿最新胜数拒绝一张已经发出的题。L30/40/45/50 的目标 32 来自 `native_targets`。
- 派题顺序和开局检查仍共用同一份 `ordered_items()` / `dispatch_priority()`；合并 238j 后尾部原生关优先于 scenario/frontier，这一点保留。

定向用例 `test_task269_claimed_native_item_is_accepted_after_target_fills` 覆盖：31 胜时只剩 1 个名额；打满 32 胜后未认领项仍被拒；已认领项放行；更早的 drill 队列变空也不撤销已发出的 native 项。

## 3. 合并 238j

`git merge origin/pkg238j` 的冲突：

- `MEMORY/2026-11-09.md`、`MEMORY/2026-11-11.md` 为文本冲突，两边内容都保留（task263/264 与 task267/268 的记录）。
- `scripts/drill_start.py` 自动合并成功：task268 的 `replay_actions()`、旧雾档案 `archive_task(... fog_observation=...)` 恢复，与 task267 的 `body_health`、`phase`、`zombie_death_phases()`、`judge(..., dying=...)` 同时保留；判据仍只有 `judge()` 一份。
- 238j 把尾关 native 的派题优先级改到 frontier 之前，旧用例 `test_decks_frontier_rotates_registered_decks_on_fresh_seeds_first` 仍按旧顺序断言。已在该用例内临时清空 `native_targets`，只验证普通 native 排在 frontier 之后，不改生产配置。
- `scripts/llmplay/accept_package.py` 同时保留 238j 的回退链验收和本次新增的模仿者导出验收。

## 4. 测试

Mac 上全部 pytest 只用 `-k`，加 `timeout 300` 与 `ulimit -f 2000000`，同一时间只开一个；没有启动引擎对局、没有开窗口。

| 范围 | 命令（`-k`） | 结果 |
|---|---|---|
| 模仿者全链路 | `test_llmplay_decks.py -k task269` | 2 passed |
| native / frontier / 合并后的 decks | `test_llmplay_decks.py -k 'native or frontier or task265 or task266 or task263 or task248 or task269'` | 22 passed |
| 238j 四项接口修复 | `test_llmplay.py -k Task267` | 4 passed |
| 238j 回退/恢复导入 | `test_llm_import_sha.py -k 'task266 or task265'` | 1 passed |
| 卡组规则 | `test_random_deck_cells.py -k 'generated_decks_satisfy_v5 or violations_are_detected or registry_is_reproducible'` | 3 passed |
| 地形 / 模仿者覆盖 | `test_replay_task_spec.py -k 'imitater or terrain or deck'` | 4 passed |
| 冻结卡池 | `test_deploy17.py -k recipe_and_frozen_pool` | 1 passed |

`python/test_drill_start.py` 顶层 `import torch`，Mac 当前环境未装 torch，未运行该文件。用同一入口直接执行了 task268 的旧雾档案断言（`archive_task` 保留 `human_v2`），并确认合并后 `replay_actions` 与 `judge` 都存在；task267 的死亡阶段判据由 `test_llmplay.py -k Task267` 覆盖。另外用最新交回的 15,953 条记录跑了完整 `workflow.derive`，1.7 秒完成，L47 `victory_decks=25`。

## 5. 台式机验收

启动前 `/mnt/c` 剩余 72 GB（门槛 25 GB）；在独立 `~/pvz-logs/task269/` 下用 `systemd-run --user --property=MemoryMax=4G`、`nice -n 19 ionice -c3` 运行，退出码 **0**，内存峰值 534 MB，用时约 73 秒。

- 包内引擎：静态 x86-64 ELF，`check_engine` 通过；SHA256 与 238g/238j 相同。
- 三进程并发领题不重复；drill 推进到 900 tick 后判失败，导入器复验同判失败。
- scenario 正常导出；resume 推进到 tick 19,140，回退子项切点 9,420；父子两份存档经统一导入器真实复验，回退链诊断保留。
- **新增模仿者导出**：L47 卡组 `[1,33,41,[48,1],39,44,37,4,17,30]` 构造、导出成功，包内 `progress()` 正常重算，`victory_decks=25`，没有 `unhashable type: list`。
- 连续两次导出前后进度均为 15,958 条，不回退。
- 验收目录已删除；未操作生产 `cq`、`~/pvz-logs` 下生产文件或正在运行的 import-1112。

**旧导入器复验说明**：本次交回的两份新 scenario（`scenario238_roof_032`，未来种子 239000064 / 239000065）带 `dying` 字段。检查两份存档终局，`dying=[]`、`alive=[]`、`success=true`；旧判据没有 `dying` 分支，但目标已不在场，`alive_targets` 同样为空，`success` 不会翻转，因此未发现预期不一致。旧 import-1112 未在本次验收中实际执行，主控部署后应再复验这两份。

## 6. 包

目录：`/Users/newbiexvwu/Downloads/pvz-llmplay-238k/`

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `llmplay_pkg.7z.001` | 20,000,000 | `262d679adc7560907eed2affb4d92c12a3aef565394427b993cacf234c9d104b` |
| `llmplay_pkg.7z.002` | 348,855 | `aa23c36f5dc98f56cf7a93a75613c3b37f82e45d16b679b45436977e504c927b` |
| `llmplay_res.zip` | 34,258,402 | `0005bb8e31ad2e2ca7649bc0b6128211ff13c8f1c333c2163984302d7d65d94e` |
| `llmplay_pkg/build/pvz-portable` | 37,676,360 | `1c3680f36aa5bd5fef97d035a97a1cdd9ca89fce2493a87f125c7bd507d29c07` |

打包命令按要求复用 238j：`--reuse-package .../pvz-llmplay-238j/llmplay_pkg`，`--partial` 用 `LLMPlay_2026-11-11_220837_HKT_out.zip`，`--resume-dir` / `--asset-dir` 指向 238j 的 `llmplay_pkg`，`--out ~/Downloads/pvz-llmplay-238k`。静态引擎与资源包未变，`check_engine` 通过。

包内 carry 为 **15,953** 条，resume 8,457、drill 927、scenario 300；`PROMPT.md` 的 carry 时间为 **2026-11-11 22:08:37 HKT**。接口包与源码经 HF `transfer/task269/` 送台式机，取用后远端文件已删除；资源包从 238j 本地复制，未上传 HF。

## 7. 部署注意

本次未部署。主控部署合并时：

1. **生产 `cq` 与导入器都要更新到同一提交**。`deck_cards` / `sorted_deck` 在 `python/pvz_env.py`，`decks.py`/`workflow.py`/`delivery.py`/`game.py` 改为调用它；只更新采集端、不更新 import-1112，旧导入器仍可能在其他调用点遇到嵌套卡组。导入器的 `_deck_cards` 本身已支持 `[48,target]`，但共用的 `drill_start.judge`、`delivery` 和进度重算需要一起换。
2. 用 238k 的两卷接口包与 `llmplay_res.zip` 更新生产包；资源包与 238j 相同。carry 已包含 15,953 条记录，首次导入自动接上，不需要再附旧分卷。
3. 本次 238j 的 drill 死亡阶段判据改变了两份新 scenario 的 `dying` 字段。旧 import-1112 的 `success` 判据从存档终局看不会翻转，但主控部署应在实际导入器上复验这两份；若出现不一致，以新判据为准并记录。
4. 生产队列的发放侧和开局检查现在共用同一份认领逻辑。部署后若仍有旧 `claims.json`，过期的认领会按 TTL 自动释放；不要手工清空正在使用中的 claims。
5. 本次只改接口/任务逻辑，未改 C++ 引擎；静态引擎 SHA256 不变，不需要重编引擎。
