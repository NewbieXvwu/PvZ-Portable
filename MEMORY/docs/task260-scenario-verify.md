# task260：scenario 交回存档复验调查（未完成）

工作分支 `scenfix`，调查基线 `ec539401`。本次已读取出题、重放与导入判据的调用链，并从台式机读到失败索引和一个完整样本的元数据。随后 SSH 连接中断，重新连接持续超时，尚未执行双引擎重放。根因未确定，未改运行代码，未追加导入，未更新生产引擎。

**已取得的现场证据**

台式机首次读取 `df -h /mnt/c`：C 盘 954 GB，已用 925 GB，剩余 30 GB，满足任务规定的 25 GB 启动条件。未启动重放实验，未创建远端实验目录。

导入脚本确认为 `~/PvZ-search/.claude/worktrees/llmimport/scripts/import_llm_run.py`，软链指向 `cq/scripts/import_llm_run.py`。导入日志开头是引擎输出，包含 BuildNum 4107、CommitDate 2026-11-03，前 32 行未出现导入命令，尚未取得完整运行参数。

读取目标库的 `merged_progress.json`，筛选 `base_task_id` 包含 `scenario` 且备注包含“判据与存档记录不一致”，得到 **141 份**。任务书的日 62、雾 54、夜 25 合计也是 141。任务书总失败数 146 与此相差 5 份，尚未读完原日志和原包，不能确认这 5 份的失败类别。

已取得以下失败记录，可按 SHA 在目标库 `sha256/<SHA>` 定位：

| 任务 | 种子 | SHA256 |
|---|---|---|
| scenario238_fog_052 | 100124384 | b5f8fa0eaccae076be65d96c9d2c4a5f55097eb487913092acf87f8f38c457b2 |
| scenario238_fog_039 | 100120230 | cdd2a940a44e1833cdb611527ff2408765a547f072af5d8f1a7232636aecc8ae |
| scenario238_fog_019 | 100122330 | c4cb337acfee2ef82deece5f69867123b0aa88260eff655ac9e15a3428e0e535 |

`scenario238_fog_052` 的元数据记录：L33，卡组 `[9,8,16,10,42,43,24,36,5,17]`，倍率 0.5，前缀操作数 126，注入操作 `[127,128,129]`。根 tick 6600，视界 1200，初始推车 6，目标为 id 66453512、类型 15、行 4、x=650 的小丑。无操作对照在 tick 7800 判失败，目标仍在场。GPT 实际结尾 tick 7811，记录目标已消失、推车未用、房子未破，局部成功。未来种子 239000104。原备注如实说明 `wait_until` 结束于根后 1211 tick。该 11 tick 超出视界本身不能证明复验差异的原因，仍需实际重放。

**代码与历史修复核查**

[`game.py`](../../scripts/llmplay/game.py) 的 `Game.load_archive → from_archive → _start → _replay_to` 负责恢复，操作日志中的注入命令也走 `_step_raw`。当前 `_replay_to` 没有逐条核对重放的 `ok` 和 `end_tick`，调查时应在这条既有路径记录实际值与原值比较。

`declare_scenario` 在 `game.py:750`，以当前在场僵尸 ID 声明目标，保存起点及注入操作索引，然后用快速快照推进一次无操作对照并恢复。C++ 的 `SNAPSHOT_FAST` 与普通快照共用 `SaveEnvironmentSnapshot` / `RestoreEnvironmentSnapshot`，快照覆盖随机流和对象数组的分配 ID。只读核查未发现足以确认本次根因的快照遗漏。

`_judge` 在 `game.py:717`，已包含 `0fea7ad4` 的修复，优先读取题目持久化的 `mowers_start` 和 `root_tick`。[`drill_start.py`](../../scripts/drill_start.py) 的 `horizon_reached` 已包含 `e91e220b` 的终局处理。task248 的记录在 [`MEMORY/2026-11-05.md`](../2026-11-05.md)，说明当时已修正前缀等待实际时长与 reset 上下文。本分支指定的 `student-bottleneck-plan.md` 中未检索到这些任务号和提交号。

[`import_llm_run.py`](../../scripts/import_llm_run.py) 的 `verify_short` 通过 `Game.load_archive` 恢复。drill 调用 `game.drill_result()`，scenario 直接调用同一 `drill_start.judge`，使用元数据里的目标、初始推车、起点和视界。当前导入器会复用失败回执，单纯重跑同一输入不会自动重新复验失败项，恢复调查后须处理这一点，保留已成功回执和已入库条目。

**修复、重新复验与 GPT 包**

尚无经过双引擎实测确认的修复。141 份判据不一致项及其余 5 份待分类项均未重新复验，新增入库 0 份，已入库 drill 与 scenario 未变动。尚不能判断 GPT 是否得到错误成功反馈，下一版 GPT 包的改动待根因确定。本次未打包。

**后续执行与上线边界**

恢复台式机 SSH 后，先核清原包 315 份 scenario 的全部失败原因，在 `~/pvz-logs/task260/` 内取日、雾、夜失败样本，分别使用 `imitater1` 和 pkg238 静态引擎重放，比较注入状态、目标 ID、各操作实际 tick、推车与终局。每次实验前检查 C 盘空间，运行使用 `nice -n 19 ionice -c3`。

若需要换引擎，必须先按任务要求完成普通对局逐位对照，再向主控交付部署步骤，不在本任务直接更新 `cq` 或 `imitater1`。若修复落在公共重放或判据中，则保留 drill 当前语义并做直接相关验证。随后通过 llmimport 的部署导入器只重试失败项，追加通过项，报告剩余类别，最后删除远端实验目录。当前没有可上线的运行代码变更。
