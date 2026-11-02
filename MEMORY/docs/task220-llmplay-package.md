# task220：GPT Pro 包改为「学生/老师败局中盘接回 + 前沿倍率 + 升级植物」

2026-11-02。交付目录 `~/Downloads/pvz-llmplay-220/`（`llmplay_pkg.7z.001` + `llmplay_res.zip`）。
分支 `llmpkg220`（基于 origin/llmpkg 的 task211 包代码）。11-02 按主控要求扩容到 ≥6000 高价值项。

## 1. 派发次序

`resume`（败局中盘接回，最高优先）> `frontier`（前沿倍率三档）>
`native`（传送带/Boss）> `main`（主线新卡组：升级植物优先，雾/屋顶广度并入）。
普通 ×1 日/夜/池重放轨退役。

## 2. 高价值项计数（预生成 6787 项）

| 轨 | 条目 | 说明 |
|---|---|---|
| 中盘接回 | **5651** | 夜 2078 / 池 1321 / 日 1218 / 雾 919 / 屋顶 115；34 关；每个败局 30/60/90 秒三档截断（1527 源三档、501 源两档、68 源一档） |
| 　门禁败局 | 4049 | r86–r89 A/B 两臂：A86 469、B86 540、A87 558、B87 552、A88 513、B88 460、A89 472、B89 485 |
| 　老师败局 | 1602 | 生产采集 `fast_search` 在课程前沿倍率上的败局（最近 6 天，24 关） |
| 前沿倍率 | **1008** | 42 个任务 × 24 胜：前沿档 14（>1.0）、拉伸档 14（前沿+0.5）、雾/屋顶档 14（各自倍率，含 1.0） |
| 传送带/Boss | **128** | 8 关 × 16 个不同种子胜局，不变 |
| 升级植物（主线内） | 120 | 日/夜/池/雾/屋顶 每会话 24 套，只算本会话新胜组；`upgrade_required` 强制带升级卡+底座 |

实际会话先吃中盘接回与前沿倍率；主线升级配额在 `main` 项里继续生效。

## 3. 中盘接回怎么生成

1. **门禁**：读 r86–r89 `round_0000NN/evaluations/*/manifest.json + registration.json + {A,B}/*/games.jsonl`，
   取日/夜/池/雾/屋顶的败局（跳过传送带关），按原 checkpoint / 原种子用 `csearch1` 重跑并 `--frames`；
   1504 个种子全部复现为败局（`won=false`），拿到逐决策帧。
2. **老师**：扫 `artifacts/hard/*/games.jsonl` 的 `fast_search` 败局，要求 frames 存在且
   `environment_task.zombie_count_multiplier` 等于该关当前课程前沿；每关最多 45 个，共 957 个候选，
   592 个有可用截断（其余破门太早）。
3. **截断**：首次破门 = 小推车 `MOWER_TRIGGERED` 或僵尸 `x<40`（进第 0 列）；
   取破门前 3000/6000/9000 tick（30/60/90 秒）各一个截断点，每个截断点是一个独立项。
   不足 600 tick 的截断点丢弃（68 个源只出了一档）。
4. **存档**：由帧还原 `ticks.jsonl`（`{action,tick,end_tick,ok}`，拒绝动作补 60 tick 等待），
   `frames.npz` 只存截断点那一帧（去掉 action/label/targets/verification），meta 写
   `resume_id` / `workflow_layer=resume` / 原 `environment_task`；`outcome.final_tick` 就是截断点。
   包内 `Game.load_archive(item["path"])` 直接重放到中盘，不做批量重放（§6.5）。
5. 任务 id 加 `_x<倍率×100>` 后缀：同一套门禁任务在 r86–r89 的倍率有 2 处变化（L9、L39），
   避免导入时同族倍率冲突；卡组/倍率/种子本身不变。

## 4. 前沿倍率三档

倍率取台式机 `curriculum_state.json` 的 `cells[*].mult_index`（排除 `demo_start` 自带 ladder 的起点格），
按关卡取最大档（last_round ≥ 80 且当前窗口有局）：

- 前沿档：>1.0 的 14 关（L1/3/4/8 2.5、L6 2.0、L9 1.75、L11 2.0、L13/16/18 1.5、L21/22/38 1.25、L31 1.5）。
- 拉伸档：同 14 关，倍率 +0.5（L1 3.0、L9 2.25、L21 1.75…）。
- 雾/屋顶档：雾 L32/33/34/36/37/39 与屋顶 L41–44/46/49 按各自倍率（多为 1.0），补广度。

新任务族 `llmplay_frontier_l{level}_m{mult*100}` 共 42 个，`deck_selection: "player"`，
GPT 只能用本关已登记胜组；种子窗口分别从 320000000（前沿）/380000000（拉伸）/400000000（雾屋顶）起，
每关 1e6 槽、纪元步长 64e6，与既有各段不重叠。每关 24 胜。

## 5. 升级植物

- 地形配额 24 套/会话，只统计**本会话**新胜组（`carried` 历史不计）。
- 项里带 `upgrade_cards` / `upgrade_wins` / `upgrade_target` / `upgrade_required`；
  `missing>0` 时新卡组必须带一张本地形升级卡，PROMPT 要求写清底座与经济计划。
- 玉米加农炮(47) 从 `blacklist.all` 移到日/夜/池/雾黑名单，只屋顶/Boss 可用。
- 所有权仍由 `pvz_tasks.profile_for_deck` 自动声明（AGENTS §8.2）。

## 6. 交付与校验

- 接口包 **13,146,416 字节**（5651 份存档约 222 MB 原始，7z 后 13 MB，嵌套前缀压缩率很高）。
- `md5 llmplay_pkg.7z.001 = 77c293a4f9bf7d5e159595470abc971b`
  `md5 llmplay_res.zip = 7c55cd5bfda1ea3f3f30300cb97a38f9`（34,258,402 字节）。
- carry 由 task211 会话 `decks_20261101_2247_out.zip` 的 `out/progress.json` 与旧 carry 合并：
  8673 条唯一存档（旧 3129 条全部包含在会话进度里，无丢失）。
- 抽样重放：老师 10/10、门禁 10/10 在包引擎里 tick 与整帧一致（雾关只差隐藏血量表示）；
  端到端 `Game.load_archive → wait → export_archive`：meta 带 `resume_id`、倍率/任务 id 正确。
- 性能：`decks.region()` 加 lru_cache（5651 条目录每次 progress 都要算地形，原先每次调用约 10 秒）；
  `resume_items` 只把队首 256 条变成项。现在 `progress()`/`next_task()` 各 <1 秒。
- 修了包内 `render_episode` 顶层 `from unified_panel import read_panel` 导致接口包无法
  `import llmplay` 的问题：改成只在生成面板的函数内延迟导入。
- 默认单片 `LLMPLAY_SHARDS=1`。`python/test_llmplay_decks.py` 26 项通过（11.7 秒）。
