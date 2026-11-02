# task220：GPT Pro 包改为「学生败局中盘接回 + 前沿倍率 + 升级植物」

2026-11-02。交付目录 `~/Downloads/pvz-llmplay-220/`（`llmplay_pkg.7z.001` + `llmplay_res.zip`）。
分支 `llmpkg220`（基于 origin/llmpkg 的 task211 包代码）。

## 1. 派发次序

`resume`（学生败局中盘接回，最高优先）> `frontier`（前沿倍率，只给 >1.0 的关卡）>
`native`（传送带/Boss）> `main`（主线新卡组：升级植物优先，雾/屋顶广度在后）。
普通 ×1 日/夜/池重放轨整体退役（配置里 `decks.replay` 删除，`decks.frontier` 取代）。

## 2. 各轨条目数

| 轨 | 条目 | 说明 |
|---|---|---|
| 中盘接回 | **365** | 夜 129 / 日 85 / 雾 77 / 池 74；覆盖 32 关；r86 门禁 A 臂 173、B 臂 192 |
| 前沿倍率 | **14 关 × 6 胜** | L1/L3/L4/L8 2.5、L6 2.0、L9 1.75、L11 2.0、L13/L16/L18 1.5、L21/L22/L38 1.25、L31 1.5 |
| 升级植物 | 每会话 6/6/6/4/6 | 日/夜/池/雾/屋顶；只算本会话新胜组，carry 历史不计 |
| 传送带/Boss | 8 关 × 16 胜 | 不变；L50 已 17 胜，L40 仍欠 |
| 主线新卡组 | 40 关循环加 2 套 | 不变，升级植物/雾/屋顶广度并入此处 |

中盘接回的卡组只有 8 种签名：门禁面板每关固定一套卡组，A/B 两臂共用；所以每条都是
「本关原卡组 × 原倍率 × 原种子」，不是新卡组实验。

## 3. 中盘接回怎么生成

1. 读 round 86 门禁 `A`/`B` 两臂（`curriculum_state.pipeline/round_000086/evaluations/`）的
   `games.jsonl`，按地形取败局（夜/日/池/雾），共 373 个（去掉 L10/L20 传送带关 8 个）。
2. 用当时的生产代码与 `csearch1` 引擎按原 checkpoint / 原种子重跑，`--frames` 落逐决策帧；
   373 局全部复现为败局（与 `games.jsonl` 的 `won/final_tick` 一致）。
3. 首次破门 = 小推车 `MOWER_TRIGGERED` 或僵尸 `x<40`（进第 0 列）；截断到破门前 **45 秒**
   （4500 tick，100 tick/秒），取不晚于该 tick 的最后一个决策帧。
4. 由帧还原 `ticks.jsonl`（`{action,tick,end_tick,ok}`，拒绝动作后补 60 tick 等待），
   `frames.npz` 只存截断点那**一帧**（避免把学生前缀动作标成 GPT 示范）。
   meta 写 `resume_id` / `workflow_layer=resume` / 原 `environment_task`；`outcome.final_tick`
   就是截断点，所以 `Game.load_archive` 直接重放到中盘。
5. 抽样 22 份（四地形 + 最早截断）在包引擎里重放：tick 与整帧完全一致；雾关只有
   「看不见的己方植物血量」生产帧写 -1、包内重放给真实值，其余（植物位置/种类、阳光、
   僵尸、小推车、波次）逐位一致。
6. 端到端：在包内 `Game.load_archive(item["path"])` → `wait` → `export_archive`，
   meta 带 `resume_id`、倍率 1.25、`resume.played` 从 0 变 1。

## 4. 前沿倍率

倍率取台式机 `~/pvz-logs/curriculum_state.json` 的 `cells[*].mult_index`（排除 `demo_start`
这类自带 `ladder` 的起点格），按关卡取最大档；只登记 >1.0 的 14 关（传送带 L10/L20 仍走支线）。
新任务族 `llmplay_frontier_l{level}_m{mult}`，种子段 320000000–359999999（每关 1e6 槽，
纪元步长 64e6），`deck_selection: player`，GPT 只能用本关已登记胜组，每关目标 6 胜。

## 5. 升级植物

- 地形配额只统计**本会话**新胜组（`carried` 历史不计），所以 carry 再大也不会把配额直接刷满。
- 项里带 `upgrade_cards` / `upgrade_wins` / `upgrade_target` / `upgrade_required`；
  `missing>0` 时新卡组必须带一张本地形升级卡，PROMPT 要求写清底座与经济计划。
- 玉米加农炮(47) 从 `blacklist.all` 移到日/夜/池/雾黑名单，只屋顶/Boss 可用；其余升级卡
  本来就在池内（香蒲 43 仅泳池/雾）。
- 所有权仍由 `pvz_tasks.profile_for_deck` 自动声明（AGENTS §8.2），没有第二份实现。

## 6. 交付与校验

- 接口包 **7,458,787 字节**（含 `resume/` 365 份存档约 17 MB 原始、1.4 MB 压缩）。
- `md5 llmplay_pkg.7z.001 = fe648730667dbadcc0987c4968054bbf`（7,458,787 字节）
  `md5 llmplay_res.zip = 7c55cd5bfda1ea3f3f30300cb97a38f9`（34,258,402 字节）。
- carry 由 task211 会话 `decks_20261101_2247_out.zip` 的 `out/progress.json` 与旧 carry 合并：
  8673 条唯一存档（旧 3129 条全部包含在会话进度里，无丢失）。
- 修了包内 `render_episode` 顶层 `from unified_panel import read_panel` 导致接口包无法
  `import llmplay` 的问题（`unified_panel` 从未进包）：改成只在生成面板的函数内延迟导入。
- `python/test_llmplay_decks.py` 26 项通过（新增中盘接回一次性、前沿倍率轮换、升级配额三条）。
