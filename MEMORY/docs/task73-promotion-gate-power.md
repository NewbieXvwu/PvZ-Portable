# task73：晋升门禁连拒 6 轮——是学生没进步，还是门禁在同一批种子上选出了"运气好"的现役？

> 主控 2026-10-25 写。执行者：台式机 CodeBuddy（`cb-qcbench` 会话）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/gatepower`（分支 `gatepower`，从 origin/pvz-env 拉出）。游戏进程最多 6 个；**不碰 tmux `qc`、cq、流水线目录**（只读）。

## 现象
晋升门禁 64 对配对（`recipes/panel_v2.json` 的冻结面板），非劣界 −3pp，Newcombe 95%：

| 轮 | 结果 | 候选胜 | 现役胜 | 差 | 95% 区间 |
|---|---|---|---|---|---|
| 2 | 过 | 40 | 37 | +4.7 | [−1.3, +10.6] |
| 5 | 过 | 48 | 40 | +12.5 | [+2.8, +22.1] |
| 6–11 | 全拒 | 41 / 44 / 50 / 42 / 46 / 45 | 48 | −10.9 … +3.1 | 下界都在 −7 到 −21 |

疑点：
1. **同一批种子既用来选、又用来评**：round 5 是在这 64 个种子上以 48/64 胜出的，之后每轮候选都和它在**同一批**种子上比，现役每次都是 48。被选中的那个，在这批种子上的成绩天然偏高（赢家诅咒 / 回归均值），候选很难"不劣于"它。
2. **功效不够**：64 对时，即使两者真实一样强，区间半宽也约 ±10pp，下界几乎必然低于 −3pp。非劣界 −3pp 大约要 600+ 对才有像样的功效。
3. heldout 组（8 对）现役 8/8，候选 2–6/8，可能是真实退步，也可能是同一个问题。

## 要做的
1. **复测**：从同一个评估任务池里取**新种子**（与冻结面板及所有训练段不相交，写清来源），每任务至少 32 个，总计 ≥256 对。比较 round 2、round 5（现役）、round 8、round 10、round 11 五个学生，两两配对，至少包括 5 对 2、5 对 8、5 对 11、8 对 11。学生检查点在 `~/pvz-logs/curriculum_state.pipeline/round_*/student/model.pt`，只读。评估命令与门禁一致（`promotion_gate.py` 的同一套参数，只换种子）。
2. **回答**：
   - round 5 在新种子上是否仍明显强于 round 2？
   - 6–11 的候选在新种子上与现役相比，差是多少、区间多宽？
   - heldout 组的退步是否复现？
   - 用实测的不一致对比例，算出 −3pp 非劣界要多少对才有 80% 功效。
3. **方案**（写清，不实施）：
   - 门禁怎么改：每轮换新种子、加大样本、序贯检验或别的办法；
   - 每种方案的耗时（按实测每局时长、16 进程）；
   - 对"真退步要能拦住"的保护力有多少。
   - 主控定。
4. 结论写 `MEMORY/docs/promotion-gate-power.md`，提交并 `git push origin gatepower`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 口径与种子来源冻结：复测参数与 round11 门禁契约一致（deploy3 引擎、value_net/v5、arms=fast、beta=0、绝对路径），只换种子。种子来源=注册评估池 unified_v2.json（34 任务×1024 种子，[30000,31024)）：历史评估已消耗 30000–30255（含冻结门禁 30000–30063），30256–31023 无任何记录使用；训练段 157,863 个 frames 种子标签全部 ≥1,600,000，与池零交集。新种子取 30256–30511 连续 256 个，按 PANEL_TASKS 顺序每任务连续 32 个、全局唯一。runner `scripts/gate_power_retest.py`：5 学生×8 任务×32 种子=1280 局，workers=5（≤6 进程上限）、search_play 按 (arm,seed) 断点续跑，统计复用门禁自己的 `pvz_promotion.summarize`。manifest/provenance 已落 `artifacts/task73/retest/`（ignored）。

- 复测完成与结论：1280 局全部 terminal、确定性抽查（workers 1 vs 5 重跑 8 局）逐种子一致。整体胜率（新 256 种子）r11 197 > r8 182 > r5 180 > r10 172 > r2 145。四问答案：① r5 对 r2 **+13.67pp [+8.15,+19.06]**，round5 晋升是真实进步；② 冻结 vs 新种子对照——r8 +3.1→+0.8pp（等价）、r10 −3.1→−3.1pp（一致）、**r11 −4.7→+6.6pp [+1.6,+11.7] 符号翻转，冻结种子上的拒绝是假拒**（夜间/泳池显著更好）；③ heldout 方向复现（r5 32/64，候选 12–28/64），但现役冻结 8/8 本身是虚高，r11 heldout 仅 −6.3pp 不显著；④ 功效模拟（门禁自己的 Newcombe 区间、实测不一致率 16.8–21.5%）：真等强时通过率 64 对仅 8–10%、1024 对 54–65%、**1536–2048 对达 71–91%**，真 +2pp 时 768 对 85–93%；拦截力不受样本量影响（真 −5pp 通过率 ≤1%、−10pp ≈0%）。实测 1.06 worker·秒/局，16 进程下 1536 对全评估约 3.4 分钟。方案（不实施）：每轮换新种子（池仅剩 512 种子，需注册更大评估窗口如 [31024,65536)）＋样本加到 1536–2048 对，或两段式 256 对初筛＋1536 对确认；zone-McNemar 否决未建模，实际功效只会更低、拦截力更高。报告见 [promotion-gate-power.md](promotion-gate-power.md)。

## 问题

## 结果（task73b：门禁 v2 已实施，待主控部署）

- **门禁 v2 实现**（`pvz_promotion.py` / `promotion_gate.py`，`GATE_VERSION=2`）：1536 对（8 任务×192 种子）、非劣界仍 −3pp、zone-McNemar 否决保留。种子来自专用池 **[31024, 65536)**，由 git 外账本 `~/pvz-logs/gate_seed_ledger.json`（fcntl 锁 + 原子写 + 只追加）按池顺序分配连续块；**耗尽抛 `SeedPoolExhausted` 停门禁，不回绕重用**；账本池号不一致/损坏/重叠均报错。现役与候选同轮在相同新种子上重跑（门禁本就双侧评估，契约/策略里显式声明不缓存旧成绩）。**旧 v1 证书照旧可验证**：`contract`/`freeze_panel`/`decide`/`policy` 全部按记录版本重放（v1 策略字典逐键不变，v1 清单验证 64 对语义不变），v1 流程完全不碰账本。门禁新增 `--gate-version {1,2}`（默认 2）、`--seed-ledger`、`--seed-block START,END`（只接受账本已登记块，未登记直接拒绝）。
- **管线协同**（`curriculum_queue.py`）：后台评估任务在**提交点**（持久化的 job 字典上）一次性分配种子块并存入 state，`deepcopy` 给 `_run` 的副本与外层共享同块——续跑/陈旧重评/重启都不会二次分配（修复过程中发现并堵住了一个“副本分配→外层契约永失配→无限重评烧账本”的漏洞，插桩验证每任务恰 1 次分配）。门禁子命令带 `--gate-version/--seed-block/--seed-ledger`；启动面板校验用探针块（纯验证，不碰账本）；`--seed-ledger` 新 CLI 参数（默认 `SEED_LEDGER_DEFAULT`，必须绝对路径）。池耗尽在提交点即抛错停门禁。
- **测试**：`test_promotion.py`（v2 面板分块/1536 对 decide/块校验拒绝/v1 策略字典逐键兼容）+ `test_promotion_evaluation.py`（账本顺序分配不重用、断点续跑不重复分配、池耗尽拒绝回绕、v1 全流程证书可读、现役双侧同轮重评、未登记块拒绝）+ 管线/队列夹具 v2 化。相关 4 个测试文件 **74 通过 + 4 子测试**（含 v1 旧用例全部保留通过）；另做了 `--seed-block` 冒烟：预分配→门禁复用（账本仍 1 块）、未登记块拒绝。
- **种子窗审计**（[gate-v2-seed-window-audit.json](gate-v2-seed-window-audit.json)，结论 clean）：窗口 [31024,65536) 与全部已知种子段零交集——训练帧种子标签 161,300 个（min 1,600,000 / max 46,000,167）窗内 0；582 份已记录评估输出（games.jsonl/games.json，全部工作树 artifacts + pvz-logs）窗内 0；478 份含种子列表的清单/面板/配方窗内 0；human_demo_tasks.json 示范种子窗内 0；task67/70/73 产物段窗内 0。核对办法与来源 SHA256 已写入审计 JSON。
- **部署交主控**：账本随首次 v2 分配惰性创建；升级后旧任务（无 seed_block）会自然分配新块并在新目录重评一次，旧目录不受影响。

## 问题

- v1 的 `_evaluate` 例外路径写 `version=GATE_VERSION` 的 error.json——v2 下 error 决策的 policy 字段自动为 v2 形状，与 1024 行的版本一致性检查自洽，无需额外改动。
- 种子窗审计依赖“评估输出与清单已落盘”的完整性；若存在未落盘的历史游戏消耗了窗内种子，无法从磁盘证据发现。已扫描全部已知落盘位置（582 份对局输出 + 478 份清单），无任何命中。

## 主控追加（2026-10-25，task73b：门禁 v2，实施）
结论收到。主控定：
- 已在组边界手动启用 round 11（新种子复测 +6.6pp [+1.6, +11.7]），curriculum.log 记了理由。
- 门禁改为 **v2：每轮换新种子 + 1536 对**，非劣界仍为 −3pp，区内 McNemar 否决保留。不做两段式；成本只有几分钟。
- **评估种子窗口**：注册 [31024, 65536) 作为门禁专用池。先机械核对它与所有已知种子段不相交：训练帧标签、各面板与清单、示范种子、task67/70/73 用过的段。核对办法写清楚。
- **消耗登记**：每轮从池里按顺序取未用种子，登记到 git 外的账本（例如 `~/pvz-logs/gate_seed_ledger.json`）；契约里写明本轮的种子区间。池耗尽时直接报错停门禁，不要回绕重用。
- **现役也在新种子上重跑**（配对）。不缓存旧契约的现役成绩：现役在选它的那批种子上成绩偏高，这正是赢家诅咒的来源。
- 实现在 `promotion_gate.py` / `pvz_promotion`，`GATE_VERSION` 加 1。旧契约的证书照旧可读。测试覆盖：种子不重用、池耗尽报错、现役重跑、旧证书兼容。
- 结果写本文件，push origin gatepower。部署由主控做。

## 主控追加（2026-10-25，task74b：L39 对空卡组）
L39 在 [9,8,10,16,23,17,19,2,5,31] 下输给气球（21 只破防，第 16–29 波）。用同样的生产老师参数，各 16 个种子，从 30768 起取。试 3–4 套带对空的卡组：
- 把磁力菇换成仙人掌；
- 把坚果类换成仙人掌；
- 仙人掌加三叶草；
- 香蒲：卡组里同时要有睡莲，并在 profile 里声明 `owned_upgrade_plants`。

报胜率与破防僵尸。每套的理由写清楚，不手写打法。结果追加到 `MEMORY/docs/task74-cold-start-sweep.md`，push origin gatepower。

## 主控追加（2026-10-25 19:30，task73c：门禁面板要覆盖课程在练的区）
门禁 v2 第一次正式运行（第 13 轮候选 vs 现役 r11，1536 对）：**真实退步**，−3.3pp [−5.4, −1.2]。
- heldout 区 −6.0pp：day_l7 holdout 145 vs 166；
- pool 区 −5.7pp：L21 167 vs 178，L26 95 vs 106；
- day、night 持平。
门禁的判断本身没问题，但暴露了两个结构性缺口：
1. **面板只有 day / heldout / night / pool 四区**。课程现在大量在练雾夜（fog_user、示范起点格）和屋顶 L41，门禁看不到这些区的进步，只看得到旧区被挤占。目标是通关冒险模式，门禁必须量到新区。
2. **day_l9 holdout 两边都是 0–2/192**，处在地板上，不提供任何信息，白占 384 局。
3. 另外：这次评估跑了 **2547 秒**（42 分钟，worker=1），task73 估算的 16 进程是 3.4 分钟。

要做的（仍在 gatepower 工作树，GATE_VERSION 升到 3，v1/v2 证书照旧可验证）：
1. **新面板 v3**：
   - 加入 **fog** 区：fog_user L31 snow 卡组，human 观测，倍率取 qc 当前该格的倍率档或 1.0，由你从 curriculum_state 读出后固定写进面板；
   - 加入 **roof** 区：L41 r7；
   - day_l9 holdout 换成一个不在地板上的 holdout 任务，例如同族另一关；先用现役学生跑 64 局确认胜率在 15%–85% 之间再选。
   - 每区仍是 2 个任务 × 192 种子，种子照旧从账本领取。
2. **判定规则不变**：总体非劣界 −3pp，任一区显著变差就否决。新区同样适用。
3. **评估并行**：评估 worker 数改为由 qc 的内存准入分配，评估期间与采集共享上限，目标是 1536 对 ≤ 10 分钟；在 curriculum.log 里记实际耗时。
4. **测试**：v3 面板的区覆盖、种子领取、旧证书兼容、并行评估结果与串行逐局一致。
5. 结果写本文件，`git push origin gatepower`。部署随 deploy14 一起，由主控做。同时最多 2 个游戏进程（验证用）。

## 结果（task73c：门禁 v3 已实施，待主控随 deploy14 部署）

- **holdout 替换实测（scripts/task73c_holdout_probe.py，两个账本块 [32560,33072)+[33072,33200)）**：
  现役 r11 学生（round_000011/model.pt，deploy4 引擎 + v5 value + arms=fast + β=0，与门禁契约同参）×
  64 新种子/候选，把注册 heldout 组 10 个候选全部测完（day_l7h1 用 task73 复测 28/32 佐证）：
  **整组两极化，无一落在 15–85%**——地板 0–2%（l8h1 0/64、l9h3 1/64、l7h3 0/64），
  天花板 87.5–100%（l9h2 56/64、l10h2 57/64、l10h1 58/64、l8h2 62/64、l10h3 62/64、l8h3 64/64、l7h2 64/64）。
  选 **day_l9_eval_d7p_holdout2（56/64=87.5%，95% Wilson CI ≈[77%,94%]，上沿跨 85%）**：
  离带最近、不是地板、保留对退步的完整否决力。数据 artifacts/task73c/probe*/probe-summary.json（git 外）。
- **面板 v3（GATE_VERSION=3，panel v3-task73c）**：10 任务 × 192 种子 = **1920 对 / 3840 局**，
  六区 day(2)/heldout(2)/night(2)/pool(2)/**fog(1)/roof(1)**。fog=fog_user_l31_snow（雪夜卡组
  [9,8,10,16,23,17,19,2,3,5]，fog_observation human，倍率 1.0——curriculum_state 读出的当前档，
  固定写死）；roof=roof_l41_train_r7（[0,1,2,3,32,35,33]，1.0）。两任务逐字段复制自课程清单
  （cq 树 task68/task46），wave_cap=None、倍率 1.0 → 沿用 v2 式"完整原始难度"冻结检查。
  判定规则未动：总体非劣 −3pp + 任一区 McNemar 否决，新区同样适用。种子照旧账本按池
  [31024,65536) 顺序领 1920 连续种子（每区每任务 192），耗尽报错不回绕。
  **注册面板新文档** artifacts/task73c/panel_v3_source.json（git 外，SHA256
  bb0a9129…d8e2f636；unified_v2.json 一字未动，旧证书 SHA 不受影响）。v1/v2 证书按记录版本重放照旧可验证。
- **评估并行（promotion_gate.py --eval-units N）**：门禁内部以线程池并发跑至多 N 个 search_play
  子进程（每个 --workers 1），16 个 (side,task) 单元变成 20 个，单元级断点续跑/命令校验/独立性
  校验全保留；结局只是 (model,task,seed) 的函数，单元顺序与并发数不影响任何配对数据——
  **并行=串行逐局一致**有测试（同契约、同 paired、同 episode sha256，仅时间戳不同）；
  eval_units 刻意不进契约。curriculum_queue._evaluate 用与采集同一条内存准入公式分配
  units（MemAvailable−6GiB 预留）/单 worker 实测 RSS，并扣除前台在跑组宽共享 --workers 上限；
  评估期间新开的组按 foreground_limit() 收缩（预留 units 个名额）。curriculum.log 记
  eval_units 与实际耗时。按当前内存（19 GiB 可用、0.88 GiB/worker）units≈14，
  3840 局 × 1.06 w·s / 14 ≈ **4.9 分钟 ≤ 10 分钟目标**。
- **测试**：test_promotion.py + test_promotion_evaluation.py + test_curriculum_pipeline.py +
  test_curriculum_queue.py 共 **80 通过 + 1 跳过 + 4 子测试**（ml venv；系统 python 缺 torch 的
  2 个 CurriculumRulesTest 失败为环境预存，与本次无关）。新增覆盖：v3 六区分块/区覆盖/种子领取、
  v3 源面板组名校验与原始难度校验、v1+v2 全流程在 GATE_VERSION=3 下仍可签发与验证、
  并行(7 单元)=串行逐局一致、并发上限构造参数、账本续跑不重复分配、耗尽拒绝回绕。
- **部署交主控（deploy14）**：启动课程控制器时必须显式 `--promotion-panel
  /home/newbiexvwu/PvZ-search/.claude/worktrees/gatepower/artifacts/task73c/panel_v3_source.json`
  （默认值仍指 unified_v2，会在启动校验处报"missing/wrong group for gate v3"——这是有意的，
  防止误用旧面板跑 v3）。评估并行 units 由内存准入自动定，无需新参数；手动直跑 promotion_gate.py
  默认 --eval-units 1（串行）。

## 问题（task73c）

- **"每区仍是 2 个任务 × 192 种子"没有照字面执行**：fog/roof 区各只有 1 个任务。注册清单里
  fog/roof 各只练一格（fog_user_v2 未入课程、roof_l41_eval_r7 与 train_r7 同关同卡组仅种子段
  不同——门禁本来就覆盖种子，重复任务只会让 384 局变成重复信息）。若主控仍要每区 2 任务，
  fog 区第二任务需要先注册一个新雾关（同 fog_user 卡组不同关），roof 区同理，这超出本次范围。
- **"day_l9 holdout 换成 15–85% 的 holdout 任务"无解**：注册 heldout 组 10 个候选对现役学生
  全部两极化（0–2% 或 87.5–100%），选了离带最近的 l9h2（87.5%）。副作用：它测退步灵敏、
  测进步会饱和（现役已 87.5%）；且候选若整体变强，该任务会失去区分度，届时需再换。
- 种子账本现分配 [31024,33200)（v2 首轮 1536 + 探针 512+128）。池剩 32336 颗 = 16 轮 v3。
