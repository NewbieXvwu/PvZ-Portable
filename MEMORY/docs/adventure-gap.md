# 冒险模式50关通关缺口（task55）

qc冻结时点：**2026-10-24 05:03:35 +08:00**；研究报告另按来源日期与SHA取证。当前38个qc单元覆盖26关。以下“已过”只是现役dw4在原版任务的已完成独立面板中观察胜率≥95%、样本≥32的盘点口径，**不是已批准的新门禁，也不证明连续冒险通关或总体胜率≥95%**。据此6关已过、26关在练/诊断、5关冷启动全败、13关未接入；“在练”包括已有研究诊断，不等于qc在采。

## 口径与证据边界

- 最新完整学生面板是task36 A（dw5i_A_s1，10-24 00:41完成，34任务3264局），**尚未上线**；现役为dw4i_s1。表中按“最新 / 现役”分列，状态只用现役，不逐关挑模型最优。A逐任务games.jsonl的3264局已检查独立seed、fast臂、searched=0及胜局数，哈希在JSON中。[SA0] [SA] [S27]
- 表内胜率用胜局/样本量表示；数值为1.0倍、同卡组族的原版难度。老师用beta1真实搜索；旧模型、不同卡组、诊断4局单列。`诊`是task46固定4seed研究，L42–45与原16局面板重叠，**不能相加样本量**；`面`是保留的正式冷启动面板。L46的1/4确有胜局，不当32seed面板或课程准入。[TR2] [TR]
- qc倍率是冻结控制器state的mult_index，对应`[1,1.25,1.5,1.75,2,2.5,3]`；不是面板覆盖的全部清单。JSON逐单元保留窗口局数与最近已完成组。qc实际收集beta0：搜索用于标签，实际执行学生；**日志wins不能叫老师胜率**。窗口可能含旧模型/历史初始化，也不替代独立面板。[Q] [QL] [QP]
- L41在04:11进入qc状态，冻结时窗口0局；雾区尚无准入cell。pool虽有8个cell与历史bootstrap窗口，最近完整第2轮768局只覆盖白天/夜间，不能说新流水线已补进泳池/屋顶训练数据。worker上限16不代表实开16，当前内存约束只容2–4；本任务未操作队列。[Q] [QL] [PLAN] [D53]
- 空白`—`表示缺少该口径证据，不填0。L2没有独立现代面板；其引擎允许表/波数与L1相近，不能复制L1胜率或据此宣布已过。普通关瓶颈列是**下一步排查入口**，证据与确定性在JSON及下文中说明。

## 50关总表

关号同时列冒险章节与引擎level。类型由`LawnApp::IsAdventureMode`及各`Is*Level`、`Board::PickBackground/HasConveyorBeltSeedBank`解析，再以单个既有deploy3引擎逐关reset+1tick核对；50关均成功。此项没有策略对局，不能证明特殊交互和终局链完整。L35是夜草；L40雾池背景但`StageHasFog=false`；L50为夜间Boss屋顶。代码与二进制分别留SHA，不把较新的源码和既有二进制说成同一构建。[E]

僵尸列是引擎卡组页的**允许表**，不是整局出现全集。JSON另存固定核对seed的已生成波表；还会有脚本增援：旗帜1、舞王8召唤伴舞9、水路普通变体10、巨人23抛小鬼24、L45空投蹦极20、L50 Boss25。尤其L45允许表没列20不等于无蹦极；L35真正罐内集合须从ScaryPotter生成而非本列读。两种表与脚本行为不能混称。[E] [D46]

ID字典：0普通，1旗帜，2路障，3撑杆，4铁桶，5报纸，6铁门，7橄榄球，8舞王，9伴舞，10鸭圈，11潜水，12冰车，13雪橇，14海豚，15小丑，16气球，17矿工，18跳跳，19雪人，20蹦极，21梯子，22投石车，23巨人，24小鬼，25僵王；定义来自引擎`ZombieDefinition`，19在允许表中不保证必刷。

卡组族：d7=`[0,1,2,3,4,5,7]`；n10=`[0,1,2,3,4,5,7,8,9,11]`；p9k=`[0,1,2,3,4,5,7,16,19]`；f8=`[0,9,2,3,4,5,16,43]`（human雾观测，43香蒲声明所有权且需先有16睡莲）；r7=`[0,1,2,3,32,35,33]`（有卷心菜32、花盆33）。d7q/n10q是按seed扰动族，不是一副固定牌，确切规则保存在清单。orig/n10_orig保留原生传送带初始语义，空deck不保证空手牌：引擎可脚本预置。†表示L30/40/45现行登记任务覆盖初始手牌后仍走传送带，**不能冒充原版开局族**。未注册的特殊关探针牌不是生产任务族。任务规则没改，旧档案仍按其保存deck回放。

| 关 | 场景/类型 | 僵尸允许表（ID） | 卡组族 | qc当前倍率 | 最新学生 / 现役dw4 | 最近老师 | 状态 / 排查入口 |
|---|---|---|---|---|---|---|---|
| 1-1（1） | 白天/普通 | 0 | d7 | d7×2 | 125/128 [SA] / 118/128 [S27] | 64/64 [D] 旧模型 | 在练 / 工程 |
| 1-2（2） | 白天/普通 | 0 | d7 | 不在qc | — / — | — | 未接入 / 工程 |
| 1-3（3） | 白天/普通 | 0,2 | d7 | d7×2 | 120/128 [SA] / 116/128 [S27] | 64/64 [D] 旧模型 | 在练 / 工程 |
| 1-4（4） | 白天/普通 | 0,2 | d7 | d7×2 | 116/128 [SA] / 122/128 [S27] | 64/64 [D] 旧模型 | 已过 |
| 1-5（5） | 白天/保龄球 | 0,2,3,4,5 | 未注册 | 不在qc | — / — | — | 未接入 |
| 1-6（6） | 白天/普通 | 0,2,3 | d7 | d7×2 | 120/128 [SA] / 122/128 [S27] | 64/64 [D] 旧模型 | 已过 |
| 1-7（7） | 白天/普通 | 0,2,3 | d7 | d7×1.75；d7q×1 | 101/128 [SA] / 116/128 [S27] | 16/16 [T52] | 在练 / 工程 |
| 1-8（8） | 白天/普通 | 0,2,4 | d7 | d7×1.5；d7q×1 | 112/128 [SA] / 116/128 [S27] | 64/64 [D] 旧模型 | 在练 / 工程 |
| 1-9（9） | 白天/普通 | 0,2,3,4 | d7 | d7×1.5；d7q×1 | 98/128 [SA] / 112/128 [S27] | 64/64 [D] 旧模型 | 在练 / 工程 |
| 1-10（10） | 白天/传送带 | 0,2,3,4 | orig | orig×2；d7q×1.5 | 127/128 [SA] / 125/128 [S27] | 64/64 [D] 旧模型 | 已过 |
| 2-1（11） | 夜草/普通 | 0,5 | n10 | n10×1.25；n10q×1 | 126/128 [SA] / 128/128 [S27] | 同族缺证；13/16 [N] n9 | 已过 |
| 2-2（12） | 夜草/普通 | 0,2,4,5 | n10 | n10×1；n10q×1 | 86/128 [SA] / 104/128 [S27] | 同族缺证；11/16 [N] n9 | 在练 / 工程 |
| 2-3（13） | 夜草/普通 | 0,2,6 | n10 | n10×1；n10q×1.25 | 128/128 [SA] / 126/128 [S27] | 同族缺证；14/16 [N] n9 | 已过 |
| 2-4（14） | 夜草/普通 | 0,2,3,6 | n10 | n10×1；n10q×1 | 62/128 [SA] / 87/128 [S27] | 32/32 [T50] | 在练 / 工程 |
| 2-5（15） | 夜草/打地鼠 | 0,2,4 | 未注册 | 不在qc | — / — | — | 未接入 |
| 2-6（16） | 夜草/普通 | 0,2,7 | n10 | n10×1.25；n10q×1.25 | 128/128 [SA] / 128/128 [S27] | 同族缺证；11/16 [N] n9 | 已过 |
| 2-7（17） | 夜草/普通 | 0,2,6,7 | n10 | n10×1；n10q×1 | 64/128 [SA] / 65/128 [S27] | 31/32 [TV4] | 在练 / 工程 |
| 2-8（18） | 夜草/普通 | 0,2,8 | n10 | n10×1.25；n10q×1 | 121/128 [SA] / 121/128 [S27] | — | 在练 / 工程 |
| 2-9（19） | 夜草/普通 | 0,2,6,8 | n10 | n10×1；n10q×1 | 40/128 [SA] / 59/128 [S27] | — | 在练 / 工程 |
| 2-10（20） | 夜草/传送带 | 0,2,6,7,8 | n10_orig | n10_orig×1 | 0/128 [SA] / 0/128 [S27] | 29/32 [TV4] | 在练 |
| 3-1（21） | 泳池/普通 | 0,2 | p9k | p9k×1 | 42/64 [SA] / 1/64 [S27] | 17/32 [T50] | 在练 / 冷启动无数据 |
| 3-2（22） | 泳池/普通 | 0,2,4,5,7 | p9k | p9k×1 | 33/64 [SA] / 1/64 [S27] | 5/16 [P] 旧模型 | 在练 / 冷启动无数据 |
| 3-3（23） | 泳池/普通 | 0,2,11 | p9k | p9k×1 | 27/64 [SA] / 0/64 [S27] | 4/16 [P] 旧模型 | 在练 / 冷启动无数据 |
| 3-4（24） | 泳池/普通 | 0,2,3,4,5,11 | p9k | p9k×1 | 12/64 [SA] / 0/64 [S27] | 3/32 [TV4] | 在练 / 叶值压平 |
| 3-5（25） | 泳池/小僵尸/传送带 | 0,2,7,11 | 未注册 | 不在qc | — / — | — | 未接入 |
| 3-6（26） | 泳池/普通 | 0,2,12,13 | p9k | p9k×1 | 12/64 [SA] / 0/64 [S27] | 2/16 [P] 旧模型 | 在练 / 叶值压平 |
| 3-7（27） | 泳池/普通 | 0,2,4,11,12,13 | p9k | p9k×1 | — / — | — | 在练 / 冷启动无数据 |
| 3-8（28） | 泳池/普通 | 0,2,14 | p9k | p9k×1 | — / — | — | 在练 / 冷启动无数据 |
| 3-9（29） | 泳池/普通 | 0,2,3,4,12,13,14 | p9k | p9k×1 | — / — | — | 在练 / 冷启动无数据 |
| 3-10（30） | 泳池/传送带 | 0,2,4,11,12,13,14 | p9k† | 不在qc | — / — | — | 未接入 |
| 4-1（31） | 雾池/普通 | 0,2,15 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [T52] | 冷启动全败 / 叶值压平 |
| 4-2（32） | 雾池/普通 | 0,2,7,15 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [TF] | 冷启动全败 / 叶值压平 |
| 4-3（33） | 雾池/普通 | 0,2,16 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [TF] | 冷启动全败 / 叶值压平 |
| 4-4（34） | 雾池/普通 | 0,2,14,16 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [TF] | 冷启动全败 / 叶值压平 |
| 4-5（35） | 夜草/砸罐子 | 0,2 | 未注册 | 不在qc | — / — | — | 未接入 |
| 4-6（36） | 雾池/普通 | 0,2,17 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-7（37） | 雾池/普通 | 0,2,4,15,17 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-8（38） | 雾池/普通 | 0,2,18 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-9（39） | 雾池/普通 | 0,2,4,16,18 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-10（40） | 雾池（无雾）/暴风雨/传送带 | 0,2,4,15,16,17,18,19 | f8† | 不在qc | — / — | — | 未接入 |
| 5-1（41） | 屋顶/普通 | 0,2,19,20 | r7 | r7×1 | 8/16 [TR] / 8/16 [TR] | 13/16 [T52] | 在练 / 冷启动无数据 |
| 5-2（42） | 屋顶/普通 | 0,2,3,4,19,20 | r7 | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面4/16 [TR] | 在练 / 冷启动无数据 |
| 5-3（43） | 屋顶/普通 | 0,2,19,21 | r7 | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面2/16 [TR] | 在练 / 冷启动无数据 |
| 5-4（44） | 屋顶/普通 | 0,2,7,18,19,21 | r7 | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面0/16 [TR] | 冷启动全败 / 冷启动无数据 |
| 5-5（45） | 屋顶/蹦极/传送带 | 0,2,4,19,21 | r7† | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面2/16 [TR] | 未接入 |
| 5-6（46） | 屋顶/普通 | 0,2,19,22 | r7 | 不在qc | — / — | 1/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-7（47） | 屋顶/普通 | 0,2,19,20,21,22 | r7 | 不在qc | — / — | 0/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-8（48） | 屋顶/普通 | 0,2,19,23 | r7 | 不在qc | — / — | 0/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-9（49） | 屋顶/普通 | 0,2,4,15,19,20,21,22,23 | r7 | 不在qc | — / — | 0/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-10（50） | Boss屋顶/僵王 | 0,2,4,15,19,20,21,22,23 | 未注册 | 不在qc | — / — | — | 未接入 |

机制/瓶颈归因与推进顺序仍在盘点；本阶段整体incomplete。

## 来源索引

日期与模型/任务口径须连同表内ID一起读；完整绝对路径、SHA、每条指标日期及qc快照元数据在[总表JSON](adventure-gap.json)。旧老师只作历史参照，不与现役学生构成严格配对比较。

| 来源ID | 日期（北京时间；旧表为归档日期） | 证据范围 |
|---|---|---|
| [E] | 2026-10-24 | one process; 50 reset + one wait tick; not complete games |
| [Q] | 2026-10-24T05:03:35.490319+08:00 | frozen controller membership/window state |
| [QL] | 2026-10-24T05:03:35.490319+08:00 | beta0 collection execution, not teacher strength |
| [QP] | 2026-10-24T05:03:35.490319+08:00 | 414 multiplier variants; only state.cells are active |
| [S27] | 2026-10-23 | completed task27 v2; dw4i production reference, dw5i comparison |
| [SA0] | 2026-10-24 00:41:40+0800 | latest completed candidate A; not deployed |
| [SA] | 2026-10-24T00:41:40+0800 | 34 paired tasks, 3264 games; raw files rechecked below |
| [TF] | 2026-10-23 | cold-start human f8; separate fast and beta1 fast_search |
| [TR] | 2026-10-23 | r7 original difficulty; L45 has overridden initial hand |
| [TR2] | 2026-10-24 | fixed four-seed teacher diagnostic, not a full evaluation panel; L42-45 overlap old panel seeds |
| [TRh] | 2026-10-23 | diagnostic multiplier .5; not current qc nor original completion |
| [TV4] | 2026-10-23 | beta1 v4 paired evaluation; new_wins is v4 |
| [T50] | 2026-10-24 | fresh deployed teacher gate; training-task seeds |
| [T52] | 2026-10-24 | 144 complete games; arm A is unaugmented beta1 teacher |
| [D] | 2026-10-21 | task17 table archived 2026-10-21 00:51+08; old v32 teacher; task20 original L10 |
| [N] | 2026-10-22 | old identity n9 teacher; NOT current n10 |
| [P] | 2026-10-22 | old identity p9k teacher; no modern same-model comparison |
| [D43] | 2026-10-24 | first v5 rejected; round2 teacher-continuation labels in progress |
| [D45] | 2026-10-24 | complete64-root terminal audit; local failure-delay improvements, no wins |
| [D46] | 2026-10-24 | independent phase2 roof diagnosis, not raw LLM conclusions |
| [D46c] | 2026-10-24 | dw4 frozen 48816 archives; no pool/fog/roof archives |
| [D46b] | 2026-10-24 | only four complete roots; not the full17-root panel |
| [D49] | 2026-10-24 | candidate promotion quality gate; two controlled demonstrations |
| [D53] | 2026-10-24 | screen47 roots complete; independent16-future confirmation7 actions; training/paired evaluation not yet complete |
| [D48] | 2026-10-24 | human demonstration tools; ten L31-34 games planned, recordings not evidenced here |
| [PLAN] | 2026-10-24 | controller timeline through04:15; qc only packed before task49 pipeline deployment |
| [R0] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R1] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R2] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R3] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R4] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R5] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |

[E]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/engine-inventory.json
[Q]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/curriculum_state.json
[QL]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/curriculum_log.log
[QP]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/curriculum_panel.json
[S27]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task27-results.json
[SA0]: /home/newbiexvwu/PvZ-search/.claude/worktrees/dw5/artifacts/task36/complete_A.json
[SA]: /home/newbiexvwu/PvZ-search/.claude/worktrees/dw5/artifacts/task36/paired_dw4i_vs_A.json
[TF]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task45-fog-results.json
[TR]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task46-roof-cold-start.json
[TR2]: /home/newbiexvwu/PvZ-search/.claude/worktrees/roof/MEMORY/docs/task46-roof-timelines.json
[TRh]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task46-roof-half-results.json
[TV4]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/value-widen-paired.json
[T50]: /home/newbiexvwu/PvZ-search/.claude/worktrees/zones/artifacts/task50-a/teacher32-v4/comparison.json
[T52]: /home/newbiexvwu/PvZ-search/.claude/worktrees/llmcand/artifacts/task52/study-v1/summary.json
[D]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/day-panel.md
[N]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/night-panel.md
[P]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/pool-panel.md
[D43]: /home/newbiexvwu/PvZ-search/.claude/worktrees/v5/MEMORY/docs/task43-value-v5-pool.md
[D45]: /home/newbiexvwu/PvZ-search/.claude/worktrees/fog/MEMORY/docs/task45-fog-terminal-diagnosis.md
[D46]: /home/newbiexvwu/PvZ-search/.claude/worktrees/roof/MEMORY/docs/teacher-diag-roof-review.md
[D46c]: /home/newbiexvwu/PvZ-search/.claude/worktrees/roof/MEMORY/docs/task46-roof-training-coverage.json
[D46b]: /home/newbiexvwu/PvZ-search/.claude/worktrees/roof/MEMORY/docs/task46-roof-terminal-root-smoke.json
[D49]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task49-A-evidence.json
[D53]: /home/newbiexvwu/PvZ-search/.claude/worktrees/forks/MEMORY/docs/task53-critical-forks.md
[D48]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task48-human-demo.md
[PLAN]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/student-bottleneck-plan.md
[R0]: /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json
[R1]: /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json
[R2]: /home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json
[R3]: /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/task45/fog_panel.json
[R4]: /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/task46/roof_panel.json
[R5]: /home/newbiexvwu/PvZ-search/.claude/worktrees/conveyor/artifacts/stage4/day_panel.json
