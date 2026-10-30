# 冒险模式50关通关缺口（task55）

qc冻结时点：**2026-10-24 05:03:35 +08:00**；研究报告另按来源日期与SHA取证。当前38个qc单元覆盖26关。以下“已过”只是现役dw4在原版任务的已完成独立面板中观察胜率≥95%、样本≥32的盘点口径，**不是已批准的新门禁，也不证明连续冒险通关或总体胜率≥95%**。据此6关已过、26关在练/诊断、5关冷启动全败、13关未接入；“在练”包括已有研究诊断，不等于qc在采。

## 口径与证据边界

- 最新完整学生面板是task36 A（dw5i_A_s1，10-24 00:41完成，34任务3264局），**尚未上线**；现役为dw4i_s1。表中按“最新 / 现役”分列，状态只用现役，不逐关挑模型最优。A逐任务games.jsonl的3264局已检查独立seed、fast臂、searched=0及胜局数，哈希在JSON中。[SA0] [SA] [S27]
- 表内胜率用胜局/样本量表示；数值为1.0倍、同卡组族的原版难度。老师用beta1真实搜索；旧模型、不同卡组、诊断4局单列。`诊`是小样本诊断：D30d为L24/L26各4局叶值探针，TR2为task46固定4seed研究；L42–45与原16局面板重叠，**不能相加样本量**；`面`是保留的较早正式面板（可能旧模型）。L46的1/4确有胜局，不当32seed面板或课程准入。[TR2] [TR] [D30] [D30d]
- T50的最近老师率来自训练任务门禁种子，TV4/学生面板来自各自评估任务；表内师生数值不是同种子配对，不能把其差值直接当作某种修法的因果收益。模型、任务ID和种子范围以JSON为准。[T50] [TV4]
- qc倍率是冻结控制器state的mult_index，对应`[1,1.25,1.5,1.75,2,2.5,3]`；不是面板覆盖的全部清单。JSON逐单元保留窗口局数与最近已完成组。qc实际收集beta0：搜索用于标签，实际执行学生；**日志wins不能叫老师胜率**。窗口可能含旧模型/历史初始化，也不替代独立面板。[Q] [QL] [QP]
- L41在04:11进入qc状态，冻结时窗口0局；雾区尚无准入cell。pool虽有8个cell与历史bootstrap窗口，最近完整第2轮768局只覆盖白天/夜间，不能说新流水线已补进泳池/屋顶训练数据。worker上限16不代表实开16，当前内存约束只容2–4；本任务未操作队列。[Q] [QL] [PLAN] [D53]
- 本表“原版难度”指既有二周目任务/profile与1.0倍率，不等于新档一周目。原生传送带另指保留脚本初始手牌与牌流；屋顶清理车等profile条件按档案保留，不补齐成另一套游戏。
- 空白`—`表示缺少该口径证据，不填0。L2没有独立现代面板；其引擎允许表/波数与L1相近，不能复制L1胜率或据此宣布已过。普通关瓶颈列是**下一步排查入口**，证据与确定性在JSON及下文中说明。

## 50关总表

关号同时列冒险章节与引擎level。类型由`LawnApp::IsAdventureMode`及各`Is*Level`、`Board::PickBackground/HasConveyorBeltSeedBank`解析，再以单个既有deploy3引擎逐关reset+1tick核对；50关均成功。此项没有策略对局，不能证明特殊交互和终局链完整。L35是夜草；L40雾池背景但`StageHasFog=false`；L50为夜间Boss屋顶。代码与二进制分别留SHA，不把较新的源码和既有二进制说成同一构建。[E]

僵尸列主体是引擎卡组页的**允许表**，L35/45/50另列脚本补充，不是整局出现全集。JSON另存固定核对seed的已生成波表；还会有脚本增援：旗帜1、舞王8召唤伴舞9、水路普通变体10、巨人23抛小鬼24、L45空投蹦极20、L50 Boss25。尤其L45允许表没列20不等于无蹦极；L35三阶段罐内类型0/4、0/4/7、0/4/8/15来自ScaryPotterPopulate，而非普通允许表；舞王仍可召伴舞9。两种表与脚本行为不能混称。[E] [D46]

ID字典：0普通，1旗帜，2路障，3撑杆，4铁桶，5报纸，6铁门，7橄榄球，8舞王，9伴舞，10鸭圈，11潜水，12冰车，13雪橇，14海豚，15小丑，16气球，17矿工，18跳跳，19雪人，20蹦极，21梯子，22投石车，23巨人，24小鬼，25僵王；定义来自引擎`ZombieDefinition`，19在允许表中不保证必刷。

卡组族：d7=`[0,1,2,3,4,5,7]`；n10=`[0,1,2,3,4,5,7,8,9,11]`；p9k=`[0,1,2,3,4,5,7,16,19]`；f8=`[0,9,2,3,4,5,16,43]`（human雾观测，43香蒲声明所有权且需先有16睡莲）；r7=`[0,1,2,3,32,35,33]`（有卷心菜32、花盆33）。d7q/n10q是按seed扰动族，不是一副固定牌，确切规则保存在清单。orig/n10_orig保留原生传送带初始语义，空deck不保证空手牌：引擎可脚本预置。†表示L30/40/45现行登记任务覆盖初始手牌后仍走传送带，**不能冒充原版开局族**。未注册的特殊关探针牌不是生产任务族。任务规则没改，旧档案仍按其保存deck回放。

| 关 | 场景/类型 | 僵尸允许表/脚本补充（ID） | 卡组族 | qc当前倍率 | 最新学生 / 现役dw4 | 最近老师 | 状态 / 排查入口 |
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
| 3-4（24） | 泳池/普通 | 0,2,3,4,5,11 | p9k | p9k×1 | 12/64 [SA] / 0/64 [S27] | 0/4 [D30d]诊；面3/32 [TV4] | 在练 / 叶值压平 |
| 3-5（25） | 泳池/小僵尸/传送带 | 0,2,7,11 | 未注册 | 不在qc | — / — | — | 未接入 |
| 3-6（26） | 泳池/普通 | 0,2,12,13 | p9k | p9k×1 | 12/64 [SA] / 0/64 [S27] | 0/4 [D30d]诊；面2/16 [P] | 在练 / 叶值压平 |
| 3-7（27） | 泳池/普通 | 0,2,4,11,12,13 | p9k | p9k×1 | — / — | — | 在练 / 冷启动无数据 |
| 3-8（28） | 泳池/普通 | 0,2,14 | p9k | p9k×1 | — / — | — | 在练 / 冷启动无数据 |
| 3-9（29） | 泳池/普通 | 0,2,3,4,12,13,14 | p9k | p9k×1 | — / — | — | 在练 / 冷启动无数据 |
| 3-10（30） | 泳池/传送带 | 0,2,4,11,12,13,14 | p9k† | 不在qc | — / — | — | 未接入 |
| 4-1（31） | 雾池/普通 | 0,2,15 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [T52] | 冷启动全败 / 叶值压平 |
| 4-2（32） | 雾池/普通 | 0,2,7,15 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [TF] | 冷启动全败 / 叶值压平 |
| 4-3（33） | 雾池/普通 | 0,2,16 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [TF] | 冷启动全败 / 叶值压平 |
| 4-4（34） | 雾池/普通 | 0,2,14,16 | f8 | 不在qc | 0/16 [TF] / 0/16 [TF] | 0/16 [TF] | 冷启动全败 / 叶值压平 |
| 4-5（35） | 夜草/砸罐子 | 0,2；罐0,4,7,8,15 | 未注册 | 不在qc | — / — | — | 未接入 |
| 4-6（36） | 雾池/普通 | 0,2,17 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-7（37） | 雾池/普通 | 0,2,4,15,17 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-8（38） | 雾池/普通 | 0,2,18 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-9（39） | 雾池/普通 | 0,2,4,16,18 | f8 | 不在qc | — / — | — | 未接入 / 冷启动无数据 |
| 4-10（40） | 雾池（无雾）/暴风雨/传送带 | 0,2,4,15,16,17,18,19 | f8† | 不在qc | — / — | — | 未接入 |
| 5-1（41） | 屋顶/普通 | 0,2,19,20 | r7 | r7×1 | 8/16 [TR] / 8/16 [TR] | 13/16 [T52] | 在练 / 冷启动无数据 |
| 5-2（42） | 屋顶/普通 | 0,2,3,4,19,20 | r7 | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面4/16 [TR] | 在练 / 冷启动无数据 |
| 5-3（43） | 屋顶/普通 | 0,2,19,21 | r7 | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面2/16 [TR] | 在练 / 冷启动无数据 |
| 5-4（44） | 屋顶/普通 | 0,2,7,18,19,21 | r7 | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面0/16 [TR] | 冷启动全败 / 冷启动无数据 |
| 5-5（45） | 屋顶/蹦极/传送带 | 0,2,4,19,21；脚本20 | r7† | 不在qc | 0/16 [TR] / 0/16 [TR] | 0/4 [TR2]诊；面2/16 [TR] | 未接入 |
| 5-6（46） | 屋顶/普通 | 0,2,19,22 | r7 | 不在qc | — / — | 1/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-7（47） | 屋顶/普通 | 0,2,19,20,21,22 | r7 | 不在qc | — / — | 0/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-8（48） | 屋顶/普通 | 0,2,19,23 | r7 | 不在qc | — / — | 0/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-9（49） | 屋顶/普通 | 0,2,4,15,19,20,21,22,23 | r7 | 不在qc | — / — | 0/4 [TR2]诊 | 在练 / 冷启动无数据 |
| 5-10（50） | Boss屋顶/僵王 | 0,2,4,15,19,20,21,22,23；脚本25 | 未注册 | 不在qc | — / — | — | 未接入 |

## 特殊机制：现有协议够到哪里

下表是缺口盘点，未实现任何新动作/观测。规模是相对工程量，小=复用已有协议与新任务验收，中=补可见状态/编码及回放，大=新交互链与训练输入/动作头协同；均不承诺训练能通关所需时间。reset+1tick只证明启动与一次更新；原生保龄球/Boss各一次plant的原始回执保存在[EP]。所有源码链接指本次advgap源码，二进制另存SHA。

| 类型/关 | 无头证据 | 动作协议 | 观测/任务缺口 | 规模与边界 |
|---|---|---|---|---|
| 原生传送带 L10/20/30 | L10/L20有完整老师结果；L30仅reset+1tick，不证明终局链 | plant/shovel/wait足以表达出牌、落点、等待；L20学生0/128但v4老师29/32 | 现有packets含重复牌/空槽/冷却，需保留原生出牌与槽位语义；L30当前p9k覆盖初始手牌，原生开局族尚缺 | **小**；协议盘点；学习与面板验收另计 |
| LittleTrouble小僵尸 L25 | reset+1tick通过，未跑完整关 | 已有种植可选择传送带卡与水陆格，无证据需要新动作 | 核对小僵尸的可见尺寸/碰撞及编码；生命值虽公开，当前编码不显式标识此变体；需原生传送带任务与数据 | **中**；原生牌为睡莲/坚果/豌豆/樱桃；不可套普通p9k手牌 |
| 暴风雨传送带 L40 | reset+1tick通过，未跑完整关 | 种植/等待可表达牌与格，不需要新增点选种类 | StageHasFog=false，human fog屏蔽不覆盖暴风雨黑屏/闪电；缺与可见光照一致的观测及闪电阶段输入，不能声称人类等价可见性 | **中**；原生牌含睡莲/仙人掌/豌豆/小喷菇/樱桃；当前f8覆盖初始手牌，另立原生族才可比较 |
| 保龄球 L5 | 原生reset+1tick通过；packet0,col0,row0种坚果被接受，未完成一局 | 现有plant已经能选行并在合法c0-c2放球；不必先加专用bowling动作 | 滚动植物mX/mY及弹跳轨迹未在完整plants公开；只有初始col、当前row/state不足以完整表示连续运动，学生编码需审查 | **中**；动作部分小；轨迹/编码/新族/面板合计中，按用户要求后置 |
| 打地鼠（用户排除） L15 | reset+1tick通过；当前deck覆盖原生3槽，不是原样原生reset | 不够：缺锤子点击僵尸；plant不能调用MouseDownWhackAZombie | 需原生锤子/点击目标/节奏与原生三槽语义；当前观测不是已验收的打地鼠输入 | **中**；用户明确不做；只记录缺口，不建议排期 |
| 砸罐子 L35 | reset+1tick通过且有罐；当前reset覆盖原生1槽，完整玩法未验证 | 不够：缺点罐、拾取/使用掉落植物卡；种植只从seedbank取卡 | grid_items直接公开封闭罐的zombie_type/seed_type，存在信息泄漏；需屏蔽隐藏内容并保留合法可见提示。coins只给type/x/y，缺掉落卡mUsableSeedType与使用状态；还需原生bank及点击合法动作 | **大**；新交互/可见性/编码/动作头与数据；按用户要求后置 |
| 蹦极 L45 | 原生reset+1tick通过；现有完整r7诊断/面板是初始手牌变体 | 基本够：传送带种植与铲子/等待可防守空投，没有必须点击蹦极的新动作 | 已有generic phase/target_row/target_col及植物bungee_state，须核对编码是否保留可见空投落点、被抓植物与高度阶段；原生初始语义及数据缺口未补 | **中**；原生牌花盆/大嘴花/南瓜/樱桃；r7面板不能宣称原生开局通过，按用户要求后置 |
| 僵王 L50 | 原生reset+1tick通过；首张卷心菜种植接受，尚无完整Boss终局证据 | 基础种植/等待可表达原生牌使用；未发现必须新增瞄准动作的证据，仍需完整链验证 | 通用僵尸血量/phase不充分代表Boss露头与冰火球；mBossMode/mBossHeadCounter/mBossFireBallReanimID未序列化，需从中提取人类可见状态而非盲目暴露隐藏计时。还缺传送带、终局判定及编码/数据验收 | **大**；原生6种牌没有玉米加农炮；不能把玉米炮瞄准协议缺口当L50必备，按用户要求后置 |
| 玉米加农炮（跨关机制） L非独立关 | 非独立冒险关类型；本次未做玉米炮交互探针 | 不够：需选择ready炮后指定像素落点；plant/shovel/wait均不能触发该点击链 | 需炮ID/ready/选中状态/落点的可见输入与动作头，并保留30tick误触保护语义；不影响当前无炮任务 | **中**；跨关机制；若含搜索、回放、动作头/编码验收，整体可升大；不作为L50前置 |

源码定位（对应上表，避免将猜测当作已实现能力）：

- 原生传送带：`Board.cpp:8577`、`LawnApp.cpp:1407`、`Challenge.cpp:1649`；[D] [TV4] [R2] [R5]。
- LittleTrouble小僵尸：`LawnApp.cpp:3122`、`Challenge.cpp:1781`、`Zombie.cpp:871`；[E]。
- 暴风雨传送带：`Board.cpp:8710`、`Challenge.cpp:1965`、`LawnApp.cpp:1643`；[E] [R3]。
- 保龄球：`Challenge.cpp:2453`、`Plant.cpp:2361`、`LawnApp.cpp:2063`；[E] [EP]。
- 打地鼠（用户排除）：`Challenge.cpp:1329`、`Board.cpp:8590`、`pvz_env.py:536`；[E]。
- 砸罐子：`Challenge.cpp:1346`、`Challenge.cpp:4057`、`LawnApp.cpp:2099`、`LawnApp.cpp:2133`、`Board.cpp:8590`；[E]。
- 蹦极：`Board.cpp:4800`、`Challenge.cpp:1807`、`LawnApp.cpp:2088`；[E] [TR] [TR2] [D46]。
- 僵王：`Challenge.cpp:1741`、`Zombie.h:150`、`Zombie.cpp:10169`、`LawnApp.cpp:2088`；[E] [EP]。
- 玉米加农炮（跨关机制）：`Plant.cpp:4207`、`Board.cpp:3581`、`pvz_env.py:536`；[E]。

另外，屋顶探针L41预置25盆（c0–c4），L45/L50为15盆（c0–c2），不能把“所有屋顶预置前五列”当作本次事实。L15/L35原生槽数分别3/1，现有reset会按传入deck重写；两关探针成功不等于保持原生玩法。砸罐的内容屏蔽与Boss可见状态均须同步检查full/lean/v2/原生推演和学生编码，不能只改展示JSON。

## 普通关瓶颈：逐关主类与证据

总表为每个未过普通关指定一个主类；这里按同一证据分组，**不是声称单一确定根因**。“冷启动无数据”具体指现役模型的来源覆盖不足或尚无独立面板，绝非磁盘上没有任何档案。传送带/小游戏单列，不硬塞进普通关分类。

- **L1,2,3,7,8,9,12,14,17,18,19：工程**（排查入口，非已证实模拟器错误）。现役面板有缺口；L14老师32/32而学生87/128，L17老师31/32而学生65/128；旧qc只打包未训练、流水线刚上线；A改善泳池却伤夜间/heldout，不能直接换。L2仅缺独立现代证据。 下一步：确认新纠正数据实际进入训练/质量门禁，并补同族评估；不要把缺证当输局或改引擎。[S27] [SA] [T50] [TV4] [PLAN] [D49]
- **L21,22,23,27,28,29：冷启动无数据**（现役训练来源缺口有直接证据；27-29正式面板缺证）。dw4冻结48816档案只含白天/夜草；qc历史pool窗口不等于dw4已经学到。A在21-23为42/64、33/64、27/64，说明数据/配比有潜力，但仍非稳定通关。 下一步：接task43老师终局信号与task53纠正，完成质量门禁；27-29先取得正式基线。[D46c] [SA] [Q] [D53]
- **L24,26：叶值压平**（task30直接覆盖两关的候选离散度实证，非唯一路径）。现役各0/64；v4 L24正式老师3/32。task30另对L24/L26各4局、每臂480决策做探针，v4候选std约.0007，100% gap落入.02门槛且8局0推翻、两臂均0胜。task43第一轮v5已否决，第二轮真实老师续走标签尚在采集。 下一步：保留真实root/续走分布与可排序终局标签，重训后按原三项验收；不堆同样全零学生续走。[D30] [D30d] [S27] [TV4] [P] [D43]
- **L31,32,33,34：叶值压平**（已证实低分差与局部反序；学习冷启动也是直接缺口）。human/xray双臂均全败；D诊断16局0射手0香蒲、候选gap≤0.011、执行合法且Python/native一致。E完成64根/12372筛查+512确认均无胜，末波/终tick一致率.600/.548，两个土豆雷仅确认延后失败。射手/香蒲在根中不可种，不能评价优劣。 下一步：在可支付且底座满足的早期状态补老师/人类示范起点数据；先建立胜负可分信号，不能直接调0.02或以降倍率替代诊断。[TF] [D45] [T52]
- **L36,37,38,39：冷启动无数据**（缺独立现代面板；不宣称全败）。已登记f8且有地形覆盖，但尚无本关独立结果；新增矿工/跳跳构成对策风险，未有实证证明其为当前主因。 下一步：复核对策覆盖/新族必要性后再预登记小面板；不能从31-34外推胜率。[E] [R3] [D45]
- **L41,42,43,44,46,47,48,49：冷启动无数据**（现役屋顶训练0份；具体搜索问题强弱分根）。r7有32卷心菜，笼统缺投掷已被反驳。L41老师13/16、学生8/16；L46诊断1/4胜，非结构性全败。42-49有4seed老师诊断；候选遗漏/halving/低分差已在少数根发现，完整17根审计尚未完成。 下一步：先让L41有效标签进训练；L42-49等待全候选终局归因和独立未来后补数据，不自动换卡或降倍率。[TR] [TR2] [T52] [D46c] [D46] [D46b]

卡组、对策型僵尸与雾记忆暂列次级风险：

- 现行p9k/f8/r7覆盖地形；未证实其为普通关主因。对策不足需新族对照，先查牌再归因策略。
- 潜水/冰车/海豚/小丑/气球/矿工/跳跳/梯子/投石车/巨人均需对策数据；引擎名单仅定位风险，不足以归因输局。
- human/xray皆全败、K4全败地板；记忆潜在价值未定，不列当前主因。

不能由允许名单直接推断需要换卡；先核对地形/对策覆盖，再用同族分叉或新族配对验证。task52三场地144局已完成，LLM新增提议仅8/510被搜索采用、0/749非法，96配对比较的胜负与末波净差均0，约132万token；当前不建议接生产路径。f8缺开雾/部分对策卡及根的可支付限制，也不能据此否定所有LLM对策先验。若要换卡，按AGENTS §8另开族、DESIGN登记、历史whatif仍用档案deck。[T52] [H48]

## 推进顺序建议（交主控决定）

以兑现现有可教信号和减少未知为优先，不预测会新增多少关。验收看原版同任务/同种子配对与独立面板，末波/延寿不能代替胜负。以下不修改生产配方、门槛、倍率或qc：

1. **先闭合现有数据→训练→质量门禁，保住已过关，吸收普通夜间与原生L20纠正**（L1,3,7,8,9,12,14,17,18,19,20）。L14/L17老师接近全胜而学生约半数，L20老师29/32对学生0/128；旧qc仅打包问题已修。A不是可直接全局替换的净改进。 依赖：task49A流水线与晋升门禁；task36配比/候选B排期；主控批准部署与数据配方。完成证据：逐来源确认新标签入训；候选通过现有配对质量门禁，再看各关原版独立面板是否稳定。限制：不承诺增加关数；task55不换生产模型/门槛。[T50] [TV4] [S27] [SA] [PLAN] [D49]
2. **优先兑现L41准入后的屋顶数据，作为新增区的低成本入口**（L41）。老师15/16冷启动、最近13/16，学生8/16；屋顶训练来源为0；已登记qc但冻结时尚0窗口局。 依赖：task46已完成准入；task50已部署deploy3；task49训练门禁；task53可区分纠正标签（若正式验收通过）。完成证据：获得实际完整采集/训练/配对评估证据，不把state注册当完成。限制：两组16样本非总体≥95%证明，须按既有准入/晋升程序推进。[TR] [T52] [D46c] [Q]
3. **合并现有终局标签工作：先完成task43/46B/53再决定数据配比与换值网**（L21,22,23,24,26,27,28,29,42,43,44,46,47,48,49）。A泳池126/320显示迁移机会；task43第一轮已否决，第二轮真实老师标签未验收；task46B少数根分别暴露候选遗漏、halving与小分差；不能只改某一阈值。 依赖：task43真实老师续走64seed+重训三门；task46B完整17根审计；task53独立确认/微调/192对评估/成本门禁。完成证据：看原版胜负而非单纯末波/存活时间；保留不同来源与任务族后训练。限制：task53已筛查9723标签、确认7动作，但微调/192对仍待验；15%成本基准旧完整轮与新引擎单组投影不同，不宣布预算已在现行混合轮通过。[SA] [D43] [D46b] [D53] [D53c]
4. **雾区先补可赢的合法早期状态/人类示范起点；胜负有信号后才量记忆**（L31,32,33,34,36,37,38,39）。E筛查/确认均无胜，部分延寿不足以教会稳定通关；K4与无记忆都落在0胜地板，尚不能判断记忆收益；对策牌和可支付时机需分开验证。 依赖：task45E固定基线；task48用户十局示范/verified重放与HF交付；task53纠正数据；新卡组族由主控登记。完成证据：验证示范/老师接管能产生胜负可分的合法状态，再做无记忆/历史输入同数据对照。限制：task55未取得实际十局示范；不写规则打法，不把记忆当已证实根因；task52 LLM候选无胜负收益，暂不生产接入。[TF] [D45] [D48] [H48] [T52]
5. **分批补未知关证据，并补传送带原生任务；不把传送带整体后置**（L2,25,27,28,29,30,36,37,38,39,40,46,47,48,49）。L2、27-29及雾后半缺独立面板；屋顶后半只有4seed诊断。L30/40当前为覆盖初始手牌变体，L25原生族未注册；L40还缺闪电可见性审计。 依赖：已有源清单/引擎谓词；task43/45/46的可教信号；任务族登记与主控小规模排期。完成证据：先核实deck×地形/可见性，再预登记原版同族小面板和独立保留种子。限制：本任务不采批量对局；不静默覆盖现行deck或历史档案。[E] [R2] [R3] [TR2]
6. **按用户顺序最后处理保龄球/砸罐/蹦极/僵王；打地鼠不做**（L5,35,45,50）。保龄球种植选行已有单步证明，蹦极基础动作可表达；砸罐有点击/掉落卡/信息泄漏缺口，Boss有可见状态与终局链缺证。 依赖：本报告特殊机制清单；新任务族/编码/动作协议各自验收；主控排期。完成证据：每类单独通过原生reset、合法动作/公开观测、回放与整局结果门禁后再建学习数据。限制：玉米炮瞄准若另立任务再做；原生L50没有玉米炮，不作为前置。[E] [EP] [D46]

task53截至本次来源读取已完成47根9723条筛查及12根384条独立16未来确认，7个动作通过校正；微调/192对评估仍待完成，不能写成学生已进步。task43第二轮仍在采64seed老师续走标签，尚无新v5三门通过证据。task46B只有4/17根442条完整标签的已审计子集。task45E则已完整交付、筛查与确认全无胜局。人类示范工具已交付，十局计划可读；本次未取得真实示范对局/HF证据，不宣称用户已打完。[D53] [D43] [D46b] [D45] [H48]

## 交付复核

task55盘点已完成；“已过”的观察口径与后续实验依赖不变。本次续做没有启动游戏或训练进程。只读验收覆盖50行/38个qc单元的倍率、35个未过普通关的完整分类、9项机制的字段与源码定位、所有指标对原始来源的数值核对、38份冻结来源SHA、task36 A全部3264局与本地链接。证据快照在本工作树忽略的`artifacts/task55/`，不随git推送；git只保存报告、结论JSON和复核工具。盘点完成不表示50关已通关，也不替代主控的后续排期。

本机原始证据复核命令：

```bash
python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/scripts/verify_adventure_gap.py
```

## 来源索引

日期与模型/任务口径须连同表内ID一起读；完整绝对路径、SHA、每条指标日期及qc快照元数据在[总表JSON](adventure-gap.json)。旧老师只作历史参照，不与现役学生构成严格配对比较。研究来源分别冻结于JSON的captured_at，qc固定为05:03:35快照；之后发生的进展不自动混入本表。

| 来源ID | 日期（北京时间；旧表为归档日期） | 证据范围 |
|---|---|---|
| [E] | 2026-10-24 | one process; 50 reset + one wait tick; not complete games |
| [Q] | 2026-10-24T05:03:35.490319+08:00 | frozen controller membership/window state |
| [QL] | 2026-10-24T05:03:35.490319+08:00 | beta0 collection execution, not teacher strength |
| [QP] | 2026-10-24T05:03:35.490319+08:00 | 414 multiplier variants; only state.cells are active |
| [RUN] | 2026-10-24T05:03:35.490319+08:00 | configuration of the group running at the qc snapshot; beta0 actual execution |
| [S27] | 2026-10-23 | completed task27 v2; dw4i production reference, dw5i comparison |
| [SA0] | 2026-10-24 00:41:40+0800 | latest completed candidate A; not deployed |
| [SA] | 2026-10-24T00:41:40+0800 | 34 paired tasks, 3264 games; raw files rechecked below |
| [TF] | 2026-10-23 | cold-start human f8; separate fast and beta1 fast_search |
| [TR] | 2026-10-23 | r7 original difficulty; L45 has overridden initial hand |
| [TR2] | 2026-10-24 | fixed four-seed teacher diagnostic, not a full evaluation panel; L42-45 overlap old panel seeds |
| [TRh] | 2026-10-23 | diagnostic multiplier .5; not current qc nor original completion |
| [TV4] | 2026-10-23 | beta1 v4 paired evaluation; new_wins is v4 |
| [D30] | 2026-10-23 12:20+0800 | L24/L26 each4 games; all480 search decisions per arm, not only sampled200 |
| [D30d] | 2026-10-23 12:20+0800 | step5-b: eight games per arm all lost; specific L24/L26 dispersion evidence |
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
| [H48] | 2026-10-24 | frozen origin/demo 206cc403ec98ef292ecde601303de1ca7bc200a9:MEMORY/docs/human-demo.md; plan, not actual recordings |
| [D53c] | 2026-10-24 | 48-game current-engine night projection only, not a complete mixed-terrain round |
| [PLAN] | 2026-10-24 | controller timeline through04:15; qc only packed before task49 pipeline deployment |
| [R0] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R1] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R2] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R3] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R4] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [R5] | 2026-10-24T05:03:35.490319+08:00 | registered tasks; does not imply qc admission or measured wins |
| [EP] | 2026-10-24 | two single actions, L5 and L50 accepted; no full games |

[E]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/engine-inventory.json
[Q]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/curriculum_state.json
[QL]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/curriculum_log.log
[QP]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/curriculum_panel.json
[RUN]: /home/newbiexvwu/PvZ-search/artifacts/hard/c3d9d7qm100_48/config.json
[S27]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task27-results.json
[SA0]: /home/newbiexvwu/PvZ-search/.claude/worktrees/dw5/artifacts/task36/complete_A.json
[SA]: /home/newbiexvwu/PvZ-search/.claude/worktrees/dw5/artifacts/task36/paired_dw4i_vs_A.json
[TF]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task45-fog-results.json
[TR]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task46-roof-cold-start.json
[TR2]: /home/newbiexvwu/PvZ-search/.claude/worktrees/roof/MEMORY/docs/task46-roof-timelines.json
[TRh]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/task46-roof-half-results.json
[TV4]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/value-widen-paired.json
[D30]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/value-widen-dispersion.json
[D30d]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/value-widen.md
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
[H48]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/human-demo-plan.md
[D53c]: /home/newbiexvwu/PvZ-search/.claude/worktrees/forks/MEMORY/docs/task53-critical-forks-deploy3-cost.json
[PLAN]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/MEMORY/docs/student-bottleneck-plan.md
[R0]: /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json
[R1]: /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json
[R2]: /home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json
[R3]: /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/task45/fog_panel.json
[R4]: /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/task46/roof_panel.json
[R5]: /home/newbiexvwu/PvZ-search/.claude/worktrees/conveyor/artifacts/stage4/day_panel.json
[EP]: /home/newbiexvwu/PvZ-search/.claude/worktrees/advgap/artifacts/task55/protocol-probes.json
