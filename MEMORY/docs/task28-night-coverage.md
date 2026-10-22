# task28（主控 2026-10-22 15:30 下发给台式机 CodeBuddy 第二条线）：2-10 DAgger + 夜间扰动卡组 n10p（补没见过的植物）

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8），再读 `MEMORY/docs/night-l20.md`（你上一个任务 task26 的结论）、
`MEMORY/docs/task25-night-n10.md`、DESIGN §14.9 与 §14.7 里 task19 的 `day_l*_train_d7p`（`deck_perturb` 的做法）。

## 背景
task26 已合进 pvz-env（042e4c79）。结论：2-10 原版开局学生 0 胜、**一棵都不种**——传送带发的 7 种卡里 5 种（10 大喷菇、12 魅惑菇、13 胆小菇、14 寒冰菇、15 毁灭菇）
从未进过任何任务卡组，identity 头在这些身份上把种植门压成负值；老师（同学生 + 搜索）6/8 胜。两件事：用 DAgger 教它在 2-10 种，
并且让这些植物在普通夜间关里也出现（换卡组泛化，和白天的 d7p 同理）。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/n10p -b n10p origin/pvz-env`，只在这里改代码、单独编译。

1. **2-10 DAgger 采集**（先开，后台跑，同时最多 2 个游戏进程）：`night_l20_train_n10_orig`（`~/PvZ-search/artifacts/stage4/night_panel_n10.json`，种子 2290000 起）偏移 0 起 32 局，
   采集学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`，参数照 `~/pvz-logs/queue103.sh` 的 `run()`（beta=0：学生走、搜索打标签），
   标签 `n20o1`，帧写进 `~/PvZ-search/artifacts/student/frames_v2`（与 q103 同布局）。用 nohup，进度写日志，完成后在“结果”记胜局与每局按类型的种植数（`planted` 键是槽号，要用卡组翻成类型）。
2. **夜间扰动卡组族**：照 task19 的 `deck_perturb`（`scripts/make_identity_tasks.py`），以 n10 `[0,1,2,3,4,5,7,8,9,11]` 为底，每种子确定性替换 1 或 2 张，
   pool 取引擎实际解锁（playthrough 2）、非升级、无引擎警告、夜间能种的植物（应包含 10/12/13/14/15）。新族 `night_lN_train_n10p`（N = 11–14、16–19，每关 8192 种子），
   种子段自己选一个与所有现有清单都不重叠的段（生成时穷举自查，写进 DESIGN）；再出留出族 `night_lN_eval_n10p_holdout{1,2}`（评估段 30000 起，每关 2 套，训练族显式排除这些卡组集合）。
   生成到 `artifacts/stage4/night_panel_n10p.json`（不进 git），**另拷一份到 `~/PvZ-search/artifacts/stage4/`**。DESIGN 新开一节登记。
3. **老师验证**（等第 1 步跑完再开，≤2 进程）：老师（dw4i_s1 推演，beta=1）在 `night_l{12,14,17}_train_n10p` 各 16 局，学生（dw4i_s1）同种子各 32 局；
   列出每局卡组、胜负、按类型种植数。老师胜率远低于 n10（2-2 / 2-4 / 2-7 老师 n10 约 13/16、11/16）的话，找出是哪些替换让它结构性打不过，写进“问题”由主控定 pool。
4. 报告写进 `MEMORY/docs/night-l20.md` 新的一节（2-10 DAgger）与 `MEMORY/docs/night-panel.md` 新的一节（n10p）。

## 规则
- **不要动任何 tmux 队列（q103 等）与别的代理会话；不要改别的工作树与 `~/PvZ-search` 主工作区**（frames_v2 只增不改；stage4 只新增 n10p 一份）。
- 分支 `n10p` 提交，可以 `git push origin n10p`；不推 pvz-env、不合并。不训模型；不改默认值；不改 `deck_for_level`；不写手工规则；不写字面密钥；
  不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。每完成一小步就提交，并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。

## 结果

- 2026-10-22 第一步（2-10 DAgger 采集）：`night_l20_train_n10_orig` 偏移 0 起 32 局（种子 2290000–2290031），`--dagger-beta 0`、标签 `n20o1`、帧写 `~/PvZ-search/artifacts/student/frames_v2/seed_22900xx/n20o1_fast_search`，2 并发 16 分钟跑完（每局 40.4 s）。**0/32 胜**、结束波 6–8（均值 6.66）、每局 271.7 决策 / 搜索 51.5 次 / 标签判“种”18.0 次。执行侧种植 **0**（beta=0 执行者是学生，与 task26 一致）；标签侧 575 个种植：毁灭菇 315、寒冰菇 113、小喷菇 93、胆小菇 27、大喷菇 18、魅惑菇 8、墓碑吞噬者 1。分析脚本 `scripts/analyze_n20o1.py`，汇总 `artifacts/task28/n20o1_summary.json`（不入 git）。报告 `MEMORY/docs/night-l20.md` §5。
- 2026-10-22 第二步（夜间扰动卡组族 n10p）：新增 `scripts/make_night_n10p.py`（照 task19 的 `deck_perturb`，底卡组 n10、`swaps=[1,2]`、池子从引擎 `DECK_OPTIONS` 现查）。生成 `artifacts/stage4/night_panel_n10p.json`（工作树 + `~/PvZ-search/artifacts/stage4/` 各一份，不进 git）：8 个训练族 `night_lN_train_n10p`（N=11–14、16–19，种子 2400000/2410000/2420000/2430000/2450000/2460000/2470000/2480000 起各 8192）+ 16 个留出 `night_lN_eval_n10p_holdout{1,2}`（种子 30000–30255）。审计：65536 训练种子 / 11794 个不同卡组；与 15 份现有清单的 407552 个已用种子穷举比对**重叠 0**；必需植物 10/12/13/14/15 分别出现在 4180/4128/4243/4011/4142 个训练卡组；reset 抽查 4 个任务均为 10 槽且类型逐位等于 `deck_for_seed`。DESIGN §14.11 登记，`MEMORY/docs/night-panel.md` §10 写族定义与审计。

- 2026-10-22 第 6 步（引擎整组检查 + 戴夫候选池 + n10p 重生成 + d7q 新族）：`SeedChooserScreen::OnStartButton` 的整组判断抽成 `CheckDeckWarnings`（弹窗路径改调它，顺序与文案不变），`CrazyDavePickSeeds` 的权重构造抽成 `BuildCrazyDaveWeights`；`LawnApp::EnvironmentDeckWarnings` / `EnvironmentDeckPool` + `main.cpp` 的 `DECK_WARNINGS` / `DECK_POOL` + `PvZEnv.deck_warnings()` / `deck_pool()`。单元测试 `python/test_deck_warnings.py` 6 项（无产阳光、泳池无水生、屋顶无花盆、夜间无阳光菇、正常卡组、戴夫池规则）全过；`test_deck_perturb.py` 加一条“`engine_warned_decks` 永不被采样”，7 项全过。`deck_perturb` 新增 `require_no_warnings` + `engine_warned_decks`（生成时穷举全部候选卡组问引擎）。**n10p 重生成**（同族名同种子段）：池子 34 → 32（去掉叶子保护伞 37、三叶草 27），每关 10615 个候选卡组里 231 个（2.2%）触发 `sun` 警告被排除，65536 个种子里 59.0% 的卡组与旧版不同，旧版 765 局（1.17%）缺经济卡的卡组已全部排除；9982 个不同卡组、5 种必需植物覆盖 4576/4531/4587/4350/4451、与 15 份清单 407552 个已用种子重叠 0。**新族 d7q**（`day_lN_train_d7q`，N=7–10，种子 2500000/2510000/2520000/2530000 起各 8192；留出 `day_lN_eval_d7q_holdout{1,2,3}` 30000–30255）：每关 2975 个候选卡组里 833 个（28.0%）触发 `sun` 警告被排除，32768 个种子 / 2130 个不同卡组 / 重叠 0。两份清单已拷到 `~/PvZ-search/artifacts/stage4/`。DESIGN §14.11 更新、§14.12 新增；`MEMORY/docs/night-panel.md` §10 更新。第 3 步老师验证已按通知停掉作废（停在 teacher L14 4 局），等这次重生成后再跑。

- 2026-10-22 第 5 步（缺经济卡统计，不跑新局）：扫 `frames_v2` 的 d7p / n10 / p9k 档案（d7p 用 meta 里的实际初始卡组，n10/p9k 用族固定卡组），逐帧数 `label`。**d7p：没有向日葵的 849 局胜率 15.1%**（128 胜、平均 15.14 波、每局老师判“种”47.7 次），**有向日葵的 3040 局 82.1%**（2495 胜、23.92 波、64.2 次），差 67pp；缺经济卡的局占 d7p 的 **21.8%**（849/3889）。夜间 n10 320 局全部带阳光菇（65.9%），泳池 p9k 896 局全部带向日葵（0.2%，与经济学无关）。n10p 只会换掉阳光菇、不会同时去掉两张经济卡（231/10615 候选卡组被整组检查排除）。结论写进 `MEMORY/docs/night-panel.md` §11；脚本 `scripts/analyze_economy_decks.py`，汇总 `artifacts/task28/economy_summary*.json`（不入 git）。

## 问题
- **任务书指的 `artifacts/stage4/night_panel_n10.json` 里没有 `night_l20_train_n10_orig`**：那份是 2026-10-22 10:10 的 9 个 eval 任务的旧清单（只有 `night_l11..l20_eval_n10`），没有 train 族、也没有 L20 原版开局任务。第 1 步改用 `~/PvZ-search/artifacts/stage4/night_panel.json`（task25 的合并清单，54 个任务，含 `night_l20_train_n10_orig`，种子 2290000–2298191）。
- **`planted` 在 beta=0 的采集里恒为空**：任务书说“用卡组把槽号翻成类型”，但 beta=0 时执行者是学生，而学生在 2-10 上一棵都不种，`planted` 全是 0（槽号字典为空，没有可翻的东西）。真正有意义的是**标签侧**（搜索想种什么）：报告里另算 `label` 的按类型计数，并注明这是“老师想种”，不是“实际种了”。
- **生成器的 `DECK_OPTIONS` 查询会短暂多开一个引擎进程**：n10p 的池子必须现查引擎，`EventWaitEnv` 每关起一次、顺序执行、每次数秒。当时 2-10 采集占着 2 个 worker，峰值 3 个引擎进程（每个只活几秒、不跑对局）。除此之外全程 ≤2 个游戏进程。


## 第 5 步（主控 2026-10-22 16:00 追加，不占游戏进程，任何时候都可以做）：没有向日葵的扰动卡组
task27 实测：统一面板里所有模型都 0 胜的两套白天留出 L7 h3 `[5,3,36,7,4,2,0]`、L8 h1 `[3,22,4,2,7,5,32]` 老师也 0/16——两套都**没有向日葵（1）**，白天只剩天上掉的阳光；
L9 h1 有向日葵，老师 1/16。d7p 训练族每种子换 1–2 张，约五分之一的种子会把向日葵换掉。用已有档案查（不跑新局）：
1. 汇总全部已采的 d7p 局（台式机 `d*p*d`、Mac `d*p*m`，`~/PvZ-search/artifacts/student/frames_v2`，用 meta 里的实际初始卡组）：按“卡组里有没有向日葵”分组，学生自走胜率与老师标注步数各多少。
2. 同样口径看夜间（有没有阳光菇 9）与泳池 p9k（向日葵）——只列数字，n10p 的 pool 若会换掉阳光菇也一并说明。
3. 结论写进 `MEMORY/docs/night-panel.md` 的 n10p 一节或单列一段：缺经济卡的卡组是不是基本全输（结构性），占训练数据多少。主控据此决定是否在扰动生成里要求保留经济卡（那是任务设计，按 AGENTS §8 另开族名）。

## 第 6 步（主控 2026-10-22 16:10 追加，**优先于第 3 步**）：扰动卡组要过引擎的整组选卡检查
用户指出：原版开局点“开始”时引擎会对整套卡组做检查，`SeedChooserScreen::OnStartButton`（src/Lawn/Widget/SeedChooserScreen.cpp:609–690）——
没有产阳光植物（向日葵 1 / 双子向日葵 / 阳光菇 9）警告 `[SEED_CHOOSER_SUN_WARNING]`，泳池关没有睡莲/海蘑菇/缠绕水草、屋顶没有花盆、有飞行僵尸没有对空植物、带了升级植物没带底座，也都警告。
task19 的 `deck_perturb` 只查了逐卡的 `not_recommended`，**整组检查没调用**（`make_identity_tasks.py` 自己写着 “interactive whole-deck OnStartButton checks not called”）。白天两套全 0 留出正是没有向日葵。
1. 把 `OnStartButton` 里的整组判断抽成引擎里一个不弹窗的查询（输入：卡组 + 关卡/模式，输出：会触发哪些警告），通过环境接口暴露给 Python。**判断直接来自引擎原代码，不在 Python 里另写一份规则**；原交互流程行为不变（弹窗仍调用同一函数）。单元测试覆盖：无产阳光、泳池无水生、屋顶无花盆、正常卡组无警告。
2. `deck_perturb` 加一个选项：排除触发任何整组警告的卡组。**n10p 还没被采集过**，直接用新选项重新生成（同族名、同种子段，在“结果”里写清前后差异与被排除的比例），再拷一份到 `~/PvZ-search/artifacts/stage4/`。
   第 3 步老师验证若已开跑，停掉作废，等重新生成后再跑。
3. 白天 d7p 已采过大量数据，不改旧族：另开新族 `day_lN_train_d7q`（L7–L10，同样做法 + 排除整组警告）与留出 `day_lN_eval_d7q_holdout{1,2,3}`，种子段自选不重叠、DESIGN 登记。旧 d7p 档案照样能用（第 5 步会量出缺经济卡的那部分有多少、是不是基本全输）。
4. （主控 16:20 补充，同属第 6 步）**替换用的卡池也改用引擎自己的“随机给卡”口径**：二周目冒险开局疯狂戴夫会替玩家随机挑 3 张卡，`SeedChooserScreen::CrazyDavePickSeeds`（SeedChooserScreen.cpp:225–255）的候选池是：
   已拥有、`SeedNotRecommendedToPick`/`SeedNotAllowedToPick` 都不命中、非升级、非模仿者；**叶子保护伞只在本关有蹦极/投石车僵尸时才进池，三叶草只在有气球僵尸或有雾时才进池，屋顶关排除火炬树桩**。
   用户指出 1-9 留出里的叶子保护伞（白天没有蹦极/投石车）毫无用处——现有 pool 就缺这层。把这份候选池判断同样抽成不弹窗的引擎查询（按关卡的僵尸名单 `mZombieAllowed` 判断，所以要在 reset 后的真实 Board 上查），
   `deck_perturb` 的替换池改用它（与第 2 条的整组检查一起作为新选项），n10p 重生成、d7q 新族都用它。不要在 Python 里列“哪种植物对哪种僵尸有用”的表——判断只来自引擎原代码。
   顺带在报告里写清：原版二周目这 3 张戴夫强制卡是真实存在的机制，我们的环境直接设卡组跳过了它；是否另开“原版二周目选卡”任务族由主控之后定，本任务不做。
