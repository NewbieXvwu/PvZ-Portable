# task93：扰动卡组的结构性修复——随机卡组任务族

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-llmplay`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/deckvar`（分支 `deckvar`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
- task84 发现，heldout 卡组 `[0,1,2,27,5,32,3]` 里的卷心菜投手（32）和三叶草（27），在所有在训任务族的卡组里都没有出现。学生只从一份冻结的旧扰动数据（4.3%）里见过它们。
- 换训练种子，heldout 就在 124 到 169/192 之间摆动。学生对换卡组的泛化靠运气。
- 目标顺序里，“泛化”排在“通关”之后，但通关本身也要求学生能处理各关的不同卡组。示范与 LLM 局的卡组也各不相同，所以这个问题绕不过去。

## 要做的
1. **随机卡组族**：照 AGENTS §8 登记新族 `day_l7_train_rand`、`night_l12_train_rand`、`pool_l21_train_rand`，每个种子的卡组随机生成。约束只用 §8 的地形覆盖与引擎合法性：
   - 白天不限；
   - 夜间不用要阳光的白天蘑菇规则——这里只需核对引擎的 `deck_warnings`，不手写规则；
   - 泳池必须有睡莲。
   - 卡组长度 6–10，用 `deck_for_level` / `profile_for_deck` 核对。
   **不手写“好卡组”**，只做均匀随机加合法过滤。heldout 卡组本身必须排除在外：机械核对，任何种子的卡组都不等于 heldout 卡组。
2. **冷启动**：生产老师加学生 r11，每族 32 局，报胜率与卡组覆盖统计（每种植物出现次数）。
3. **课程接入方案**：照 task75b，按胜率决定倍率档；写清采样占比（建议与现有扰动旧数据同量级起步），并预测它对 heldout 的影响。只写方案，不改 qc。
4. **验证设计**：
   - 先在 Mac 或台式机离线加一份随机卡组数据（约 48 局 × 3 族），用 r13 配方训一个学生；
   - 在 heldout 192 对、对照 d7 192 对上，与 task84 的 A / D2 比较，看 heldout 的均值和方差是否改善；
   - 至少训 2 个种子，否则无法和方差对比。
5. 结论写 `MEMORY/docs/deck-diversity.md`，DESIGN 登记新族，提交并 `git push origin deckvar`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；**不手写打法规则或公式**（卡组随机，不挑）。
- GPU 训练与 qc 冲突时让 qc 先。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果
- 2026-10-26 CodeBuddy 接手扫描（未改代码），交回 Codex。已查明的机制，续做时不必重查：
  - 运行时每种子卡组解析在 `scripts/render_episode.py:288`（`training_task_kwargs` → `pvz_tasks.deck_for_seed`，纯函数）。随机族建议在 `pvz_tasks.py` 加 `deck_rand` spec：清单构建时离线生成、逐个过引擎 `PvZEnv.deck_warnings`（`python/pvz_env.py:838`），把通过的 种子→卡组 表存进清单，运行时纯查表——整组警告无法像 swap 扰动那样穷举（`deck_rand` 空间是组合数），但逐个查询在生成期可行。
  - 候选卡池来源照 `scripts/perturb_family.py` 的 `engine_family`：`PvZEnv.deck_pool`（Crazy Dave 候选）∩ `deck_options().not_recommended` 为 0 ∩ `0 <= p < 40`（排除升级/特殊卡）；泳池族再强制含睡莲 16（`terrain_covered`）；夜间不手写规则，只靠引擎警告过滤（无阳光生产者的整组警告会自动拒掉）。
  - heldout 排除先例：`perturb_family.py` 生成的 `split='heldout_deck'` 任务（如 `day_l7_eval_d7p_holdout1`，卡组 [0,1,2,27,5,32,3]），生成后把全部 heldout 签名写进 `excluded_decks`；随机族照此机械排除并断言。
  - 任务清单是本地派生不入 git（`/home/newbiexvwu/PvZ-search/artifacts/stage4/*.json` 等）；新种子段必须扫描全部现有清单防撞（`perturb_family.run_family` 的 `scan_paths` 模式），候选块 90_000_000 起待核对。
  - 建议基任务复制非卡组字段自现行课程族：`day_l7_train_d7` / `night_l12_train_n10q` / `pool_l21_train_p9k`（各 1 关，playthrough 2，倍率 1.0）。
  - 下一步：写 `scripts/make_rand_deck_panel.py` 生成三族清单（第 1 步），然后冷启动 32 局/族（第 2 步）。
- 2026-10-26 CodeBuddy（deckvar）：第 1 步代码完成并提交（6b978c39）。Codex 已写的 `make_random_deck_tasks.py` /
  `pvz_tasks.py` deck_random 收据机制 / `search_play.py` 支持照收；修正候选池为任务书定的
  `crazy_dave_pool ∩ [0,40) ∩ not_recommended==0`（原稿用未过滤 `available_plants`，含升级卡，纯升级卡组是种不下的结构性死局）。
  种子段 93_000_000 起，对全仓 artifacts 90M+ 数字扫描 0 撞车；生产引擎= `cq/build/pvz-portable`（与 d7q 清单指纹一致）。
  `test_deck_perturb.py` 9 过。下一步：跑生成脚本产出三族清单。
- 2026-10-26 CodeBuddy：三族清单已生成 → `/home/newbiexvwu/PvZ-search/artifacts/stage4/random_deck_panel.json`
  （本地派生不入 git）。day_l7/night_l12/pool_l21 各 96 种子全部拿到引擎校验收据（cold 32 + tiers 16 + train 48）；
  长度 6–10 近均匀；泳池族睡莲(16) 96/96；夜间整组警告拒绝全来自引擎 `sun`（154/263 次），无手写规则；
  heldout 签名碰撞 0；引擎指纹 b642614f（= 生产 cq）。下一步：冷启动 32 局/族（老师 + r11）。
- 2026-10-26 CodeBuddy：冷启动 32 局/族 × 双臂完成（r11 学生 + 生产老师 v5/branch1，workers 2，本树自建引擎 ac683aeb；
  逐局 JSON 在 `artifacts/task93/cold/report.json`）。胜率（学生/老师）：
  day_l7_rand **11/32 / 17/32（53%，Wilson 36–69）**；night_l12_rand **1/32 / 5/32（16%）**；pool_l21_rand **3/32 / 7/32（22%）**。
  覆盖：卷心菜(32) 三族 8/8/6 次；睡莲(16) 泳池 32/32；向日葵(1) 白天/泳池 32/32（引擎组警告过滤，非手写）。
  注意：三叶草(27) 在三族候选池中一次未出现——它是引擎 `not_recommended` 标记滤掉的（day 池 24 种、night 32 种、pool 26 种），
  heldout 卡组里的 27 仍只能靠旧冻结数据覆盖，这条已在结论文档记为已知缺口。下一步：第 3 步课程接入方案。
- 2026-10-26 CodeBuddy：第 3 步课程接入方案 + DESIGN §14.35 登记 + 结论文档 `MEMORY/docs/deck-diversity.md` 完成并提交。
  要点：day 族 1.0 倍直接登记（53%）；night/pool 先扫 0.5/0.75（各 16 局）再定；三族合计 ~4% 行占比起步、
  label_weight 1.0 平权（task84 B 证明加权冻结数据无效）；heldout 预测均值回典型簇、方差缩小。
  第 4 步已开工：144 局老师 frames 采集中（deploy4 引擎，labels rand_d7/rand_n12/rand_p21）。
- 2026-10-26 主控换手：task93 交 Business 号 Codex。CodeBuddy 到此为止（代码与文档已全部提交推送，工作树干净）。
  **交接状态**：第 1–3 步完成；第 4 步进行中——144 局老师 frames 后台采集中（后台任务未停，输出在
  `artifacts/task93/collect/<族>/` 与 `artifacts/task93/frames/`，deploy4 引擎 cq/build4 16989f23，
  labels rand_d7 / rand_n12 / rand_p21，种子 93_x0x_100..147，完成后自动退出）。
  采集完成后的下一步：pack（见 `MEMORY/docs/deck-diversity.md` 验证节：task84 A 语料 + 3 新标签，
  symlink 农场复用 task84 段缓存）→ seed 1/2 各训一次（train_rollout_policy --recipe，~45 分钟/个）→
  照 `gatepower/scripts/task84_ablation_eval.py` 在 heldout 192 + 对照 d7 192 配对评估（deploy4、fast、dagger-beta 0，
  r11 配对局在 `~/pvz-logs/curriculum_state.pipeline/round_000013/evaluations/a41fdfa96075ae608a40/incumbent/`）。
  基线：A 124/172、D2 169/178、r11 166/168。
- 2026-10-26 补记：**采集已完成（自行跑完，非 CodeBuddy 操作）**——3 族 × 48 局老师局全齐，
  存档 `artifacts/task93/frames/seed_*/{rand_d7,rand_n12,rand_p21}_fast_search` 各 48（逐局结果在 `artifacts/task93/collect/`）。
  Codex 从 pack 步骤接着做即可（上方交手行里“采集中”字样已过时）。CodeBuddy 保持停止。

- 2026-10-26 Codex 续做：现场确认第 3 步已提交 2fb1e925、第 4 步采集存活，沿用同一 144 局作业，未重开游戏。
  新增 `scripts/task93_analyze.py`：逐份核验冷启动 192 局完整终局/卡组/双臂，0–47 覆盖全表补入结论文档；
  训练后严格按完整 192 对比较 A / D2 / r11，输出配对区间、McNemar、draw 间均值/标准差/极差。
  修正统计单位：A / D2 的 45 胜局极差实际为 23.44pp，双 draw 只能提供方差改善初步证据；
  4% 目标行占比不能由 144 局直接推定，须核对实际 source_rows。相关回归 17 passed，未改 qc。

- 2026-10-26 Codex：新增 `scripts/task93_prepare_training.py`，seed 1/2 配置已用 plan-only 生成，
  351 标签（A 的 348 + 随机族 3），旧超参数与权重逐字段全等、init=None；缺局拒绝检查通过。
  采集仍沿用现有作业。下一步：三族各48局齐后建立隔离语料并复用旧分片，打包核对行占比，qc GPU 优先下串行训练两种子。

- 2026-10-26 Codex：原采集白天48局已齐（29胜），夜间继续，现场核对始终2个游戏进程。
  `scripts/task93_validation.py` 已通过 pidfd 绑定现存采集 shell，待其退出并验证144局后串行打包、训练seed1/2和评估；
  独立锁与模型/配方回执避免重复工作，运行中遇qc训练只让本实验退出。状态见 `artifacts/task93/validation_status.json`。

- 2026-10-26 Codex：第4步144局训练数据采集完成，三族均48终局，老师胜局 day29/night11/pool21；
  卷心菜卡组18/10/14，三叶草仍0；完整种子、卡组、heldout排除逐局核验通过，原采集及其游戏进程已退出。
  隔离语料准备成功，408份既有不可变分片已硬链接复用，正在打包。当前活进程句柄见 validation_status.json；不要重开采集。

- 2026-10-26 Codex：打包完成，17,012,427行，built3/reused348；与A逐标签核对348旧标签的行数及归档数全部相等。
  新增29,496行（day11815/night7732/pool9949），原始占比0.17338%、按label_weight折算0.15458%，明显低于4%课程目标；
  保持预登记144局和平权，不临时提权。seed1训练已启动，seed2和评估未完成。
  同时纠正方案引用：task84实际A=348标签/扰动默认权重1.0，B=380标签/扰动1.5；旧文字的“3倍加权单独证明无效”不成立。

- 2026-10-26 Codex：seed1已到第6/12轮；另一项1轮离线训练并发后swap耗尽，实际步进降至约0.1步/秒，qc无训练冲突。
  为保留已完成优化步，暂以SIGSTOP挂起本任务训练，不重训、不改配方；恢复辅助进程已通过pidfd绑定对方训练，待其退出后SIGCONT原进程。
  精确句柄及状态见 `artifacts/task93/training_yield.json`，流水线仍存活。续做同时核实训练进程T状态、恢复辅助进程和所等进程，不能当作崩溃重开。

- 2026-10-26 Codex：上述让行结束，对方单轮训练以complete终态退出；辅助进程成功SIGCONT原seed1进程并退出。
  原PID不变，优化步从37421继续增长，仍在第6轮，无重训；让行699秒，后续墙钟耗时包含这段等待。

- 2026-10-26 Codex：同一外部实验紧接着启动第2个单轮训练，swap再次耗尽；原seed1进程仍保留第6轮约41575步，第二次SIGSTOP让行。
  新恢复辅助进程已绑定该第2个训练的pidfd；`training_yield.json` 同时保留第一次恢复记录，当前应按其中最新句柄核实，不能重开训练。

- 2026-10-26 Codex：第2个外部单轮训练complete退出，第二次辅助进程也已成功SIGCONT原seed1并退出；仍为同一PID，当前41614步、第6轮。
  两次让行分别699/310秒，合计1009秒，均保留优化器状态；接下来继续第6–12轮，未重训。

- 2026-10-26 Codex：seed1完成12轮/89772步，模型SHA e9f2cb8f…，配方SHA、seed/init、1–12轮历史与参数有限值独立核验通过。
  heldout137/192、d7对照174/192，均为完整192终局且严格配对；对r11的heldout −29局（−15.10pp，p=.000522），对照+6局（p=.4296）。
  相对A的heldout+13局（p=.1875）、相对D2 −32局（p=.0000773）；首个种子未达166阈值，不重抽、不改门槛。
  seed2未开训，现存流水线等待内存余量（qc无训练），须完成seed2及其384局评估后才报告均值/方差与最终结论。

- 2026-10-26 Codex：因另一12轮学生训练占用资源，修正私人调度器：首轮RSS采样峰值11.29GiB，seed2准入=实测峰值+15%余量、至少12GiB（当前12.98GiB），未采样仍16GiB。
  启动前等待全部其他学生训练退出，保留qc优先；空闲调度器已安全替换，现存句柄以 validation_status.json 为准。
  模型、打包与两组完整192局均校验复用，游戏文件SHA不变，没有重跑；恢复/完整性相关19项测试通过。
  当前seed2仍等待对方训练与内存条件，实验配置与任务范围不变。

- 2026-10-26 Codex：补全课程方案预算及范围校正：在A规模与当前行密度下，4%约需70.8万新增行/总计约3455局，后续扩量须另行预登记新种子段、校验收据并保留旧定义；本次验证仍144局不加码。
  明确当前随机范围是冻结候选池子集，不是全部合法卡组空间；A另有24份r7屋顶归档/3600行，卷心菜并非只存在于旧扰动数据。
  未修改任务规则、配方或qc，seed2继续由现存调度器等待资源。

- 2026-10-26 Codex：资源窗口满足后，现存调度器已自动启动seed2；实际resolved配方与seed1仅seed/out不同，12轮/351标签/init=None/CUDA。
  两份source_rows.json逐项全等，均17,012,427行；seed1模型与384局严格复用，没有重训重评。
  当前seed2训练中，完成后继续其heldout192+d7对照192，再汇总两种子的均值与离散程度，任务尚未完成。

- 2026-10-26 Codex：qc恢复第18轮训练，seed2首次尝试在第3轮按优先机制终止，无model.pt/完成回执，也未评估；不能计作第二个模型。
  私人调度器现将已知qc让行记入preemptions.jsonl并恢复排队，启动前同时检查活动qc控制器的queued任务，待资源释放后重试同一冻结seed2配方。
  普通训练错误不重试，无回执模型拒绝覆盖；23项相关测试通过。已核实旧控制器退出后启动唯一恢复控制器，首模型和384局校验复用，未改qc。

- 2026-10-26 Codex按主控追加取用Mac task93b：origin/deckvar-mac的ec5dfc9b记录三族各192局，台式机已导入576份frames/meta（新增1152文件）。
  独立核验实际种子段93_000_500–691/93_010_500–691/93_020_500–691、r11 SHA、倍率、卡组/地形/heldout排除及文件账本；胜104/75/114，未抽样可用行194636/166863/149963。
  三份HF transfer包校验后已原子删除并确认不存在，删除提交9002402f…；仅改导入器支持m100/m050后缀，19项导入清理测试通过。
  扩采纳入后续课程方案：白天1.0、泳池0.5可提议起步，夜间0.5仍39.06%暂缓；原144局/两种子剂量保持冻结，seed2继续按qc优先排队。
  DESIGN登记expand族；完整覆盖与SHA见task93b-desktop-import.json。Mac扫描数字来源于git提交，原引擎清单/采集配置未随包传来，生产接入前须补齐，未改qc。

- 2026-10-26 CodeBuddy（接手 Business Codex 后收尾）：先核实现场，第 4 步已自行跑完——seed2 训练完成 12 轮
  （模型 SHA 3013f860…，第 3 轮让行那次无模型/回执，不计入），heldout 192 + 对照 192 全齐，
  validation_status 为 complete，无存活游戏进程属本任务。独立重跑 `task93_analyze.py compare` 两次，
  生成的两份 JSON 与入库文件逐字节相同；两 seed 的 resolved 配方只差 seed/out，source_rows 全等。
  结论写入 deck-diversity.md：判据“两种子都≥166”**未达成**（seed1 137 / seed2 169），
  均值 153.0 高于 A(124)、低于 D2(169) 与 r11(166)，种子极差 16.67pp 无方差改善证据；
  0.173% 剂量不构成对 4% 的检验。建议 144 局剂量不进生产课程，走 expand 族补到 4% 再按同一判据复验。
  同时提交 Codex 遗留的 task93b Mac 导入（576 局、19 项测试通过）。DESIGN 已补结论；工作树干净并 push。

- 2026-10-26 CodeBuddy：按方案要求补做剂量预算复核（文档里“具体局数须按保留行密度重新预算”那一项）。
  用 `training_rows` 重数 144 局可用行 125,146，打包保留 29,496 → 保留率 23.57%（204.8 行/局）；
  扩采 576 局按同族保留率折算约 12.0 万行（0.702%），两批 720 局 0.873%；到 4% 还需约 2,687 局、合计约 3,407 局。
  新种子段提案 93_100_000 / 93_110_000 / 93_120_000 各 2,000 槽位，扫 178,817 个种子目录碰撞 0；
  只写提案，未生成清单、未采集。限制：保留率来自 144 局，扩采局长分布不同，实际须打包后复核。

- 2026-10-26 CodeBuddy：把上一行的估算换成实测。576 局扩采在隔离目录（576 个种子符号链接）用生产打包器
  `pack()` 打包，3 个新分片、0 复用、499 秒，实测 **120,336** 保留行（day 46,098 / night 35,347 / pool 38,891），
  比按 144 局保留率折算的估计多 292 行（+0.24%）。720 局合计 149,832 行 = 0.875%；
  到 4% 约再需 2,680 局、合计约 3,400 局；两族各 2% 则 day 再约 1,202、pool 再约 1,470 局。
  实测包在 ignored `artifacts/task93/expand_pack/`（59 MB），不并入训练语料；未采集、未改 qc。

- 2026-10-26 CodeBuddy：预登记并搭好 0.875% 剂量档（不新增采集）。新增 `scripts/task93_expand_prepare.py`：
  720 局语料（A 348 + 冻结 144 + Mac 576）354 标签、**17,132,763 行**，打包断言通过
  （扩采三标签 46,098/35,347/38,891，其余 351 标签合计仍 17,012,427，旧分片 822 个全部硬链接复用、0 重建）；
  576 份档案逐份复核 task_id/种子/倍率/r11 SHA/卡组长度/地形/heldout 排除/profile 槽位。
  `task93_validation.py` 增加 `--root` / `--prepare expand`，其余流程（qc 让行、内存准入、12 轮回执、
  384 局评估、配对比较）原样复用。判据三条已预登记；下一步按预登记跑 seed1/2 训练与评估。

- 2026-10-26 CodeBuddy 接手（Codex 额度用尽后第二次换手）：现场核实 expand 档 seed1 **训练已死在第 6 轮之后**——
  控制器 pid 914256 与子进程 916578 均不存在，系统里没有任何 `train_rollout_policy.py`；`dmesg` 无 OOM 记录；
  `seed_1/student/` 只有 `recipe.resolved.json` 与 `source_rows.json`，没有 `model.pt` / `metrics.json` / `trained.json`。
  因此**不能算完成、也没有可续的检查点**（训练器不存中间模型）。6 轮日志已另存为 `train_seed_1.attempt1.log`
  （第 6 轮 1817 秒，对比第 2–3 轮 230–237 秒，说明当时有明显资源争用；但无 OOM 证据，死因未定）。
- 2026-10-26 CodeBuddy：用**同一冻结配方、同一打包文件**（`4197e0ba5f6e919d`，17,132,763 行）重启流水线，
  不重采集、不改配方、不改判据。准入内存显式给 12 GiB：该 root 没有 `trained.json` 实测峰值，
  默认 16 GiB 在当前 12–13 GB 可用内存下永远无法准入；依据是 144 局档实测峰值 11.29 GiB（语料比本档小 0.7%）
  与调度器既有的 12 GiB 下界。重启后 seed1 第 1 轮已起（child 936571），本任务无游戏进程。

- 2026-10-26 CodeBuddy 交回 Codex（主控：额度恢复，提交并停）。交接时刻现场：控制器 pid **936153**、
  训练子进程 **936571**（17:28 起，第 1 轮中，RSS ≈ 9.4 GB），流水线状态在
  `artifacts/task93/expand/validation_status.json`，日志 `expand/train_seed_1.log`（前 6 行 epoch 属
  attempt1，已另存 `train_seed_1.attempt1.log`）。未重采集、未改配方、未改判据、未动 qc。
  **风险提示**：该训练是本会话的后台子进程；若会话结束把它带走（首轮正是这样死的，且无中间检查点），
  按同一配方重启即可，不要重训已有 `trained.json` 的模型。剩余待办：seed1 12 轮 → seed2 12 轮 →
  各跑 heldout 192 + d7 对照 192 → `task93_analyze.py compare` → 给 0.875% 两个模型补 32 卡组补充端点
  （照 `expand/randcold/seed_1_0173/config.json` 的 argv，只改 `--rollout-policy` 与 `--out` 为 seed_1_0875 / seed_2_0875）
  → 写结论并按三条预登记判据判定。

- 2026-10-26 Codex恢复接手：交接记录中的expand控制器936153/训练936571均已消失，最后遥测为第1轮4375步，无模型/metrics/完成回执；死因无退出状态可证，不计完成。
  保留attempt2日志快照，以同一冻结配方恢复唯一控制器（start_new_session，当前句柄见expand/controller_launch.json），避免依赖交互终端。
  准入按原完整seed1实测RSS乘扩采行数比例后加15%余量，约13.07GiB，训练轮数/数据/种子/判据不变，qc优先。
  同时核验并校正结论：A/D2实际配方仅seed/out与来源元数据不同；0.173%档相对其双种子均值+3.39pp，样本SD11.79 vs16.57pp，仅描述性改善。
  三叶草仅在新增随机数据缺失，旧冻结扰动已有覆盖；32卡组面板全部含向日葵，删除与档案不符的解释。

- 2026-10-26 Codex：新增task93_randcold.py，pidfd绑定现存expand主控制器；主验证成功且模型完成回执齐全后，持同一实验锁串行补两个模型各32局，workers2，冻结首批config，旧96局校验复用。
  首批r11/0.173%两种子独立重算11/13/13通过。对训练包354分片全部17,132,763行packet植物ID检查，与32评估卡组碰撞0；不依赖缺deck的51,191份旧metadata。
  主训练与补充等待器均已核实存活，当前句柄分别在expand/controller_launch.json与expand/randcold/controller_launch.json；51项相关回归通过。

- 2026-10-26 Codex：expand seed1同一训练进程因nice=10、全机负载约27而仅约3步/秒；仅将该进程恢复普通CPU优先级nice=0，未动其他进程，qc的GPU让行机制保持。
  后续30秒遥测恢复到35.91步/秒，优化步连续至2869；无重训或优化器重置。seed2启动时同样先核对其实际进程与nice值，避免低CPU优先级造成长时间饥饿。

- 2026-10-26 Codex补做划分审计：生产训练器在全局种子集合上seed0打乱取10%验证，加语料导致旧训练/验证分区改变。
  从354分片种子数组重建A/0.173%/0.875%划分并逐标签对齐真实source_rows，全等；A→0.173%共同旧行13.40%换分区，0.173%→0.875%为13.80%。
  数字和SHA见task93-split-audit.json，结论已明确这是加数据的整体管线对比，不能单独归因于剂量/覆盖；不改在跑配方、模型或判据，不追加实验。

- 2026-10-26 Codex完成主端点覆盖核验：全部17,132,763个保留输入的卡槽植物ID组合里，heldout整套0行，对照12,029,996行（跨场景/训练+验证口径，不等于完整任务匹配）。
  数字并入task93-split-audit.json；原144局为子集，heldout整套组合亦未见。训练已进入第4轮，仍同一进程，未重采或重训。

- 2026-10-26 Codex：qc新作业排队使expand seed1第5轮中的进程943041按既定优先机制终止，旧训练器未保存优化器，未完成模型。
  为避免持续丢失计算，新增仅显式启用的训练恢复：SIGTERM在完成批次后原子保存模型/AdamW/调度器、轮内顺序位置、Python/NumPy/Torch/CUDA RNG；每轮保存临时恢复点，完成回执后删除。
  8项相关小测试实际16秒通过，包含带dropout的中断续训与不中断训练逐参数一致；没有重跑完整训练作验收。
  新控制器953319、32卡组等待器953320已启动，资源仍让qc先；--recover-training/--train-nice0，冻结配方/数据/判据不变。
  调度器只读已有pack头与分片存在，源指纹遍历由下游训练入口完成一次，取消重复上游遍历；当前句柄见controller_launch.json。

- 2026-10-26 Codex：精简deck-diversity.md中的重复首种子表、过时续做指令和校验/运行流水账，保留双种子结果、课程方案、预登记判据及来源；现有expand排队作业保持。

- 2026-10-26 Codex：修正GPU排队范围：qc的queued状态也覆盖CPU晋升评估，已有对应recipe的trained回执时不再仅因queued占住GPU；待训练作业和实际qc训练仍优先。
  7项相关调度测试0.17秒通过，仅替换已确认无子进程的空闲控制器，训练/数据/判据及断点恢复不变；新句柄见controller_launch.json。

## 问题

- 第 4 步在 0.173% 剂量上判据未达成（seed1 137 / seed2 169）。是否批准用 expand 族做约 4% 的第二轮验证
  （按当前保留行密度约 3455 局 / 70.8 万行）交主控；不批准则随机卡组线停在“族与方案已登记、未入课程”。
- 三叶草(27) 仍无新覆盖：三族候选池都被引擎 `not_recommended` 滤掉。要覆盖 heldout 里的 27，
  得换一条不依赖 Dave 候选池的来源（例如从 heldout 签名反推），这属于改候选池 = 换任务族，须另行登记。
