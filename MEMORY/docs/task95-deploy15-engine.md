# task95：deploy15 引擎——把 task82 / task85 / task75 之后的 C++ 改动过门禁

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-mem`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy15`（分支 `deploy15`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
deploy14 的引擎 `76ff36dc` 是用 a8d00858 的源码编的。在那之后，pvz-env 又合进了几处 C++ 改动：
- **task82**：三处未初始化字段在构造时初始化（`Challenge::mCloudsCounter`、`Projectile::mLastPortalX`、`Coin::mCollectX/Y`）；
- **task85**：L50 Boss 的可见观测（`src/LawnApp.cpp`、`src/main.cpp`、`src/EnvironmentRollout.*`）；
- 其他合入后的改动：用 `git diff a8d00858 origin/pvz-env -- src` 列全。

它们要进生产：L50 的学生观测需要 Boss 字段；task82 让跨进程哈希稳定，以后门禁不必再容忍粒子线差异。

## 要做的
1. 列出 `a8d00858..origin/pvz-env` 的全部 `src/` 改动，逐项说明对游戏逻辑的影响，预期都是“无”。
2. 照 task70 A1 的 PGO 流程构建，**构建完成立即把二进制拷到 `artifacts/task95/` 留档**（B8 吃过亏：PGO 重建不能逐位复现）。
3. **逐位门禁**：对 `76ff36dc` 做 A2 的做法，六族 × 老师 / 学生 × 8 种子，要求逻辑 0 差异。
   - 另外核对：新引擎同进程、多进程的 state_hash 应当全同（task82 的效果），旧引擎的“粒子线”差异应当消失。
4. **吞吐 ABBA**：照 A4，中位配对比要求 ≥ 0.98。
5. **L50**：新引擎在 `boss_l50_orig_eval` 上跑 4 局。确认 Boss 观测字段存在，且终局与不读这些字段时的旧引擎一致。
6. 写 `~/pvz-logs/qc_deploy15.sh`，结构照 `qc_deploy14.sh`，**只换引擎**，代码与面板沿用 deploy14。只写，不执行。
7. 结论写 `MEMORY/docs/task70-engine-redeploy.md` 新的 B9 节，`git push origin deploy15`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认，不加开关。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 和二进制不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- [5b-观测路径核对，2026-10-26 20:05 CST] 发现 `observation=6` 向量路径不一致（详见“问题”）：C++ 按 106 通道加载/分配但 `Encode` 只传 101，Boss 4 个空间通道被丢弃；Python `v2boss` 为 105。当前无 obs=6 模型，故不影响 build2 对现有策略的门禁结论，但 v2boss 学生暂不可用，需新任务修复后重建重跑。

- [1b-源码范围复核，2026-10-26 19:50 CST] 对照当前 `origin/pvz-env`（tip `e0faf684`）：`a8d00858..origin/pvz-env -- src` 仍是 7 文件 +137/−11，与[1]一致；`a630fbd8..origin/pvz-env -- src` 为空、部署树 `HEAD` 的 `src` 与 `a630fbd8` 也为空差异。即构建所用源码等于当前远端 tip 的 `src`（tip 只在 src 之外前进），候选引擎未落后。已记入 `task95-engine-gate.json` 的 `source_src_tip_verified_against`。

- [6d-端到端前置检查，2026-10-26 19:45 CST] 把脚本截到 preflight 结束（仅函数定义+只读检查，`L`/`O` 重定向到 `/tmp`）用 bash 实跑：exit 0，日志写出“preflight passed”。覆盖引擎 SHA、回滚引擎 SHA（`cq/build5` 仍为 76ff36dc）、deploy14c 代码守卫（改动集恰五文件 + 五个 blob 同哈希）与四条证据断言（六族、吞吐、L50、哈希suite）。仓库脚本与`~/pvz-logs`副本未改动，未触碰 qc。

- [6c-证据适用性，2026-10-26 19:15 CST] 核对门禁证据能否套用到 deploy14c 代码：`command_for` 在部署树与 cq 当前代码中**逐字节相同**（AST 源片段 SHA 一致）；`search_play.py` 与 `pvz_tasks.py` 的差异全部在 `deck_random` 分支（`deck_for_seed` 无该字段时提前返回原路径），四个生产清单（stage4 倍率、boss_l50、roof、fog）`deck_random` 出现次数均为 0。因此六族与 L50 的门禁命令、牌组与生产实际一致，build2 的门禁结论可直接用于部署；`select_student` / `train_rollout_policy` 不在引擎对局路径上。

- [6b-代码基线复核，2026-10-26 19:05 CST] 发现生产代码在 16:15 已由 deploy14c 前移：`cq` 的 `scripts python recipes configs` 相对 deploy14fix `0b8dcc26` 恰好只差五个文件（curriculum_queue / pvz_tasks / search_play / select_student / train_rollout_policy），且五个 blob 与 `fd1a0b61` 逐一相同；`qc_setup.log` 记录 deploy14c 16:15:52 用该代码重启、引擎/面板/rates/state 未动。原脚本的`diff --quiet 0b8dcc26`守卫会因此拒绝部署，若照 deploy14 结构checkout则会把 deploy14c 撤掉。已改为只读断言：改动集必须恰好这五个文件、五个 blob 必须等于 `fd1a0b61` 的版本，不 checkout 代码；在真实 `cq` 上逐条验证通过。脚本仍只写未执行。

- [3b-确定性完成，2026-10-26 18:46 CST] 六族正式任务确定性核验PASS：候选83aed3be在6/6族（每族8独立进程+同进程4次reset，0/60/660tick）均为1条哈希线且两模式一致；旧引擎同法6/6族不稳定（多进程2–4种、同进程1–2种）。deck由A2清单按地形推导（pool含16、roof含33、fog含16）。证据`artifacts/task95/state_hash_suite_build2.json`，汇总写入`MEMORY/docs/task95-engine-gate.json`的`six_family_hash_suite`。

- [6-定稿，2026-10-26 18:47 CST] `scripts/qc_deploy15.sh`切到最终候选build2：`ENGINE`指向`pvz-portable-build2`、`ENGINE_SHA` 83aed3be，前置检查改读`gate_b2/gate_evidence.json`、`abba3/throughput_abba.json`，并新增六族哈希suite断言（pass、6族stable、引擎路径一致）。按脚本参数单独运行前置Python块通过（只读，未执行部署），两份脚本`bash -n`通过且内容一致，已同步`~/pvz-logs/qc_deploy15.sh`；**仍只写未执行**。

- [5完成，2026-10-26 18:45 CST] 候选83aed3be在boss_l50_orig_eval上四种子85000000..3对拍PASS：4/4逐帧动作流、全部逻辑字段与终局一致（final_tick 5895–7720，均输但terminal=true）；候选Boss块存在、七个字段齐全（107–144帧/局，动作head/idle/summon），旧引擎无Boss块。证据`artifacts/task95/boss_l50.json`。

- [3b完成，2026-10-26 18:44 CST] 候选83aed3be六族门禁96局完成：6族×老师/学生×8种子，完整动作流与逻辑字段0差异，PASS。旧臂36个games.jsonl与192个帧档经`diff -rq`确认与原始门禁副本逐字节一致后复用，未重跑；state_hash差96/96属task82初始化字段变化。证据`artifacts/task95/gate_b2/gate_evidence.json`。同一驱动已顺序启动L50四种子对拍。

- [3b-验证链续接] 六族确定性补充检查已接到现有六族门禁→L50作业之后：等待已核实的验证主进程退出，核对两阶段均结束且没有本任务残留游戏进程，再串行运行task95_hash_suite.py。状态写入artifacts/task95/hash_suite_queue_status.json；不会与当前2个游戏进程并跑。

- [3b-确定性范围准备] 新增task95_hash_suite.py，沿用A2六族的正式清单、首个种子及profile/deck推导，不复用固定草地卡组。每族对旧/新引擎各做8个独立进程、同进程4次reset，并比较reset与两段wait后的全部哈希/实际tick轨迹；候选还要求两种进程模式相互一致。4项纯比较测试通过，当前不启动游戏，等六族门禁及L50退出后串行运行此补充核验。

- [3b首组核验] 83aed3be白天老师组8局已独立提前比较：8对完整逐帧动作流、全部逻辑字段均0差异，未使用截断前缀比较。剩余组继续运行，最终通过与否仍等全部96对的完整门禁。

- [6-候选绑定，2026-10-26 17:08 CST] 部署前置检查增加实际引擎路径/Boss运行SHA绑定，并从16对原始用时重新计算吞吐门槛，避免把07703a57的证据混用于83aed3be。仍保持旧候选路径与失败证据，最终候选全部验收后才统一切换；外部脚本同步，仅做语法检查，不执行。

- [4b完成，2026-10-26 17:03 CST] 共享库候选83aed3be补齐完整16对串行ABBA，逐对old/new中位数0.9934474352≥0.98，决策数16/16相同，PASS。接续保留原9对及第10对new臂，未重测已完成记录；三次完整实测及输入指纹见task95-throughput.json，原07703a57两次失败均保留。
- [3b进行中，2026-10-26 17:03 CST] 83aed3be同进程/多进程探针各1种哈希，通过。gate_b2的old臂36个文件、frames_old的192个文件与原始门禁副本逐字节一致；复用96局旧结果，已启动候选96局（workers=2），后续按序运行L50完整对拍，本任务始终最多2个游戏进程。

- [7-进行中，2026-10-26 12:14 CST] task70新增B9节，已写入源差异、PGO归档指纹、96对零差异、确定性探针及仅换引擎脚本的证据；吞吐与L50明确标记待实测，不提前判定候选通过。

- [6-复核] 部署脚本已纳入本分支scripts/qc_deploy15.sh并同步到指定外部路径，两份bash -n通过、内容一致，未执行。以当前qc_deploy14.sh的deploy14fix 0b8dcc26作为代码基线（保留seeds-exhausted修复，不退回早期7722c0d1）；仅复制已留档07703a57引擎，生产代码/面板/rates均不改。执行前强制检查六族门禁、严格吞吐、4种子Boss和哈希探针，首组判定必须出现新增rc记录；任一门禁未完成或失败即拒绝部署。

- [5-准备] 修正既有L50脚本对旧引擎也强制检查新Boss字段的问题，并改为从boss_l50_orig_eval正式清单取4种子，用与A2相同的dw4i_s1、beta=0生产形态命令完整对拍；检查候选Boss块、逐帧动作与全部逻辑字段，不再只等待至输局。旧字段缺失、候选字段缺失、终局/动作差异等4项纯单测通过；等吞吐阶段结束后运行，避免本任务负载相互干扰。

- [3] 接手续做时发现六族×老师/学生×8种子的192局已经落盘，复用现有动作归档重跑纯比较，96对逻辑字段与动作流全部0差异，不重跑对局。对照76ff36dc已核实并复制到本任务留档目录，新引擎仍为07703a57；跨新旧完整state_hash差96/96属于初始化字段变化，不作为逻辑失败。保留已完成的新引擎探针（多进程/同进程各1条哈希线），补存旧引擎对照记录（此次多进程3条、同进程4条）。结论及指纹见task95-engine-gate.json。
- [4-准备] 新增task95_throughput_abba.py，原样调用A4的16对串行ABBA工作负载，按每对old_seconds/new_seconds的中位数≥0.98验收，并要求决策数相同；不沿用历史new/old≤1.03的较松门槛。阈值、配对完整性与决策差异等5项纯单测通过。

- [1] src/ 改动清单（`git diff a8d00858 origin/pvz-env -- src`，7 文件 +137/−11），逐项影响判定（预期全部为“无”，证实）：
  1. **task82 三处未初始化字段**：`Challenge.cpp` `mCloudsCounter[6]=0`（SaveGame 每种模式都序列化该数组；TreeOfWisdomInit 在首次 update 前赋玩法值 → 纯哈希稳定性，无逻辑影响）；`Coin.cpp` `mCollectX/Y=0`（Collect() 用前必赋值；快照保存未收集金币 → 仅哈希）；`Projectile.cpp` `mLastPortalX=-1`（冒险局只有 SaveGame 读；DataArray 槽复用 → 仅哈希）。效果：同二进制跨进程 state_hash 稳定。
  2. **task85 Boss 观测**：`EnvironmentRollout.cpp/.h` 新增 observation=6（v2boss）布局（棋盘 +4 Boss 空间通道=14、标量 +2=9），只在加载 observation=6 的网络时激活——生产网络全部 ≤5，行为不变；头部校验放宽 observation ≤6。`LawnApp.cpp` `WriteObservationBoss` 给 lean / v2 / 完整观测追加 `"boss":{...}` JSON 块（仅 L50 `GetBossZombie()` 非空时输出）；特权僵尸观测加增量字段 `summon_counter`（task81 投石车弹药，纯增量，Python 不校验版本号）。
  3. **main.cpp**：DEMO_INPUT 块纯缩进变化；新增 `DEMO_CLICK_PACKET` / `DEMO_CLICK_CELL` 测试缝（只在 demo && visible 会话生效，rollout/评估协议不用）。
  对六族门禁与全部现行任务的预期影响：逻辑零。
- [2] PGO 两阶段构建完成（build_pgo.sh 流程；phase1 插桩 + bench workload 141 份 .gcda + phase2 profile-use，接续原脚本中断处手动跑完 workload 与 phase2，flags 与脚本一致）。新引擎 SHA256 `07703a5781e70d2c3b1b3ad7b0c473ddfaf2a313ac2ce4f5724035ba44e184be`，**构建完成立即留档** `artifacts/task95/engine/pvz-portable`（SHA 已核对一致）。
- [6] `~/pvz-logs/qc_deploy15.sh` 已写（只写未执行）：结构照 qc_deploy14.sh，仅换引擎（ENGINE_SHA 07703a57…，preflight 校验 `artifacts/task95/gate/gate_evidence.json` pass）；代码 REV 7722c0d1、469 任务面板、teacher_rates 全部沿用 deploy14，回滚目标改为 deploy14（build5 76ff36dc）。
- [3-探针] `scripts/task95_state_hash_probe.py`（新增）：同任务（L7 seed30000）8 个独立进程 reset + 2 个等待点、另加同进程连续 4 次 reset。**新引擎：多进程 8/8 全同、同进程 4/4 全同（各 1 种哈希）**；旧引擎 76ff36dc 对照：多进程 8 次 4 种哈希线、同进程 4 次 4 种——task82 之前的不稳定完整复现，task82 效果确认。证据 `artifacts/task95/state_hash_probe.json`。

- [4] 吞吐 ABBA 实测**未达标**，两次独立测量一致：首跑（12:41，原 A4 工作目录）16 对中位 old/new **0.9477**；重跑（15:20–15:56，新目录 `artifacts/task95/abba2/`，已确认每局都是重新跑的，不是复用旧记录）中位 **0.9688**（new/old 1.0322，min 0.846 / max 1.127）。两次均 <0.98，16/16 局决策数相同。证据 `artifacts/task95/throughput_abba.json`、`artifacts/task95/abba2/throughput_abba.json`。
- [4-归因] 排除源码回归：逐函数对比两引擎的机器码大小，游戏代码逐项相同（`Board::Update` 0xa25 相同、`Plant::Update` 0x1367 相同、`RunEnvironment` 0x31b7→0x31ed、`Zombie::Update` 0xde3→0xac8），mimalloc 159 个符号逐个同尺寸。差异在**构建配置**：`/usr/local/lib/libopenmpt.a`（10-25 17:02 新装）让 `find_library` 选中静态 libopenmpt，候选 68.7 MB / .text 4.55 MB，deploy14 是 10.4 MB / 2.48 MB，且候选不再 `NEEDED libopenmpt.so.0`。bench 引擎净吞吐（低负载时段 8 对）old 232 688 / new 235 859 ticks·CPU-s⁻¹，说明慢的部分来自负载下的布局/内存足迹，不是每 tick 的指令数。
- [2b] 按 deploy14 的链接配置重建对照候选：`build2`，显式 `-DOPENMPT_LIBRARY=/usr/lib/x86_64-linux-gnu/libopenmpt.so`，PGO 两阶段同 build_pgo.sh 流程（插桩 → bench workload → 141 份 .gcda → profile-use），profile 目录 `artifacts/perf/pgo2`。SHA256 `83aed3bed2b74392b20367915d84908faf9384a986a115893baecdccf65f2029`，10.65 MB / .text 2.70 MB，`NEEDED` 与 deploy14 完全一致（含 `libopenmpt.so.0`、`libz.so.1`）；构建完成即留档 `artifacts/task95/engine/pvz-portable-build2`。引擎净吞吐配对 bench（8 次交替）old 282 762 / build2 281 321 ticks·CPU-s⁻¹（比 0.995）。

- [2b-吞吐，进行中被叫停] build2 对 deploy14 的 ABBA 跑到 **7/16 对**（全是 day 段）时主控通知 Codex 额度恢复、要求交还：中位 new/old **1.004**（即 old/new 0.996），7 对决策数全同；pool 段未跑，**不作判定**。部分记录留在 `artifacts/task95/abba3/run.log`。后台进程已停，未留下本任务的 pvz 进程。

## 问题

- **[Boss 向量观测不一致，2026-10-26 20:05 CST]** 核对 `observation=6`（v2boss）时发现 C++ 与 Python 通道数不一致，且 C++ 编码时丢掉 Boss 通道：
  - C++ `src/EnvironmentRollout.cpp:102` 按 `Channels+14` 读取首层权重、`:163` 按 106 通道分配 buffer、`:203` Boss 起点 `Channels+9` 只写到 `+12`（4 个通道），`+13` 空置；但 `:477` `Encode()` 首层卷积传入 `Channels+9`（101），Boss 的 4 个空间通道从不进入网络，且 kernel 的 `target*channels` 步长与 106 通道权重不匹配。
  - Python `python/pvz_student_observation.py:8` 定义 `v2boss=(13,9)`，`:67-77` 写 4 个 Boss 通道（9..12），首层权重 105 通道；`MEMORY/docs/boss-l50.md` 也写“Python 与 C++ 使用 4 个 Boss 空间通道”。
  - 后果：Python 训出的 v2boss 模型（105 通道）在引擎里会因权重多读 1 个通道被拒；即便手工做 106 通道模型，`Encode` 只用 101，Boss 特征被丢弃、卷积索引错位。**observation=6 目前不可用**。
  - 影响与处置：当前无任何 obs=6 模型（`dw4i_s1` 是 v2/101，两个 value 网 92，artifacts 内无 v2boss），生产网络 ≤5，因此 build2 对现有策略无影响，六族与 L50 门禁结论不变；但 task85 的“L50 学生观测”尚不能用。修复需改 C++ 三处 14→13 与 `Encode` 的通道数，再重建 PGO 并重跑六族+L50+吞吐，属于新任务；本次不擅自改源码，否则作废已门禁的二进制与全部证据。

- [代码基线前移，2026-10-26 19:05 CST] task95 脚本 17:08 定稿后，deploy14c 在 16:15 已把生产代码从 deploy14fix 换成 `0b8dcc26`+`fd1a0b61` 五文件（日志时间戳早于定稿，但当时未核对）。因此“代码沿用 deploy14”的字面做法（checkout `0b8dcc26`）现在会**回退**正在运行的 deploy14c。按“只换引擎”的本意，deploy15 保留当前生产代码、只换引擎；守卫改为断言当前代码恰为 deploy14c 状态。新引擎的六族/L50/吞吐证据是在 deploy15 树的固定 A2 夹具上取得的，与 cq 的策略侧代码无关；若主控希望先回退到 deploy14fix 再部署，需要另跑门禁，不能直接执行本脚本。

- [续做核实，2026-10-26 16:55 CST] 磁盘实际已有9对完整build2 ABBA记录及第10对new臂，已校验build2与归档均为83aed3be并续跑，不重采已有记录。静态/动态libopenmpt链接差异已经确认，但函数尺寸相同不能单独排除源码性能回归；最终仍以build2完整16对吞吐和独立逻辑门禁为准。

- **待办已清空**：build2（83aed3be）六族96对0差异、L50四种子、六族哈希suite、16对吞吐全部通过；B9、结论指纹与`qc_deploy15.sh`的候选SHA/证据路径均已切到build2。脚本仍只写未执行，等主控决定部署窗口。
- **吞吐差 3–5%，两次 ABBA 都不过 0.98（已解决）。** 已定位到构建配置（静态 libopenmpt 把二进制从 10.4 MB 撑到 68.7 MB），不是 task82/task85 的源码行为；照 deploy14 的共享库链接重建 build2（83aed3be）后，重跑六族逐位门禁、L50、六族哈希suite与完整16对ABBA，全部通过。原始 07703a57 仍留档，不覆盖。
