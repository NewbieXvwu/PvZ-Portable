# task29（主控 2026-10-22 17:40 下发给台式机 CodeBuddy 第二条线，task28 完成后接着做）：自动课程——按“关卡 × 倍率”调压力的采集控制器

先读仓库根目录 `AGENTS.md`（优先级最高），再读 DESIGN §3.2（自适应难度前沿）、§13.7（`day_l7_z13`…`z20` 倍率族）、§14.5 第 3 条（自动课程）、§14.6（DAgger 带锚标签），
以及 `MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条。

## 背景
加宽阶段把所有关都放回了原版难度（倍率 1.0），手写固定的采集清单（queue101–103）。结果：已经“通关”的关压力太小。用户 5 倍速看 dw4i_s1 打原版 1-10（胜率约 98%）：
乱扔樱桃炸弹（常扔在没僵尸的地方）、后期坐着不用卡、该种豌豆不种、乱种坚果。原因多半是：没有压力时任何动作都能赢，
带锚的 DAgger 标签只有在搜索优势超过 margin 时才推翻学生，所以学生的坏习惯在轻松局面里原样保留下来。白天 L7 阶段我们用 1.5–2.0 倍（z15–z20）逼出了真本事，
加宽后却忘了 DESIGN §3.2 / §14.5 约定的自动课程。本任务把它补上，并让它取代手写的采集清单。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/curr -b curr origin/pvz-env`，只在这里改代码、单独编译。

1. **倍率任务族**：先核实 `zombie_count_multiplier` 在原版传送带开局（`conveyor_start: original`）、夜间（含坟头最后一波）、泳池关上都生效且语义正确（读源码 + 冒烟，写清它放大的是什么）。
   为现有训练族生成倍率变体（新族名，如 `day_l10_orig_train_m150`；倍率档 1.25 / 1.5 / 1.75 / 2.0 / 2.5 / 3.0），覆盖：白天 `day_l{1,3,4,6,7,8,9}_train_d7`、`day_l10_orig_train`、
   夜间 `night_lN_train_n10` 与 `night_l20_train_n10_orig`、泳池 `pool_lN_train_p9k`、换卡组 `day_lN_train_d7q` / `night_lN_train_n10p`。种子段自选、与全部现有清单穷举不重叠，DESIGN 新开一节登记。
2. **课程控制器** `scripts/curriculum_queue.py`（取代手写的 queue 脚本；采集参数照 `~/pvz-logs/queue103.sh` 的 `run()`，beta=0）：
   - 状态按“关卡 × 卡组族 × 倍率”分组，记录**学生自走**（beta=0 采集局本身就是学生在打）的近期胜率（滑动窗口，样本数一起记）。
   - 每轮按 DESIGN §3.2 的元规则分配局数：优先胜率在 [0.5, 0.85] 的组；某关当前倍率胜率 > 0.85 就把这关的倍率升一档，< 0.5 就降一档（不低于 1.0）；
     原版难度（1.0）每关保留一小部分（如每轮 10–15%）防遗忘；1.0 下也 < 0.5 的难关照样采（那是老师标注的价值所在）。
   - 状态落盘（json，可续跑），每轮写一行日志：每组局数、胜率、倍率变化。停止标志文件 `~/pvz-logs/stop_curr`（轮次边界停，不能被 SIGTERM 打断子进程）。
   - 标签命名与偏移自动分配、写进状态文件，不与已用偏移冲突（读 DESIGN 登记的已用段）。帧写 `~/PvZ-search/artifacts/student/frames_v2`。
   - 先用 `--dry-run` 打出第一轮分配表（用 q101–q103 已有结果初始化胜率），写进“结果”。
3. **压力面板**：`scripts/unified_panel.py` 加 v3 项：每区挑代表关（白天 L7、L10 原版；夜间 2-2、2-8、2-10 原版；泳池两关）在 1.0 / 1.5 / 2.0 倍各 64 局，
   汇总每关的“压力上限”（胜率过 50% 的最大倍率，按 §13.7 的做法插值）。原版胜率饱和后，用它看模型是否真的变强。
4. 不开正式采集：控制器写好、dry-run 与 1 轮小规模冒烟（≤2 进程，每组 2 局）通过即可。何时用它替换 q103 由主控定（q103 第 2 轮后）。

## 规则
- **不要动任何 tmux 队列（q103 等）与别的代理会话；不要改别的工作树与 `~/PvZ-search` 主工作区**（frames_v2 只增不改；stage4 只新增）。同时最多 2 个游戏进程。
- 分支 `curr` 提交，可以 `git push origin curr`；不推 pvz-env、不合并。不训模型；不改默认值；不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。
- 元规则之外不写手工规则（不按植物 / 僵尸写任何特例）。每完成一小步就提交并在“结果”里记一行；疑问写“问题”，按最佳判断继续。

## 结果

- 2026-10-22 第 1 步（倍率语义核实 + 倍率任务族）：源码口径 `Board::SpawnZombieWave`（`src/Lawn/Board.cpp:4913`）——`aSpawnCount = std::lround(NumberZombiesInWave(wave) * multiplier)`，类型按 `mZombiesInWave[wave][i % aBaseCount]` 循环，所以放大的是**每波常规出怪数量**（`.5` 远离零，名单前几张多 0–1 次）；**不放大**最后一波之后的额外刷怪：坟头 `SpawnZombiesFromGraves`（Board.cpp:4850，每坟一只）、泳池 `SpawnZombiesFromPool`（Board.cpp:4728，固定 2–3 只）、屋顶 `SpawnZombiesFromSky`。冒烟 `scripts/smoke_zombie_multiplier.py`（同关同种子、不放植物、5 tick 步进数每波新增）：白天 L7、夜间 L12、泳池 L21、原版传送带 L10、夜间原版 L20，7 档（1.0/1.25/1.5/1.75/2.0/2.5/3.0）全部 `count == lround(base × mult)`，干净观测 19/16/14/14/14 条（被小推车碾死的波次标 dirty 不参与断言）。生成器 `scripts/make_multiplier_tasks.py`：7 个源族 37 个源任务 × 6 档 = **222 个任务**，命名 `<base>_m125..m300`，除 id/seeds/multiplier 外逐字段复制（含 `deck_perturb`）。种子段 5,000,000 起每档 500,000、每任务 2048（slot = 族序号×32 + 关卡−1），454,656 个种子；与 257,619 个清单文件 / 506,880 个已用种子穷举比对**重叠 0**，逐种子 `deck_for_seed` 卡组合法。清单 `artifacts/stage4/multiplier_panel.json`（工作树 + `~/PvZ-search/artifacts/stage4/` 各一份，9.0 MB，不进 git）；DESIGN §14.11 登记；静态测试 `python/test_multiplier_tasks.py` 4 项通过。


- 2026-10-22 第 2 步（课程控制器）：`scripts/curriculum_queue.py`（采集参数照 q103 的 `run()`：beta=0、`--verify-samples 9 --student-candidates 2`、静态价值 `value_branch1`、价值网 v3；`--workers` 上限 2，解释器默认 `~/.venvs/ml/bin/python3`）。分组 = 关卡 × 卡组族 × 倍率；1.0 档直接用源清单里的原任务（`--manifest` 按任务分别传，倍率变体用 `multiplier_panel.json`）。元规则只有 DESIGN §3.2 那一条：窗口胜率 > 0.85 升一档、< 0.5 降一档（不低于 1.0），窗口 = 每组最近 256 局；每轮预留 12% 给 1.0 档防遗忘（`guard_budget = max(per_group, 12%)`），其余优先给 [0.5, 0.85] 的格，没有边缘格就按“离 0.675 最近”排；1.0 档也 < 0.5 的难关照样采。状态 json（`--state`，默认 `~/pvz-logs/curriculum_state.json`）记窗口、倍率档、已用偏移与每轮计划，**偏移在跑之前落盘**（崩溃不重采）；停止标志 `~/pvz-logs/stop_curr` 只在两组之间检查、不 SIGTERM 子进程。**dry-run（用 q101–q103 已有结果初始化）**：37 个训练格；第一轮 240 局 / 每组 48 → 5 组：`day_l10_orig_train` 1.0（99.2%，n=256）、`night_l14_train_n10` 1.0（68.5%，n=130）、`night_l12_train_n10` 1.0（79.7%，n=128）、`night_l17_train_n10` 1.0（52.3%，n=128）、`day_l10_orig_train` 1.25（无样本）；同时 9 个格按窗口胜率升到 1.25（白天 L1/3/4/6/7/8/9、夜间 L18、原版 L10）。完整表 `artifacts/task29/dry_run_round1.txt`。**冒烟**（每组 2 局、2 进程、`--round-games 4`）：2 组 4 局全部 rc=0 —— `c1d10origm100`（day_l10_orig_train 偏移 256，2/2 胜，253 s）、`c1n14n10m100`（night_l14_train_n10 偏移 128，2/2 胜，339 s）；状态 round=1、偏移 128→130 / 256→258、窗口写入、帧写 `artifacts/task29/smoke_frames/seed_*/`，日志 `artifacts/task29/smoke.log`（正式跑默认帧目录 `~/PvZ-search/artifacts/student/frames_v2`）。单测 `python/test_curriculum_queue.py` 8 项（升降档、1.0 下限、样本不足不动、滑动窗口、12% 防遗忘份额、边缘优先、偏移耗尽报错、停止标志不启动任何子进程）全过。**未开正式采集**。

- 2026-10-22 第 3 步（v3 压力面板）：`scripts/unified_panel.py` 新增 `--generate-pressure` / `--pressure`。每区代表关：白天 L7、原版传送带 L10；夜间 2-2（L12）、2-8（L18）、原版 2-10（L20）；泳池 L21、L26 —— 7 关 × 3 档（1.0/1.5/2.0）× 64 局 = 21 个任务 / 1344 局，任务 id `press_<区>_l<关卡>_m<倍率×100>`，除 id/seeds/倍率外逐字段复制对应评估任务（`day_l7_eval_d7`、`day_l10_orig_eval`、`night_l1{2,8}_eval_n10`、`night_l20_eval_n10_orig`、`pool_l2{1,6}_eval_p9k`），种子取评估段前 64 个（30000 起，与既有评估同种子、便于配对，不新增训练种子）。压力上限按 DESIGN §13.7 插值：相邻两点线性插值求 50% 交点；最高档仍 > 50% 记 `≥`，1.0 都没过 50% 记 `—`（`pressure_ceiling`）。清单 `artifacts/task29/pressure_manifest.json`（21 项）。**冒烟**（`--pressure-seeds 2`、`--pressure-tasks press_day_l7_m100,press_day_l7_m150`、2 进程）：2 组各 2 局 rc=0，均 2/2 胜，输出表 `day L7 100%/100%/— ≥1.50`（最高档未测，按 censored 记 `≥`）。**完整 1344 局没跑**（按第 4 条不开正式采集，留给主控在替换 q103 时跑）。单测 `python/test_pressure_panel.py` 6 项（任务复制/倍率/id、两段插值、censored、1.0 未过 50%、空测点）全过。

## 问题
- **DESIGN 章节号会和 task28 撞**：本分支从 origin/pvz-env（1c983ed9）开出，那边没有 task28 的 DESIGN 条目；task28 的 n10p 分支把 §14.11 / §14.12 用在夜间扰动族与 d7q 上，本任务也用 §14.11 登记倍率族。合并时按 task28 先、task29 后的顺序把本节改成 §14.13（内容不变，只改标题编号）。
