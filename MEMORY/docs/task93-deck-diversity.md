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

## 问题
