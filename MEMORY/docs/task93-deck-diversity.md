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

## 问题
