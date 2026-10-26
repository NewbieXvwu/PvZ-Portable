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

## 问题

## 主控追加（2026-10-26 08:20，task93b：Mac 侧做倍率扫描并扩采）
执行者：Mac Codex（plus2 号，tmux `codex-plus2b`）。工作树是 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/deckvar-mac`（分支 `deckvar-mac`，从 origin/deckvar 拉出）。台式机的三族清单已拷到该树的 `artifacts/stage4/random_deck_panel.json`。Mac 上每个任务最多 6 个游戏进程。台式机那边的第 4 步验证由 Business 号 Codex 继续做，**两边不重复**。
1. **夜间 / 泳池倍率扫描**：照第 3 步方案，对 `night_l12_train_rand` 与 `pool_l21_train_rand` 各扫 0.5 倍和 0.75 倍。
   - 每档 16 局，生产老师，学生 r11；
   - r11 路径：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/fog36/artifacts/task79/models/r11/model.pt`；
   - 叶值 v5：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/hard/value_v5/model.pt`；
   - branch1：`/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt`。
   报胜率与 Wilson 区间，按 task75b 的规则定起步档。
2. **扩采**：
   - 三族各再采 192 局老师 frames。白天 1.0 倍；夜间、泳池用第 1 步定的档。
   - 种子用 93_000_000 段里台式机没用过的子段：从 93_x05_000 起，先机械核对台式机清单与已用段。
   - 照 Mac 现有的标签导入管道打包，经 HF `transfer/` 送到台式机（参考 `scripts/import_mac_labels.py` 的命名规则）。只传数值标签，台式机取用后删除。
3. 结果写回本任务书的“结果”节，标明 Mac 侧；`git push origin deckvar-mac`。合并由主控做。
