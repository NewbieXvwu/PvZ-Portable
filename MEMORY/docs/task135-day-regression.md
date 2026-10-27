# task135：r26 学生在门禁 v5 白天区退步的诊断

主控 2026-10-28 派给 Mac codex-plus2。只读分析，不训练、不部署。

## 背景

- 门禁 v5（只看冒险模式原版难度、记分板卡组）首评 r26：overall +0.4pp 通过；唯一否决是 `significant_zone_regression:day`：day 359 vs 383（gain 24 / harm 48，414 对）。night −2.7pp，pool +8.4pp，fog +1.5pp。
- 数据在台式机 `~/pvz-logs/curriculum_state.pipeline/round_000026/`（`complete.json`、`evaluations/`、`evaluation.log`、`selection.log`、`candidates.json`），在役学生是 r11。r27 的评估约 04:00 出来，也可以一起看。

## 要回答

1. 白天的 48 个 harm 集中在哪些关、哪些种子：逐关列出 pairs / gain / harm。
2. 输掉的局是怎么输的：抽几局 harm 局，用现有工具（`episode_query` / `render_episode` 的文字版）对比新旧学生同种子的关键决策差异，比如开局经济、某类植物用量。早先发现新学生种三叶草 3558 次、卷心菜 1346 次，r11 分别是 2754 次和 5147 次，看这是否仍成立、是否就是原因。
3. 这是训练数据配比造成的系统性退步，还是候选间的种子抽签：r26 有 4 个候选，`selection.log` 里各候选的白天成绩如何，被选中的是否恰好白天最差？
4. 给出 1–2 条可执行建议，只能是数据 / 配比 / 挑选层面，不要手写策略。

## 约束

- 台式机是生产机：只读文件，最多 2 个游戏进程，跑之前看 `free -g`，测试限内存（AGENTS §6.5 第 7 条）。能在 Mac 上跑的就在 Mac 上跑（把需要的模型和存档拷过来）。
- 结论写在本文件末尾，推送分支 `dayreg`。
