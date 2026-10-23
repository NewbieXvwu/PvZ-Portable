# task43：价值网 v5——泳池补决策级标签，让叶值在全败区重新随局面变化

> 主控 2026-10-23 16:50（台式机时间）写。执行者：台式机 agent_switch 链（Codex 18:16 回来前由 CodeBuddy cb-cuda 先做）。
> 工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/v5`（分支 `v5`，从 pvz-env e59c26ca 拉出，主控已建好）。
> 同时最多 **4** 个你自己的游戏进程（qc 采集 16–18 进程在跑）。不碰 tmux `qc` 与别的代理会话，不重启 WSL。

## 为什么
- 价值网 v4 10-23 起是老师叶子的唯一默认（DESIGN §14.17）。依据和已知限制都在 `MEMORY/docs/value-widen.md` 第 5 步。
- 限制是：在全败的泳池局面（L24/L26），v4 给所有候选的叶值都是 ≈0.002，候选间 std 只有 0.0007，是 v3 的 1/49。于是搜索从不推翻学生（0%，v3 是 19.8%），等于老师在这些局面里不教东西。
- 根因：泳池整局胜负标签几乎全是负（0/256），常数标签教不出排序。修法是换**标签粒度**，不是改公式。
- 计划已经写好：`value-widen.md` 末尾"v5：泳池决策级标签"一节。**先通读那一节，照它做**；本任务书只补执行细节和门槛。

## 步骤
1. **找现成通道，不新写采集器**：
   - 分叉 / 决策级数据的做法在 DESIGN §14.6 与 `scripts/value_ranking_check.py --dump`、`scripts/generate_value_data.py`。先读懂它们怎样在一个搜索决策上把每个候选推演到终局、记实际胜率。
   - 再读 `scripts/train_value_widen.py`（v4 的训练入口）与 `python/pvz_value_net.py` 的编码 v2，确认决策级样本能用同一编码进训练。
   - 写一段"通道说明"进本文"结果"并提交。
2. **L21 可行性采集**（泳池可赢关，任务族见 `scripts/make_multiplier_tasks.py` 的 `FAMILIES` 与 DESIGN §14.15）：
   - 老师配置照 qc 的 `command_for`（读 `scripts/curriculum_queue.py`，不要猜），叶子用 v4；
   - 先 4 个种子冒烟，确认每个决策都记下了候选和终局结果；
   - 再采到 ≥ 64 个种子，按计划估约 1.3 万决策、3–5 万候选-终局对；
   - 数据放 `artifacts/task43/`，不进 git。
   - **中途检查**：用这批数据离线看 v4 在 L21 决策上的候选排序（成对一致率）。如果排序已经不错，说明问题只在全败区，记下来继续。
3. **训练 v5**：`train_value_widen.py --init value_net/v4` 热启动。
   - 泳池决策级对与既有各区数据混训，泳池样本先 1:1；
   - 验证集报成对一致率（分区报），同时报白天、夜间的外部校准不掉；
   - 产物放 `artifacts/task43/v5/model.pt`，不进 git；
   - 训练按 `MemAvailable ≥ 10 GiB` 门禁，GPU 可用就用。
4. **验收**（同第 5 步的尺子，三条全过才算）：
   - (a) 泳池 L24/L26 各 4 局，`search_play.py --leaf-probe`：候选间 std ≥ 0.02，top1−top2 中位数超过 0.02 的推翻阈值；
   - (b) 泳池 L21，v5 对 v4 同种子 32 对：净胜局 ≥ −2，且配对检验不显著变差；
   - (c) 白天 L7 d7 与夜间 L14 n10 各 32 对：判据同 (b)。
   - 工具用 `scripts/value_widen_ab.py` 与 `scripts/value_leaf_dispersion.py`（v4 用过的那套），种子用新段：评估段 31000 起。
5. **结论**：
   - 三条全过：把 v5 上传 HF `transfer/value_v5.pt`（`scripts/hf_sync.py` 或 `value_widen_publish.py`，下载回验 SHA256），在"结果"写建议。**不改默认值**，换默认由主控做。
   - 任一条不过：写清是哪条、差多少，以及你认为下一步该补什么数据。

## 规则
- 只用绝对路径，不 `cd` 进别的工作树。`PVZ_NO_POPUP=1`；跑测试要设 `PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources`。
- 引擎用 `/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy2/build/pvz-portable`（PGO 构建，与 qc 同一份）。**不要在本工作树之外编译**；要编自己的就编本工作树的 `build/`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；不手写规则或公式；不加开关（`--leaf-probe` 是现成的诊断，可以用）。
- 台式机 C 盘只剩约 35 GB。中间数据放 `artifacts/task43/`，每步结束删能重新生成的大文件，并在"结果"写占用。
- 每完成一小步就提交、`git push origin v5`，在"结果"记一行。疑问写"问题"，按最佳判断继续，不要停下来等。

## 结果

## 问题
