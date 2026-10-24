# task65：被吃植物里把零费植物单独计一列

> 主控 2026-10-24 写。执行者：台式机 CodeBuddy（`cb-qcalloc` 会话）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/eaten`（分支 `eaten`，从 origin/pvz-env 拉出，主控已建好）。游戏进程最多 2 个；**不碰 tmux `qc`、不重建生产引擎 deploy3**。

## 为什么
- 用户 10-24 指出：夜间关后期，零费的小喷菇严重溢出，可以毫无顾忌地往前线种，当垫材挡僵尸。被吃掉是有意的打法，不是损失。
- 现在的"被吃植物"统计（引擎 `Board::mPlantsEaten`，env 事件 `plants_eaten`）不分费用。它出现在下面这些地方：
  - 示范 meta 与各类诊断报告的 outcome；
  - `episode_query.py whatif` 排序，在丢车之后作为第二判据；
  - `render_episode.py` 的叙述；
  - `search_play.py` 帧里的 `eaten`。
- 不分费用的结果：拿小喷菇当垫材的局会被算成"损失惨重"，`whatif` 也会把这种好打法排在后面。

## 要做的
1. **引擎**：在 env 事件里新增 `plants_eaten_free`，与 `plants_eaten` 并列。`plants_eaten` 的含义不变，是全部被吃的植物；新字段只计其中费用为 0 的。
   - 费用用游戏自己的函数判断，例如 `Plant::GetCost` / `GetCostForSeedType`，按种植时实际的费用算，传送带关卡另行处理；不写死植物编号。
   - 在 `mPlantsEaten++` 的各处（Board.cpp、Zombie.cpp，共约 5 处）旁边计数。
   - **不改存档格式**：新计数不进 SaveGame，存档哈希不变。参照 `branchAfter/branchBefore` 的差分做法，在每步、每个分支前后取差。要证明快照恢复、分支推演下差分仍然正确，并写测试。
2. **Python 侧**：
   - 事件字典、outcome、`totals` 里都带上 `plants_eaten_free`；
   - `human_demo.py`、`search_episode.py`、`roof_phase2.py`、`render_episode.py` 的报告里，被吃数显示为"被吃 N 株（其中零费 M）"；
   - `episode_query.py whatif` 的排序改用**付费植物被吃数**（`plants_eaten − plants_eaten_free`）作第二判据；
   - 旧存档没有新字段时，按旧口径处理，写明。
3. **验证**：用用户两局示范的动作带重放，报出 `plants_eaten` / `plants_eaten_free` 两个数：
   - `demo_fog_31_fume_rebaselined`，原 10 株；
   - `demo_fog_31_snow_m200`，原 10 株。

   两局存档在 Mac：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/human_demo/raw/seed_20261031/`。需要时从 Mac 拷过来（只读，几 MB），或者在台式机上用同样的 task 设置重放；拷不过来就在"问题"里写明，由主控在 Mac 上跑。
4. 测试全过，push origin eaten，结果写本文件。**主控合并；引擎随下次部署生效。**

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则（零费由游戏自己的费用函数判定，这是统计口径，不是打法规则）。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

## 问题
