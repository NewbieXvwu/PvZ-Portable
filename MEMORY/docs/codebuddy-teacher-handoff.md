# 交接书：老师侧两个对照——坚果包扎术、把学生的次选加进候选（计划第 11 项改方向）

写给台式机 WSL 上的 CodeBuddy（2026-10-18，主控 Claude 起草）。先读仓库根目录 `AGENTS.md`（体积与交付规则，**优先级最高**），
再读 `MEMORY/docs/codex-shovel-report.md`（铲子第一阶段报告，你的前一个会话写的）和
`MEMORY/docs/student-bottleneck-plan.md` 末尾 2026-10-18 10:30 / 10:50 两条（为什么改方向）。

## 0. 开工前

1. 机器：台式机 WSL，主仓库 `~/PvZ-search`（**只读**，别改它的工作区、别重编它的 `build/`）。
   新建 worktree：`cd ~/PvZ-search && git fetch origin && git worktree add .claude/worktrees/teacher -b teacher origin/pvz-env`，
   在 worktree 里单独编译（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j3`）。
   学生 v25 用观测 v2，**必须用新编的二进制**（主仓库 `build/` 是旧的）。**只在 `teacher` 分支本地提交；不 push，不合并。**
2. 文件（绝对路径，worktree 里的 `artifacts/` 是空的）：
   - 任务清单 `~/PvZ-search/artifacts/stage4/day_l7_hard.json`（已有 `day_l7_z18_eval` / `z19_eval` / `z20_eval`：1.8 / 1.9 / 2.0 倍，种子 30000–31023）
   - 学生 v25 `~/PvZ-search/artifacts/student/v25/model.pt`
   - 叶子价值网络 `~/PvZ-search/artifacts/value_net/v3/model.pt`，静态价值网络 `~/PvZ-search/artifacts/hard/value_branch1/model.pt`
   - 资源目录 `~/.cache/pvz-research-resources`
3. **资源纪律**：台式机同时在跑正式采集与训练（负载 15–20）。你同时最多 **4 个游戏进程**，不碰 GPU。不要动 `~/pvz-logs` 下除
   `codebuddy_teacher_questions.md`（有问题写这里）以外的任何文件。
4. 老师参数一律：`--rollout-policy <v25> --verify-samples 9 --value <v3> --static-value <branch1>`，其余默认；`--arms fast_search`。

## 1. 第一步：找一个老师自己也会输的压力档

老师（v25 续走）在 1.7 倍已赢约 98%，对照分不出好坏。先在 `day_l7_z19_eval` 和 `day_l7_z20_eval` 各跑前 64 个种子（`--seeds 64`），
选老师胜率最接近 70% 的那一档（都高于 90% 就再加 2.2 倍：照样在本地清单里加 `day_l7_z22_eval`，并在报告里写明）。后面两个对照都在这一档、
同一批 128 个种子（30000–30127）上配对。

## 2. 对照 A：坚果包扎术（零新动作）

- 原版机制：买了包扎术后，坚果血量低于 2/3 时可以在它身上种一个新坚果（`Board::CanPlantAt`，`src/Lawn/Board.cpp` 约 2827 行起）。
  环境已支持：`python/pvz_env.py` 的 `PlayerProfileContext.first_aid_owned`；合法动作列表走游戏自己的 `CanPlantAt`，打开后会自动出现。
- 要做的是**接线**：任务清单里加一个字段（例如 `"first_aid_owned": true`），让 `scripts/render_episode.py` 的
  `load_training_task` / `training_task_kwargs` 与各处 `profile_for_deck(...)` 的调用把它带进 profile（搜索 `profile_for_deck` 找全调用点：
  `search_play.py`、`render_episode.py`、`backfill_observation_v2.py`、`watch_policy.py` 等）。字段缺省 = 关闭，旧任务逐位不变。
- 这是**改任务**：按 `AGENTS.md` §8 新开任务名（例如 `day_l7fa_z20_eval`，只差包扎术），在本地清单里加上，并在 `DESIGN.md` 任务清单处登记（在你的分支里改）。
- 检查学生 / 老师 / 价值网络的特征代码有没有“一个格子只有一株植物”或“只能种空格”的假设（例如学生的格子合法性掩码），有就报告并修。
- 验收：
  - 关闭时（旧任务）同种子逐局一致：胜负、决策数、推翻数。
  - 选定压力档上 128 局配对（老师，包扎术开 vs 关）：胜局、各自独胜、McNemar p、丢车、每局用包扎术的次数（在坚果上种坚果）、耗时。
  - 学生 v25 自己（`--arms fast`）在两种任务上各 1024 局的胜局（看学生不经训练会不会用、有没有被新合法动作搞乱）。

## 3. 对照 B：把学生自己的前几名种植动作加进老师候选

- 背景：铲子实验里老师常拿铲子当“杠杆”——铲一下扰动棋盘，让学生在下一步换个反应去种。根子是老师的种植候选太窄：
  每张卡只取静态价值网络挑的 1 个格子（`--per-card 1`），学生自己想种的别的格子 / 学生的次选都不在候选里。
- 要做的是 `scripts/search_play.py` 加开关（例如 `--student-candidates k`，默认 0 = 关闭，关闭时旧行为逐位不变）：
  把学生策略里概率最高的 k 个**种植**动作（不论学生最终选的是不是等待）加进候选，去重后交给续走比较。
  学生的打分接口在 `python/pvz_rollout_policy.py`（`act()` 附近），注意 v2 学生要带 `StudentHistory`（看 `watch_policy.py` / `render_episode.py` 的用法）。
- 验收：
  - 关闭时同种子逐局一致。
  - 选定压力档上 128 局配对：k = 2 vs 关闭（时间够再加 k = 4）。报告胜局、独胜、McNemar p、丢车、每局搜索次数 / 推翻次数 / 耗时，
    以及“推翻落在学生候选上”的次数（新候选到底被选中多少次）。
  - 注意赢家诅咒：候选越多，越容易选中被高估的那个；如果胜局没涨、推翻却大涨，就是这个信号。

## 4. 文档（中文、说人话）

- 报告写进 `MEMORY/docs/codebuddy-teacher-report.md`：改了什么（提交列表）、一致性、压力档测量、A / B 两个配对、
  2–3 个具体局面（用 `scripts/episode_query.py` 的 frame / whatif 讲清楚老师为什么这样选），以及你对下一步的建议。
- `MEMORY/2026-10-18.md`（或当天日期）追加记录。每完成一节就提交一次。
