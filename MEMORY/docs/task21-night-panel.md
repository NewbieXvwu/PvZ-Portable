# task21（主控 2026-10-21 下发给台式机 CodeBuddy 第二条线）：夜间 L11–L20 原版难度评估面板

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8 卡组与地形），再读 `MEMORY/docs/task17-day-panel.md`（白天面板，本任务照它的做法）、
`MEMORY/docs/day-panel.md` 与 DESIGN §14.7。

## 背景
“加宽”阶段：一个模型、原版难度、按地形配卡组，打通冒险模式。白天面板已有（task17），学生在白天 7 卡上 L1–L10 基本都在 87% 以上。
下一个区是夜间草地（11–20）：天上不掉阳光、有墓碑格、蘑菇醒着。先要知道现在站在哪：现役学生与老师在每一关原版难度下各赢多少。
学生从没见过夜间关和阳光菇，数字低是预期的，这一步只是量出起点、把面板搭好。

本任务与 task20（另一个代理在 `conveyor` 工作树做传送带）**并行**，互不依赖。

## 要做的
工作区（台式机）：`~/PvZ-search/.claude/worktrees/night`（分支 `night`，主控已从 origin/pvz-env 建好），只在这里改代码、单独编译
（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。

1. **清单**：`scripts/make_day_panel.py` 加夜间关（或写 `scripts/make_night_panel.py`，能复用就复用），生成 `artifacts/stage4/night_panel.json`（不进 git）：
   每关一个评估任务 `night_lN_eval`（倍率 1.0、不限波、playthrough 2，其余字段照白天面板），卡组用 `deck_for_level`（不要另写）。
   种子用评估段 30000–31023（每关 1024 个）。
   先查清每一关：是不是夜间草地（`pvz_constants.background_for_level`）、有没有特殊玩法（例如 2-5 / 2-10 是否小游戏或传送带）、
   卡组是否只含该关已解锁的植物（按引擎实际 playthrough=2 的列表，task19 的 `DECK_OPTIONS` 可用）、墓碑格是否会让某些卡组结构性输局。
   特殊关能跑就收并在报告里写清语义，不能跑就写进报告。不要改 `deck_for_level`；觉得卡组该换，写进“问题”由主控决定。
2. **跑数**（同时最多 2 个游戏进程：台式机采集队列与另一个代理也在用 CPU）：
   - 学生 v34（`~/PvZ-search/artifacts/student/v34/model.pt`，slot 头）与 identity 学生
     （`~/PvZ-search/.claude/worktrees/identity/artifacts/task19/identity_dw3_s1/model.pt`）：每关 128 局；
   - 老师（学生 v34 + 搜索，参数照 `~/pvz-logs/queue100.sh` 里采集轮的写法）：每关 16 局，只为知道老师在各关的上限。
   用 `scripts/eval_panel.py` / `scripts/search_play.py`，Python 用 `~/.venvs/ml/bin/python3`，资源目录 `~/.cache/pvz-research-resources`。
3. 在 DESIGN §14 下登记新任务族 `night_l*_eval`（AGENTS §8 第 4 条）。

## 交付
- 结果写 `MEMORY/docs/night-panel.md`：每关一行（关卡、卡组、特殊玩法、v34 / identity 胜局 / 128、老师胜局 / 16、平均到达波数），
  加一段“哪几关老师也打不过”（多半是卡组或地形问题，不是策略问题）和“哪几关学生明显不如老师”（下一步采集重点）。
- 分支 `night` 提交，可以 `git push origin night`；不推 pvz-env、不合并。

## 规则
- **不要动任何 tmux 队列（q100 等）与别的代理的会话；不要改 `rowpool`、`identity`、`conveyor` 工作树与 `~/PvZ-search` 主工作区。**
- 不训模型；不改默认值；不写手工规则；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；不 `git add -f`；`.pt/.npz/.log` 不进 git。
- 只用绝对路径。每完成一小步就提交，并在本文末尾“结果”里记一行做到哪；疑问写“问题”一节，按最佳判断继续，不要停下来等。

## 结果

- 2026-10-21 清单：`scripts/make_night_panel.py` 生成 `artifacts/stage4/night_panel.json`（不进 git），9 个任务 `night_lN_eval`（L11–14、16–20，各 1024 种子 30000–31023），卡组 `deck_for_level` = `[0,1,9,2,3,4,5]`。2-5（L15）经源码核对是**打地鼠**（`LawnApp::IsWhackAZombieLevel`，Adventure `mLevel == 15`），默认排除；2-10（L20）是 mini-boss/传送带，收进来并在报告标注。

- 2026-10-21 编译与冒烟：`build/` 在本工作树单独编译成功（Release + BLAS，131 步）。冒烟 L11/L20 能跑；L15（打地鼠）能 reset 也能跑，但 2 局都是 wave 2 结束、丢 3–5 车，语义退化，确认排除。
- 2026-10-21 学生 v34（slot 头）128 局/关：**9 关全 0 胜**（L11–14、16–20）。夜间关没见过是预期，但还有一层混淆：v34 的槽位头按槽位索引出卡，夜间卡组 `[0,1,9,2,3,4,5]` 把 2 号槽从樱桃换成阳光菇、后面整体错位，见“问题”。

- 2026-10-21 学生与老师跑数：identity 学生 9 关合计 1/1152（只有 2-1 赢 1 局），老师 8 关合计 3/128（2-1 2/16、2-6 1/16，其余 0）。**师生一起卡在同一处**：平均停在 9–10 波、丢 4 车左右。报告见 `MEMORY/docs/night-panel.md`。

## 问题

## 第二步（主控 2026-10-21 22:55 追加）

### 第二步结果

- 2026-10-21 identity 学生 64 局/关（5 关）：对照 0/320、n8 4/320、n9 13/320。n9（带双发射手）最好但仍在地板；详见 `night-panel.md` §9.1。
- 2026-10-21 老师（identity 推演）16 局/关（5 关）：对照 3/80（3.8%）、n8 44/80（55.0%）、n9 58/80（72.5%）。**卡组是第一步“老师也打不过”的原因**；n9 上学生 4.1% 对老师 72.5%，差约 68pp，是下一步采集第一优先。详见 `night-panel.md` §9.2–9.3。

- 2026-10-21 第二步清单：`make_night_panel.py` 加 `--deck/--suffix`，生成 `artifacts/stage4/night_panel_n8.json`（`night_lN_eval_n8` = `[0,1,2,3,4,5,8,9]`）与 `night_panel_n9.json`（`night_lN_eval_n9` = `[0,1,2,3,4,5,7,8,9]`）；三套卡组同种子段 30000–31023，`deck_for_level` 未改。DESIGN 我的登记改为 §14.9（§14.8 留给 task20）。第一步的老师 L20 16 局（v34 推演）作废并停止，改 identity 推演。

报告很好。两点更正与下一步：
1. **老师的推演策略要换成 identity 学生**：第一步老师 = v34 + 搜索，搜索的推演用的是 v34（slot 头），在夜间卡组上同样槽位错位，
   所以“老师也打不过”混着错位效应。之后老师一律用 identity 学生做 `--rollout-policy`
   （`~/PvZ-search/.claude/worktrees/identity/artifacts/task19/identity_dw3_s1/model.pt`，C++ CPU 前向已支持 PVZRN006；CUDA 不支持，别开 CUDA 推演）。
2. **卡组靠实测选，不凭感觉**：新开任务族（不覆盖 `night_l*_eval`，DESIGN 登记；你和 task20 都写了 §14.8，你这边改成 §14.9），候选：
   - `night_lN_eval_n8`：`[0,1,2,3,4,5,8,9]`（白天 6 卡 + 小喷菇 + 阳光菇）
   - `night_lN_eval_n9`：`[0,1,2,3,4,5,7,8,9]`（学生训练卡组 + 小喷菇 + 阳光菇）
   - 现有 `night_lN_eval`（`[0,1,9,2,3,4,5]`）作对照
   卡组照 AGENTS §8 核对解锁与所有权；种子同评估段 30000–31023。
3. **跑数**（同时最多 2 个游戏进程）：三套卡组 × L11–L14、L16 × 老师（identity 推演）16 局，再加 identity 学生 64 局。
   表格加进 `MEMORY/docs/night-panel.md` 新的一节，写清哪套卡组老师赢得最多、学生离老师多远。
   不改 `deck_for_level`；最后选哪套由主控定。
