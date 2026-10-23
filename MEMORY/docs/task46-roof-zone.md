# task46：接入屋顶区（冒险 41–49）第一阶段——族登记、屋顶动作与观测、老师冷启动实测

> 主控 2026-10-23 20:30（台式机时间）写。执行者：台式机 Codex（Business 账号，codex4）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/roof`（分支 `roof`，从 pvz-env 拉出，主控已建好）。
> 同时最多 **4** 个你自己的游戏进程。不碰 tmux `qc` 与别的代理会话。

## 为什么
- 目标是通关冒险模式。现在的课程覆盖白天、夜间、白天泳池、传送带，**屋顶一关都没有**。雾夜泳池另派 task45。
- 第 50 关（僵王）按用户 10-20 的决定放最后，本任务不做。
- 屋顶的特点：
  - 斜坡：直射植物在前几列打不到，要用投掷类；
  - 每格要花盆（本环境开局预置 c0–c4，见 AGENTS §8）；
  - 蹦极、投石车、梯子僵尸；
  - 夜里的屋顶关。
- `deck_for_level` 现在给 41–49 的是 `DECK + (FLOWER_POT,)`。它按地形覆盖，**但没有投掷植物**。这多半是结构性输局，要先实测（AGENTS §8 第 1 条：打不过先考虑卡组）。

## 步骤
1. **现状盘点**（只读）。写进本文"结果"：
   - 引擎能否无头 reset、跑完 L41–L49 一局；
   - 开局预置花盆的实际位置；
   - 引擎整组检查（`SeedChooserScreen::OnStartButton`，DESIGN 里 `PvZEnv.deck_pool` 那节）对现卡组报什么；
   - 各关僵尸名单（从引擎读）；
   - 观测与动作覆盖：花盆上种植的合法动作、斜坡射程有没有体现在观测里（不需要手写，只要观测里有植物位置和类型就够）、玉米加农炮落点之类的新动作（第一阶段**不做**新动作头，只记录缺口）。
2. **卡组**（改卡组 = 新任务族，AGENTS §8 第 4 条）：
   - 卡组规则的唯一实现是 `scripts/pvz_tasks.py` 的 `deck_for_level`；
   - 用引擎 `CrazyDavePickSeeds` 的候选池（`PvZEnv.deck_pool`）做"卡组附近扰动 + 老师实测过滤"，做法同 DESIGN 里 task19 / 扰动族；
   - 给屋顶找到老师能赢的卡组：投掷类卷心菜投手 / 玉米投手 / 西瓜投手是否解锁，按冒险进度从引擎读；
   - 新卡组用新族名登记，旧族名不覆盖。
3. **族登记**：
   - 新任务族 `roof`，含训练 / 评估、原版与倍率，结构照 `zone_panel.py` 的 `Family` 与 pool 族（DESIGN §14.15）；
   - DESIGN.md 登记 §14.x；
   - 评估面板每关 32 种子，种子段避开现有所有段（读 DESIGN 的偏移台账）。
4. **老师冷启动实测**（预登记后再跑）：
   - 老师照 qc 的 `command_for`：学生 dw5i_s1 / dw4i_s1、价值网 v4、静态网 branch1；
   - L41–L45 原版难度各 16 局，至少两套卡组：现卡组与第 2 步选出的卡组；
   - 同时报学生自己（beta 0）的胜率；
   - 输局写清死因：波数、哪路、什么僵尸。
   - 判断：老师 ≥ 50% 的关可以进课程；< 20% 的先查卡组，再考虑倍率 0.5 起步。
5. **接进课程**（只做到可用，不动 qc）：
   - `make_multiplier_tasks.py` 的 `FAMILIES` 加 roof 族；
   - `curriculum_queue.py` dry run 能列出 roof 的 cell 与命令；
   - 主控决定何时让 qc 采。
6. 测试：`PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources python -m pytest python -q` 全过。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 引擎若要改，要过：非屋顶关逐帧重放 0 差异 + 已有引擎测试。编译只编本工作树的 `build/`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；**不手写规则或打法**。卡组可行性用引擎自己的检查与实测，不写"必须带 X"。
- 台式机 C 盘紧（约 28 GB），产物放 `artifacts/task46/`，可再生的大文件用完即删。
- 每完成一小步就提交、`git push origin roof`，在"结果"记一行。疑问写"问题"，按最佳判断继续。
- 与 task45（雾，codex3，分支 `fog`）并行：两边都改 `FAMILIES` / DESIGN，冲突主控合并时解。**不要改对方的族。**

## 结果

## 问题
