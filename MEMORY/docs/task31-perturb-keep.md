# task31（主控 2026-10-22 19:20 下发给台式机 CodeBuddy 第二条线）：扰动卡组“哪些卡不能换掉”按实测定

先读 `AGENTS.md`、`MEMORY/docs/task28-night-coverage.md`（尤其第 3、5、6 步与“问题”最后一条）、`MEMORY/docs/night-panel.md` 的 n10p 与经济卡两节、DESIGN §14.11–§14.13。

## 背景
引擎的整组检查只管“有没有产阳光植物 / 水生 / 花盆 / 对空 / 升级底座”。task28 第 3 步实测：n10p 里把**小喷菇 8** 换掉的卡组老师 1/10、学生 0/17，是 n10p（45.8%）比 n10（75.0%）掉分的主因。
白天 d7p 没有向日葵的局胜率 15.1%（引擎检查已排除）。用户的要求是随机给的卡组“起码有用”——不能靠我们手写“哪张卡重要”，要**按实测**：换掉某张底卡后老师基本赢不了，这张卡就不参与替换。

## 要做的（工作区 `git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/keep -b keep origin/pvz-env`）
1. **按“被换掉的底卡”统计老师胜率**（先用已有档案：task28 第 3 步的 n10p 老师局、d7p 全部采集局按 meta 初始卡组；样本不够的底卡再补老师局，≤2 进程，老师 = dw4i_s1 推演 + beta=1，参数照 `~/pvz-logs/queue103.sh` 的 `run()`）。
   每张底卡给：被换掉的局数、老师胜率与 95% 区间、对照（同关不换这张时的胜率）。
2. **判据只有一条，写成参数**：被换掉后老师胜率的 95% 上界 < 0.25 的底卡列为“保留卡”（其余照常可换）。不按植物写任何特例；结果是数据决定的一张表。
3. 新族：`night_lN_train_n10q` / `night_lN_eval_n10q_holdout{1,2}`（n10p 的做法 + 保留卡），白天 `day_lN_train_d7r` / `day_lN_eval_d7r_holdout{1,2,3}`（d7q + 保留卡，若白天没有保留卡就不建 d7r，写明）。
   种子段自选、穷举不重叠，DESIGN 新开一节登记（保留卡表与判据写进去）；清单拷到 `~/PvZ-search/artifacts/stage4/`。再给 `scripts/make_multiplier_tasks.py` 加上新族的倍率变体（课程控制器之后换用）。
4. 报告写进 `MEMORY/docs/night-panel.md` 新的一节。

## 规则
- 不要动任何 tmux 队列（q103、qc 等）与别的代理会话；不改别的工作树与主工作区（frames_v2 只读；stage4 只新增）。同时最多 2 个游戏进程。
- 分支 `keep` 提交，可以 push；不推 pvz-env、不合并。不训模型；不改默认值；不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。每完成一小步就提交并在“结果”里记一行。

## 结果

## 问题
