# task28（主控 2026-10-22 15:30 下发给台式机 CodeBuddy 第二条线）：2-10 DAgger + 夜间扰动卡组 n10p（补没见过的植物）

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8），再读 `MEMORY/docs/night-l20.md`（你上一个任务 task26 的结论）、
`MEMORY/docs/task25-night-n10.md`、DESIGN §14.9 与 §14.7 里 task19 的 `day_l*_train_d7p`（`deck_perturb` 的做法）。

## 背景
task26 已合进 pvz-env（042e4c79）。结论：2-10 原版开局学生 0 胜、**一棵都不种**——传送带发的 7 种卡里 5 种（10 大喷菇、12 魅惑菇、13 胆小菇、14 寒冰菇、15 毁灭菇）
从未进过任何任务卡组，identity 头在这些身份上把种植门压成负值；老师（同学生 + 搜索）6/8 胜。两件事：用 DAgger 教它在 2-10 种，
并且让这些植物在普通夜间关里也出现（换卡组泛化，和白天的 d7p 同理）。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/n10p -b n10p origin/pvz-env`，只在这里改代码、单独编译。

1. **2-10 DAgger 采集**（先开，后台跑，同时最多 2 个游戏进程）：`night_l20_train_n10_orig`（`~/PvZ-search/artifacts/stage4/night_panel_n10.json`，种子 2290000 起）偏移 0 起 32 局，
   采集学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`，参数照 `~/pvz-logs/queue103.sh` 的 `run()`（beta=0：学生走、搜索打标签），
   标签 `n20o1`，帧写进 `~/PvZ-search/artifacts/student/frames_v2`（与 q103 同布局）。用 nohup，进度写日志，完成后在“结果”记胜局与每局按类型的种植数（`planted` 键是槽号，要用卡组翻成类型）。
2. **夜间扰动卡组族**：照 task19 的 `deck_perturb`（`scripts/make_identity_tasks.py`），以 n10 `[0,1,2,3,4,5,7,8,9,11]` 为底，每种子确定性替换 1 或 2 张，
   pool 取引擎实际解锁（playthrough 2）、非升级、无引擎警告、夜间能种的植物（应包含 10/12/13/14/15）。新族 `night_lN_train_n10p`（N = 11–14、16–19，每关 8192 种子），
   种子段自己选一个与所有现有清单都不重叠的段（生成时穷举自查，写进 DESIGN）；再出留出族 `night_lN_eval_n10p_holdout{1,2}`（评估段 30000 起，每关 2 套，训练族显式排除这些卡组集合）。
   生成到 `artifacts/stage4/night_panel_n10p.json`（不进 git），**另拷一份到 `~/PvZ-search/artifacts/stage4/`**。DESIGN 新开一节登记。
3. **老师验证**（等第 1 步跑完再开，≤2 进程）：老师（dw4i_s1 推演，beta=1）在 `night_l{12,14,17}_train_n10p` 各 16 局，学生（dw4i_s1）同种子各 32 局；
   列出每局卡组、胜负、按类型种植数。老师胜率远低于 n10（2-2 / 2-4 / 2-7 老师 n10 约 13/16、11/16）的话，找出是哪些替换让它结构性打不过，写进“问题”由主控定 pool。
4. 报告写进 `MEMORY/docs/night-l20.md` 新的一节（2-10 DAgger）与 `MEMORY/docs/night-panel.md` 新的一节（n10p）。

## 规则
- **不要动任何 tmux 队列（q103 等）与别的代理会话；不要改别的工作树与 `~/PvZ-search` 主工作区**（frames_v2 只增不改；stage4 只新增 n10p 一份）。
- 分支 `n10p` 提交，可以 `git push origin n10p`；不推 pvz-env、不合并。不训模型；不改默认值；不改 `deck_for_level`；不写手工规则；不写字面密钥；
  不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。每完成一小步就提交，并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。

## 结果

## 问题
