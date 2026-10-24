# task67：雾夜冷启动是不是一直是卡组问题——用户卡组上的老师冷启动面板

> 主控 2026-10-24 写。执行者：Mac CodeBuddy（`cb-demo` 会话，接着 task64 做）。工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo`（分支 `demo`）。游戏进程最多 4 个；不停 p5d，也不碰其他会话。

## 为什么
task64 的问题 2 指出：同一关（L31）、同一老师，卡组不同，冷启动结果差很多。
- task45 用 `fog_l31_eval_f8` 卡组加叶值 v4：0/384。
- task64 用用户的大喷菇线卡组加 v5：2/8。并且老师一旦接到 8 株（第 1 波）就 8/8。

AGENTS §8 说过："打不过的时候必须先考虑卡组。"雾夜族至今冷启动全败，可能一直是卡组的结论，不是策略的结论。这件事要用足够的局数确认。

## 要做的
1. **面板**：用老师的生产参数（与 task64 相同：`fast_search`、`--dagger-beta 1`、`--verify-samples 9`、`--student-candidates 2`、叶值 v5、静态网 branch1、学生 dw4i_s1），在下列族上做冷启动，每族 **32 个不同种子**。冷启动用不同种子即可，不需要示范。
   - L31：`demo_fog_31_fume`、`demo_fog_31_snow`（1.0 倍）；
   - L32：`demo_fog_32_magnet`、`demo_fog_32_freeze`（1.0 倍）；
   - L33：`demo_fog_33_cactus_snow_m200` 的卡组在 **1.0 倍**下（需要新开 1.0 倍族就登记 `demo_fog_33_cactus_snow`），以及 `demo_fog_33_blover`；
   - L34：`demo_fog_34_cactus` 的寒冰射手卡组在 1.0 倍下（同上登记），以及 `demo_fog_34_blover`。
   - 对照：同关原有的 qc 雾族卡组（task45 的 `fog_l3x_*` 面板卡组）。每关 32 局，同一批种子，配对比较。
   - 种子要与示范种子 20261031–34 分开，并写清楚。
2. **报告**：
   - 每族老师胜率，附 95% 区间；与对照卡组配对比较。
   - 输局按末波和破防僵尸归类，照 task64 的做法。
   - 判断：
     - 若用户卡组的老师冷启动胜率明显高于对照（例如 ≥25%，且配对显著），给出建议：用哪几个族接入 qc 课程，起始倍率多少。
     - 若仍接近 0，说明瓶颈不在卡组。这时建议改用 task64 的示范起点做倒推课程。
   - 结论写 `MEMORY/docs/fog-user-decks.md`，提交并 `git push origin demo`。**接不接 qc 由主控定。**

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 新开族必须登记 `configs/human_demo_tasks.json` 与 DESIGN（AGENTS §8），旧族不变。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

## 问题
