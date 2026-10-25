# task75：原生传送带关 L25 / L30 / L40 接入——从"未接入"到有老师冷启动数据

> 主控 2026-10-25 写。执行者：Mac Codex（新 Plus 号，tmux `codex-plus2`）。工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/conveyor2`（分支 `conveyor2`，从 origin/pvz-env 拉出，主控已建好）。游戏进程最多 6 个；Mac 上的 p5d 采集与 CodeBuddy 会话照常跑，不要停。

## 为什么
冒险模式 50 关里还有 13 关未接入（`MEMORY/docs/adventure-gap.md`）。其中三关是原生传送带，用户没有要求后置：
- L25（3-5）：Little Trouble 小僵尸，传送带；
- L30（3-10）：泳池传送带；
- L40（4-10）：暴风雨传送带，无雾，有闪电黑屏。

L10、L20 的原生传送带族已经在 qc 里，L20 学生从 0 练到了 45/48。照同样的路接这三关。

## 要做的
1. **原生族**：照 L10/L20 原生族（`day_l10_orig`、`night_l20_orig`）的做法，为三关登记原生开局族：
   - 不用固定卡组覆盖原生出牌；
   - 保留原生槽位、重复牌、空槽语义；
   - 按 AGENTS §8 登记 DESIGN 与清单，种子段与现有全部段不相交（机械核对）。
2. **观测审查**：
   - **L25**：小僵尸的尺寸、碰撞、血量在完整 / lean / v2 / C++ 学生输入里能不能和普通僵尸区分开。区分不开就补一个人类可见的字段，例如体型。
   - **L40**：暴风雨的黑屏和闪电。人在黑屏时看不见场上，闪电时才能看见。现在 `StageHasFog=false`，human 口径什么都不遮，等于透视。按 task66 的思路，用游戏自己的绘制逻辑判定"此刻人能不能看见"，做成 `fog_observation` 的一个新族或新取值。原族逐位不变。
   - **L30**：确认原生开局与 p9k 覆盖开局的差别。
3. **老师冷启动**：生产老师参数——叶值 v5、静态网 branch1 用台式机那份（从 HF 或主控处取，核对 SHA 24858410），`--dagger-beta 1 --verify-samples 9 --student-candidates 2`。每关 32 局，报胜率、Wilson 区间、输局末波与破防僵尸。过 50% 的关建议进 qc，起始倍率由主控定。
4. **测试**：原族/旧任务逐位不变；新族 reset 与整局跑通；观测遮挡的不变式照 task66 的四类测试来写。
5. 结论写 `MEMORY/docs/native-conveyor.md`，提交并 `git push origin conveyor2`。部署由主控做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不加开关，不改生产默认。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥，不打印凭证；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 2026-10-25：登记六个独立原生传送带任务与 60M 训练 / 61M 评估种子段，生成器不改旧面板。L40 的 storm 观测协议与遮挡在下一步实现，尚未启动冷启动测量。
- 种子机械审计：53889 个 JSON、748 项含种子任务、968772 个现有唯一种子与新增24672个种子零交集，包含倍率清单。清单静态测试 1 passed。

## 问题

Mac 上的模型路径：静态网 `/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt`（主控已从台式机拷来，SHA 24858410…）；叶值 v5 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/hard/value_v5/model.pt`（8a636d33…，只读）；学生 dw4i_s1 同 demo 工作树的现有路径。
