# task101：冒险模式 50 关实测记分板（deploy14 之后）

> 主控 2026-10-26 写。执行者：Mac Codex（plus2 号，tmux `codex-plus2`）。工作树新建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/scoreboard`（分支 `scoreboard`，从 origin/pvz-env 拉出，执行者自建）。Mac 上每个任务最多 6 个游戏进程。

## 为什么
`adventure-gap.md`（task55，10-24）是读报告拼出来的。这两天变化很大：
- deploy14 上线：雾夜示范格、fog79b、原生 L25、门禁 v3；
- 24 个示范与 LLM 起点格；
- L30 / L40 / L50 的诊断。
目标排第二的是“通关冒险模式”，需要一张**实测**的总表，回答：现在每关原版倍率（1.0）下打得过吗？下一步该把示范、GPT 跑局和训练投到哪几关？

## 要做的
1. **总表**：第 1–50 关逐关一行。
   - 卡组用该关现行训练族的卡组；同关有多个族的取胜率最高的那个，并注明族名。屋顶用示范族卡组；L30 / L40 / L50 用原生传送带族。
   - 倍率 1.0，16 个种子（新开种子段，机械核对）。
   - 两臂：学生 r11 单独打（C++ fast，β=0）；生产老师（v5 + branch1，`--dagger-beta 1 --verify-samples 9 --student-candidates 2`）。
   - r11：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/fog36/artifacts/task79/models/r11/model.pt`；v5、branch1 的路径见 task93b。
   - 引擎用 Mac 的 `build/pvz-portable`（pvz-env HEAD）。
   - 打地鼠、保龄球、砸罐子等特殊小游戏关按 task55 的结论标“未接入”，不跑。
2. **每行列出**：学生胜率、老师胜率、Wilson 区间，输局的中位末波与破防僵尸前三，以及 task55 的旧状态，便于对比。
3. **分类**（只引用证据，不写打法规则）：
   - 学生 ≥ 80%；
   - 老师 ≥ 80% 但学生 < 80%：可教，靠课程；
   - 老师 < 50%：要示范、卡组或机制。
4. **建议顺序**：哪几关最值得投入用户示范、GPT 多平台跑局（任务清单里有的）和训练。由主控定。
5. 报告写 `MEMORY/docs/adventure-scoreboard.md`，表格另存 `adventure-scoreboard.json`。`git push origin scoreboard`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
