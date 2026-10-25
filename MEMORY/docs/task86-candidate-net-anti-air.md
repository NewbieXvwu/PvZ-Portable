# task86：静态候选网漏掉对空植物——用“强加全部合法格”的老师标签重训候选网

> 主控 2026-10-25 写。执行者：Mac Codex（plus2 号，tmux `codex-plus2`）。工作树沿用 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/conveyor2`（先 `git merge origin/pvz-env`）。Mac 上每个任务最多 6 个游戏进程。

## 证据
- **task75c**：L40 原生 0.5 倍，候选强加全部合法格后老师 **6/16 → 15/16**，多种了仙人掌 298 张、三叶草 97 张。现行静态候选网（branch1）在气球关几乎不提议对空植物，属于**候选遗漏**。
- 雾夜 L39 在多套卡组下也是气球破防（task74b / task79），很可能是同一个缺口。
- 屋顶（task76）强加候选只有很小的增益（200 对：gained 5 / lost 1），瓶颈不在候选。
- L30 强加候选后仍是 0/16，属于叶值或学习覆盖问题，不在本任务范围。

## 要做的
1. **复核 L39**：雾夜 L39 用 `fog79_l39_snow_blover_v2` 与 `fog79_l39_snow_cactus_v2` 两族，学生 r11，生产老师参数。照 task75c 的方法，同种子配对 16 局，跑“现行候选”与“强加全部合法格”两组。报胜率差与气球破防数。
2. **数据**：用“强加全部合法格”的老师在候选遗漏关（L40 0.5 倍、L39，以及第 1 步确认的其它关）上采集决策级数据。标签取老师最终选择的动作，记录它是否在原候选集里。种子段要独立并做机械核对。同时保留原有多区数据，防止旧区退化。
3. **重训候选网**：沿用 branch1 的结构与训练脚本，新数据与原数据混合，混合比例写在预登记里。评估分三项：
   - **候选召回**：老师最终动作落在候选集里的比例，按区报，对比 branch1；
   - **配对胜率**：L40 0.5 倍、L39 各 32 局，新网 vs branch1；
   - **旧区不退化**：day / night / pool / fog L31 / roof L41 各 32 局，配对比较，非劣界 −3pp。
4. 预登记写在动手训练之前。结论写 `MEMORY/docs/candidate-net-anti-air.md`，`git push origin conveyor2`。新网只交付路径与 SHA，是否上生产由主控定。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认，不加开关。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；**不手写打法规则或公式**，也不写“看到气球就加仙人掌”之类的候选规则。改进只能来自数据和训练。
- 模型与资源路径（只读）：
  - 资源：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`
  - branch1：`/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt`
  - 叶值 v5：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/hard/value_v5/model.pt`
  - 学生 r11：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/fog36/artifacts/task79/models/r11/model.pt`
  - Python（带 numpy）：`/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3`
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题
