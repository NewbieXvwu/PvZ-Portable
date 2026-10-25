# task76：屋顶 L42–L49 冷启动全败——v5 之后，候选遗漏还是不是最大缺口？

> 主控 2026-10-25 写。执行者：台式机 Codex（Plus 号，tmux `codex-clean` 会话，task62 已收尾）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/roofcand`（分支 `roofcand`，从 origin/pvz-env 拉出）。游戏进程最多 6 个；**不碰 tmux `qc`、cq、流水线目录**。

## 为什么
- task46（10-24，叶值 v4）：屋顶 17 个根的终局真值里，**候选生成遗漏是最大缺口**；v4 短叶值与终局胜负的一致率只有 53.9%。
- 之后叶值换成了 v5。v5 让雾夜 L31 老师冷启动从约 16% 升到 29–31/32（task70 B3）。
- task72 在雾区用"强加全部合法格"的反事实证明：那里瓶颈不在候选。屋顶需要同样的检验，而且要在 v5 下做。
- task74（cb-qcbench）正在跑屋顶 L42–L49 的 v5 老师冷启动面板，约 1 小时出数：`MEMORY/docs/task74-cold-start-sweep.md`（分支 gatepower）。

## 要做的
1. 先用 L42、L43 开工，不必等 task74；task74 出数后，扩到所有老师胜率 < 50% 的屋顶关。
2. **候选覆盖**：照 task72 第 1、2 步。
   - 逐决策记录老师候选集与全部合法动作（含花盆格、投手），统计好动作被静态网漏掉的比例。
   - 诊断构建里"强加全部合法格"作反事实，同种子配对，每关至少 32 局，只读诊断，不进生产。
3. **叶值对照**：照 task72b 第 3 步，选一批输局局面做终局真值（每动作 8 个未来），算 v5 叶值与真值的一致率，与 task46 的 v4 53.9% 对比。
4. **判断**：
   - 若强加候选能显著翻盘 → 写静态候选网多区重训方案草稿：数据（决策级、多区）、标签、评估（候选召回率 + 配对胜率），主控定。
   - 若翻不了，按 task72b 的顺序说清瓶颈：局面已死、经济、叶值尺度。
5. 结论写 `MEMORY/docs/roof-candidates-v5.md`，提交并 `git push origin roofcand`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认、不加开关；反事实只在诊断代码路径里。
- 生产参数：引擎 `/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable`，叶值 `artifacts/value_net/v5/model.pt`，静态网 `artifacts/hard/value_branch1/model.pt`，学生用 qc 现役（`~/pvz-logs/curriculum_state.json` 的 `student`）。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 输入冻结与预注册完成：指定build4引擎、现役round_11学生、v5、branch1（另留v4匹配叶值对照）均复制到roofcand自己的ignored目录并逐文件SHA核验；L41–L49登记r7卡组均通过实际引擎解锁/警告核对。先做L42/L43各32种子；主比较强加全部合法植物格，另列含铲子的全合法臂，避免混淆候选遗漏与动作空间。终局按task72b固定学生续走、每动作8未来，与task46老师续走的历史53.9%分开报告。计划见task76-roof-candidates-v5-plan.json；未改qc/cq/流水线。

## 问题

- task74任务书/结果尚未在已拉取gatepower分支或同名本地路径找到；已向主控索取路径/提交号，先按要求推进L42/L43。
