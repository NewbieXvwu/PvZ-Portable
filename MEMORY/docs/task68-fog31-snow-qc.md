# task68：雾夜第一个冷启动族 `fog_user_l31_snow` 接进 qc 课程

> 主控 2026-10-24 写。执行者：Mac CodeBuddy（`cb-demo` 会话，接着 task67 做）。工作树新建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/fogq`（分支 `fogq`，从 origin/pvz-env 拉出，主控已建好）。最多 2 个游戏进程；不碰 p5d 与其他会话。**只改代码、清单与测试，部署由主控在台式机做。**

## 为什么
task67（`MEMORY/docs/fog-user-decks.md`，demo 分支）的结论：
- L31 用用户卡组（阳光菇/小喷菇/大喷菇/睡莲/高坚果/窝瓜/缠绕海草/樱桃/坚果/**寒冰射手**），老师冷启动 **25/32**（78%，配对 p<1e-6）。同一批种子下，原 qc 雾族卡组是 0/32。
- 唯一的差别是寒冰射手换掉了路灯花。老师会把路灯花反复种下（每局 4.3 株），把经济节奏带偏。
- 这是雾夜区第一个过 50% 的冷启动族。**主控决定接入 qc，起始倍率 1.0。**

## 要做的
1. **登记任务族**（AGENTS §8：新族名，旧族不动，DESIGN 登记）：
   - 族名 `fog_user_l31_snow`；关卡 31；卡组 `[9,8,10,16,23,17,19,2,3,5]`；`fog_observation: human`。任务族的其他字段，例如二周目、香蒲所有权，照 task67 面板的用法。
   - 训练种子段、评估种子段都要新开，与示范种子 20261031–34、task67 面板 20261101–20261132、现有 qc 各族都不重叠，写进 DESIGN。
2. **接入课程清单**：
   - 照 task50 / zones 接入 L41 屋顶格的做法：`scripts/make_multiplier_tasks.py` 的 `FAMILIES` / `CURRICULUM_LEVELS`，以及 `artifacts/stage4/multiplier_panel.json` 的生成方式。
   - 在 qc 清单里加一个 cell：`fog_user_l31_snow`，倍率阶梯从 1.0 起，与其他族同样的升降档规则。
   - 重新生成清单时，**现有 cell 的 task_id、种子、偏移一律不变**。用测试或比对证明，其余 cell 逐字节相同。
3. **老师证据**：`~/pvz-logs/teacher_rates.json` 的格式见 `curriculum_queue.py`（`--teacher-rates`）。准备一条 `fog_user_l31_snow: 0.78`（来源 task67），给出要追加的 JSON 片段，主控部署时写入台式机。
4. **学生能吃这个族的数据**：
   - 确认 qc 生产学生（v2 观测）在 `fog_observation: human` 的任务上能正常采集、打标签、打包、训练。task45 已验证过非雾存档的学生输入逐位一致，这里要验证雾族一局能完整走通。
   - 在 Mac 上用 `curriculum_queue.py` 的同一条 `search_play` 命令（`command_for`），对新 cell 跑 2 局冒烟，产出的帧能被 `pack_training_rows.py` 打包。
5. **promotion 面板不动**：晋升门禁的冻结面板不加雾族，避免改变门禁合同。在报告里写明这一点，并说明新族的学生进展怎么看：qc 自己的组胜率窗口。
6. 测试全过（`python/test_curriculum_*.py`、`make_multiplier_tasks` 相关），提交并 `git push origin fogq`，结果写本文件。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

## 问题
