# task45：接入雾夜泳池区（冒险 31–40，35 除外）第一阶段——族登记、雾的观测、老师冷启动实测

> 主控 2026-10-23 20:30（台式机时间）写。执行者：台式机 Codex（Business 账号，codex3）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/fog`（分支 `fog`，从 pvz-env 拉出，主控已建好）。
> 同时最多 **4** 个你自己的游戏进程（qc 16 进程 + 别的任务在跑）。不碰 tmux `qc` 与别的代理会话。

## 为什么
- 目标是通关冒险模式。现在的课程（`scripts/make_multiplier_tasks.py` 的 `FAMILIES`）覆盖白天、夜间、白天泳池、传送带。**雾夜泳池（31–40）和屋顶（41–50）一关都没有。**屋顶另派 task46。
- 雾区的已知难点见 `MEMORY/docs/student-bottleneck-plan.md` 2026-10-20 00:30 用户答复那条：
  - 雾（`Board::UpdateFog`、`ClearFogAroundPlant`；路灯花半径 4、火炬树桩半径 1 开雾，三叶草吹雾）；
  - 气球僵尸、潜水 / 海豚等新僵尸。
- 用户定了"雾关先实验再定"，问题有两个：
  1. 能不能学会"看不见也往雾里放、种路灯花开雾"；
  2. 不透视比透视差多少。
- 现状：观测只给 `fog` 标志，僵尸列表全给，等于透视。

## 步骤
1. **现状盘点**（只读）。把下面每项的答案写进本文"结果"：
   - 引擎在雾关能否无头 reset / 跑完一局；
   - `deck_for_level` 给 31–40 的卡组：`(0,9,2,3,4,5,LILY_PAD,CATTAIL)`；
   - 引擎 `SeedChooserScreen::OnStartButton` 的整组检查对它报不报警告（DESIGN 里已有可调用的检查，见 `PvZEnv.deck_pool` 那节）；
   - 本区出现的僵尸名单（按关，从引擎读，不手抄）；
   - 观测里哪些字段暴露了雾里的信息。
2. **雾的观测（两种都做，用同一个实现、一个参数，不是开关堆）**：
   - 在 C++ 观测里按 `mGridCelFog`（或引擎实际用的雾量）把雾中格子里的僵尸屏蔽掉，得到"人类同等信息"的观测；
   - 保留"透视"作对照；
   - 二者由任务族决定，族名里写明（如 `fog_l31_train_f8` 与 `fog_l31_train_f8x`），**不加运行开关**；
   - 学生观测编码（`python/pvz_student_observation.py`）要能表示"这格有雾"（雾量通道），不透视时的僵尸通道就是屏蔽后的；
   - 一致性门禁：非雾关观测逐位不变（4 个存档整局逐帧重放 0 差异）。
3. **族登记**（AGENTS §8）：
   - 新任务族 `fog`，含训练 / 评估、原版与倍率，结构照 `zone_panel.py` 的 `Family` 与 pool 族（DESIGN §14.15）；
   - 在 DESIGN.md 登记 §14.x（接在现有最后一节之后）；
   - 评估面板：每关 32 种子，种子段避开现有所有段（读 DESIGN 的偏移台账）。
4. **老师冷启动实测**（预登记后再跑）：
   - 老师照 qc 的 `command_for`：学生 dw5i_s1 或 dw4i_s1，泳池 dw5i 更强；价值网 v4；静态网 branch1。L31–L34 原版难度各 16 局，透视与不透视两臂；
   - 同时报学生自己（beta 0）的胜率；
   - 输局写清死因：波数、哪路破、是不是雾里来的。
   - 判断：老师 ≥ 50% 的关可以直接进课程采集；< 20% 的关先查卡组（AGENTS §8 第 1 条：打不过先考虑卡组），再考虑倍率降到 0.5 起步；
   - 写清"进课程"的建议，包括哪些关、哪种观测、起始倍率。
5. **接进课程**（只做到可用，不动 qc）：
   - `make_multiplier_tasks.py` 的 `FAMILIES` 加 fog 族；
   - `curriculum_queue.py` 跑 dry run，能列出 fog 的 cell 与命令；
   - 主控决定何时让 qc 采。
6. 测试：`PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources python -m pytest python -q` 全过。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 引擎改动要过：非雾关逐帧重放 0 差异 + 已有引擎测试。
- 编译只编本工作树的 `build/`，用 `scripts/build_pgo.sh` 或普通 Release 都行。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；**不手写规则或打法**，雾的处理只在观测与任务层。
- 台式机 C 盘紧（约 28 GB），产物放 `artifacts/task45/`，可再生的大文件用完即删。
- 每完成一小步就提交、`git push origin fog`，在"结果"记一行。疑问写"问题"，按最佳判断继续。
- 与 task46（屋顶，codex4，分支 `roof`）并行：两边都改 `FAMILIES` / DESIGN，冲突主控合并时解。**不要改对方的族。**

## 结果

## 问题
