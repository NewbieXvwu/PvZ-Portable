# task27（主控 2026-10-22 15:15 下发给台式机 Codex，额度用完由 CodeBuddy 接手）：面板 v2（夜间 n10）+ 泳池分支变基 + 留出死卡组诊断 + 第二次合训 dw5i

先读仓库根目录 `AGENTS.md`（优先级最高），再读 `MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条、
`MEMORY/docs/task23-dw4i-unified-panel.md`（你上一个任务，含第 5、6 步结论）、`MEMORY/docs/unified-panel.md`、
`MEMORY/docs/task25-night-n10.md`、DESIGN §14.7–§14.11。

## 背景
task23 已合进 pvz-env（25b1672a）。决定：**宽度保持 64，不上 MoE**（128 宽留出与夜间反而掉、推演慢 3.1 倍）；
**夜间卡组定 n10**（n9 + 墓碑吞噬者，同种子 +29.4pp，空槽控制组排除了“扰动”解释）。
queue103 正用 dw4i_s1 采集（`~/pvz-logs/queue103.log`，4 轮 × [夜间 n10 弱关 `n{N}t{r}` + 白天早关 `d{l}dag{r}d` + 泳池 p9k `p{N}dag{r+1}` + d7p `d{l}p{r+6}d`]），
q102 已停（夜间 n9 弱关、泳池第 1 轮、d7p）。下一个模型 dw5i 要把这些都学进去。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/dw5 -b dw5 origin/pvz-env`，只在这里改代码、单独编译。

1. **泳池分支变基**：`origin/pool`（task22，已完成）把 `scripts/make_night_panel.py` 重构成了 `scripts/zone_panel.py`，
   与 pvz-env 里 task25 对它的改动（n10 族、`night_l20_*_n10_orig` 原版开局）冲突。把 pool 的提交合进 dw5，解决冲突时**两边功能都保留**；
   验收：用合并后的脚本重新生成 `night_panel.json`、`night_panel_n10.json`、`pool_panel.json`，与 `~/PvZ-search/artifacts/stage4/` 下现用的三份**逐任务逐种子一致**（写一个比对脚本，结果记进“结果”）。不一致就保留现用文件并写进“问题”。
2. **统一面板 v2**：`scripts/unified_panel.py` 加一个版本选项（v1 保持不变，可复现旧数字），v2 的夜间换成 `night_l{11,12,13,14,16,17,18,19}_eval_n10` + `night_l20_eval_n10_orig`（每关 128），
   再加泳池 p9k 五关（task22 选的那 5 关，`pool_lN_eval_p9k`，每关 64）；汇总按区（白天、换卡组留出、夜间、泳池）。
   汇总里的种植统计**按植物类型**输出（用每局卡组把槽号翻成类型；`planted` 键是槽号，这次已经踩过坑）。
   跑 v2：dw4i_s1、identity_dw3_s1（作为没见过夜间/泳池的对照）。
3. **留出死卡组诊断**：统一面板里所有模型都是 0 的三套留出（L7 h3、L8 h1、L9 h1）：列出卡组；老师（dw4i_s1 推演，beta=1，参数照 `~/pvz-logs/queue103.sh` 的 `run()`）每套 16 局；
   看死在第几波、什么僵尸。老师也 0 → 卡组结构性打不过（按 AGENTS §8，这类卡组不该进留出，写进“问题”由主控定换不换）；老师能赢 → 是学生的泛化缺口。
4. **dw5i 训练**：等 q103 第 2 轮全部结束（看 `queue103.log` 里 `d10p8d` 那行）再冻结数据。先 `pgrep -f train_rollout_policy` 确认没人在训。
   配方照 dw4i（64 宽、identity 头，`scripts/run_dw4_study.py` 的参数），标签 = dw4i 的全部标签 + q102 全部 + q103 已完成轮次 + HF `transfer/` 里已上传而还没导入的 Mac 轮次
   （导入到 `~/PvZ-search/artifacts/student/frames_v2`，已存在的不覆盖）。训 1 个种子，跑统一面板 v2 与 v1；与 dw4i_s1 并列写进 `unified-panel.md` 新的一节，
   重点看：夜间 n10、泳池、白天早关 L1/3/4/6 有没有回到 identity_dw3_s1 的水平、留出。模型传 HF `transfer/dw5i_s1_model.pt`。

## 规则
- 同时最多 4 个游戏进程（q103 在用 16 个）；**不要动任何 tmux 队列与别的代理会话；不要改别的工作树与 `~/PvZ-search` 主工作区**（frames_v2 只增不改、stage4 清单只读）。
- 分支 `dw5` 提交，可以 `git push origin dw5`；不推 pvz-env、不合并。不改默认值；不写手工规则；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；
  不 `git add -f`；`.pt/.npz/.log` 不进 git；大于 100 KB 的结果 json 别进 git（放 artifacts，报告里写路径与 SHA）。
- 1、2、3 可以先做（不等数据），4 等 q103 第 2 轮。每完成一小步就提交，并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。
  你额度用完时 `agent_switch.sh` 会让 CodeBuddy 接手、额度恢复后交回你，所以进度一定要落在提交和“结果”里。

## 结果

## 问题

## 第 0 步（主控 2026-10-22 15:45 追加，先于其他步骤做）：种植统计改按植物类型记录
用户指出：对局汇总里的 `planted` 按**卡槽序号**计数，与“按植物身份”的设计方向不一致（传送带关槽位会移动，槽号连跨局都不可比）。动作接口按槽号（和人点卡槽一样）不改，**统计与报告**改成按类型：
1. `src/main.cpp`（整局 C++ 推演，约 469/525/579 行）与 `scripts/search_play.py`（约 773–878 行）：在**执行种植之前**读出该槽的植物类型（传送带关种完槽会移动，不能事后查），
   新增键 `planted_types`（`{植物类型: 棵数}`，JSON 对象）；原键改名 `planted_slots`，不再写 `planted`，免得新旧文件同名不同义。
2. 写一个公用读取函数（如 `python/pvz_episode_stats.py` 的 `planted_by_type(game, deck)`）：有 `planted_types` 直接用；旧文件只有 `planted` 时用该局卡组把槽翻成类型；
   传送带关（`conveyor_start` 或 L10/L20 类）的旧文件翻不了，返回 None 并在报告里注明，不要猜。
   把读 `planted` 的脚本（`git grep -n planted scripts python`：`dw4_night_ab_report.py`、`analyze_l20_conveyor.py`、`sim_skip_agreement.py`、`cuda_root_corpus.py`、`unified_panel.py` 等）改用它。
3. 顺手修同类错误：`search_play.py` 的 `replant_events` 里 `"same_type": occupant.get("type") == action["packet"]` 是拿植物类型比槽号，改成比较该槽的植物类型。
4. 测试：单元测试覆盖翻译函数（含传送带返回 None）；冒烟 1 局 C++ 整局路径与 1 局 Python 路径，核对两边 `planted_types` 与逐动作重放一致。
   编译只在 dw5 工作树的 `build/`，**不要重编主工作区 `~/PvZ-search/build/`**（q103 在用；它采完的旧文件照旧是槽号，由第 2 条的读取函数兼容）。
