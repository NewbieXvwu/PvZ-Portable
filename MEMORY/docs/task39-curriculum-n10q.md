# task39：课程控制器换上 n10q（夜间扰动族），并让已有状态能增删格子

> 主控 2026-10-23 09:30 写。执行者：Qoder（DeepSeek-Flash），工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/curr39`（分支 `curr39`，从 pvz-env c02924e7 拉出）。
> 只做代码 + 单测 + 干跑；**正式队列 qc 的停 / 换 / 重启由主控做**，你不要碰 tmux 会话 `qc`、`~/pvz-logs/curriculum_state.json`、`~/pvz-logs/stop_curr`、cq 工作树。

## 背景（先读，不用再查）

- task31（DESIGN §14.14）实测：夜间 n10p 扰动族里，**8 小喷菇 / 9 阳光菇**被换掉后老师胜率只有 4.8% / 10.2%（结构性输局），于是建了新族 `night_lN_train_n10q` = n10p + `keep_plants: [8, 9]`。
- 课程控制器 `scripts/curriculum_queue.py` 正在桌面机 tmux `qc` 里跑（16 进程），清单 `cq` 工作树的 `artifacts/stage4/multiplier_panel.json`，状态 `~/pvz-logs/curriculum_state.json`。
- 问题 1：控制器只在**状态文件不存在时**用 `make_state` 建格子（`main` 里 `if args.state.exists(): state = json.loads(...)`）。已有状态不会自动加 n10q 的 8 个格子，也不会去掉 n10p。
- 问题 2：n10p 应退出课程（它的数据保留、清单保留、评估保留——只是不再排新局）。不能从 `make_multiplier_tasks.FAMILIES` 里删 n10p：族序号决定倍率种子段，删了 n10q 的种子会平移。
- 问题 3：`scripts/make_multiplier_tasks.py` 第 46–49 行注释说“倍率族还没开过采集局，没有数据要迁移”——**已经不对了**：课程控制器 10-22 起已经跑过 `_m125` / `_m150` 的局（`curriculum_state.json` 的 `offsets` 里 `day_l7_train_d7_m125: 96` 等）。档距 500k→700k 后这些 task_id 的种子列表整体平移。实际影响：控制器按 task_id 记偏移，偏移继续往后走，取到的都是这个 task_id 下没打过的新种子，所以**不会重复**；只是同一个 task_id 前后两段种子来自不同的段。要把这件事如实写进注释和 DESIGN §14.13 的末尾（一句话），并在“结果”里记录你核对过 `games.jsonl` 里实际记录的是种子值本身（不是靠清单 + 偏移反推）——如果有任何工具靠“清单 + 偏移”反推种子，列出来写进“问题”，不要改。

## 要做的事（按顺序，每步提交一次，并在本文末“结果”记一行）

### 第 1 步：族的“被取代”关系写成数据
在 `scripts/make_multiplier_tasks.py` 里 `FAMILIES` 旁边加一个常量：

```python
# 课程里被新族取代的族（旧族的清单、种子段、数据、评估全部保留，只是课程不再给它排新局）。
# 依据：task31 / DESIGN §14.14 —— n10p 换掉 8/9 号卡的局是结构性输局。
SUPERSEDED = {'night_n10p': 'night_n10q'}
```

`FAMILIES` 本身**一个字都不改**（族序号 = 种子段，改了就平移）。`build_variants` 照常生成 n10p 的倍率变体（清单要能重放旧局）。

### 第 2 步：控制器的训练格子集合排除被取代的族
`curriculum_queue.training_base_ids()` 跳过 `SUPERSEDED` 里的族（按 FAMILIES 的第一个字段，即族短名）。`init_from_runs` 已经用 `base_ids` 过滤，会自然跟着变。

### 第 3 步：加载已有状态时对齐格子（核心）
新函数 `reconcile_cells(state, tasks, names, windows_from_runs) -> list[str]`，在 `main` 读入已有状态之后、`plan_round` 之前调用，返回日志行并逐行 `log_line` 写进 curriculum.log：

- 清单里有、`training_base_ids()` 里有、状态里没有的 base id → 按 `make_state` 同样的字段新建格子（`mult_index=0`、`last_round=0`、`windows` 取 `init_from_runs` 的结果，没有就空）。日志：`cell added: night_l12_train_n10q`。
- 状态里有、但已不在 `training_base_ids()` 里的格子 → 从 `state['cells']` 移到 `state['retired_cells']`（原样保存，含 windows 与 mult_index，**不删**），日志：`cell retired: night_l12_train_n10p (superseded by night_n10q)`。
- `retired_cells` 里的格子如果又回到训练集合（比如以后撤销取代）→ 原样搬回 `cells`，日志 `cell restored: ...`。
- `offsets` 一律不动（偏移按 task_id 记，退役格子的偏移留着，以后恢复时接着用）。
- `make_state` 改为调用同一套逻辑（空状态 + reconcile），不要留两份建格子的代码。

为了让 reconcile 能用 `init_from_runs` 的窗口：现在 `init_from_runs` 只在状态不存在时调用（看 `main`），新格子需要它的结果；如果 `--init-glob` 没给，就传空字典，新格子从空窗口起步（这是预期行为，n10q 没有旧局）。

### 第 4 步：单测
在 `python/test_curriculum_queue.py` 加（全部用临时目录与小清单，不碰真实状态）：
1. 旧状态（含 n10p 格子、不含 n10q）+ 含两族的清单 → reconcile 后 `cells` 有 n10q 无 n10p，`retired_cells` 有 n10p 且 windows / mult_index 原样，`offsets` 不变，返回的日志行数正确。
2. 再跑一次 reconcile → 无变化、无日志（幂等）。
3. 把 `SUPERSEDED` 临时打补丁成空 → n10p 从 `retired_cells` 原样搬回。
4. `make_state` 结果与“空状态 + reconcile”逐字段相同。
5. `training_base_ids()` 不含任何 `_n10p` 且含 8 个 `_n10q`。
跑 `python3 -m unittest test_curriculum_queue test_multiplier_tasks test_keep_families`（在 `curr39/python` 下）全过。

### 第 5 步：桌面真实干跑（只读）
1. 在你的工作树里重新生成倍率清单：`cd /home/newbiexvwu/PvZ-search/.claude/worktrees/curr39 && /home/newbiexvwu/.venvs/ml/bin/python3 scripts/make_multiplier_tasks.py --dry-run`，再不带 `--dry-run` 跑一次（它会写工作树里的 `artifacts/stage4/multiplier_panel.json`；**先看脚本是否还会拷贝到主目录 `~/PvZ-search/artifacts/stage4/` 或别的工作树——如果会，加一个只写工作树的方式或先备份目标再说，绝不覆盖 cq 工作树里的那份**）。确认 `night_panel_n10q.json` 在 `~/PvZ-search/artifacts/stage4/` 存在（task31 生成过）。
2. 复制真实状态：`cp ~/pvz-logs/curriculum_state.json $CLAUDE_TMP/state_copy.json`（`$CLAUDE_TMP` 用 `/home/newbiexvwu/pvz-logs/task39_tmp/`，自己建）。
3. `curriculum_queue.py --dry-run --manifest <工作树清单> --state <副本> --log <副本目录>/curr.log --round-games 768 --games-per-group 48`，确认：日志里 8 条 `cell added ... n10q`、8 条 `cell retired ... n10p`；计划里出现 n10q 组、没有 n10p 组；其他格子的倍率档与真实状态一致。把计划输出贴进“结果”（截前 30 行即可）。
4. **不要**真跑局（不去掉 `--dry-run`），不起任何游戏进程。

### 第 6 步：报告
本文末“结果”逐步一行；“问题”写任何不确定的地方。DESIGN §14.13 末尾加一句倍率种子平移的说明（见背景问题 3），§14.14 末尾加一句“课程控制器从 task39 起用 n10q 取代 n10p（`SUPERSEDED`），n10p 格子进 `retired_cells`”。提交到 `curr39` 分支并 `git push origin curr39`（只推这个分支）。

## 约束
- 只用绝对路径；不 `cd` 进别的工作树；不碰 tmux `qc` / 其他代理会话 / `~/pvz-logs/curriculum_state.json` 原件 / cq 工作树。
- 不加命令行开关（这是重构，不是实验）。
- 禁止 `git add -f`；`.json` 清单不进 git（被 .gitignore 拦就对了）。
- 不手写“哪张卡重要”的规则：`SUPERSEDED` 是族级的取代关系，依据写在注释里指向 §14.14 的实测表。

## 结果

- 第 1 步：`make_multiplier_tasks.py` 加 `SUPERSEDED = {'night_n10p': 'night_n10q'}`，FAMILIES 未改；修正种子段注释（档距 500k→700k 后种子平移，控制器按 task_id 偏移续采不会重复，`games.jsonl` 记录种子值本身）。
- 第 2 步：`training_base_ids()` 跳过 `SUPERSEDED` 中的族；实测 37 个 base id，含 8 个 `_n10q`、0 个 `_n10p`。
- 第 3 步：新增 `reconcile_cells(state, tasks, names, windows_from_runs)`：新增训练格、退役非训练格到 `retired_cells`、可恢复，`offsets` 不动；`make_state` 改为空状态 + reconcile；`main` 读入已有状态后调用并逐行 `log_line`（未给 `--init-glob` 时新格从空窗口起步）。

## 问题
