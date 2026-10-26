# task103：冒险模式记分板做成可重复的北极星指标

## 背景

task101 一次性测出了 45 关 × 16 种子 × 学生 / 老师的记分板，见 `MEMORY/docs/adventure-scoreboard.md`。北极星是“一个模型、原版难度通关冒险模式”，记分板就是它的直接度量。每次 qc 晋升新学生都应该能重测。

## 要做的

1. 把 task101 的测量做成 `scripts/adventure_scoreboard.py`：
   - 参数：学生模型、叶值、静态网、种子数、关卡子集、worker 数、只测学生 / 只测老师；
   - 输出 json 与 md，格式与现有记分板一致；
   - 可以和上一份记分板做差异对比：每关胜率变化，以及 Wilson 区间是否分离。
2. 种子从 DESIGN 登记的新段取。同一份记分板内，两臂用同一批种子。不和门禁面板、训练种子重叠，在 DESIGN 登记段号。
3. 测试：
   - 小参数冒烟：2 关 × 2 种子；
   - 差异对比的单元测试。
4. 用 r11 在 Mac 上跑一份完整记分板：只测学生臂，每关 32 种子，用于和 task101 对比方差。
   - 台式机的 qc 现役学生由主控换上后再跑；
   - 先在本文件里写清台式机的运行命令。
5. 分支 `scoreboard2`，推送；合并由主控决定。

最多 8 个游戏进程。不要手写规则。

## 台式机运行命令

主控换上 qc 现役学生后，将其绝对路径传入 `STUDENT_MODEL`，在台式机的 scoreboard2 检出目录执行。本任务按主控最新指示使用 2 个游戏进程。默认同时测学生与老师，两臂同种子。

台式机目前**没有** scoreboard2 工作树，第一次运行先建检出（不合并 pvz-env）：

```bash
git -C /home/newbiexvwu/PvZ-search fetch origin scoreboard2
git -C /home/newbiexvwu/PvZ-search worktree add -b scoreboard2 \
  /home/newbiexvwu/PvZ-search/.claude/worktrees/scoreboard2 origin/scoreboard2
```

```bash
REPO=/home/newbiexvwu/PvZ-search/.claude/worktrees/scoreboard2
PYTHON=/home/newbiexvwu/.venvs/ml/bin/python3        # 系统 /usr/bin/python3 没有 torch
STUDENT_MODEL=/主控指定的现役学生/model.pt
ENGINE=/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build5/pvz-portable

"$PYTHON" "$REPO/scripts/adventure_scoreboard.py" \
  --student "$STUDENT_MODEL" \
  --value /home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt \
  --static-value /home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt \
  --executable "$ENGINE" \
  --resource-dir /home/newbiexvwu/.cache/pvz-research-resources \
  --seeds 32 --workers 2 \
  --previous "$REPO/MEMORY/docs/adventure-scoreboard-r11-32.json" \
  --out "$REPO/artifacts/task103/qc-scoreboard"
```

`--student-only` / `--teacher-only` 分别只测一臂。台式机实际运行由主控安排，本任务不切换 qc、不在 qc 忙时插队（核对时 load average 20.4、4 个 curriculum_queue 进程）。

路径已按台式机实况核对（2026-10-26 只读检查，未动 tmux qc）：

| 输入 | 状态 |
|---|---|
| `cq/build5/pvz-portable` | 存在，SHA `76ff36dc…`，与 deploy14 在役引擎一致 |
| `~/PvZ-search/build5/pvz-portable` | **不存在**（早先那版命令写的就是这里，已改） |
| `~/PvZ-search/build/pvz-portable` | 不存在（search-wt 工作树没有引擎） |
| `value_net/v5/model.pt` | 存在，SHA `8a636d33…`，与 Mac r11-32 的叶值同字节 |
| `hard/value_branch1/model.pt` | 存在，SHA `24858410…`，与 Mac 的静态网同字节 |
| `/usr/bin/python3` | 有，但**没有 torch**，必须走 `~/.venvs/ml` |
| 台式机本地 `scoreboard2` ref | 无；`ls-remote` 能看到 `baf2d443`，所以先 `fetch` 再 `worktree add -b` |
| `~/.cache/pvz-research-resources` | 存在，含 `main.pak` |

因此学生 / 叶值 / 静态网三项与 Mac 那份可比，**只有引擎不同**（台式机 `76ff36dc…` vs Mac `6f7d2c3f…`）。跑完先核 `scoreboard.json` 的 `engine_sha256`：跨引擎的胜率差不能全归给模型。

**2026-10-26 台式机实跑冒烟**：按上面命令建好检出后，用 `--levels 1 --seeds 1 --seed-offset 9000 --workers 1 --student-only` 跑了 1 局（学生用台式机现役 `dw4i_s1`），RC=0、1 胜，`scoreboard.json`/`.md` 正常生成，记录的 `engine_sha256` 为 `76ff36dc…`、叶值与静态网为 `8a636d33…`/`24858410…`，与 Mac 同字节。命令本身已验证可跑，正式测量只剩换 `STUDENT_MODEL` 与种子数。

注意：**台式机上没有 r11 的检查点**（Mac 那份在 `worktrees/fog36/artifacts/task79/models/r11/`，台式机三个候选路径都不存在），现役学生是 `dw4i_s1`（`12270380…`）。正式测量时 `STUDENT_MODEL` 必须传主控指定的现役路径，不要照抄 Mac 的 r11 路径。

**续跑**：每关的 `runs/<task_id>/games.jsonl` 已完整覆盖（臂, 种子）且全部终局时，生成器跳过该关、只重算汇总；中断后重跑同一条命令即可，不会重开已完成的对局。2026-10-26 在台式机用 1 关冒烟目录验证：重跑耗时 2 秒，局数保持 1→1。

## 结果

2026-10-26：已实现可重复入口与冻结任务清单，支持分臂测量、逐关差异比较和 Wilson95% 区间分离判定。种子块 [103000000,103500000) 已审计并登记 DESIGN §14.35。差异比较及汇总测试 7 项通过，L1/L12 × 2 种子 × 双臂冒烟共 8 局全部到终局，JSON/MD 已生成。

Mac r11 学生单臂全量测量采用每关 32 种子、2 个游戏进程，运行目录为 `artifacts/task103/r11-32/`。主控解冻原进程树 PID 90913 后继续完成，未重启。2026-10-26 09:23（America/New_York）确认进程退出，45关共1440局全部到终局。正式结果为 [JSON](adventure-scoreboard-r11-32.json) 与 [记分板及逐关对比](adventure-scoreboard-r11-32.md)，保留 task101 的独立样本。

与 task101 学生臂相比，总胜局由408/720（56.67%）变为814/1440（56.53%），相差−0.14个百分点。45关 Wilson95% 平均区间宽度由27.33个百分点缩至18.19个百分点，缩窄33.44%，逐关区间分离0/45。最大胜率变化为 L23 −21.875个百分点（15/16→23/32）与 L38 +21.875个百分点（11/16→29/32），两关区间均重叠。新旧种子独立，未将区间重叠解读为配对显著性结论。本次22关学生胜率≥80%，老师臂未重测。

按 task110 调查结论更正 L25 为“原生已接入，本面板未测”，同步 task101 的 JSON/MD 与生成器。保留45关冻结任务清单、跳过关卡记录及 task55旧状态列，没有新增 L25 样本。7项统计测试通过，既有 L1/L12 × 2种子 × 双臂冒烟结果保留，不重复开局。

2026-10-26 接手复核：Codex 额度中断时已完成全部实现与 Mac 测量，改动仅暂存未提交、分支未推送。接手后逐项复核：7 项测试通过；记分板 JSON 重算得学生 814/1440（56.53%）、task101 408/720（56.67%）、平均 Wilson95% 宽度 18.19% vs 27.33%（缩窄 33.44%）、逐关区间分离 0/45、L23 −21.875pp 与 L38 +21.875pp、22 关学生≥80%、L25 分类已更正，均与正文一致；暂存区无 .pt/.npz/.log，artifacts 忽略自查为空。未重跑任何对局，直接提交并推送 scoreboard2。

2026-10-26 主控指示交接（Codex 额度恢复）：全部改动已提交并推送，分支 tip `cbd4acd3`，工作区干净，台式机无遗留进程与临时脚本。任务书 5 项均完成，另补三件：台式机命令实机核对（venv/引擎/worktree）并 1 局冒烟跑通、跨引擎对照（同模型同种子 1440 局，引擎差异无可测影响）、关卡级续跑。剩余仅“主控换上现役学生后的 45×32 正式测量”（需学生路径）与合并决定。

2026-10-26：加关卡级续跑。`runs/<task_id>/games.jsonl` 已完整覆盖（臂, 种子）且全部终局时跳过该关、只重算汇总，避免中断后从第一关重来（本任务的 r11 测量就撞上过 SIGSTOP）。新增 3 项测试（完整/缺局/重复/非终局/错臂/缺失/损坏文件），全套 10 项通过；台式机用已有冒烟目录实测重跑 2 秒、局数 1→1，未重开对局。

2026-10-26：台式机跨引擎对照完成。用 r11（台式机 `worktrees/multiseed/artifacts/task92b/v1_r11/model.pt`，SHA `1cf82f5e…` 与 Mac 同字节）、同一批 32 种子、`--student-only --workers 2`，在 deploy14 引擎 `76ff36dc…` 上跑满 45 关 1440 局，rc=0。结果 [JSON](adventure-scoreboard-r11-desktop-engine.json) / [记分板](adventure-scoreboard-r11-desktop-engine.md)：总胜局 814/1440，与 Mac 引擎 `6f7d2c3f…` 那份**完全相同**；45 关里 39 关逐关胜局一致，其余 6 关各差 1 局（L21/L23/L27 −1，L24/L26/L29 +1）正负抵消，逐关 Wilson95% 分离 0/45，平均区间宽度 18.19%→18.31%。结论：引擎版本对本面板无可测影响，跨机器比较可用。本次测的是任务书点名的 r11，不是台式机现役 `dw4i_s1`，所以“主控换上现役学生后的正式测量”仍未做。

2026-10-26：台式机命令实跑冒烟通过。按文档建好 scoreboard2 检出后，1 关 × 1 种子 × 学生单臂、`--seed-offset 9000`、1 worker，RC=0 且 1 胜，json/md 正常生成；记录的 `engine_sha256` 为 `76ff36dc…`（deploy14 在役引擎），叶值 `8a636d33…`、静态网 `24858410…` 与 Mac 同字节。顺带确认台式机上没有 r11 检查点，现役是 `dw4i_s1`，正式测量须由主控指定 `STUDENT_MODEL`。

2026-10-26：台式机命令按实机核对后重写（只读检查，未动 tmux qc）。查出三处会直接失败的地方：`~/PvZ-search/build5/pvz-portable` 不存在（deploy14 在役引擎在 cq 工作树的 `build5/`，SHA `76ff36dc…`）；系统 `/usr/bin/python3` 没有 torch，须走 `~/.venvs/ml/bin/python3`；台式机没有 scoreboard2 工作树，需先 `git worktree add`。叶值 `8a636d33…`、静态网 `24858410…` 与 Mac r11-32 同字节，只有引擎不同（`76ff36dc…` vs `6f7d2c3f…`）。台式机本地还没有 scoreboard2 ref（`ls-remote` 能看到），所以命令先 `fetch` 再 `worktree add -b`；资源目录与 `main.pak` 已在位。

2026-10-26：修正台式机命令的引擎路径。qc 从 cq 工作树跑、deploy14 把引擎拷到 `cq/build5/`，原命令写的 `~/PvZ-search/build5/` 少了一层 cq 工作树；同时把 `REPO` 改为 `git rev-parse --show-toplevel`，避免在 search-wt 工作树里执行本分支的脚本。命令块已过 `bash -n`，并补了跑完先核 `engine_sha256`/`model_sha256` 的说明（台式机引擎与 Mac 不同，跨引擎差异不能全归给模型）。

## 问题

此前 Mac 出行暂停时主控以 SIGSTOP 冻结 PID 90913，落盘719/1440局。该暂停已由主控解除，原进程续跑及剩余721局汇总均已完成。

本任务 Mac 测量、正式汇总与 task101 对比已完成。台式机 qc 测量仍按上方命令由主控安排，本分支未切换 qc、未启动台式机测量。

台式机命令原先把引擎写成 `~/PvZ-search/build5/`（少一层 cq 工作树），2026-10-26 已修正；命令块过 `bash -n`。
