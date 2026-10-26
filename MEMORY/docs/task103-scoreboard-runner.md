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

主控换上 qc 现役学生后，将其绝对路径传入 `STUDENT_MODEL`，在 **scoreboard2 的检出目录**执行（不是 `~/PvZ-search`——那是 search-wt 分支的工作树）。本任务按主控最新指示使用 2 个游戏进程。默认同时测学生与老师，两臂同种子。

```bash
REPO=$(git rev-parse --show-toplevel)          # scoreboard2 的检出目录
STUDENT_MODEL=/主控指定的现役学生/model.pt
ENGINE=/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build5/pvz-portable   # deploy14 在役引擎

python3 "$REPO/scripts/adventure_scoreboard.py" \
  --student "$STUDENT_MODEL" \
  --value /home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt \
  --static-value /home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt \
  --executable "$ENGINE" \
  --resource-dir /home/newbiexvwu/.cache/pvz-research-resources \
  --seeds 32 --workers 2 \
  --previous "$REPO/MEMORY/docs/adventure-scoreboard-r11-32.json" \
  --out "$REPO/artifacts/task103/qc-scoreboard"
```

`--student-only` / `--teacher-only` 分别只测一臂。台式机实际运行由主控安排，本任务不切换 qc。

引擎路径说明：qc 从 cq 工作树跑，deploy14 把引擎拷到 `cq/build5/`（见 `MEMORY/docs/task70-engine-redeploy.md`）；仓库文档里没有 `~/PvZ-search/build5/`，早先那版命令少了一层 cq 工作树，已改。要换引擎就改 `ENGINE`，脚本会把实际 SHA256 记进 `scoreboard.json`。

跑完先核 `scoreboard.json` 的 `engine_sha256` 与 `model_sha256`：台式机引擎与 Mac 那份 r11-32（`6f7d2c3f…`）不同，跨引擎的差异不能全归给模型；学生 / 叶值 / 静态网三项 SHA 应与 r11-32 的 `1cf82f5e…` / `8a636d33…` / `24858410…` 逐项对照。

## 结果

2026-10-26：已实现可重复入口与冻结任务清单，支持分臂测量、逐关差异比较和 Wilson95% 区间分离判定。种子块 [103000000,103500000) 已审计并登记 DESIGN §14.35。差异比较及汇总测试 7 项通过，L1/L12 × 2 种子 × 双臂冒烟共 8 局全部到终局，JSON/MD 已生成。

Mac r11 学生单臂全量测量采用每关 32 种子、2 个游戏进程，运行目录为 `artifacts/task103/r11-32/`。主控解冻原进程树 PID 90913 后继续完成，未重启。2026-10-26 09:23（America/New_York）确认进程退出，45关共1440局全部到终局。正式结果为 [JSON](adventure-scoreboard-r11-32.json) 与 [记分板及逐关对比](adventure-scoreboard-r11-32.md)，保留 task101 的独立样本。

与 task101 学生臂相比，总胜局由408/720（56.67%）变为814/1440（56.53%），相差−0.14个百分点。45关 Wilson95% 平均区间宽度由27.33个百分点缩至18.19个百分点，缩窄33.44%，逐关区间分离0/45。最大胜率变化为 L23 −21.875个百分点（15/16→23/32）与 L38 +21.875个百分点（11/16→29/32），两关区间均重叠。新旧种子独立，未将区间重叠解读为配对显著性结论。本次22关学生胜率≥80%，老师臂未重测。

按 task110 调查结论更正 L25 为“原生已接入，本面板未测”，同步 task101 的 JSON/MD 与生成器。保留45关冻结任务清单、跳过关卡记录及 task55旧状态列，没有新增 L25 样本。7项统计测试通过，既有 L1/L12 × 2种子 × 双臂冒烟结果保留，不重复开局。

2026-10-26 接手复核：Codex 额度中断时已完成全部实现与 Mac 测量，改动仅暂存未提交、分支未推送。接手后逐项复核：7 项测试通过；记分板 JSON 重算得学生 814/1440（56.53%）、task101 408/720（56.67%）、平均 Wilson95% 宽度 18.19% vs 27.33%（缩窄 33.44%）、逐关区间分离 0/45、L23 −21.875pp 与 L38 +21.875pp、22 关学生≥80%、L25 分类已更正，均与正文一致；暂存区无 .pt/.npz/.log，artifacts 忽略自查为空。未重跑任何对局，直接提交并推送 scoreboard2。

2026-10-26：修正台式机命令的引擎路径。qc 从 cq 工作树跑、deploy14 把引擎拷到 `cq/build5/`，原命令写的 `~/PvZ-search/build5/` 少了一层 cq 工作树；同时把 `REPO` 改为 `git rev-parse --show-toplevel`，避免在 search-wt 工作树里执行本分支的脚本。命令块已过 `bash -n`，并补了跑完先核 `engine_sha256`/`model_sha256` 的说明（台式机引擎与 Mac 不同，跨引擎差异不能全归给模型）。

## 问题

此前 Mac 出行暂停时主控以 SIGSTOP 冻结 PID 90913，落盘719/1440局。该暂停已由主控解除，原进程续跑及剩余721局汇总均已完成。

本任务 Mac 测量、正式汇总与 task101 对比已完成。台式机 qc 测量仍按上方命令由主控安排，本分支未切换 qc、未启动台式机测量。

台式机命令原先把引擎写成 `~/PvZ-search/build5/`（少一层 cq 工作树），2026-10-26 已修正；命令块过 `bash -n`。
