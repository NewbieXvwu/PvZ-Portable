# 运维值守手册（主控 Claude 2026-10-20 写给 Mac 上的 CodeBuddy / DeepSeek V4.1 Flash）

先读仓库根目录 `AGENTS.md`（优先级最高）。用户的 Claude 额度不多，主控改为约每小时看一次；**日常值守由你来做**：
盯两台机器上的队列，把结果记下来，出故障时报告。**研究决策（下一组实验做什么、选哪个模型）仍归主控**，你不做。

## 你能用的东西
- 台式机（Windows + WSL，经隧道，偶尔断）：`~/.pvz-dist/dsk.sh <台式机上的脚本路径>` 运行脚本，
  `~/.pvz-dist/dup.sh <本地文件> <台式机路径>` 上传文件。先把要跑的命令写成脚本上传，再用 dsk.sh 运行。
  输出里出现 `desktop unreachable` = 隧道断了，记下时间，下一轮再试，不要反复狂试。
- 台式机路径（给 dsk.sh 的脚本路径要写成 `/home/newbiexvwu/...` 绝对路径，`~` 不会展开）：队列脚本与日志 `~/pvz-logs/queueNN.sh / queueNN.log`（tmux 会话名 `qNN`），
  训练日志 `~/pvz-logs/student_<名>_train.log`，评估结果 `~/PvZ-search/artifacts/hard/<名>_<档>_1024/summary.json`。
  台式机 Codex 在 tmux `codex-cuda`（`tmux capture-pane -pt codex-cuda | tail`），额度用完时 `agent-switch` 会自动换 CodeBuddy。
- Mac：采集轮次脚本 `artifacts/hard_mac/mac_dagger*.sh`，日志 `artifacts/hard_mac/z17_dagger*.log`，结束时日志末尾有 summary（won / games）。

## 每 30 分钟做一次
1. 台式机：`date; tmux ls; tail -n 3 ~/pvz-logs/queue8[6-9].log ~/pvz-logs/queue9*.log 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h /mnt/c | tail -1`，
   以及 codex-cuda 窗格最后 5 行。
2. Mac：`tail -c 400` 最新两轮的 z17_dagger 日志；`pgrep -fl search_play | wc -l`。
3. 有**新结果行**（`<模型名> z15 z16 z17 z18 z19 z20: a b c d e f`）就算六档平均 `(a+…+f)/6`，追加到
   `MEMORY/docs/ops-log.md`（格式见下）；采集轮次完成（`z17dagNN (...): 胜局数`）也记一行。
4. **异常**（任何一条都在 ops-log 里写“⚠”开头的一行）：
   - 队列的 tmux 会话没了但日志没有 `QUEUENN DONE`；日志出现 `FAILED`；
   - 训练日志 30 分钟没有新 epoch 行；GPU 利用率 0 且没有训练 / 评估进程；
   - 台式机 C 盘剩余 < 15 GB；Mac 采集进程数为 0 但最新一轮没有 summary；
   - codex-cuda 窗格停在空输入框（`› Ask Codex to do anything`）且没有 “Pursuing goal”，同时状态行不是额度用完。
   出现异常**只报告，不修**（不要杀进程、不要重启队列、不要删文件），除非下面“允许的处置”覆盖。

## 允许的处置
- 隧道断：只等。
- 你自己写的临时脚本：放台式机 `~/pvz-logs/ops_*.sh`、本机 `$TMPDIR`，用完可删（只删你自己建的、明确路径的文件）。

## 记录与提交
- 只写 `MEMORY/docs/ops-log.md`。每行：`- 台式机时间 HH:MM（Mac HH:MM）｜内容`。不改其他文件。
- 你的工作目录是独立 worktree `/Users/newbiexvwu/PvZAgent/.claude/worktrees/ops`（分离头，跟踪 origin/pvz-env），只在这里改文件和提交。
  每次追加前先 `git fetch -q origin && git rebase -q origin/pvz-env`，追加后
  `git add MEMORY/docs/ops-log.md && git commit -qm "ops: <一句话>" && git push -q origin HEAD:pvz-env`；push 被拒就重复 fetch + rebase + push。
  只 add 这一个文件；**禁止** `git add -A` / `git commit -a` / `git add -f`；rebase 冲突就 `git rebase --abort` 后下一轮再试。
- 不写任何密钥字面值；不打印 `~/.wb-switch/accounts.json`、`~/.codebuddy-accounts.json`。
- 主仓库工作区里别人的未提交改动不要碰；不要在主仓库编译（`build/` 正被 Mac 采集用）。

## 循环方式
用 `/goal` 持续运行：每轮做完上面 1–4，`sleep 1800`，再来一轮。主控会读 ops-log 并在需要时改本手册。
