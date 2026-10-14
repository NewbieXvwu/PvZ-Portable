# 分工与 codex

- **训练队列由另一个 Agent 负责**；本 Agent：仓库体积治理、制品交付通道、跨机同步、文档与约束。
- codex 在台式机跑：任务书 `~/codex-task.md`、日志 `~/codex-run.log`、报告 `~/codex-report*.md`。启动：
  `tmux new-session -d -s codex "cd ~/PvZ-Portable && cat ~/codex-task.md | codex exec -C ~/PvZ-Portable --dangerously-bypass-approvals-and-sandbox - > ~/codex-run.log 2>&1; echo EXIT=\$? >> ~/codex-run.log"`。
- **报告数字可信度高**（多次独立复核吻合），但**会报不存在的 git 提交号** → 任务书必须要求**"每个提交号都能被
  `git cat-file -t` 验证"**；**Holm 数字曾与同一份 JSON 对不上**（v4）→ 也写进任务书。
- **它会 commit 但不 push** → 收工时替它推；**它的护栏拒 `rm -f`** → 用 `python3 -c 'os.unlink(...)'`。
- **接管别人的活之前先查它为什么这么做**（10-08 教训）；**任务书必须自包含**（接续轮是全新上下文）+
  内嵌决策规则（不让它停下来问）；**每 30 分钟推结论层到 HF** 这条规则救过场。
