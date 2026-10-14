# 通宵 Codex 任务检查 — 执行记录

## 2026-10-10 11:55（首次执行）

- **任务**：检查台式机通宵跑的 Codex 任务（RL 微调猜想验证），整理成说人话汇报。
- **结果**：Codex **仍在跑**（未结束）→ 未重启、未打断。GPU 队列做到 **B8**。
- **连接**：`scripts/win_ssh.py wsl`（PVZ_DESKTOP_HOST/PORT）。首次 ssh 报 connection refused，
  重试即通（端口转发瞬时抖动，**不必惊慌，重试 3 次**）。
- **进度**：A1–A8 完成；A9 跑完 v3 起点+100k 两个节点后被暂停给 B8 让路；
  B1–B5 失败（CUDA current_stream / B2 显存 OOM）、B6/B7 完成且不显著、B8 在跑、B9/B10/criticwarm 排队。
- **独立复核**：自己重算了基线(217/256=84.77%)、B6/B7/B5/A9(1024)/A2/idle，与报告**逐项吻合**。
  → **codex 报告可信**。
- **交付**：`MEMORY/docs/reports/codex-night-report-2026-10-10.md`；结论已追加到 `memory/2026-10-10.md`。
- **遗留风险**：① 源码修复未提交、备份只在 `/tmp`（易失）；② 队列 100k 预算太短（A9 证明 100k 无变化）。

## 下次执行要点

- 若 codex 已结束（`grep -a "^EXIT=" ~/codex-run.log` 有输出），读 `~/codex-report.md` 的最终版，
  重点看 **B8/B9/B10/criticwarm** 结果与 A9 的 1024 局完整曲线。
- 提醒用户：**替 codex 推它的 commit**（它 commit 但不 push）。
- 报告文件命名沿用 `MEMORY/docs/reports/codex-night-report-YYYY-MM-DD.md`（不放进仓库根目录，
  遵守 AGENTS.md §5 文档纪律）。
