# task142：把 task137/138/140/141 合成一次部署（deploy14p）

主控 2026-10-28 派给 Mac Qoder。集成分支 `prodint` 已由主控合好并推送（b2b36cc9 = deckcells + groupwidth + fixtest + coalesce，只在日志文件有冲突，已解决，控制器自动合并、可编译）。

## 要做的

1. 在 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/prodint` 跑**相关**单测，确认合并后四项改动没有互相打架：随机卡组格（task137）、可扩容 worker 池（task138）、持久化错误有界重试（task140）、合并排队训练（task141）。严格遵守 AGENTS §6.5 第 8 条：只用 `-k`，`timeout 300`，`ulimit -f 2000000`，同一时间只开一个 pytest。合并引入的问题就在 `prodint` 上修。
2. 写 `scripts/qc_deploy14p.sh`，取代 14l / 14m / 14n / 14o 四个脚本：
   - **基线**：台式机生产文件当前的 sha256 在 `/Users/newbiexvwu/pvz-logs/live_prod_shas.txt`（控制器 = cf94a8cb，挑选三件 = task136 的 6dae2c2e）。预检只核对这些。
   - **替换**：`prodint` 相对生产所改动的全部运行文件，包括 14l 的新文件（`deck_rules.py`、`make_random_deck_cells.py`、`configs/random_deck_cells_v5.json`）、`growable_pool.py`、`search_play.py`、控制器、挑选面板等。逐个列清楚。
   - **start_qc**：以台式机现用的 `/Users/newbiexvwu/pvz-logs/live_qc_deploy14l.sh` 为准（主控改过：`EXTRA_ARG` 由 `qc_deploy14l.mode` 决定）。新增一个模式文件 `qc_deploy14p.mode`：激活时写 `deckcells`，让 `--extra-manifest` 生效；未激活和回滚分支等于当前生产参数。
   - **stop_qc 不排空**：直接 `tmux kill-session -t =qc`，再按 PID 结束 `search_play` / `train_rollout_policy`，先确认 `/tmp/tmux-1000/default` 存在。主控会在门禁决定刚出来的时刻执行。
   - **首组 rc 超时不回滚**，只记日志；只有 `rc != 0` 或 qc 退出才回滚。
   - 预检要便宜：只做 sha256、ast 解析和面板条目数。**不要**跑 dry-run、对局或全量测试。
3. 推送 `prodint`，结论写在本文件末尾：改了哪些文件、测试结果、部署脚本要点。不碰台式机。
