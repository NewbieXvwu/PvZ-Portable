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

## 结论（2026-10-28，Mac Qoder，分支 prodint）

### 合并后的相关单测

全部按 AGENTS §6.5-8 执行（`ulimit -f 2000000`、`timeout 300`、同一时间一个 pytest、只用 `-k`）：

| 范围 | 结果 |
|---|---|
| task137 随机卡组格 `test_random_deck_cells.py` | 11 passed |
| task138 可扩容池 `test_growable_pool.py` | 8 passed |
| task140 有界重试 `test_qc_resilience.py -k "round_boundary_retries or persistent_state_error"` | 2 passed |
| task141 合并排队 `test_curriculum_queue.py -k "supersede or backlog"` | 3 passed |
| 挑选/门禁集成 `test_selection_holdout.py` | 8 passed, 1 skipped |
| 挑选生命周期 `test_multiseed_production.py` | 15 passed |
| `test_curriculum_pipeline.py` 中用到选择桩的 9 个用例 | 9 passed |
| `test_qc_resilience.py -k twice_failed_background_snapshot` | 1 passed |
| `test_background_failure_retry.py` | 3 passed |

四项改动之间没有行为冲突。合并本身没有引入运行时问题；失败的是四处**旧测试桩/旧断言没有跟上分支语义**，已在 prodint 上修好（4beab441）：

1. `test_multiseed_production.py` 把 `load_panel` 桩成单参数，task137 改成 `load_panel(manifest, extra_manifest)` → 改成 `lambda *_`。
2. `test_curriculum_pipeline.write_selection` 只写 `total`，task137 的 holdout / 随机卡组汇总要读 `score['wins']` → 桩现在按挑选面板写 `wins={task_id: 0}`。
3. `test_multiseed_production` 仍按“38 冒险 + 3 随机 + 2 未见过 = 43 任务、416 局”断言；task137 的评估批是 8 套 v5 卡组 → 48 任务、456 局（`configs/random_deck_cells_v5.json` 登记 eval=8）。
4. `test_background_failure_retry` 仍期待“同一轮第二次失败就停 qc”；task98 起的行为是隔离快照、采集继续（控制器日志 `snapshot isolated, collection continues`）→ 断言改为隔离，且断言日志里不再有 `stopping qc`。

以上 4 条在 deckcells 基线上也同样失败，属于分支继承的过期测试，不是合并冲突。

### prodint 相对生产的运行文件（7 个，deploy14p 全部替换）

| 文件 | 来源 |
|---|---|
| `scripts/curriculum_queue.py` | task137 `--extra-manifest` / 随机卡组配额 + task140 有界重试 + task141 合并排队 |
| `python/pvz_selection_refresh.py` | task137 八套评估卡组、12 个随机训练格、按组汇总胜局 |
| `scripts/search_play.py` | task138 父进程调度、可扩容 worker 池 |
| `scripts/growable_pool.py` | 新增（task138） |
| `scripts/deck_rules.py` | 新增（task137） |
| `scripts/make_random_deck_cells.py` | 新增（task137） |
| `configs/random_deck_cells_v5.json` | 新增（task137，24 套登记：16 训练 + 8 评估） |

其余差异只有测试、`python/conftest.py`、部署脚本和文档，不参与生产运行，14p 不替换。本次不改 C++ 可执行文件。

### `scripts/qc_deploy14p.sh` 要点（5bd236fb）

- `REV=4beab441`（运行文件固定在该提交；已推送），逐文件用 `git show $REV:$file | sha256sum` 对照，并在预检加 `git cat-file -e $REV^{commit}`，**台式机需先 `git -C /home/newbiexvwu/PvZ-search fetch origin prodint`**，否则脚本只报 ABORT、不动机器。
- 基线只核对 `live_prod_shas.txt` 的 6 个生产文件；控制器 / `pvz_selection_refresh.py` / `search_play.py` 额外接受本次自己的 sha（5ef3377f / 616b5d5e / 4f4ba541），所以激活后重跑或 `--rollback` 不会卡在基线。`select_student.py`、`pvz_promotion.py`、`pvz_training_rows.py` 不在替换列表，仍只接受生产 sha。
- `start_qc` 逐字复制台式机现用 `live_qc_deploy14l.sh` 的四个分支（v2sleep / gatev5 / rebalance / 默认），只把 `EXTRA_ARG` 的来源从 `qc_deploy14l.mode` 换成 `qc_deploy14p.mode`；激活时写 `deckcells`，回滚时删除该文件，未激活分支参数与当前生产完全一致。
- `stop_qc` 不排空：确认 `/tmp/tmux-1000/default` 存在 → `tmux kill-session -t =qc` → 按 PID `TERM` 掉 `search_play` / `train_rollout_policy`。
- 首组 rc：等 120×30 秒；`rc=0` 成功退出；60 分钟还没出新组只记日志、qc 保持运行；只有 `rc != 0` 或 qc 退出才回滚（恢复备份文件、删模式文件、用生产参数重启）。
- 备份目录在全部 `.before` 写完后落 `backed_up` 标记；手动 `--rollback` 必须先看到该标记，避免从不完整的备份目录误删生产文件。
- 预检便宜：替换前只做文件 sha、提交存在性、可执行文件；替换后做 sha 校验、`ast.parse`、生成并核对面板条目（144 任务 / 16 格 / 294912 唯一 seed / 最低倍率 0.5）。没有 dry-run、对局或全量测试。
- 14l/14m/14n/14o 保留在仓库作历史记录，不再用于部署。

### 给主控

- 部署前：`git fetch origin prodint`。
- 执行时机按任务书：门禁决定刚写出时运行；脚本会直接杀 qc，最多丢当前组和至多一份打包。
- 若台式机当前不是 `live_prod_shas.txt` 的状态（例如还挂着 14l/14m/14n/14o），脚本会在预检 ABORT，不会覆盖；先恢复到生产 sha 再跑。
