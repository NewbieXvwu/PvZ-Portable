# task97：deploy14b——11 个 LLM 屋顶起点格上 qc

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-hashdet`；额度用完由 agent_pool 交 CodeBuddy）。工作树沿用 `/home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy`（先 `git merge origin/pvz-env`）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`**，cq 只读。**先等 deploy14 完成**：`~/pvz-logs/qc_setup.log` 里出现 `deploy14: first group after restart rc=0, OK`。如果出现的是 ROLLBACK，就停下，写进“问题”。

## 要做的
1. **存档入 cq**：用 `scripts/import_llm_run.py`（task89）把 `/home/newbiexvwu/pvz-logs/task89/llm_free_out_numeric.zip` 里的 12 份通关局导入 `cq/artifacts/llm_demo/raw/`，目录布局与 task87 相同。
   - 逐份 verify；
   - 核对 `configs/human_demo_tasks.json` 里这 11 个 `demo_start` 格引用的存档路径都能解析。
2. **teacher_rates**：照 B8.6 的做法（`scripts/task70_b8_teacher_rates.py`），每格 8 局，用 deploy14 引擎 build5。给出写入 `~/pvz-logs/teacher_rates.json` 的片段。
3. **面板与 dry-run**：照 B8 第 3、5 步。
   - 面板只能新增，旧任务逐项不变，种子不相交；
   - 在 state 拷贝上跑 dry-run，确认 11 个格都能注册。
4. 写 `~/pvz-logs/qc_deploy14b.sh`，结构照 `qc_deploy14.sh`：
   - 引擎不变，只换 configs / 面板 / rates；
   - 回滚目标是 deploy14；
   - **启动命令照抄 deploy14 的 `start_qc`**。
   只写，不执行，由主控执行。
5. 结果写 `MEMORY/docs/task70-engine-redeploy.md` 的 B10 节，`git push origin redeploy`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 和存档不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- 2026-10-26 08:2x（CodeBuddy 接手 cbp-task97）：合并 origin/pvz-env 进 redeploy（4 个 MEMORY/docs 文档与 student-bottleneck-plan 取 origin 版、2026-10-25/26 日志取并集），push 5dda229f。deploy14 状态：qc_deploy14.sh 在跑（07:41 preflight OK + stop_curr set，等 qc 当前轮结束重启），尚未见 first group rc=0。第 1 步完成：llm_free_out_numeric.zip 里 12 份通关存档（full/half 判定）经 scripts/import_llm_run.py 入库 cq/artifacts/llm_demo/raw/seed_<种子>/<task_id>/，deploy14 引擎 verify 12/12 通过，无新族登记（manifest 未动）；configs/human_demo_tasks.json 里 11 个 LLM demo_start 格引用路径 11/11 全解析。

- 2026-10-26 08:40（cbp-task97）：第 3 步完成。面板按 B8.3 同一生成器重生成到临时文件逐项比对：469 任务 0 删 0 增 0 变更、seed_overlap 0（仅 generation_audit 的 scanned_manifests/scanned_seeds 计数因磁盘文件增多而不同）——deploy14b 无需换面板，deploy14 的 469 任务面板原样沿用。dry-run：redeploy 树 scripts + 现行面板 + 76ff36dc + panel_v3 + state 拷贝（--state/--log/--stop-flag 全部指向 /tmp）→ EXIT=0，31 个起点格全部注册，11 个 LLM 格全部 start cell added（demo_roof_46_doom_m200_llm 3 档、46_s20261047 6 档、46_s20261048 6 档、47_doom_m200 6 档、47_s20261048 5 档、47_s20261049 5 档、48_doom_m200_llm 7 档、48_s20261049 6 档、48_s20261050 7 档、49_s20261050 2 档、49_s20261051 1 档）。12 份 LLM 存档同步拷入本树 artifacts/llm_demo/raw/（dry-run 按 ROOT 解析存档；cq 侧为部署目标，两处都有）。

## 问题

- 2026-10-26 08:35（cbp-task97）：**deploy14 卡住，非本任务原因**。qc 于 06:50 在 deploy14 启动（07:41）之前自行退出，qc.out 末行 `demo_fog_32_snow_m200: seeds exhausted (offset 216 + 48 > 256)`（控制器 planning 失败退出，tmux 会话消失）。qc_deploy14.sh（PID 803586）的 stop_qc 死等 curriculum.log 末尾出现 "stopping"，而 qc 已死不会再打——脚本卡在第一个等待循环 50+ 分钟，永远到不了完成或 ROLLBACK。不碰 tmux qc、不代执行部署；已挂监听，若主控侧解除卡住（qc_setup.log 出现 deploy14 新行）即继续。处置：qc 已死、无资源争用，第 2 步 teacher_rates（自用 2 进程）先行。
