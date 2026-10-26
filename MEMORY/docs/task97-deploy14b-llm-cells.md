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

- 2026-10-26 11:51：第 4 步完成，`scripts/qc_deploy14b.sh` 已原样安装到 `/home/newbiexvwu/pvz-logs/qc_deploy14b.sh`，仅写入，未执行。沿用 deploy14 的 `start_qc` 原文、build5、469 项面板、gate v3 与现行代码/训练热修复；预检锁定 baseline/config/rates/引擎/12 存档 SHA。组边界停妥后备份配置/rates/state，合并 11 格；隔离 state dry-run 全注册后重启，只有新完成组 rc=0 才记成功；失败回滚到部署前 deploy14 数据，代码与引擎保持原样。`bash -n`、3 个内嵌 Python 语法检查、只读预检、启动函数逐字一致检查通过；隔离 rates 合并保留全部旧字段/底数，基底 SHA 不匹配正确拒绝。上一小步提交 `1b11647c` 已 push。

- 2026-10-26 11:47（Codex 续做）：已读 CodeBuddy 三笔交付，不重跑对局。qc_setup.log 确认 deploy14 在 08:55:47 首组 rc=0 成功，08:37 的回滚阻塞已解除。复核 cq 12 份 verified 存档的 ticks SHA 与原 zip 的 12 个通关局逐份相符；11 格 ×8 原始 games/config 与生产模型 SHA、起点决策、未来种子和终局逐项一致。冻结 `configs/deploy14b/`：以 cq 的 deploy14 配置为底，仅合入 11 个 LLM 起点格（67→77 个任务，已有 L47 只增加 demo_start），不夹带 task81 的 7 个倍率起点格。469 项面板原样不变，新增未来块 2816 种子彼此、与旧起点块和面板均不相交。用 cq 现行控制器、build5、冻结配置/rates、正式 state 的本树拷贝 dry-run EXIT=0，11/11 全注册，未创建门禁账本。结论和输入 SHA 见 `task97-deploy14b-audit.json`。

- 2026-10-26 08:2x（CodeBuddy 接手 cbp-task97）：合并 origin/pvz-env 进 redeploy（4 个 MEMORY/docs 文档与 student-bottleneck-plan 取 origin 版、2026-10-25/26 日志取并集），push 5dda229f。deploy14 状态：qc_deploy14.sh 在跑（07:41 preflight OK + stop_curr set，等 qc 当前轮结束重启），尚未见 first group rc=0。第 1 步完成：llm_free_out_numeric.zip 里 12 份通关存档（full/half 判定）经 scripts/import_llm_run.py 入库 cq/artifacts/llm_demo/raw/seed_<种子>/<task_id>/，deploy14 引擎 verify 12/12 通过，无新族登记（manifest 未动）；configs/human_demo_tasks.json 里 11 个 LLM demo_start 格引用路径 11/11 全解析。

- 2026-10-26 08:40（cbp-task97）：第 3 步完成。面板按 B8.3 同一生成器重生成到临时文件逐项比对：469 任务 0 删 0 增 0 变更、seed_overlap 0（仅 generation_audit 的 scanned_manifests/scanned_seeds 计数因磁盘文件增多而不同）——deploy14b 无需换面板，deploy14 的 469 任务面板原样沿用。dry-run：redeploy 树 scripts + 现行面板 + 76ff36dc + panel_v3 + state 拷贝（--state/--log/--stop-flag 全部指向 /tmp）→ EXIT=0，31 个起点格全部注册，11 个 LLM 格全部 start cell added（demo_roof_46_doom_m200_llm 3 档、46_s20261047 6 档、46_s20261048 6 档、47_doom_m200 6 档、47_s20261048 5 档、47_s20261049 5 档、48_doom_m200_llm 7 档、48_s20261049 6 档、48_s20261050 7 档、49_s20261050 2 档、49_s20261051 1 档）。12 份 LLM 存档同步拷入本树 artifacts/llm_demo/raw/（dry-run 按 ROOT 解析存档；cq 侧为部署目标，两处都有）。

- 2026-10-26 08:50（cbp-task97）：第 2 步 teacher_rates 完成但整体**按 ROLLBACK 铁律停下**（见问题 2）。11 个 LLM 格各 8 局老师冷启动（deploy14 引擎 76ff36dc，生产模型 dw4i_s1 + v5 + 台式机 branch1，β=1，起步 = 阶梯 index 0 决策、种子取该格未来块前 8 个；证据 artifacts/task70/b8_teacher_rates/teacher_rates.json，13 个旧格自动跳过）：46_llm **7/8**、46_s20261047 **8/8**、46_s20261048 **8/8**、47_doom_m200 **8/8**、47_s20261048 **8/8**、47_s20261049 **8/8**、48_llm **8/8**、48_s20261049 **8/8**、48_s20261050 **8/8**、49_s20261050 **1/8**、49_s20261051 **7/8**。待 deploy14 重跑成功后可写入 ~/pvz-logs/teacher_rates.json 的片段：

```json
"demo_roof_46_doom_m200_llm": {"wins": 7, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_46_doom_m200_s20261047": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_46_doom_m200_s20261048": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_47_doom_m200": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_47_doom_m200_s20261048": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_47_doom_m200_s20261049": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_48_doom_m200_llm": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_48_doom_m200_s20261049": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_48_doom_m200_s20261050": {"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_49_doom_m200_s20261050": {"wins": 1, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"},
"demo_roof_49_doom_m200_s20261051": {"wins": 7, "games": 8, "model": "dw4i_s1 + v5 + desktop branch1", "source": "T97"}
```

第 4 步（qc_deploy14b.sh）与第 5 步（B10 节）未做：deploy14 未完成、已 ROLLBACK，按任务书铁律停在此处。

## 问题

- 2026-10-26 11:47：先前 ROLLBACK 已由主控重跑解除（08:55:47 OK；11:43 另有训练 plane_of 热修复）。deploy14b 只改本任务配置与 rates，保留当前 scripts/python/recipes 和引擎；回滚也不把热修复降回旧版本。redeploy 接手时已有 `python/test_curriculum_queue.py` 未提交改动，保留，不并入本任务提交。

- 2026-10-26 08:35（cbp-task97）：**deploy14 卡住，非本任务原因**。qc 于 06:50 在 deploy14 启动（07:41）之前自行退出，qc.out 末行 `demo_fog_32_snow_m200: seeds exhausted (offset 216 + 48 > 256)`（控制器 planning 失败退出，tmux 会话消失）。qc_deploy14.sh（PID 803586）的 stop_qc 死等 curriculum.log 末尾出现 "stopping"，而 qc 已死不会再打——脚本卡在第一个等待循环 50+ 分钟，永远到不了完成或 ROLLBACK。不碰 tmux qc、不代执行部署；已挂监听，若主控侧解除卡住（qc_setup.log 出现 deploy14 新行）即继续。处置：qc 已死、无资源争用，第 2 步 teacher_rates（自用 2 进程）先行。
- 2026-10-26 08:50（cbp-task97）：**deploy14 重跑后 dry-run 失败 → ROLLBACK（08:37:22），按任务书铁律 task97 在此停下**。经过：主控在 origin/deploy14fix（0b8dcc26，修 qc 种子块用完即退，正是 06:50 宕机根因）上更新了 qc_deploy14.sh（REV=0b8dcc26、stop_qc 增加 qc-already-down 分支）并于 08:37 重跑：teacher_rates +18 bases 成功、cq took 0b8dcc26 成功，dry-run 失败——`FileNotFoundError: cq/artifacts/task79b/fog79b_panel.json`（load_panel 顺着面板/清单引用找 family 子清单，cq 树 artifacts 不进 git、缺这份文件；其他树有）。08:37:27 已回滚到 deploy13（7dc58414 lineage + build4），qc 重生并正常运行。修复方向（供主控参考）：把 fog79b_panel.json（以及其他 load_panel 引用的 artifacts 子清单）拷进 cq，或让 cq checkout 一并覆盖 artifacts 里被引用的清单文件。
- 2026-10-26 08:35（cbp-task97）：deploy14 第一次执行卡住（非本任务原因）。qc 于 06:50 在 deploy14 启动（07:41）之前自行退出，qc.out 末行 `demo_fog_32_snow_m200: seeds exhausted (offset 216 + 48 > 256)`；qc_deploy14.sh 的 stop_qc 死等 curriculum.log 的 "stopping" 行，脚本卡住 56 分钟（07:41–08:37）。后由主控侧解除（deploy14fix 重跑），已消解。
- 2026-10-26 08:45（cbp-task97）：teacher_rates 有一个值得主控注意的数：`demo_roof_49_doom_m200_s20261050` 起点格老师只有 **1/8**（阶梯 index 0 起步），远低于其他 10 格的 7–8/8；若 deploy14b 要上，建议复核这一格是否保留（task87 扫描时该局 50% 线本来就只有 2 档）。
