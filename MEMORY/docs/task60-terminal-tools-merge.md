# task60：三条"接终局"诊断线合成一套，备好合入 pvz-env

> 主控 2026-10-24 10:05（台式机时间）写。执行者：台式机 Codex（plus 账号，`codex-merge` 会话）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge`（分支 `termmerge`，从 origin/pvz-env 拉出，主控已建好）。**以读代码、改代码、跑测试为主**，游戏进程最多 2 个，只用于复现核对；**不碰 tmux `qc`、不改别的工作树**。合入 pvz-env 由主控做。

## 为什么
- 三个分支各写了一套"在决策点对候选做续走到终局"的工具，都还没合入 pvz-env：
  - `fog`（task45 追加 E，16 个提交）：`fog_terminal_*`；
  - `roof`（task46 追加 B 与第三阶段，30 个提交）：`roof_terminal_*`、`terminal_candidate_audit.py`、`terminal_candidate_report.py`；
  - `forks`（task53，16 个提交）：`critical_forks.py` 及数据、评估脚本。task53 的任务书本来就要求"只有一种实现，三条线以后改用它，主控合并时删重复"。
- 这三个分支落后 pvz-env 49–161 个提交，越拖越难合。
- 它们还改了生产文件：
  - `python/pvz_env.py`、`python/pvz_frame_store.py`（fog）；
  - `python/pvz_training_pack.py`、`scripts/pack_training_rows.py`（forks）；
  - `scripts/search_play.py`、`scripts/episode_query.py`（roof）。

  其中 `pack_training_rows.py` 和 `curriculum_queue.py` 的同一条流水线，正在 `qcpack` 分支（task59）上改。

## 要做的
1. **以 `forks` 的 `critical_forks.py` 为唯一实现。**
   - 逐个读 fog 和 roof 的终局脚本，列出它们有、而 `critical_forks.py` 没有的能力（例如屋顶的候选审计、雾区的透视对照）。
   - 缺的补进 `critical_forks.py`。
   - 补齐后删掉重复脚本和它们的测试。
   - 各报告里的复现命令改成指向新实现，旧证据（json 与报告）原样保留。
2. **只想复现结论时**，用 2 个进程挑一两个点核对：屋顶取一个根，雾区取一个根，看新实现给出的终局结果与旧证据是否一致，写进结果。
3. **生产文件改动逐处审**，每处写一行：为什么需要、是否改变生产行为。
   - 改变生产行为、又不是诊断必需的，就撤掉。
   - 留下的必须由测试覆盖。
   - `pack_training_rows.py` / `pvz_training_pack.py` 的改动要与 origin/qcpack 对照，避免将来冲突；有冲突就写进"问题"，由主控定顺序。
4. **合并方式**：在 `termmerge` 上依次合入 forks、roof、fog，冲突逐个解决，保留双方的 MEMORY 日志条目。之后：
   - Mac 与台式机测试全部通过：`python -m pytest python/`，或至少相关的测试文件，写明跑了哪些；
   - 跑一次 `scripts/loose_ends.py`。
5. 推 `origin termmerge`，结果记在本文件。**主控审阅后合入 pvz-env。**

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`；引擎用 `/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3/build/pvz-portable`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写规则或公式。
- 不要 `git stash`；不要改写已推送的历史。
- 每完成一小步就提交、`git push origin termmerge`，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 统一实现第一步：`critical_forks.py` 补时间窗选根、旧轨迹导入、旧命名空间与 0→1 起决策转换、完整静态/halving/锚定/核验追踪；`critical_forks_data.py` 补阻断归因、human/xray 关卡/reset-seed 聚类、老师路径叶值与严格/阈值/非终局排序分列、逐指标多挑战者独立确认。`critical_forks_sampling.py` 仅保留旧计划的 CPU 取点/命名点去重，不含游戏或终局循环。
- 已先查引用并推 tag `task60-pre-terminal-unification`，再删 10 个重复脚本和 7 个重复测试文件；有价值的行为测试迁入 `test_critical_forks_unified.py`。旧 JSON 全部保留，报告追加新入口说明。
- 首轮公共与生产相关回归：42 passed（`test_critical_forks.py`、`test_critical_forks_unified.py`、`test_roof_counterfactual.py`、`test_frame_store_chunks.py`）；后续最终全套测试待记录。

### 能力逐项对照

| 原脚本 | forks 原有 / 本次补齐 |
|---|---|
| fog_terminal_diagnostic | 原有全候选、真实终局、收入/学生历史恢复、断点与 CPU/空间门禁；补旧轨迹和未来命名空间适配、8000/10000 tick 时间根 |
| terminal_candidate_audit | 原有铲子覆盖、老师搜索追踪；补 best/student 索引、核验前后值、要求样本数，统一 schema=2 |
| fog_terminal_report | 原有新未来确认、符号检验/Holm、价值数据导出；补每个终局指标/v4 最优候选确认和 human/xray 配对 seed 聚类；tick 只在双败时作为存活改善，赢局耗时仅描述 |
| terminal_candidate_report | 补候选生成遗漏/空间外铲子/halving/叶排序/初始门槛/核验拒绝/执行的互斥归因，静态保留另列重叠计数；少于两个有效 reset 聚类不报区间 |
| roof_terminal_pipeline | 纯取点功能迁 `critical_forks_sampling.py`：CPU 计时多阶段、整面板预算、命名点去重、保留 reset 覆盖的稀疏后备、冻结后不按新 CPU 改根；自动提交/推 roof 的旧一次性流程退役 |
| fog_terminal_pipeline | 公共 `screen → plan-confirm → confirm → report-confirm/diagnose` 显式阶段取代旧 watcher/自动推 fog；全量验证未过不能选确认；task53 的 PID+启动身份等待器保留 |
| roof_terminal_verify | 公共 panel 重建身份并验证精确候选×未来覆盖、根头、真实终局、tick、实际 reset 防线、全动作链 SHA/决策连续性；不信任标签自述身份 |
| 两个 repro 脚本 | `task53_repro.py` 共用语义比较，增加旧 schema 对照；世界/搜索未来与每一步老师动作精确比较，v4 Python/native 仅允许 1e-6 概率误差 |
| roof_terminal_diagnosis | 报告生成收敛为 `critical_forks_data.py diagnose` 的通用 JSON，含各关/观测臂、排序和阻断明细；旧叙述结论不重写 |

### 生产改动逐处审查（相对 task60 起点 b69024fa）

| 文件 / 改动处 | 必要性与生产影响 / 覆盖 |
|---|---|
| pvz_env.py: rollout_predict 的 trace OBS 屏障 | 诊断需等旧引擎关闭 trace；只在 trace_path 非空时增加只读往返，不改变动作或普通生产调用；`test_predict_trace_is_complete_when_response_precedes_stream_close` 与真实 fog native trace 测试 |
| pvz_frame_store.py: 固定 ZIP 日期 | 已撤回，不是诊断必需且会改变全部生产 NPZ 字节；最终与起点一致，原数组/流写回归保留 |
| pvz_training_pack.py: fingerprint_paths | 显式训练行必须进分段/全包身份，防旧包复用；普通 meta 无该键时指纹保持原值；`test_exact_training_rows_preserve_history_and_wait_confidence` |
| pvz_training_pack.py: training_rows | 为纠正标签保留完整真实前缀但只训练指定决策；正整数/递增/唯一校验；精确行及四项非法行回归 |
| pack_training_rows.py: import training_rows | 仅接入同一公共契约，不另造实现；精确行端到端打包测试 |
| pack_training_rows.py: explicit_rows + selected_mask | 诊断导出行不受 wait 随机抽样影响，越界报错；普通存档仍走原 mask；缓存/非缓存同帧同历史测试 |
| pack_training_rows.py: 两条 keep 路径 | 缓存 hints 与原始 JSONL 都应用同一显式行掩码；普通 mask 不变；同上双路径覆盖 |
| pack_training_rows.py: retention | 显式行已全保留，权重不能再乘 1/wait_keep；只改变 opt-in 纠正档案；wait_keep=.01、label_weight=.875 实测 |
| pack_training_rows.py: 缓存 count/weight | 缺有效标签必须报错、显式 wait 权重正确；同上缓存及非法标签回归 |
| pack_training_rows.py: 非缓存 weight | 与缓存路径同契约；同上双路径覆盖 |
| pack_training_rows.py: 非缓存计数校验 | 防指定行无有效标签被静默丢弃；非法标签回归 |
| episode_query.py: 基线与各候选终 tick 两处显示 | 诊断必须区分末波/死亡时间；仅展示文本，重放策略不变；`test_teacher_whatif_compares_rerun_baseline_and_reports_ticks` |
| search_play.py: fast 快路径 observer 条件 | 传 observer 的诊断必须经过逐决策回调；普通 fast 继续原生整局，fog 限制保留；`test_fast_play_observer_gets_each_decision_and_default_keeps_native_path` 两臂覆盖 |

### 现行复现命令

以下均在本树运行，`--out` 必须是新的绝对目录；源工作树仅只读取证。模型和原始证据继续留本机 ignored 目录/HF，不进 git。

```bash
PVZ_NO_POPUP=1 /home/newbiexvwu/.venvs/ml/bin/python /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/scripts/critical_forks.py /home/newbiexvwu/PvZ-search/.claude/worktrees/fog/artifacts/task45/phase-e/plan.json import-legacy --jobs /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/artifacts/task60/fog-selection.json --out /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/artifacts/task60/repeat-fog --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3/build/pvz-portable --workers 1
PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 /home/newbiexvwu/.venvs/ml/bin/python /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/scripts/critical_forks.py /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/artifacts/task60/repeat-fog/plan.json screen
PVZ_NO_POPUP=1 /home/newbiexvwu/.venvs/ml/bin/python /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/scripts/task53_repro.py /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/artifacts/task60/repeat-fog/plan.json --legacy-source /home/newbiexvwu/PvZ-search/.claude/worktrees/fog/artifacts/task45/phase-e --out /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/artifacts/task60/repeat-fog-proof.json
```

屋顶同样三步：源计划换为 `/home/newbiexvwu/PvZ-search/.claude/worktrees/roof/artifacts/task46/phase3/plan.json`，选择换 `artifacts/task60/roof-selection.json` 的绝对路径，输出用新的 `repeat-roof`，`--legacy-source` 换源计划父目录。两份选择分别固定首个未带 teacher archive 的已选根、wait 候选、k=0，不按终局好坏选点。完整重算可用源 `selection-active.json` 代替选择；仍显式 `--workers 1` 或 `2`。

新面板沿用 forks 的冻结 archive jobs；公共 `pick_roots` 支持时间根，`select_points` 支持破防/叶值平坦根，均只读原档案。`inspect/screen/confirm` 共用一个采集循环。
全量只读验证：`/home/newbiexvwu/.venvs/ml/bin/python /home/newbiexvwu/PvZ-search/.claude/worktrees/termmerge/scripts/critical_forks_data.py <绝对plan路径> screen`；详细报告将最后的 `screen` 改为 `diagnose`。
独立确认用 `critical_forks_data.py <plan> plan-confirm` → `critical_forks.py <plan> confirm --jobs <out/confirmation.json>` → `critical_forks_data.py <plan> report-confirm`；所有占位参数都必须替换成绝对路径。报告不会自动提交、覆盖旧 JSON 或操作生产。

- 合入 origin/fog；DESIGN 与两日日志的追加段全部保留，生产增量为 trace 同步与 NPZ 时间戳确定性，后续逐项验证。

- 合入 origin/roof；两日日志冲突保留双方条目，search_play 快路径条件同时保留 fog_diagnostics 与 observer，后续审查测试。

- 合入 origin/forks；仅当天 MEMORY 日志冲突，双方条目全部保留。后续统一实现与测试继续。

## 问题

- 已与 origin/qcpack 对照：它的 `pvz_training_pack.py` 与起点相同；`pack_training_rows.py` 只新增 `pack_memory_limit` 并替换 pack 层内存并发估算。本次只改 archive 行选择/权重，语义互补，无需引入 qcpack 或修改课程队列；最终再以 merge-tree 检查文本冲突，主控合并时须同时保留两者，禁止用整文件覆盖解决。
- 本机是 Linux；Mac 实机尚未运行，不把本机通过写成 Mac 验收。最终记录实际执行的测试文件、跳过和环境限制，主控可按同一文件集在 Mac 复核。
