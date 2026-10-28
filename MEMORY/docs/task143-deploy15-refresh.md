# task143：把 deploy15（v2sleep 输入 + build7）刷新到 prodint 之上

主控 2026-10-28 派给 Mac Qoder。只准备，不部署。

## 背景

- task131（`origin/obsup` 5423811f）做了 v2sleep 输入的上线准备：加宽热启动、旧缓存补零复用、台式机上的 `build7`、`qc_deploy15.sh`。它的基线是 deploy14i。
- 生产之后又换了几次代码：14j / 14k，以及即将上线的 14p（`origin/prodint` 4beab441 起）。deploy15 的基线已经过时。
- task135 核对过：生产学生每轮都是**从头训练**（`recipe.resolved.json`）。如果属实，task131 的“加宽热启动”在生产里根本用不上，切换输入版本只需要新配方 + 新引擎 + 新训练行。请先确认这一点，它决定 deploy15 要不要带热启动代码。

## 要做的

1. 新建分支 `obsup2`，以 `origin/prodint` 为底，合入 `origin/obsup`，解决冲突。
2. 确认生产训练是不是从头训。是的话，把热启动从部署路径里去掉，或者保留但不启用，写明理由。
3. 写 `scripts/qc_deploy15.sh`：基线为 14p 换上之后的文件；start_qc 照抄 `scripts/qc_deploy14p.sh`，再加 15 的激活分支（build7、`obs=v2sleep` 的新配方）；stop_qc 不排空；首组超时不回滚；预检要便宜。
4. 估算切换后第一轮重打训练行的耗时和磁盘占用，只读台式机数据，不跑训练。
5. 测试严格遵守 AGENTS §6.5 第 8 条。结论写在本文件末尾，推送 `obsup2`。

## 结论（2026-10-28，Mac Qoder，分支 obsup2）

### 1. 生产确实从头训练，热启动已从部署路径移除

- 只读核对：`curriculum_state.json` 的 r29–r31 作业全部 `init: null`；`round_000029/seeds/*/student/recipe.resolved.json` 也是 `obs=v2`、`init: null`、12 epochs。task135 的结论成立。
- 因此删掉 obsup 在 `scripts/curriculum_queue.py` 里加的两段（入队时把 `recipe['init']` 冻成现役学生、并复制到 B 配方）。删除后控制器与生产/14p 逐字节相同（sha `5ef3377f`），deploy15 不需要替换控制器。
- `python/pvz_rollout_policy.py` 的 `warm_start_student` 与 trainer 的 `if args.init` 分支保留但不启用：生产配方永远不带 `init`，只有将来显式写 `init` 的配方才会走它；`python/test_observation_upgrade.py` 里加宽/单步训练用例仍在验证这条能力。
- 对应测试从“入队冻结现役”改为“入队后四条候选配方仍是 `init: null`”。理由：冻结会让首个 v2sleep 轮次与其它轮次不可比，还把训练绑在一个回滚后可能不存在的检查点上。

### 2. `scripts/qc_deploy15.sh`（未执行）

- 基线：14p 换上之后的六个文件（控制器 `5ef3377f`、选择 `616b5d5e`、search_play `4f4ba541`，其余三项仍为生产 sha）。预检同时要求 `qc_deploy14p.mode=deckcells` 与 task137 训练面板存在，否则 ABORT——deploy15 假设 14p 已生效。
- 相对生产替换 10 个文件：`python/pvz_env.py`、`pvz_frame_store.py`、`pvz_recipe.py`、`pvz_rollout_policy.py`、`pvz_student_observation.py`、`pvz_value_net.py`、`scripts/pack_training_rows.py`、`scripts/train_rollout_policy.py`、两份 `recipes/*_v2sleep.json`。C++ 可执行文件不重建。
- 激活分支：`--executable` 切 `obsup/build7/pvz-portable`（task131 编译，sha256 `c35633bc…dec8`，脚本里按此固定），配方切 A15/B15，`--gate-version 5`、`--candidates-per-recipe 2`、`--training-parallelism 1` 与 14p 相同，并继续带 `--extra-manifest artifacts/task137/training_panel.json`。
- `start_qc` 四个分支逐字照抄 14p；唯一的差别是 extra manifest 改成无条件传入（deploy15 只有在 deckcells 生效时才允许启动，回滚目标也是 14p）。
- `stop_qc` 不排空：先确认 tmux socket，`kill-session` 后按 PID `TERM` 掉 `search_play` / `train_rollout_policy`。`stop_curr` 不被写。
- 首组 rc：120×30 秒；`rc=0` 记完成；60 分钟没出新组只记日志、qc 保持运行；只有 `rc != 0` 或 qc 退出才回滚（恢复 `.before`、删 mode、按 14p 参数重启）。
- 备份目录在全部 `.before` 写完后落 `backed_up` 标记，`--rollback` 必须先看到该标记。
- 预检很便宜：文件 sha、提交存在性、两份配方的 `obs/init/precompute_workers`、build7 存在且 sha 匹配、task137 面板 144 任务/16 格/294912 唯一 seed/最低倍率 0.5、`ast.parse` 八个 Python 文件。没有 dry-run、对局或全量测试。
- 删除 task131 的 `scripts/qc_deploy15_state.py`、`configs/deploy15_baselines.json` 及其测试，换成基于 14p 风格的脚本；`python/test_deploy15.py` 改测四条 `start_qc` 分支和 watchdog 提取。

### 3. 第一轮重打训练行的估算（只读台式机，未跑训练）

实测数据（`~/pvz-logs/curriculum_state.pipeline`）：

| 项 | 实测 |
|---|---|
| 标签段 | 874 个 npz，11.2 GiB，均值 13.2 MiB |
| 行数 | 21,132,605（round 29） |
| 打包速率（近期） | 22.6–29.0 s / 段 / worker（r22–r29，4–7 并发） |
| 打包速率（早期/冷缓存） | 57–123 s / 段 / worker（r2–r20） |
| 旧 obs 缓存 | `cache/v2` 8.0 GiB，74,933 个条目 |
| 共享训练映射 | `frames/.merge/shared-v1` 11.3 GiB |
| 磁盘 | `/dev/sdd` 1007 GB，已用 127 GB，空闲 829 GB |

- **缓存不用重编码**：v2sleep 只对缺字段的旧档案用零填充视图复用 v2 缓存。抽 400 份档案解压检查 `"sleep"/"boss"/"bungee"` 键命中 0，说明 74,933 个旧条目全部走视图（每条约 1 KB，合计约 73 MB）；只有 build7 之后新打的局会完整编码。
- **段必须全量重建**：段指纹包含 obs 版本，874 个 v2 段全部失效。旧段不会被删除（没有段回收），所以常驻磁盘净增约 11.2 GiB；新段体积与旧段同量级。
- **耗时**：按近期 25 s/段/worker 计，874 段
  - 2 个建造者（配方 `precompute_workers=2`）：约 **3.0 小时**（区间 2.7–3.5 h）；
  - 4 个建造者：约 1.5 小时；6 个：约 1.0 小时；
  - 若冷缓存回到早期速率，2 个建造者约 4–5 小时。
  首个 v2sleep 轮次的训练会等这段打包；采集不受影响，所以“首组 rc”仍会很快出现，部署脚本不会因此回滚。若想缩短，可把配方 `precompute_workers` 提到 4（机器近期实测本来就跑 4–7 个建造者），本任务保持 task131 已验收的 2。
- **共享映射**：首次训练开始时按新段签名重建 `shared-v1`（先删后写，11.3 GiB），磁盘不叠加；两遍读写约几分钟。
- **磁盘**：稳态净增约 11.3 GiB 段 + 0.1 GiB 缓存，共享映射同量级；峰值约 +11.5 GiB。数据目录从约 32.9 GiB 到约 44.5 GiB，829 GB 空闲无压力。

### 4. 测试

全部按 AGENTS §6.5 第 8 条：`ulimit -f 2000000`、`timeout 300`、同一时间一个 pytest、只用 `-k`。

| 范围 | 结果 |
|---|---|
| `test_observation_upgrade.py`（含入队不再冻结、加宽、缓存复用、单步训练） | 15 passed |
| `test_deploy15.py`（四条 start_qc 分支 + watchdog 提取） | 4 passed |
| `test_sleep_inputs.py` | 2 passed, 2 skipped |
| `test_value_widen_training.py` + `test_bungee_inputs.py` | 8 passed, 1 skipped |
| `test_curriculum_queue.py -k "supersede or backlog or build_plan or init_requeues or restore_when_superseded"` | 7 passed, 1 failed |
| `test_curriculum_pipeline.py -k "enqueue or recipe or candidate or selection"` | 2 passed |

唯一失败是既有问题 `test_pack_memory_limit_matches_build_plan_per_builder`（selinc/prodint 上同样失败；本次在 `ulimit -f` 下更早以 EFBIG 失败，属于同一用例的大文件写法）。没有跑三个大文件的整套。

### 5. 需要主控知道的两件事

1. **deploy14p 的 `start_qc` 原本不能被看门狗提取**：`local extra=""` 让 `qc_watchdog.sh` 的变量闭包报 `unresolved deployment variable: extra`，14p 生效后若 qc 崩溃，看门狗会拒绝重启。已在 `prodint` 修正（`e2f8cc16`，顶层声明 `EXTRA_ARG`，行为不变，提取出来的 launcher 通过 `bash -n`），**台式机上的 `~/pvz-logs/qc_deploy14p.sh` 需要重新复制**。obsup2 已合入该修正。
2. 台式机 `~/pvz-logs/qc_deploy15.sh` 还是 task131 的旧版本（10-27 15:24），执行本任务的新脚本前要先覆盖；否则看门狗可能挑到旧 15。build7 仍是 task131 的二进制（sha 已固定），本任务不重建。
