# task62：WSL 彻底清理——本地只留在用的，能放 HF / GitHub 的不留本地，不留重复副本

> 主控 2026-10-24 16:40（台式机时间）写。执行者：台式机 Codex（plus 账号，`codex-clean` 会话）。盘点范围是整个 `/home/newbiexvwu`；文档提交在工作树 `cleanup`（见规则）。**不碰 tmux `qc` 及其在用数据**；游戏进程 0 个。

## 为什么
- 用户 16:30 要求：WSL 不应该占 181 GB，里面一定塞满了垃圾和不要的旧文件，要彻底清理干净。能放 HF 和 GitHub 的就不要在本地存，更不要复制那么多份。
- 现状：
  - `ext4.vhdx` 在 C 盘，181 GB；WSL 内已用 137 GB。
  - 今天两次因 C 盘不足 8 GiB 触发 `disk_guard.sh`，停了 qc。
- 主控 16:30 的粗盘点（`du`，未去硬链接）：

| 位置 | 大小 | 初判 |
|---|---|---|
| `~/PvZ-search/artifacts/student/frames` | 28G | 旧 v1 存储：10496 个 jsonl + 57521 个 npz |
| `~/PvZ-search/artifacts/student/frames_v2` | 14G | 现行正式帧（qc 写这里），**保留** |
| `~/PvZ-search/artifacts/student/v2_cache` | 8.4G | 特征缓存；现行配方 `read_cache_dir` 指向它 |
| `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27` | 27G | `training_frames` 18G（真实文件拷贝）、`cache` 7.1G、`hf_mac` 1.7G |
| `~/PvZ-search/.claude/worktrees/memprobe/artifacts/task58` | 7.2G | task58 正在用（codex-mem） |
| `~/pvz-logs/curriculum_state.pipeline` | 15G | qc 流水线在用：段 7.8G、缓存、`.merge` |
| `~/PvZ-search/artifacts/hard` 4.2G、`artifacts/night` 3.8G | | 旧实验 |
| 其余工作树约 40 个，合计约 17G | | 多为已合并分支 |
| `~/.codex` 4.4G（`packages` 3.3G）、`~/PvZ-Portable` 3.2G、`~/.debug` 2.5G、`~/.npm` 1G、`~/.nv` 1G、`~/pvz-logs` 下的大 log | | 缓存或旧物 |

## 要做的

### 第一阶段：盘点 + 方案（不删任何东西）
1. **全量盘点**：所有 ≥100 MB 的目录。
   - 去重口径：用 `du` 一次统计（硬链接只算一次），并对可疑目录查 `stat -c %h` 的链接数；符号链接写出它指向哪里。
   - 每一项写清：是什么、谁产生的、**谁还在引用**。引用的查法：
     - 正在运行的进程：`/proc/*/fd`、`cwd`；
     - qc 的 state 与流水线：`~/pvz-logs/curriculum_state.json`、`curriculum_state.pipeline/round_*/recipe.json`；
     - cq 的配方：`recipes/dw4i2.json` 的 `data`、`cache_dir`、`read_cache_dir`；
     - tmux 会话里的任务（qoder-cut、qoder-mix、qoder-b3、codex-mem 等）；
     - 仓库代码里写死的路径：`git grep`。
2. **分类**：
   - **A 留**：生产在用、正在跑的任务在用。
   - **B 先上传再删**：有长期价值、但不需要常驻本地。例如已结束实验的模型、需要留作证据的原始数据。走 HF：`scripts/hf_sync.py`，或 huggingface_hub 加 `$HF_TOKEN` / `$PVZ_HF_REPO`。上传后下载回验 SHA256，再删本地。
   - **C 直接删**：
     - 可再生的缓存；
     - 重复副本（同一份数据在多处）；
     - 已合并且已推送分支的工作树（`git worktree remove`，分支保留）；
     - 已被 npz 取代的旧 jsonl（用户 10-23 已批准，按 `scripts/convert_frames_npz.py` 的五道校验，或证明 frames_v2 已覆盖）；
     - 旧日志：大 log 删，或压缩成小摘要。
   - **D 存疑**：写理由，交主控定。
3. **特别核对**：
   - `artifacts/student/frames` 与 `frames_v2`、`dw4/.../task23/training_frames`、`dw5/.../task27/training_frames`、`curriculum_state.pipeline/frames` 之间的关系：哪些是硬链接、哪些是真拷贝、哪些已被别处覆盖。目标是正式帧只保留一份。
   - `v2_cache`、`curriculum_state.pipeline/cache`、`dw5/task27/cache`、`artifacts/student/recipe_cache` 等缓存之间是否重复。生产只该留一份。
   - 若改动生产路径（例如让配方的 `read_cache_dir` 指向流水线缓存、删掉 `v2_cache`），写成提案交主控，不要直接改 cq。
4. 方案写进本文件"结果"：表格列出每项的路径、大小、类别、理由、预计腾出的空间。**写完提交 push 后停下，等主控批准。**

### 第二阶段：执行（主控批准后）
- 按批准的清单逐项**用明确路径删除**；B 类先上传、回验、再删。
- 每删一批记一行：删了什么、腾出多少。
- 最后跑 `sudo fstrim -v /`，报 WSL 内已用空间（`df -h /`）。

## 规则
- **永远不删**：
  - `frames_v2`；
  - qc 流水线在用的东西；
  - 生产模型：`artifacts/student/dw4i_s1`、`artifacts/value_net/v4`、`v5`，`artifacts/hard/value_branch1`；
  - `deploy3` 的构建；
  - cq 工作树；
  - 正在跑的任务的数据；
  - 任何不确定的东西（归 D 类）。
- 删除只用明确路径，不用通配符批量删；删工作树前确认分支已推送到 origin（`git branch -r --contains`），用 `git worktree remove`，**不删分支**。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；密钥只用环境变量，不写字面值。
- 不 `git stash`，不改写已推送的历史，不碰 tmux `qc` 及其他会话。
- 文档与代码改动提交在工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup`（分支 `cleanup`，主控已建好），`git push origin cleanup`；不在主检出上提交。

## 结果

## 问题

## 主控批准（2026-10-24 17:05，台式机时间）——第二阶段按此执行

原则（用户 16:30）：本地只留正在用的；能放 HF / GitHub 的不留本地；同一份数据只留一份。第一阶段的 A 判定偏保守："某会话曾经用过"不算在用，只有**正在运行的进程**和**生产配方 / qc state** 才算。

按以下顺序执行，每批记 df 前后：
1. **§5 的 C 清单全部批准**（约 8.3 GiB）。另外批准以下各项：
   - `~/pvz-logs` 下没有被打开的大日志，单个 ≥10 MB（`*_tui.log`、`*_eval.log`、`*_panel.log` 等），直接删。被 fd 打开的不动。
   - `frames_v2/.feature_cache`（2.5 GiB，是可再生缓存，不是帧，不受"frames_v2 永不删"保护）。
   - `~/PvZ-search/artifacts/student/v2_cache`（8.4 GiB）。理由：现行段已全部建好，`read_cache_dir` 只在段要重建时才读，读不到会自动重算（`candidate.exists()` 检查，不会报错）；生产只留 `curriculum_state.pipeline/cache` 一份。删前确认没有 `pack_training_rows.py` 进程在跑。同时在 cleanup 分支把 `recipes/*.json` 里的 `read_cache_dir` 去掉，或改为 null（看 `pvz_recipe` 支持哪种），主控合并后随下次部署生效。
   - `dw5/artifacts/task27/cache`（7.1 GiB）和 `dw5/artifacts/task27/training_frames/packs`（打包产物，可再生）。确认当时没有 `train_rollout_policy.py` / `pack_training_rows.py` 在用 dw5 的路径。`training_frames` 里的链接视图和 `unified_v2.json` 保留。
2. **B 类先上传 HF、回验、再删**：
   - 范围：`artifacts/hard/branch`、`artifacts/hard/branch3`、`artifacts/night`、`artifacts/relabel`、`~/PvZ-Portable/artifacts`、`~/pvz-lane`。`pvz-lane` 是 seed_300xx 和 out_d27S*，属于 10 月初 lane/d27 实验的产物，归档即可。
   - 上传方式：每个目录打成 tar，大于 4 GB 的分片；放 HF `$PVZ_HF_REPO` 的 `archive/2026-10-24/<名>.tar[.partN]`，附带 MANIFEST（相对路径、字节数、SHA256）。下载回验 SHA 后再删本地。
   - 在 `MEMORY/docs/` 记一份归档索引，写清 HF 路径和取回命令。`train_value_widen.py` 等代码里写死的这些默认路径不用改，在索引里写明"需要时先从 HF 取回"。
3. **旧 `artifacts/student/frames` 整目录退役**：
   - 与 frames_v2 同 inode 的文件：删旧名不释放空间，但也要删，目录不留。
   - 其余文件（约 5632 个 jsonl episode、7 个 npz、约 16k 个同名但未证同物的 npz）：按 2 的方式打 tar 分片上传 HF `archive/2026-10-24/student_frames_v1/`，回验后删整个目录。
   - 先查一遍有没有符号链接指向旧 frames：qc state、生产配方、dw4/task23 视图里有的必须先改成指向 frames_v2 的同物文件，或者证明这些链接不在生产链上。否则不删。
4. **工作树**：§6 中所有 B / C 类工作树都退役。
   - B 类先把 ignored 证据打 tar 上传 HF、回验；C 类直接移除。一律用 `git worktree remove <绝对路径>`，不删分支。
   - 另外批准退役：b3、vnet、fog、roof、qcalloc、qcbench、v5、termmerge、trainmem。这几个分支已合入 pvz-env，它们的会话也已结束；会话（tmux）保留，只是不再有工作目录。对其中 ignored 的证据，按 B 类处理。
   - 保留：cq、deploy3、dw4、conveyor（这两个在生产链接链上）、cleanup、memprobe（task58 在跑），以及未合入的 cutoff、dw5、llmcand、perf、shovel、vcpp。deploy、deploy2 是旧引擎，可以移除。
   - prunable 的元数据可以 `git worktree prune`。
5. 最后 `sudo fstrim -v /`，报 `df -h /`。**WSL 关机与开启 sparse 由主控做，你不要关 WSL。**

不做：memprobe 的 task58 数据（在跑）、`curriculum_state.pipeline` 下的任何东西、主检出的 tracked 变动。
