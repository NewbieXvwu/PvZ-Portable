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
