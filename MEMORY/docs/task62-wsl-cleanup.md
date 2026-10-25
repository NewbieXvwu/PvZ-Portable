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

> 当前结论：221ae160误删在用stage4导致qc停机。六份清单已恢复并核验，主控于2026-10-24 21:35重启qc；详见“事故”一节。此前“清理完成”结论已被该事故纠正，禁止继续删除。

### 第一阶段盘点与处置提案（2026-10-24，待主控批准）

本阶段只做只读盘点、分类和方案，未删除/移动/转换任何已有文件，未上传 HF，未改生产配方，未对任何 tmux pane 发指令、按键或停止/重启会话。tmux 仅 `list-panes` 读取任务归属。后续删除、压缩、工作树移除、`fstrim` 均未执行。

#### 1. 口径和现场变化

- 16:04（本机 `date -Is`，UTC+08）开始；任务书的 16:40 与本机时钟不一致，本报告按本机采样时间记录，不能把二者拼成先后证明。
- 全家目录仅运行一次 `du -x -B1 --null /home/newbiexvwu`，未加 `-l`、未跟随符号链接；退出 0、stderr 空。按十进制 **100,000,000 B** 筛出全部 354 个目录，逐项列在 §8。GiB = 2³⁰ B。家目录分摊体积 **150,920,466,432 B = 140.556 GiB**。
- `df -B1 /` 同期已用 **167,965,769,728 B（约 156.43 GiB）**。`df` 包含家目录外系统空间，不能用家目录合计替代；任务书 137 GB 已非本次实测。Windows 的 vhdx 181 GB 未重测，Linux 内删文件后的宿主回收也不能提前承诺。
- 一次 `du` 对硬链接只计首次遇到的路径；父子目录含同一批块，**所有表的父子行不能累加**。例如正式帧本身的可达块大于其分摊值，删除旧链接也不会释放正式帧仍持有的块。最终腾出空间必须在获批执行后按实际 `df` 差值复测。
- 全家目录 `lstat`/符号链接扫描零读目录错误；发现 318,685 个符号链接及 103 个逻辑大小 ≥100 MB 的普通文件（含硬链接重复名）。相关数据链在 §3 展开；软件包内的命令/库链接属于其安装目录，随该目录 A/C/D 分类，不当成另一个副本计量。
- `/proc/*/{cwd,exe,fd}` 和关键缓存的 `maps` 已核对；33 个进程在 fd 快照中不可读或已退出，maps 快照 30 个不可读/已退出。因此“未见引用”仅指可见快照，执行前仍须复核，不是不存在引用的证明。

#### 2. 正在引用什么（A 类保护）

任务书“游戏进程 0 个”已过时：快照中 qc 主进程 **323267**，search_play **328239/328241/328242**，游戏 **328249/328250** 均存在。其命令路径直接指向 `deploy3/build/pvz-portable`、`frames_v2`、`hard/value_branch1/model.pt`、`value_net/v5/model.pt`、`.cache/pvz-research-resources`、pipeline round 2 学生模型；`games.jsonl` 打开于 `artifacts/hard/c4d10origm250_48/`。**整个在用 qc 链不列入可执行删除清单。**

`curriculum_state.json` 58 处路径引用中包含：cq 的 multiplier_panel、主检出 stage4 面板、生产 dw4i_s1、pipeline round 2/3 recipe/model、**dw5/task27/unified_v2.json**、deploy3、value v5 和资源缓存。当前 student 是 round 2；round 3 为 queued。round 2/3 的实际配方均为：

| 字段 | 实际值 |
|---|---|
| data | `~/pvz-logs/curriculum_state.pipeline/frames` |
| cache_dir | `~/pvz-logs/curriculum_state.pipeline/cache` |
| read_cache_dir | `~/PvZ-search/artifacts/student/v2_cache` |

cq 的 `recipes/dw4i2.json` 仍为 `data=.claude/worktrees/dw4/artifacts/task23/training_frames`、`cache_dir=artifacts/student/recipe_cache/dw4i`、`read_cache_dir=artifacts/student/v2_cache`。队列生成 round recipe 时才改 data/cache_dir；不能因 round 已改路径就删除基础配方来源。主检出的 `recipe_cache` 当前不存在，预计释放 0。

| 会话/进程 | 现场归属/引用 | 处理 |
|---|---|---|
| qc（pane PID 323264） | cwd 在 `/mnt/c/Users/NewbieXvwu`，子进程路径指向 cq/deploy3/主检出数据/pipeline | A，cwd 不在数据目录不代表数据没用 |
| qoder-b3 / qoder-cut / qoder-mix，47203/47206/47213 | 当前 cwd 均 dw5；前两者 fd 仍含 b3/cutoff 会话日志；167362 正 tail dw5/task36/train_A.txt | A：dw5、b3、cutoff 及任务数据，不因暂时无训练进程而删 |
| codex-mem，285170/285178 | memprobe/task58 所在工作树 | A：memprobe 全部 |
| codex-cuda | perf | A |
| codex-merge | termmerge | A |
| codex2、codex3 | vcpp、fog | A |
| codex5、codex5-p、cb-v5 | v5；v5 的证据软链还指向 vnet/task30/v4 | A：v5 和 vnet |
| codex7、cb-roof | roof | A |
| codex-qb、cb-qcbench | qcbench | A |
| cb-llmcand、cb-qcalloc | llmcand、qcalloc | A |
| codex-clean | 本报告 cleanup | A，交付后也不自行移除 |
| mac_import_loop 94373、qc_deploy10.sh 331301、disk_guard 502 | pvz-logs 内脚本与输出；导入正式帧/部署生产构建 | A，保留相关脚本、状态与打开日志 |

四组明确禁止删除的模型目录均实际存在：`student/dw4i_s1`（model 703150 B）、`value_net/v4`（611163 B）、`value_net/v5`（611739 B）、`hard/value_branch1`（584603 B）。它们虽未达 100 MB，仍全部 A；cq、deploy3 以及资源缓存同理。

#### 3. 帧之间的关系与唯一性

| 位置 | 一次 du 分摊 GiB | 实体/关系 | 类别、预计释放 |
|---|---:|---|---|
| `~/PvZ-search/artifacts/student/frames` | 27.891 | 68017 个 episode；57521 npz + 10496 jsonl，元数据 68017 份 | D，当前 0；未来全部获准处置的物理上限约 17.204 GiB，不能报 27.891 |
| `~/PvZ-search/artifacts/student/frames_v2` | 13.764 | 88118 份 meta；文件计数会随 qc 增长；仅实体可达文件块约 23.790 GiB，含共享块与 2.503 GiB 的内置 .feature_cache | A，永远保留，0 |
| `~/PvZ-search/.claude/worktrees/dw4/artifacts/task23/training_frames` | 0.735 | 实体普通文件 0；episode 内 meta/frames 全为符号链接，目录元数据占空间 | A，生产间接依赖，0 |
| `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames` | 17.172 | episode 文件也是链接；实体普通文件仅 40 个，大头是 packs | A，qoder/task36 配方仍用，0；任务退役后 pack 可再生候选另议 |
| `~/pvz-logs/curriculum_state.pipeline/frames` | 8.181 | 52469 个 episode 目录链接；实体普通文件 210 个为分段/索引 pack | A，全部保留，0 |

`stat -c '%h %d:%i %s %F %N'` 的实证：`frames/seed_1830362/d10dag1_fast_search/frames.npz` 与 frames_v2 同路径文件均为 **nlink=2、dev:inode=2096:868616、162270 B**。dw4/dw5 的同名文件则是指向 frames_v2 的 **99 B 符号链接**，不是 99 B 的残缺帧，也不是真拷贝。

全量 inode 比较：旧 frames 与 frames_v2 共享 **83030 个普通文件 inode**（41515 帧 + 41515 meta），共享文件块 **11,474,546,688 B = 10.686 GiB**。旧 frames 的 57521 npz 中 57514 有 v2 同名文件，但只有 41515 已由 inode 证明同物；另 15999 不能仅凭同名判同内容。另有 **7 个 npz 不在 v2**。10496 jsonl（占文件块 **14,957,928,448 B = 13.931 GiB**）只有 **4864** 找到 v2 同 episode 的 npz，尚余 **5632** 没有对应文件。单凭“v2 已上线”不足以删旧目录。

相关符号链接按完整根路径及保留相对路径的规则列下表；每行表示源根下实际已存在的链接 `源根/relative → 目标根/relative`，并非创建或删除用的通配符。所有这些主要链均无断链。其余重命名的 replay 链单列样例。

| 源根 | 目标根 | 链接数 | 相对路径一致数 |
|---|---|---:|---:|
| `~/PvZ-search/.claude/worktrees/conveyor/artifacts/task20/frames_v2` | `~/PvZ-search/artifacts/student/frames_v2` | 50858 | 50858 |
| `~/PvZ-search/.claude/worktrees/dw4/artifacts/task23/training_frames` | `~/PvZ-search/.claude/worktrees/conveyor/artifacts/task20/frames` | 512 | 512 |
| `~/PvZ-search/.claude/worktrees/dw4/artifacts/task23/training_frames` | `~/PvZ-search/artifacts/student/frames_v2` | 97120 | 97120 |
| `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames` | `~/PvZ-search/.claude/worktrees/conveyor/artifacts/task20/frames` | 512 | 512 |
| `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames` | `~/PvZ-search/artifacts/student/frames_v2` | 113056 | 113056 |
| `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/training/frames` | `~/PvZ-search/artifacts/student/frames` | 8 | 0 |
| `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/training/frames` | `~/PvZ-search/artifacts/student/frames_v2` | 16 | 0 |
| `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data/by-terrain/day` | `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data` | 15 | 15 |
| `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data/by-terrain/night` | `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data/by-terrain/pool` | `~/PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/l20gap/artifacts/task56/pack_view` | `~/PvZ-search/artifacts/student/frames_v2` | 36 | 36 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/cohort-pack/day_l10_orig_train_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/cohort-pack/day_l7_train_d7_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/cohort-pack/night_l14_train_n10_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/cohort-pack/pool_l24_train_p9k_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51-deploy3/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/cohort-pack/day_l10_orig_train_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/cohort-pack/day_l7_train_d7_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/cohort-pack/night_l14_train_n10_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/cohort-pack/pool_l24_train_p9k_m150/frames` | `~/PvZ-search/.claude/worktrees/qcbench/artifacts/task51/frames/cq` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/acceptance/state.pipeline/frames` | `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/acceptance/frames` | 32 | 32 |
| `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/pack_bench/parallel/data` | `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/smoke/frames` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/pack_bench/serial/data` | `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/smoke/frames` | 16 | 16 |
| `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/pack_bench/warmup/data` | `~/PvZ-search/.claude/worktrees/qcpipe/artifacts/task49/smoke/frames` | 16 | 16 |
| `~/PvZ-search/artifacts/hard/vdata` | `~/PvZ-search/artifacts/student/frames` | 1024 | 1024 |
| `~/PvZ-search/artifacts/hard/vdata_ctx` | `~/PvZ-search/artifacts/student/frames` | 2048 | 2048 |
| `~/PvZ-search/artifacts/hard/vdata_z15` | `~/PvZ-search/artifacts/student/frames` | 512 | 512 |
| `~/pvz-logs/curriculum_state.pipeline/frames` | `~/PvZ-search/.claude/worktrees/dw4/artifacts/task23/training_frames` | 256 | 256 |
| `~/pvz-logs/curriculum_state.pipeline/frames` | `~/PvZ-search/artifacts/student/frames_v2` | 52213 | 52213 |

关键补充：pipeline 的 256 条 dw4 episode 链，最终落到 **conveyor/task20/frames** 的 `d10orig1_fast_search` 256 集（dw4 和 dw5 各有 512 个 meta/frames 文件链接指向该根）。故 **conveyor 即使已合并也不能整个移除**。`_prepare_corpus` 明确合并基础训练集与正式帧，缺标签即拒绝，必须先补齐真实帧才能提出改路径。

forks 的重命名 replay 例：`forks/artifacts/task53/training/frames/seed_2030167/replay_pool/frames.npz` → `~/PvZ-search/artifacts/student/frames/seed_2030167/p24dag3_fast_search/frames.npz`。此外 `hard/vdata`、`vdata_ctx`、`vdata_z15` 共 3584 条目录链接还指向旧 frames；删旧根会断这些证据链。

v5 的 `v5_smoke`、`v5`、`sparse64/v5b`、`sparse64/v5` 中的 `episodes.json`、`boards.npy`、`scalars.npy`、`rows.npz` 分别链接到 `vnet/artifacts/task30/v4/` 下同名文件。vnet 需随 v5 会话保留，不能将约 1 GiB 的 boards.npy 当独立旧缓存删。

dw5 的 `packs/e39ba7cbc076380a.npz` 为 **8744068761 B**，同级 `.npz.d/` 展开数组约 **8744206336 B** 分摊；这是 pack 容器与 memmap 展开副本，不是又一整套正式 episode。当前任务仍用，先 A。pipeline 的 `.merge` 当前目录存在；包括未来可能生成的稀疏/映射文件，一律纳入 qc 在用链，不按旧临时文件删。

全家目录仅发现 4 个既有断链：perf 工具包里 3 个 Debian changelog 链，以及 `.codex/app-server-control/app-server-control.sock` → `/tmp/codex-daemon-1000/…`。均归所在活动目录 A/D，不清理会话 socket。通用链接还包括各工作树 `CLAUDE.md → AGENTS.md`、构建 `libmimalloc.so → libmimalloc.so.2 → libmimalloc.so.2.1`、测试 venv 的 python → `/usr/bin/python3.14`；它们是引用，不重复计量目标文件。

#### 4. 缓存核对与生产路径提案

| 路径 | 分摊 GiB | 普通文件/硬链接情况 | 当前消费者与分类 |
|---|---:|---|---|
| `~/PvZ-search/artifacts/student/v2_cache` | 8.367 | 119123 npz，均 nlink=1 | cq、round 2/3 read_cache_dir，A |
| `~/pvz-logs/curriculum_state.pipeline/cache` | 6.698 | 209876 npz，均 nlink=1 | round 2/3 cache_dir，A |
| `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache` | 7.088 | 227144 npz，均 nlink=1 | dw5 的任务现场；dw6i/task36 仍用其 training_frames，不能据无 fd 宣布 cache 可删，A |
| `~/PvZ-search/artifacts/student/recipe_cache` | 不存在 | 0 | 基础配方声明路径，当前无占用，0 |
| `~/PvZ-search/artifacts/student/frames_v2/.feature_cache` | 2.506 | 21744 npz，均 nlink=1 | 正式帧目录内部，按禁止删除规则 A |
| `~/PvZ-search/artifacts/student/.feature_cache` | 0.494 | 位于正式帧目录之外的旧缓存 | 当前配方/proc 未引用，可再生，C |

缓存同名并不证明相同：`cache_episode` 以 resolve 后源路径命名，内部还校验 encoder + source 的 digest。不同参数/编码器/压缩版本都可能共存。实际比较如下（SHA 只抽样，绝不外推为全量重复证明）：同相对路径且同 st_size 的名字排序，以 `max(1, 同尺寸数量 // 16)` 为步长取前 16 个，分别流式算整个文件 SHA256；没有同尺寸候选则不取样。

| 两组 | 同相对文件名 | 同字节数 | SHA256 样本相等/样本数 | 结论 |
|---|---:|---:|---|---|
| v2_cache / pipeline_cache | 76542 | 0 | 0/0 | 不能判为字节重复 |
| dw5_cache / pipeline_cache | 195264 | 11226 | 0/16 | 不能判为字节重复 |
| v2_cache / dw5_cache | 76542 | 38271 | 16/16 | 存在重复样本，未全量验证 |
| feature_cache / v2_cache | 0 | 0 | 0/0 | 不能判为字节重复 |

**生产路径提案（D，待主控单独裁定，不是本次可执行项）：**保留 `v2_cache` 与 pipeline/cache 原样；未来先对全部源帧、encoder、digest、候选/packet/cell/identity 辅助缓存逐项建对应清单，证明 pipeline cache 的有效内容和缺口，离线验证一次训练读取与重建成本，再由主控安排生产切换窗口。只有主控批准、生产不再读旧路径且全量覆盖证明通过后，才能把 `read_cache_dir` 统一到单一缓存。当前不能为了“生产只留一份”直接删 v2_cache，也不能仅凭 16 个相同样本删 dw5 cache。

旧 jsonl 的后续提案（D）：逐文件依据 `convert_frames_npz.verify()` 做原始字节一致、JSON 对象一致、encoder 签名一致、hints 帧数一致、labelled rows/全部训练数组首维和索引范围一致等五组校验；仅存在同名 npz 不算通过。没有 v2 对应的 5632 集 jsonl + 7 集 npz 先确认长期价值、走 B 的 HF 全量归档回验；需要并入正式帧的标签须主控审批路径/配方变化。还须先修正 hard/forks 入链及其证据引用。当前未运行会写 npz 的转换器，未作任何删除。

#### 5. 第一批可批准的 C 清单与 B 清单

以下均为**未来批准后**的候选；C 表中的路径互不包含，预计释放只计一次。执行前再次检查 fd/cwd/maps、会话、配置与入链；出现新引用就转 A/D。不把分类 C 本身当作删除授权。

| 明确路径 | 分摊 GiB | 类别 | 产生者/引用、处置理由 | 预计可释放 GiB |
|---|---:|---|---|---:|
| `~/.npm/_cacache` | 0.956 | C | npm 下载内容，可重取；可见 fd/maps 未引用 | 0.956 |
| `~/.nv/ComputeCache` | 0.992 | C | CUDA 可再生编译缓存；执行前确认没有新 GPU 任务 | 0.992 |
| `~/.debug` | 2.549 | C | 性能符号/库缓存；66 个文件中共享硬链接块不能回收 | 0.073 |
| `~/.codex/packages/app-server-daemon/releases/0.157.1-x86_64-unknown-linux-musl` | 0.364 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.364 |
| `~/.codex/packages/app-server-daemon/releases/0.158.0-x86_64-unknown-linux-musl` | 0.414 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.414 |
| `~/.codex/packages/app-server-daemon/releases/0.159.0-x86_64-unknown-linux-musl` | 0.414 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.414 |
| `~/.codex/packages/app-server-daemon/releases/0.159.1-x86_64-unknown-linux-musl` | 0.414 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.414 |
| `~/.codex/packages/app-server-daemon/releases/0.159.2-x86_64-unknown-linux-musl` | 0.414 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.414 |
| `~/.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl` | 0.414 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.414 |
| `~/.codex/packages/app-server-daemon/releases/0.160.0-x86_64-unknown-linux-musl` | 0.416 | C | 旧安装版本；current/auto-update-version 均为 0.160.1，无可见 maps/exe 引用 | 0.416 |
| `~/PvZ-Portable/logs` | 0.724 | C | 旧 PPO/t5 日志，当前无可见 fd；先将结论留文档/HF，再用明确文件清单处理 | 0.724 |
| `~/PvZ-Portable/build/core.2439` | 0.188 | C | 旧 core 转储，可见进程无引用 | 0.188 |
| `~/PvZ-search/artifacts/student/.feature_cache` | 0.494 | C | 旧 student 根的特征缓存；不是受保护 frames_v2 内缓存 | 0.494 |
| `~/PvZ-search/.claude/worktrees/conveyor/artifacts/task20/cache` | 1.045 | C | 旧特征缓存可重建；仅处理 cache，保留生产必需 frames 和整个工作树 | 1.045 |
| `~/PvZ-search/.claude/worktrees/identity/artifacts/task19/cache` | 1.011 | C | 旧特征缓存可重建；原始帧/结果先保留，不整工作树删 | 1.011 |

C 这批预计约 **8.333 GiB**（不含工作树移除、不含旧帧、不含任何生产缓存）。`.debug` 的 du 为 2.549 GiB，但 nlink=1 文件仅 **78422016 B = 0.073 GiB**；其余 **2657828864 B** 与其他路径硬链接，不能虚报为可释放。目录元数据和并发增长会使最终 df 有差异。

B 表不是“已远端备份”的声明：本阶段未上传/下载，当前释放均为 0。列出的完整父目录包含其子目录，不能再加子目录体积。

| 明确路径 | 分摊 GiB | 类别 | 产生者/仍有引用 | 上传回验并解除引用后的空间上限 GiB |
|---|---:|---|---|---:|
| `~/PvZ-search/artifacts/hard/branch` | 2.284 | B | 旧分支价值训练数据；train_value_widen.py 默认引用，离线重放仍需要，先 HF 完整归档 | ≤2.284 |
| `~/PvZ-search/artifacts/hard/branch3` | 1.567 | B | 旧分支价值训练数据；train_value_widen.py 默认引用，离线重放仍需要，先 HF 完整归档 | ≤1.567 |
| `~/PvZ-search/artifacts/night` | 3.793 | B | 旧夜间价值样本；代码 git grep 无直接引用，可见进程无引用，先存 HF；不据此声称已备份 | ≤3.793 |
| `~/PvZ-search/artifacts/relabel` | 0.369 | B | 历史重标样本；未证实远端完整回验，先保留证据 | ≤0.369 |
| `~/PvZ-Portable/artifacts` | 0.413 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | ≤0.413 |

上述互斥 B 目录合计上限 **8.426 GiB**；若仍有外部硬链接或需保留结论文件，实际更小。其他非活动工作树的 ignored 证据见 §6，尚未合并进此估算。`~/pvz-lane`（0.632 GiB）因生成任务未唯一确认，归 D，当前预计释放 0，不纳入 B 的可归档退役范围，先由主控确认归属。

B 的实施方案：以明确目录建立含相对路径、字节数、SHA256、源 revision、引用关系的 MANIFEST，保留模型、原始数据与结论；先检查现有 HF 是否真的覆盖该清单。`scripts/hf_sync.py push-evidence` 只接受含 archive_manifest/report/training_state 的完整证据树，遇外部软链会拒绝，不能直接给它 dw4/dw5 的链接视图。满足格式的目录可走该脚本；其余由 `huggingface_hub` 加 `$HF_TOKEN` / `$PVZ_HF_REPO` 上传明确文件，并在清单中记录规范源文件和别名，不能盲目跟随外部链接制造重复。固定上传 commit revision，**下载回独立校验目录逐文件 SHA256/字节数回验**，保存回执与可恢复路径；不以 ls/上传成功日志替代回验。确认无活动/生产引用后，才按主控批准的明确本地路径整目录处置；不零散打断 archive_manifest，也不使用 tar/scp/Git LFS 作为常规交付通道。本阶段不会创建这套上传/删除作业。

`pvz-logs` 的大旧 log 已看到 `dw3s2_eval.log`（163208442 B）、`qoder_tui.log`（169834136 B）、`cb_cuda_tui.log`、`cb_shovel_tui.log` 等，但它们与驻留会话的历史证据归属尚需主控确认，当前 D，不纳入 C 总额。`qc.out`（采样约 58 MB）、mac_import_loop.out、qc_deploy10.out 有打开 fd，A，不能截断/轮转。Agent 的会话 JSONL/SQLite/日志也不当作无用缓存整目录删。

#### 6. 工作树逐个审计

通过 `git ls-remote --heads origin` 校验真实远端；采样 `origin/pvz-env` 为 **4c1946ed7cbb2e523bf231df9fb1682f6cc7b9a7**。该对象最初不在本地，随后仅以 SHA fetch 对象（`--no-write-fetch-head --no-tags`，不 checkout、不改工作区）后用 `merge-base --is-ancestor HEAD <远端SHA>` 验合并性。不能把原本缺对象的 128 退出当作“未合并”。远端同 SHA 列证明该 HEAD 已推送；非同 SHA 但被远端主线包含的也有远端可达证明。`git branch -r --contains` 已逐个读取；删除前仍按任务书重做并结合 ls-remote 防陈旧 refs。

工作树状态采用 `GIT_OPTIONAL_LOCKS=0 git status --porcelain --ignored=matching`，只读；clean 不包括 ignored 文件的安全性。活动 A 优先于已合并/推送，未合并 D 不移除。B 树先上传回验 ignored 证据，C 树只有可再生构建/缓存；全部只提案，分支保留。此表“预计”是整棵树候选上限，**不得与 §5/§8 子项相加**。

| 工作树（均为 `~/PvZ-search/.claude/worktrees/` 下，另列主检出） | 分支 / HEAD | 实时远端同 HEAD 分支 | 已合入远端 pvz-env | tracked 变动 / untracked 项 / ignored 项 | 类别与依据 | 当前/后续预计释放 GiB |
|---|---|---|---|---|---|---|
| `~/PvZ-Portable` | pvz-env / 7194113da7 | 无（看主线可达性） | 是 | 0 / 0 / 41 | D：原始检出与共享 Git 库共存，不能整目录删；代码已被 origin/pvz-env 包含 | 0 |
| `~/PvZ-search` | search-wt / 930d776d8d | 无（看主线可达性） | 是 | 410 / 0 / 17 | D：主检出含生产数据及 410 条已有 tracked 变动；不在这里改动/提交 | 0 |
| `~/PvZ-search/.claude/worktrees/advgap` | advgap / 88839b609b | advgap | 是 | 0 / 0 / 3 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/b3` | b3 / b30a75fbca | b3 | 是 | 0 / 0 / 7 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/cleanup` | cleanup / 6fb40bd896 | cleanup | 是 | 0 / 0 / 0 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/conveyor` | conveyor / cc74fcc3f1 | conveyor | 是 | 0 / 0 / 5 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/cq` | cq / c84ad7d14d | 无（看主线可达性） | 是 | 63 / 0 / 10 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/curr` | curr / f69ec11795 | curr | 是 | 0 / 0 / 9 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/curr39` | curr39 / eec0d13a33 | curr39 | 是 | 0 / 0 / 3 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/cutoff` | cutoff / 69a8f0ed89 | cutoff | 否 | 0 / 0 / 3 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/dayearly` | dayearly / 1e7105efd8 | dayearly | 是 | 0 / 0 / 3 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/deploy` | detached / 2a8dc97c8b | 无（看主线可达性） | 是 | 0 / 0 / 4 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续≤0.106 |
| `~/PvZ-search/.claude/worktrees/deploy2` | detached / f2ae755ab0 | 无（看主线可达性） | 是 | 0 / 0 / 4 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续≤0.106 |
| `~/PvZ-search/.claude/worktrees/deploy3` | detached / 2b743c6216 | 无（看主线可达性） | 是 | 0 / 0 / 4 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/dw4` | dw4 / 3c02244a76 | dw4 | 是 | 1 / 0 / 4 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/dw5` | dw5 / 7974e32435 | dw5 | 否 | 0 / 1 / 8 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/fog` | fog / 15b671aded | fog | 是 | 0 / 0 / 10 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/forks` | forks / 0e9738d78e | forks | 是 | 0 / 0 / 8 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续≤0.459 |
| `~/PvZ-search/.claude/worktrees/identity` | identity / e8b2782e0f | identity | 是 | 0 / 0 / 11 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续≤1.180 |
| `~/PvZ-search/.claude/worktrees/keep` | keep / 7d98a5fecc | keep | 是 | 0 / 0 / 5 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/l20diag` | l20diag / 86b82ac40a | l20diag | 是 | 0 / 0 / 4 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/l20gap` | l20gap / 75f16e28de | l20gap | 是 | 0 / 0 / 3 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/llmcand` | llmcand / cd4da073dc | llmcand | 否 | 0 / 0 / 4 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/macpool` | macpool / a4589db5f6 | macpool | 是 | 0 / 0 / 10 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/memprobe` | memprobe / bdb77e641b | memprobe | 否 | 0 / 0 / 7 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/n10p` | n10p / d4caa1a862 | n10p | 是 | 0 / 0 / 5 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/night` | night / 81810b50b9 | night | 是 | 0 / 0 / 5 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/nightn10` | nightn10 / 615f35b6d4 | nightn10 | 是 | 0 / 0 / 4 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/oom` | oom / f4a3f17c07 | oom | 是 | 0 / 0 / 2 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/perf` | perf / 84c02629d1 | perf | 否 | 0 / 0 / 11 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/pool` | pool / c9d5066704 | pool | 是 | 0 / 0 / 5 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/q101` | detached / 9ccf553735 | 无（看主线可达性） | 是 | 0 / 0 / 3 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/qcalloc` | qcalloc / 539cb2d859 | qcalloc | 是 | 0 / 0 / 7 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/qcbench` | qcbench / 828c503e4c | qcbench | 是 | 0 / 0 / 9 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/qcpack` | qcpack / 25f1a529a6 | qcpack | 是 | 0 / 0 / 6 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/qcpipe` | qcpipe / b48fac4fe6 | qcpipe | 是 | 0 / 0 / 10 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续≤0.136 |
| `~/PvZ-search/.claude/worktrees/recipe` | recipe / 174b16b551 | recipe | 是 | 0 / 0 / 10 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续≤0.119 |
| `~/PvZ-search/.claude/worktrees/roof` | roof / 575d7da0bb | roof | 是 | 0 / 0 / 10 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/rowpool` | detached / 4703cf11d2 | 无（看主线可达性） | 是 | 0 / 0 / 3 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/rowpool2` | detached / d6c8e5eb60 | 无（看主线可达性） | 是 | 0 / 0 / 3 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/shovel` | shovel / c7af988ac8 | 无（看主线可达性） | 否 | 0 / 0 / 3 | D：尚不能证明已合并到远端 pvz-env；保留分支和工作树 | 0 |
| `~/PvZ-search/.claude/worktrees/slimworker` | slimworker / 67c12e57f2 | 无（看主线可达性） | 是 | 0 / 0 / 3 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/teacher` | teacher / 2fdab81c90 | 无（看主线可达性） | 是 | 0 / 0 / 4 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/termmerge` | mactest / b5d29130aa | mactest | 是 | 0 / 0 / 8 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/trainmem` | buildpeak / ff597f6da9 | buildpeak | 是 | 0 / 0 / 7 | C：已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；后续<0.0932 |
| `~/PvZ-search/.claude/worktrees/v5` | v5 / ced8ef48de | v5 | 是 | 0 / 0 / 3 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/vcpp` | vcpp / b73fd8cf97 | vcpp | 否 | 0 / 0 / 5 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/vnet` | vnet / e6c68f3ac1 | 无（看主线可达性） | 是 | 0 / 0 / 5 | A：会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `~/PvZ-search/.claude/worktrees/zones` | zones / 960737888c | zones | 是 | 0 / 0 / 14 | B：已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；后续≤0.273 |

主检出已有 410 条 tracked 变动、cq 63 条、dw4 1 条，dw5 有 `.codebuddy/` untracked；均非本任务改动，未 stash、恢复或纳入 cleanup 提交。dw4 的变动和其生产入链使它更不能直接移除。

4 条 worktree 元数据指向不存在的路径：`~/pvz-merge/artifacts/cuda/task7/origin`、`/tmp/pvz-cuda-task5-origin`、`/tmp/task36-premerge`、`/tmp/teacher_base`；`git worktree list` 标 prunable。当前无实体目录，预计释放 0，元数据归 D，不运行 prune，也不删除分支。

未来移除每一棵获批 C/B 树前：确认分支 HEAD 可从实时 origin 恢复且已合并；检查全局软链入链、proc/session、dirty/untracked/ignored；保存 B 的完整 HF 回验回执。仅使用单个明确路径的 `git worktree remove <该绝对路径>`，遇拒绝就停查原因，不加 force 扫过证据，不删分支。对 detached 树保留其远端主线可达 SHA 记录。cleanup 自身在本阶段不移除。

#### 7. 代码引用证据、待主控裁定项与执行门槛

以下 `git grep -n -I -F` 作用域为 `scripts/ python/ recipes/ dsh/ CMakeLists.txt`，以 cleanup 当前代码为基准；关键生产字段另在 cq 工作树复核，结果一致。命中数仅用于定位，路径未命中不等于无运行/数据链引用。

| 路径模式 | 代码命中行数 | 关键消费者 |
|---|---:|---|
| `artifacts/student/frames` | 25 | 包含 frames_v2 的子串命中；旧帧另有 backfill 示例及 hard/forks 数据链 |
| `frames_v2` | 35 | curriculum_queue、import_mac_labels、训练/审计；明确生产目标 |
| `v2_cache` | 7 | recipes/dw4i.json、dw4i2.json、dw5i.json、identity_dw3_s1.json；dagger_data_audit/design；run_dw4_day_diagnosis |
| `recipe_cache` | 4 | 4 个 recipes 写缓存路径 |
| `task23/training_frames` | 2 | dw4i/dw4i2 data |
| `task27/training_frames` | 1 | dw5i data；dw5 树另有 dw6i data、task36_data_ledger/row_weights 默认路径 |
| `task27/cache` | 1 | dw5i provenance 的原始缓存路径 |
| `task27/unified_v2` | 2 | panel_pressure/panel_v2 manifest；state 也仍引用 |
| `curriculum_state.pipeline` | 1 | test_incremental_pack 的生产 pack 门禁；运行 recipe/state 另有直接引用 |
| `artifacts/hard` | 40 | curriculum_queue 默认输出、static value、train_value_widen 旧 branch/branch3 |
| `artifacts/night` | 0 | 此代码作用域无直接命中，仅有存量原始样本 |
| `pvz-lane` | 0 | 未发现直接代码引用；来源未唯一确认，不能当垃圾 |
| `PvZ-Portable` | 36 | dsh 历史证据路径等，且 .git 是 shared git-common-dir |
| `.debug` | 5 | prof_symbolize 的 debug_root/符号缓存用途，非逐字 home 路径 |
| `.nv/ComputeCache` | 0 | 无直接代码命中；驱动生成，可重建 |
| `.codex/packages` | 0 | 无仓库代码命中；安装系统 current 链引用 |

需要主控裁定的独立议题：

1. 批准 §5 哪些明确 C 路径；不包含活动会话、生产树、旧帧或缓存统一。优先这批约 8 GiB 的可再生物与旧安装，先缓解磁盘压力。
2. B 原始模型/样本的 HF 归档范围、长期保留标准，以及非活动工作树 ignored 证据的退役范围。上传和下载回验都需要临时空间；按目录分批回验，不能先删后验。
3. 旧帧中 v2 缺失/不同 inode 的内容如何保留，是否允许补齐正式帧并改 hard/forks 引用；未定前 D 整根不动。已被 npz 取代的 jsonl 仍需逐文件五组校验；此前用户对旧 jsonl 的批准不扩展为任意删目录。
4. 统一生产 read_cache_dir 的独立变更是否立项；本次不改 cq，也不清生产 .merge、segments、cache 或历史 round。
5. cutoff、dw5、llmcand、memprobe、perf、shovel、vcpp 等未合入当前远端主线的树保留（即使已推自有分支）；活动会话结束和合并证据都要重查。pvz-lane 具体生成任务归属、日志的结论提炼范围仍存疑。

第二阶段只有在主控批准明确清单后开始；每批记录具体路径、回验凭证（B 类）、前后 df 与腾出空间。最后才按任务书运行 `sudo fstrim -v /` 并报 `df -h /`；本阶段没有运行。不得以全表某类标签批量 glob 删除，表中汇总行也不是删除对象。

#### 8. 全量 ≥100 MB 目录清单（354 项）

本节为那一次 du 的完整筛选结果，保留原始 B 数便于复核。路径全部相对 `/home/newbiexvwu/`；根以 `.` 表示。分类及消费者说明对每一行展开；“随项”表示后续释放已包含在 §5/§6 对应根项估算，**本行不重复计入总额**；A/D 当前预计 0。混合父目录用 D 或 A 的保护含义，不表示其所有子项同类。C/B 在主控批准前当前实际释放亦为 0。

| 路径 | 分摊 B | GiB | 类别 | 是什么/产生者、谁在引用/理由 | 当前预计/获批后随项上限 GiB |
|---|---:|---:|---|---|---|
| `.` | 150920466432 | 140.556 | D | 家目录汇总，包含 A/B/C/D；不可整目录处置 | 0 |
| `.codebuddy` | 458960896 | 0.427 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codebuddy/logs` | 112586752 | 0.105 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codebuddy/traces` | 152506368 | 0.142 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex` | 4666048512 | 4.346 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex-plus` | 115429376 | 0.108 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/.tmp` | 102096896 | 0.095 | D | 插件临时/安装现场；用途与恢复边界未确认，不当成缓存直接删 | 0 |
| `.codex/.tmp/plugins` | 102072320 | 0.095 | D | 插件临时/安装现场；用途与恢复边界未确认，不当成缓存直接删 | 0 |
| `.codex/packages` | 3507662848 | 3.267 | A | 安装版本集合；current 指向 0.160.1，仅下列旧版本另列 C | 0 |
| `.codex/packages/app-server-daemon` | 3507658752 | 3.267 | A | 安装版本集合；current 指向 0.160.1，仅下列旧版本另列 C | 0 |
| `.codex/packages/app-server-daemon/releases` | 3507646464 | 3.267 | A | 安装版本集合；current 指向 0.160.1，仅下列旧版本另列 C | 0 |
| `.codex/packages/app-server-daemon/releases/0.157.1-x86_64-unknown-linux-musl` | 391274496 | 0.364 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.364 |
| `.codex/packages/app-server-daemon/releases/0.157.1-x86_64-unknown-linux-musl/bin` | 359313408 | 0.335 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.335 |
| `.codex/packages/app-server-daemon/releases/0.158.0-x86_64-unknown-linux-musl` | 444289024 | 0.414 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.414 |
| `.codex/packages/app-server-daemon/releases/0.158.0-x86_64-unknown-linux-musl/bin` | 360636416 | 0.336 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.336 |
| `.codex/packages/app-server-daemon/releases/0.159.0-x86_64-unknown-linux-musl` | 444440576 | 0.414 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.414 |
| `.codex/packages/app-server-daemon/releases/0.159.0-x86_64-unknown-linux-musl/bin` | 360792064 | 0.336 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.336 |
| `.codex/packages/app-server-daemon/releases/0.159.1-x86_64-unknown-linux-musl` | 444579840 | 0.414 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.414 |
| `.codex/packages/app-server-daemon/releases/0.159.1-x86_64-unknown-linux-musl/bin` | 360931328 | 0.336 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.336 |
| `.codex/packages/app-server-daemon/releases/0.159.2-x86_64-unknown-linux-musl` | 444481536 | 0.414 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.414 |
| `.codex/packages/app-server-daemon/releases/0.159.2-x86_64-unknown-linux-musl/bin` | 360833024 | 0.336 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.336 |
| `.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl` | 444825600 | 0.414 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.414 |
| `.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl/bin` | 361172992 | 0.336 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.336 |
| `.codex/packages/app-server-daemon/releases/0.160.0-x86_64-unknown-linux-musl` | 446836736 | 0.416 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.416 |
| `.codex/packages/app-server-daemon/releases/0.160.0-x86_64-unknown-linux-musl/bin` | 363188224 | 0.338 | C | 旧 daemon 安装副本；current 不指向且可见 proc exe/maps 无引用，可重新安装 | 当前0；随项≤0.338 |
| `.codex/packages/app-server-daemon/releases/0.160.1-x86_64-unknown-linux-musl` | 446914560 | 0.416 | A | 安装版本集合；current 指向 0.160.1，仅下列旧版本另列 C | 0 |
| `.codex/packages/app-server-daemon/releases/0.160.1-x86_64-unknown-linux-musl/bin` | 363261952 | 0.338 | A | 安装版本集合；current 指向 0.160.1，仅下列旧版本另列 C | 0 |
| `.codex/sessions` | 644845568 | 0.601 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/sessions/2026` | 644841472 | 0.601 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/sessions/2026/09` | 149794816 | 0.140 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/sessions/2026/09/30` | 116035584 | 0.108 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/sessions/2026/10` | 495042560 | 0.461 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/sessions/2026/10/18` | 116776960 | 0.109 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.codex/sessions/2026/10/23` | 123785216 | 0.115 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.debug` | 2736926720 | 2.549 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home` | 2672246784 | 2.489 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu` | 2672242688 | 2.489 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs` | 2658078720 | 2.476 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml` | 2658074624 | 2.476 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib` | 2658070528 | 2.476 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14` | 2658066432 | 2.476 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages` | 2658062336 | 2.476 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia` | 1612177408 | 1.501 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13` | 1336979456 | 1.245 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib` | 1336975360 | 1.245 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcublasLt.so.13` | 508280832 | 0.473 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcublasLt.so.13/af45ff0258f38021736f07f164839af070746eb2` | 508276736 | 0.473 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcufft.so.12` | 290877440 | 0.271 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcufft.so.12/0db5a851657fea4fd1eefa47bd25f74d096bee61` | 290873344 | 0.271 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcusolver.so.12` | 156467200 | 0.146 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcusolver.so.12/8cb6a819b895b12cde901eb564c01cecd588aa97` | 156463104 | 0.146 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcusparse.so.12` | 167366656 | 0.156 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libcusparse.so.12/6ee62830e97c8f8b2f064511fd43d110bfa7a5fa` | 167362560 | 0.156 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libnvrtc.so.13` | 115343360 | 0.107 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib/libnvrtc.so.13/43d8e89d0c12671c26b8e79474e37b6fd1374a65` | 115339264 | 0.107 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cusparselt` | 233934848 | 0.218 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cusparselt/lib` | 233930752 | 0.218 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0` | 233926656 | 0.218 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0/ea393370ca2a968b8b528a6d814f48c51fc99052` | 233922560 | 0.218 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/torch` | 996941824 | 0.928 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/torch/lib` | 996937728 | 0.928 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so` | 446328832 | 0.416 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so/9b68c24087e9be32cee95fd7675596f0f73c7441` | 446324736 | 0.416 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/torch/lib/libtorch_cuda.so` | 512262144 | 0.477 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.debug/home/newbiexvwu/.venvs/ml/lib/python3.14/site-packages/torch/lib/libtorch_cuda.so/7aff4247d43ad11f1db856b4fba7194b76d93dd2` | 512258048 | 0.477 | C | 性能符号缓存；无可见进程 fd/maps 引用，prof_symbolize 可重新取；多数硬链接不能释放块 | 当前0；整根仅≤0.073，共享块不计 |
| `.local` | 1493843968 | 1.391 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/bin` | 220684288 | 0.206 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt` | 1193844736 | 1.112 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64` | 1121169408 | 1.044 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/bin` | 126627840 | 0.118 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib` | 923410432 | 0.860 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules` | 923406336 | 0.860 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@deepseek-ai` | 532459520 | 0.496 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@deepseek-ai/dsh` | 532455424 | 0.496 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@deepseek-ai/dsh/node_modules` | 532320256 | 0.496 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai` | 227614720 | 0.212 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/libreoffice-kit-wasm` | 152518656 | 0.142 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@deepseek-ai/dsh/node_modules/@deepseek-ai/libreoffice-kit-wasm/assets` | 151871488 | 0.141 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@tencent-ai` | 308645888 | 0.287 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@tencent-ai/codebuddy-code` | 308641792 | 0.287 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.local/opt/node-v24.21.0-linux-x64/lib/node_modules/@tencent-ai/codebuddy-code/dist` | 106614784 | 0.099 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.npm` | 1026928640 | 0.956 | C | npm 内容缓存；无可见进程引用，可重新下载 | 当前0；随项≤0.956 |
| `.npm/_cacache` | 1026875392 | 0.956 | C | npm 内容缓存；无可见进程引用，可重新下载 | 当前0；随项≤0.956 |
| `.npm/_cacache/content-v2` | 1015107584 | 0.945 | C | npm 内容缓存；无可见进程引用，可重新下载 | 当前0；随项≤0.945 |
| `.npm/_cacache/content-v2/sha512` | 1015103488 | 0.945 | C | npm 内容缓存；无可见进程引用，可重新下载 | 当前0；随项≤0.945 |
| `.npm/_cacache/content-v2/sha512/02` | 124133376 | 0.116 | C | npm 内容缓存；无可见进程引用，可重新下载 | 当前0；随项≤0.116 |
| `.nv` | 1064820736 | 0.992 | C | CUDA 编译缓存；无可见 fd/maps 引用，可重建；执行前再查 GPU 任务 | 当前0；随项≤0.992 |
| `.nv/ComputeCache` | 1064816640 | 0.992 | C | CUDA 编译缓存；无可见 fd/maps 引用，可重建；执行前再查 GPU 任务 | 当前0；随项≤0.992 |
| `.nv/ComputeCache/5` | 125423616 | 0.117 | C | CUDA 编译缓存；无可见 fd/maps 引用，可重建；执行前再查 GPU 任务 | 当前0；随项≤0.117 |
| `.nv/ComputeCache/6` | 161402880 | 0.150 | C | CUDA 编译缓存；无可见 fd/maps 引用，可重建；执行前再查 GPU 任务 | 当前0；随项≤0.150 |
| `.nv/ComputeCache/9` | 160518144 | 0.149 | C | CUDA 编译缓存；无可见 fd/maps 引用，可重建；执行前再查 GPU 任务 | 当前0；随项≤0.149 |
| `.nv/ComputeCache/9/1` | 112152576 | 0.104 | C | CUDA 编译缓存；无可见 fd/maps 引用，可重建；执行前再查 GPU 任务 | 当前0；随项≤0.104 |
| `.qoder` | 527540224 | 0.491 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.qoder/bin` | 181460992 | 0.169 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.qoder/bin/qodercli` | 181456896 | 0.169 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.qoder/logs` | 238854144 | 0.222 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.qoder/logs/runs` | 174047232 | 0.162 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.qodersec` | 146726912 | 0.137 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.qodersec/bin` | 146239488 | 0.136 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs` | 3115708416 | 2.902 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml` | 3115704320 | 2.902 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib` | 3115565056 | 2.902 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14` | 3115560960 | 2.902 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages` | 3115556864 | 2.902 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia` | 1637109760 | 1.525 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia/cu13` | 377425920 | 0.352 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia/cu13/lib` | 367468544 | 0.342 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia/cudnn` | 966426624 | 0.900 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia/cudnn/lib` | 965304320 | 0.899 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia/nccl` | 253513728 | 0.236 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/nvidia/nccl/lib` | 252657664 | 0.235 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/torch` | 254312448 | 0.237 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton` | 936255488 | 0.872 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton/_C` | 528265216 | 0.492 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton/backends` | 389885952 | 0.363 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton/backends/nvidia` | 386441216 | 0.360 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton/backends/nvidia/lib` | 290672640 | 0.271 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton/backends/nvidia/lib/cupti` | 133566464 | 0.124 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `.venvs/ml/lib/python3.14/site-packages/triton/backends/nvidia/lib/cupti-blackwell` | 156397568 | 0.146 | A | Agent/工具或运行环境；进程 exe/cwd/fd 引用，历史部分另议 | 0 |
| `PvZ-Portable` | 3388129280 | 3.155 | D | 原始检出与共享 Git 库共存，不能整目录删；代码已被 origin/pvz-env 包含 | 0 |
| `PvZ-Portable/.git` | 1915596800 | 1.784 | A | 所有 linked worktree 共用对象库与 refs；必须保留，不做历史重写/gc | 0 |
| `PvZ-Portable/.git/objects` | 1904087040 | 1.773 | A | 所有 linked worktree 共用对象库与 refs；必须保留，不做历史重写/gc | 0 |
| `PvZ-Portable/.git/objects/pack` | 1886855168 | 1.757 | A | 所有 linked worktree 共用对象库与 refs；必须保留，不做历史重写/gc | 0 |
| `PvZ-Portable/artifacts` | 443912192 | 0.413 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | 当前0；随项≤0.413 |
| `PvZ-Portable/artifacts/research` | 314961920 | 0.293 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | 当前0；随项≤0.293 |
| `PvZ-Portable/artifacts/research/bc_finetune_level7_v3` | 314957824 | 0.293 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | 当前0；随项≤0.293 |
| `PvZ-Portable/artifacts/research/bc_finetune_level7_v3/output` | 314953728 | 0.293 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | 当前0；随项≤0.293 |
| `PvZ-Portable/artifacts/research/bc_finetune_level7_v3/output/runs` | 314949632 | 0.293 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | 当前0；随项≤0.293 |
| `PvZ-Portable/artifacts/research/bc_finetune_level7_v3/output/runs/run_1` | 314945536 | 0.293 | B | 早期实验模型/证据，未证明 HF 已完整回验，先归档上传 | 当前0；随项≤0.293 |
| `PvZ-Portable/build` | 225972224 | 0.210 | D | 旧构建和 core 混合；仅 core.2439 单列 C，其他用途不整目录假定 | 0 |
| `PvZ-Portable/logs` | 777302016 | 0.724 | C | 旧 PPO/t5 日志；可见 proc 无引用，先提炼结论后按明确文件清单处理 | 当前0；随项≤0.724 |
| `PvZ-Portable/logs/t5_research` | 771502080 | 0.719 | C | 旧 PPO/t5 日志；可见 proc 无引用，先提炼结论后按明确文件清单处理 | 当前0；随项≤0.719 |
| `PvZ-search` | 113875263488 | 106.055 | D | 主检出含生产数据及 410 条已有 tracked 变动；不在这里改动/提交 | 0 |
| `PvZ-search/.claude` | 50429943808 | 46.967 | D | 工作树汇总，逐工作树看审计表；不可整目录删 | 0 |
| `PvZ-search/.claude/worktrees` | 50429939712 | 46.967 | D | 工作树汇总，逐工作树看审计表；不可整目录删 | 0 |
| `PvZ-search/.claude/worktrees/b3` | 998842368 | 0.930 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts` | 973271040 | 0.906 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42` | 973217792 | 0.906 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/full` | 572133376 | 0.533 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/full/trajectories` | 560291840 | 0.522 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/full/trajectories/day_l10_orig` | 118870016 | 0.111 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/full/trajectories/day_l7` | 133279744 | 0.124 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/full/trajectories/night_l14` | 247529472 | 0.231 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/round2` | 333168640 | 0.310 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/round2/full` | 321196032 | 0.299 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/round2/full/trajectories` | 313692160 | 0.292 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/b3/artifacts/task42/round2/full/trajectories/night_l14` | 248303616 | 0.231 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/conveyor` | 1489498112 | 1.387 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts` | 1442721792 | 1.344 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts/task20` | 1442439168 | 1.343 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts/task20/cache` | 1122566144 | 1.045 | C | task20 可再生特征缓存；生产链接只读同级 frames，不读此 cache | 当前0；随项≤1.045 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts/task20/cache/cells` | 364621824 | 0.340 | C | task20 可再生特征缓存；生产链接只读同级 frames，不读此 cache | 当前0；随项≤0.340 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts/task20/cache/identity_scene` | 162095104 | 0.151 | C | task20 可再生特征缓存；生产链接只读同级 frames，不读此 cache | 当前0；随项≤0.151 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts/task20/cache/packets` | 562712576 | 0.524 | C | task20 可再生特征缓存；生产链接只读同级 frames，不读此 cache | 当前0；随项≤0.524 |
| `PvZ-search/.claude/worktrees/conveyor/artifacts/task20/frames_v2` | 211881984 | 0.197 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/deploy` | 113610752 | 0.106 | C | 已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；随项≤0.106 |
| `PvZ-search/.claude/worktrees/deploy2` | 113696768 | 0.106 | C | 已合并且远端可达，无会话/生产入链；只在复查 clean 与 ignored 均可再生后移除工作树 | 当前0；随项≤0.106 |
| `PvZ-search/.claude/worktrees/deploy3` | 115433472 | 0.108 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw4` | 1271132160 | 1.184 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw4/artifacts` | 1223892992 | 1.140 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw4/artifacts/task23` | 1223839744 | 1.140 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw4/artifacts/task23/training_frames` | 789553152 | 0.735 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5` | 28413698048 | 26.462 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts` | 28366118912 | 26.418 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27` | 28225634304 | 26.287 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache` | 7610994688 | 7.088 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache/cells` | 552062976 | 0.514 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache/identity_scene` | 239005696 | 0.223 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache/packets` | 455725056 | 0.424 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache/v2` | 6364196864 | 5.927 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/hf_mac` | 1585057792 | 1.476 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/hf_mac/transfer` | 1584783360 | 1.476 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/hf_probe` | 209379328 | 0.195 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/hf_probe/transfer` | 209346560 | 0.195 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames` | 18437722112 | 17.171 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames/packs` | 17517293568 | 16.314 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames/packs/e39ba7cbc076380a.npz.d` | 8744206336 | 8.144 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/dw5/artifacts/task36` | 140402688 | 0.131 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/fog` | 396738560 | 0.369 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/fog/artifacts` | 275136512 | 0.256 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/fog/artifacts/task45` | 275079168 | 0.256 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/fog/artifacts/task45/phase-e` | 195993600 | 0.183 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/fog/artifacts/task45/phase-e/episodes` | 186892288 | 0.174 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/forks` | 493039616 | 0.459 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.459 |
| `PvZ-search/.claude/worktrees/forks/artifacts` | 464257024 | 0.432 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.432 |
| `PvZ-search/.claude/worktrees/forks/artifacts/task53` | 464203776 | 0.432 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.432 |
| `PvZ-search/.claude/worktrees/forks/artifacts/task53/roots` | 302333952 | 0.282 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.282 |
| `PvZ-search/.claude/worktrees/forks/artifacts/task53/value-data` | 138952704 | 0.129 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.129 |
| `PvZ-search/.claude/worktrees/identity` | 1267331072 | 1.180 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤1.180 |
| `PvZ-search/.claude/worktrees/identity/artifacts` | 1195433984 | 1.113 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤1.113 |
| `PvZ-search/.claude/worktrees/identity/artifacts/task19` | 1194889216 | 1.113 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤1.113 |
| `PvZ-search/.claude/worktrees/identity/artifacts/task19/cache` | 1085206528 | 1.011 | C | task19 可再生特征缓存；无可见会话或生产配方引用，原帧保留 | 当前0；随项≤1.011 |
| `PvZ-search/.claude/worktrees/identity/artifacts/task19/cache/cells` | 362496000 | 0.338 | C | task19 可再生特征缓存；无可见会话或生产配方引用，原帧保留 | 当前0；随项≤0.338 |
| `PvZ-search/.claude/worktrees/identity/artifacts/task19/cache/identity_scene` | 161030144 | 0.150 | C | task19 可再生特征缓存；无可见会话或生产配方引用，原帧保留 | 当前0；随项≤0.150 |
| `PvZ-search/.claude/worktrees/identity/artifacts/task19/cache/packets` | 561676288 | 0.523 | C | task19 可再生特征缓存；无可见会话或生产配方引用，原帧保留 | 当前0；随项≤0.523 |
| `PvZ-search/.claude/worktrees/llmcand` | 167399424 | 0.156 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe` | 7732035584 | 7.201 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts` | 7705436160 | 7.176 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58` | 7705382912 | 7.176 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models` | 7598198784 | 7.076 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s1` | 243224576 | 0.227 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s1/test` | 121393152 | 0.113 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s1/test_ieee_fp32` | 121397248 | 0.113 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s2` | 243531776 | 0.227 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s2/test` | 121548800 | 0.113 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s2/test_ieee_fp32` | 121552896 | 0.113 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s3` | 242294784 | 0.226 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s3/test` | 120926208 | 0.113 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-human-s3/test_ieee_fp32` | 120938496 | 0.113 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s1` | 236224512 | 0.220 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s1/test` | 117891072 | 0.110 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s1/test_ieee_fp32` | 117903360 | 0.110 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s2` | 233168896 | 0.217 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s2/test` | 116355072 | 0.108 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s2/test_ieee_fp32` | 116383744 | 0.108 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s3` | 236359680 | 0.220 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s3/test` | 117960704 | 0.110 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/convgru-xray-s3/test_ieee_fp32` | 117968896 | 0.110 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s1` | 257949696 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s1/test` | 128753664 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s1/test_ieee_fp32` | 128765952 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s2` | 257613824 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s2/test` | 128581632 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s2/test_ieee_fp32` | 128602112 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s3` | 257888256 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s3/test` | 128724992 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-human-s3/test_ieee_fp32` | 128733184 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s1` | 254607360 | 0.237 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s1/test` | 127082496 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s1/test_ieee_fp32` | 127094784 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s2` | 254099456 | 0.237 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s2/test` | 126828544 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s2/test_ieee_fp32` | 126840832 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s3` | 254947328 | 0.237 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s3/test` | 127254528 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/current-xray-s3/test_ieee_fp32` | 127262720 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s1` | 258834432 | 0.241 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s1/test` | 129196032 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s1/test_ieee_fp32` | 129216512 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s2` | 258465792 | 0.241 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s2/test` | 129015808 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s2/test_ieee_fp32` | 129028096 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s3` | 258244608 | 0.241 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s3/test` | 128909312 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-human-s3/test_ieee_fp32` | 128913408 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s1` | 258084864 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s1/test` | 128819200 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s1/test_ieee_fp32` | 128843776 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s2` | 249335808 | 0.232 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s2/test` | 124456960 | 0.116 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s2/test_ieee_fp32` | 124456960 | 0.116 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s3` | 256774144 | 0.239 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s3/test` | 128176128 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/spatial_ssm-xray-s3/test_ieee_fp32` | 128176128 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s1` | 255819776 | 0.238 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s1/test` | 126861312 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s1/test_ieee_fp32` | 128536576 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s2` | 256663552 | 0.239 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s2/test` | 127266816 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s2/test_ieee_fp32` | 128974848 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s3` | 256679936 | 0.239 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s3/test` | 127324160 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-human-s3/test_ieee_fp32` | 128933888 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s1` | 257413120 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s1/test` | 128147456 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s1/test_ieee_fp32` | 128843776 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s2` | 253923328 | 0.236 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s2/test` | 126410752 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s2/test_ieee_fp32` | 127090688 | 0.118 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s3` | 257593344 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s3/test` | 128278528 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack4-xray-s3/test_ieee_fp32` | 128892928 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s1` | 258027520 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s1/test` | 128802816 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s1/test_ieee_fp32` | 128802816 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s2` | 257376256 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s2/test` | 128471040 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s2/test_ieee_fp32` | 128483328 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s3` | 258162688 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s3/test` | 128860160 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-human-s3/test_ieee_fp32` | 128880640 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s1` | 258392064 | 0.241 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s1/test` | 128978944 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s1/test_ieee_fp32` | 128991232 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s2` | 258531328 | 0.241 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s2/test` | 129048576 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s2/test_ieee_fp32` | 129060864 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s3` | 257957888 | 0.240 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s3/test` | 128761856 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/memprobe/artifacts/task58/models/stack8-xray-s3/test_ieee_fp32` | 128774144 | 0.120 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf` | 825106432 | 0.768 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts` | 617484288 | 0.575 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts/perf` | 617426944 | 0.575 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts/perf/batch_prototype` | 254693376 | 0.237 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts/perf/batch_prototype/0822825277b17b8a81661d5ec43815239454cc3b` | 128032768 | 0.119 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts/perf/batch_prototype/adcf9b3fb71db0af197674d4216f12efc2e67912` | 125751296 | 0.117 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts/perf/tools` | 170266624 | 0.159 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/perf/artifacts/perf/tools/usr` | 139214848 | 0.130 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/qcbench` | 623919104 | 0.581 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/qcbench/artifacts` | 575754240 | 0.536 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/qcbench/artifacts/task51` | 514101248 | 0.479 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/qcbench/artifacts/task51/history` | 432275456 | 0.403 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/qcpipe` | 145526784 | 0.136 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.136 |
| `PvZ-search/.claude/worktrees/recipe` | 128180224 | 0.119 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.119 |
| `PvZ-search/.claude/worktrees/roof` | 576618496 | 0.537 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/roof/artifacts` | 456687616 | 0.425 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/roof/artifacts/task46` | 456630272 | 0.425 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/roof/artifacts/task46/phase2` | 371511296 | 0.346 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/v5` | 2570493952 | 2.394 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/v5/artifacts` | 2545807360 | 2.371 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/v5/artifacts/task43` | 2545754112 | 2.371 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/v5/artifacts/task43/l21_dump` | 2144428032 | 1.997 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/v5/artifacts/task43/sparse64` | 384339968 | 0.358 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vcpp` | 236691456 | 0.220 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vcpp/artifacts` | 178638848 | 0.166 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vcpp/artifacts/task37` | 178581504 | 0.166 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vnet` | 1172045824 | 1.092 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vnet/artifacts` | 1043251200 | 0.972 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vnet/artifacts/task30` | 1043193856 | 0.972 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vnet/artifacts/task30/v4` | 997593088 | 0.929 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/vnet/build` | 102703104 | 0.096 | A | 会话/生产引用链保护；具体消费者见工作树与链接表 | 0 |
| `PvZ-search/.claude/worktrees/zones` | 293486592 | 0.273 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.273 |
| `PvZ-search/.claude/worktrees/zones/artifacts` | 171696128 | 0.160 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.160 |
| `PvZ-search/.claude/worktrees/zones/artifacts/task50-a` | 121671680 | 0.113 | B | 已合并/推送，但 ignored 实验证据不由 Git 保存；先 HF 回验再移除工作树 | 当前0；随项≤0.113 |
| `PvZ-search/artifacts` | 63397404672 | 59.043 | D | 实验与生产混合汇总，按下列子项处置 | 0 |
| `PvZ-search/artifacts/hard` | 4477394944 | 4.170 | A | qc 默认输出/计数来源；现有 search_play 正在写 games.jsonl，不能整目录删 | 0 |
| `PvZ-search/artifacts/hard/branch` | 2452164608 | 2.284 | B | 旧分支价值训练数据；train_value_widen.py 默认引用，离线重放仍需要，先 HF 完整归档 | 当前0；随项≤2.284 |
| `PvZ-search/artifacts/hard/branch3` | 1682092032 | 1.567 | B | 旧分支价值训练数据；train_value_widen.py 默认引用，离线重放仍需要，先 HF 完整归档 | 当前0；随项≤1.567 |
| `PvZ-search/artifacts/hard/branch3/z15d_shards` | 1682087936 | 1.567 | B | 旧分支价值训练数据；train_value_widen.py 默认引用，离线重放仍需要，先 HF 完整归档 | 当前0；随项≤1.567 |
| `PvZ-search/artifacts/night` | 4073144320 | 3.793 | B | 旧夜间价值样本；代码 git grep 无直接引用，可见进程无引用，先存 HF；不据此声称已备份 | 当前0；随项≤3.793 |
| `PvZ-search/artifacts/night/vdata` | 4058439680 | 3.780 | B | 旧夜间价值样本；代码 git grep 无直接引用，可见进程无引用，先存 HF；不据此声称已备份 | 当前0；随项≤3.780 |
| `PvZ-search/artifacts/relabel` | 396623872 | 0.369 | B | 历史重标样本；未证实远端完整回验，先保留证据 | 当前0；随项≤0.369 |
| `PvZ-search/artifacts/relabel/frames` | 316436480 | 0.295 | B | 历史重标样本；未证实远端完整回验，先保留证据 | 当前0；随项≤0.295 |
| `PvZ-search/artifacts/student` | 54396141568 | 50.660 | D | 正式帧、旧帧、生产模型与缓存混合；分项看 | 0 |
| `PvZ-search/artifacts/student/.feature_cache` | 530378752 | 0.494 | C | 旧学生目录特征缓存；当前配方、proc 未引用，编码器可重建 | 当前0；随项≤0.494 |
| `PvZ-search/artifacts/student/frames` | 29947662336 | 27.891 | D | 旧帧未被完全覆盖且有 hard/forks 入链；不能整目录删除；唯一块上限另算 | 0 |
| `PvZ-search/artifacts/student/frames_v2` | 14779056128 | 13.764 | A | 正式帧，qc 与导入服务在用；整棵永久保留，含其 .feature_cache | 0 |
| `PvZ-search/artifacts/student/frames_v2/.feature_cache` | 2690383872 | 2.506 | A | 正式帧，qc 与导入服务在用；整棵永久保留，含其 .feature_cache | 0 |
| `PvZ-search/artifacts/student/frames_v2/.feature_cache/v2` | 2690379776 | 2.506 | A | 正式帧，qc 与导入服务在用；整棵永久保留，含其 .feature_cache | 0 |
| `PvZ-search/artifacts/student/v2_cache` | 8983879680 | 8.367 | A | cq 配方与 round 2/3 read_cache_dir 仍指向此处；不得直接换路径 | 0 |
| `PvZ-search/artifacts/student/v2_cache/aux_v1_k8` | 142708736 | 0.133 | A | cq 配方与 round 2/3 read_cache_dir 仍指向此处；不得直接换路径 | 0 |
| `PvZ-search/artifacts/student/v2_cache/cells` | 401956864 | 0.374 | A | cq 配方与 round 2/3 read_cache_dir 仍指向此处；不得直接换路径 | 0 |
| `PvZ-search/artifacts/student/v2_cache/v2` | 8439209984 | 7.860 | A | cq 配方与 round 2/3 read_cache_dir 仍指向此处；不得直接换路径 | 0 |
| `pvz-lane` | 678096896 | 0.632 | D | 独立 seed 轨迹原始数据，生成任务未能唯一归属；git grep 无引用，交主控确认归属和引用 | 0 |
| `pvz-logs` | 17269030912 | 16.083 | A | 生产状态、守护脚本及当前日志；大旧日志按单文件另列，不整目录删 | 0 |
| `pvz-logs/curriculum_state.pipeline` | 15978369024 | 14.881 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/cache` | 7191506944 | 6.698 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/cache/cells` | 417558528 | 0.389 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/cache/identity_scene` | 220753920 | 0.206 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/cache/packets` | 422961152 | 0.394 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/cache/v2` | 6130229248 | 5.709 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/frames` | 8783777792 | 8.181 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/frames/packs` | 8364085248 | 7.790 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |
| `pvz-logs/curriculum_state.pipeline/frames/packs/segments` | 8364027904 | 7.790 | A | qc 的 frames 链接视图、分段 pack、cache、round 模型与状态恢复链全部在用 | 0 |

**阶段边界：本报告提交并 push origin cleanup 后，第一阶段完成；停止等待主控批准，不进入第二阶段。**

### 第二阶段执行记录（主控 17:05 批准）

按批准的 1→5 顺序执行。下表 df 单位 B，差值 = 前已用 − 后已用；qc 并发增长、页缓存/延迟回收和归档暂存会影响净变化，不能把它等同于单一目录 du。每行路径均为实际明确对象。 初期第二步的“归档回验”行在临时 tar 清除前采样，故可出现负值；各阶段总净变化以下方阶段汇总的首尾 df 为准，不累加重叠窗口。

| 本机时间 | 步骤 | 路径/批次 | df 已用前 | df 已用后 | 净减少 B | 状态与证据 |
|---|---|---|---:|---:|---:|---|
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.npm/_cacache` | 168604545024 | 167577673728 | 1026871296 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.nv/ComputeCache` | 167577677824 | 166512861184 | 1064816640 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.debug` | 166512861184 | 166433763328 | 79097856 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.157.1-x86_64-unknown-linux-musl` | 166433767424 | 166042492928 | 391274496 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.158.0-x86_64-unknown-linux-musl` | 166042492928 | 165598203904 | 444289024 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.159.0-x86_64-unknown-linux-musl` | 165598203904 | 165153763328 | 444440576 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.159.1-x86_64-unknown-linux-musl` | 165153763328 | 164709183488 | 444579840 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.159.2-x86_64-unknown-linux-musl` | 164709183488 | 164264701952 | 444481536 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl` | 164264701952 | 163819876352 | 444825600 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:04+08:00 | 1-C及追加缓存 | `~/.codex/packages/app-server-daemon/releases/0.160.0-x86_64-unknown-linux-musl` | 163819876352 | 163373039616 | 446836736 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:05+08:00 | 1-C及追加缓存 | `~/PvZ-Portable/logs` | 163373043712 | 162595741696 | 777302016 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:05+08:00 | 1-C及追加缓存 | `~/PvZ-Portable/build/core.2439` | 162595741696 | 162394333184 | 201408512 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:05+08:00 | 1-C及追加缓存 | `~/PvZ-search/artifacts/student/.feature_cache` | 162394337280 | 161863958528 | 530378752 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:09+08:00 | 1-C及追加缓存 | `~/PvZ-search/.claude/worktrees/conveyor/artifacts/task20/cache` | 161863958528 | 160741408768 | 1122549760 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:13+08:00 | 1-C及追加缓存 | `~/PvZ-search/.claude/worktrees/identity/artifacts/task19/cache` | 160741408768 | 159656206336 | 1085202432 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:14+08:00 | 1-C及追加缓存 | `~/PvZ-search/artifacts/student/frames_v2/.feature_cache` | 159656206336 | 156965822464 | 2690383872 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:17+08:00 | 1-C及追加缓存 | `~/PvZ-search/artifacts/student/v2_cache` | 156965822464 | 147981946880 | 8983875584 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:22+08:00 | 1-C及追加缓存 | `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27/cache` | 147981946880 | 140370956288 | 7610990592 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:22+08:00 | 1-C及追加缓存 | `~/PvZ-search/.claude/worktrees/dw5/artifacts/task27/training_frames/packs` | 140370960384 | 122853666816 | 17517293568 | 完成：主控逐项批准；删除前复核可见 fd/cwd/exe 与相关 pack/train 进程；未递归跟随软链 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/cb_cuda_tui.log` | 122853670912 | 122810265600 | 43405312 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/cb_shovel_tui.log` | 122810265600 | 122777595904 | 32669696 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/codex_cuda_tui.log` | 122777595904 | 122763276288 | 14319616 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/dw3s2_eval.log` | 122763276288 | 122600067072 | 163209216 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/iter2_teacher_eval.log` | 122600067072 | 122576740352 | 23326720 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/night_vdata_a100.log` | 122576740352 | 122565074944 | 11665408 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/night_vdata_b25.log` | 122565074944 | 122553430016 | 11644928 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:22+08:00 | 1-未打开大日志 | `~/pvz-logs/night_vdata_b50.log` | 122553430016 | 122541764608 | 11665408 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/night_vdata_b75.log` | 122541764608 | 122530103296 | 11661312 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/probe_day_l9.log` | 122530103296 | 122519871488 | 10231808 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/qoder_tui.log` | 122519871488 | 122350030848 | 169840640 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/student_v2b_eval.log` | 122350034944 | 122326745088 | 23289856 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/student_v3_eval.log` | 122326745088 | 122303426560 | 23318528 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/student_v3all_eval.log` | 122303426560 | 122280136704 | 23289856 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/v34_panel.log` | 122280136704 | 122256830464 | 23306240 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/weak_update_000000_rerun.log` | 122256830464 | 122246537216 | 10293248 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:31:23+08:00 | 1-未打开大日志 | `~/pvz-logs/weak_update_000034.log` | 122246537216 | 122234875904 | 11661312 | 完成：单文件 ≥10,000,000 B；删除前再次检查 fd，qc.out 等打开日志保留 |
| 2026-10-24T16:33:37+08:00 | 1-配方变更 | `~/PvZ-search/.claude/worktrees/cleanup/recipes/dw4i.json`；`~/PvZ-search/.claude/worktrees/cleanup/recipes/dw4i2.json`；`~/PvZ-search/.claude/worktrees/cleanup/recipes/dw5i.json`；`~/PvZ-search/.claude/worktrees/cleanup/recipes/identity_dw3_s1.json` | 122237718528 | 122237718528 | 0 | 完成：移除可选 read_cache_dir；pvz_recipe 默认为 None；全部 recipes schema 验证通过；未改 cq 或 pipeline 已保存配方 |
| 2026-10-24T16:34:57+08:00 | 2-B-归档回验 | `~/PvZ-search/artifacts/hard/branch` | 122237755392 | 124691021824 | -2453266432 | 完成：HF archive/2026-10-24/hard_branch.MANIFEST.json @ 64688af9c6861b94bd74fa3d159265d8ab5ce25f；全部部件强制下载、SHA256/字节数一致；源元数据稳定；详细清单在 HF MANIFEST |
| 2026-10-24T16:34:58+08:00 | 2-B源目录删除 | `~/PvZ-search/artifacts/hard/branch` | 122238795776 | 119786631168 | 2452164608 | 完成：HF 固定 revision 64688af9c6861b94bd74fa3d159265d8ab5ce25f 已下载回验，索引 hard_branch |
| 2026-10-24T16:35:42+08:00 | 2-B-归档回验 | `~/PvZ-search/artifacts/hard/branch3` | 119786635264 | 121471156224 | -1684520960 | 完成：HF archive/2026-10-24/hard_branch3.MANIFEST.json @ 474831852f2e4594a062e779f87715e6f337355a；全部部件强制下载、SHA256/字节数一致；源元数据稳定；详细清单在 HF MANIFEST |
| 2026-10-24T16:35:42+08:00 | 2-B源目录删除 | `~/PvZ-search/artifacts/hard/branch3` | 119788965888 | 118106873856 | 1682092032 | 完成：HF 固定 revision 474831852f2e4594a062e779f87715e6f337355a 已下载回验，索引 hard_branch3 |
| 2026-10-24T16:38:59+08:00 | 2-B-归档回验 | `~/PvZ-search/artifacts/night` | 118106873856 | 122182909952 | -4076036096 | 完成：HF archive/2026-10-24/night.MANIFEST.json @ 82108867dac86e39d5ad0bed729757eaf248d75c；全部部件强制下载、SHA256/字节数一致；源元数据稳定；详细清单在 HF MANIFEST |
| 2026-10-24T16:38:59+08:00 | 2-B源目录删除 | `~/PvZ-search/artifacts/night` | 118118555648 | 114045411328 | 4073144320 | 完成：HF 固定 revision 82108867dac86e39d5ad0bed729757eaf248d75c 已下载回验，索引 night |
| 2026-10-24T16:42:00+08:00 | 2-B-归档回验 | `~/PvZ-search/artifacts/relabel` | 114045411328 | 114441355264 | -395943936 | 完成：HF archive/2026-10-24/relabel.MANIFEST.json @ ba30a8018bb6354acf0745721e69ec8644de7b0f；全部部件强制下载、SHA256/字节数一致；源元数据稳定；详细清单在 HF MANIFEST |
| 2026-10-24T16:42:00+08:00 | 2-B源目录删除 | `~/PvZ-search/artifacts/relabel` | 114054791168 | 113658167296 | 396623872 | 完成：HF 固定 revision ba30a8018bb6354acf0745721e69ec8644de7b0f 已下载回验，索引 relabel |
| 2026-10-24T16:43:43+08:00 | 2-B-归档回验 | `~/PvZ-Portable/artifacts` | 113658167296 | 114111021056 | -452853760 | 完成：HF archive/2026-10-24/portable_artifacts.MANIFEST.json @ 6a6978e1446bf411a137b83e47d0e354066b52d1；全部部件强制下载、SHA256/字节数一致；源元数据稳定；详细清单在 HF MANIFEST |
| 2026-10-24T16:43:43+08:00 | 2-B源目录删除 | `~/PvZ-Portable/artifacts` | 113663692800 | 113219944448 | 443748352 | 完成：HF 固定 revision 6a6978e1446bf411a137b83e47d0e354066b52d1 已下载回验，索引 portable_artifacts |
| 2026-10-24T16:47:44+08:00 | 2-B-归档回验 | `~/pvz-lane` | 113219944448 | 113900937216 | -680992768 | 完成：HF archive/2026-10-24/pvz_lane.MANIFEST.json @ dab418c1b8d1bd9f45a5ffb8483d484cdae4e918；全部部件强制下载、SHA256/字节数一致；源元数据稳定；详细清单在 HF MANIFEST |
| 2026-10-24T16:47:44+08:00 | 2-B源目录删除 | `~/pvz-lane` | 113231912960 | 112553816064 | 678096896 | 完成：HF 固定 revision dab418c1b8d1bd9f45a5ffb8483d484cdae4e918 已下载回验，索引 pvz_lane |
| 2026-10-24T16:47:44+08:00 | 1-配方回归核验 | `~/PvZ-search/.claude/worktrees/cleanup/python/test_recipe.py` | 112554008576 | 112554008576 | 0 | 完成：旧测试强制访问已移除的可选字段导致 KeyError；已改为验证默认 None 和剩余必选路径可移植；pytest test_recipe.py：8 passed、18 subtests passed（13.19s） |
| 2026-10-24T16:48:14+08:00 | 2-阶段汇总 | `~/PvZ-search/artifacts/hard/branch`；`~/PvZ-search/artifacts/hard/branch3`；`~/PvZ-search/artifacts/night`；`~/PvZ-search/artifacts/relabel`；`~/PvZ-Portable/artifacts`；`~/pvz-lane` | 122237755392 | 112555196416 | 9682558976 | 完成：6 组均完成 HF 固定 revision 下载 SHA256/字节数回验后删除；所有上传/下载暂存已清；night 拆为 4GB+余片；完整索引 task62-archive-index.md |
| 2026-10-24T17:55:37+08:00 | 3-旧帧归档回验 | `~/PvZ-search/artifacts/student/frames` | 112578510848 | 120607309824 | -8028798976 | 完成：代理传输大量发送失败降速；仅终止自有两个上传进程（均退出143），小流量实测代理0.04MiB/s、直连1.2MiB/s，改归档进程直连复用原分片/Xet数据块；固定 HF revision 782c24f44ac9c2ec6847580c683fd10bd58ccb10 强制下载所有部件，SHA256/字节数及源稳定性通过；暂存已清 |
| 2026-10-24T17:57:41+08:00 | 3-退役旧离线视图链接 | `~/PvZ-search/artifacts/hard/vdata`；`~/PvZ-search/artifacts/hard/vdata_ctx`；`~/PvZ-search/artifacts/hard/vdata_z15`；`~/PvZ-search/.claude/worktrees/forks/artifacts/task53/training/frames` | 120611647488 | 120596934656 | 14712832 | 完成：3592 条明确软链已随 student_frames_v1 tar 回验后逐一 unlink；全家清查结果与生产 state/recipe/dw4 视图交叉证明不在生产链；未递归删除其余 hard 数据 |
| 2026-10-24T17:57:54+08:00 | 3-旧帧整根退役 | `~/PvZ-search/artifacts/student/frames` | 120596934656 | 102123933696 | 18473000960 | 完成：HF 固定 revision 782c24f44ac9c2ec6847580c683fd10bd58ccb10 已下载回验，索引 student_frames_v1 |
| 2026-10-24T17:57:54+08:00 | 3-阶段汇总 | `~/PvZ-search/artifacts/student/frames` | 120609099776 | 102123933696 | 18485166080 | 完成：旧根已不存在；83030 个共享文件在 frames_v2 的 inode/字节数逐一复核未变；3592 条旧入链已归档退役；索引含取回方法 |
| 2026-10-24T17:59:38+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/advgap` | 102125350912 | 102127976448 | -2625536 | 完成：HF archive/2026-10-24/worktrees/worktree_advgap.MANIFEST.json @ 6529e3c1d3df9caad3e45c1bb6012ddf3f1058db；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T17:59:38+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/advgap` | 102127980544 | 102075891712 | 52088832 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 88839b609bd97ef637692ec6d97246c23ba29f1a 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:01:12+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/b3` | 102075891712 | 102095044608 | -19152896 | 完成：HF archive/2026-10-24/worktrees/worktree_b3.MANIFEST.json @ 3abfd4eba6ebed56dcfdbc785d08a08c15aa8af0；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:01:12+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/b3` | 102095044608 | 101096439808 | 998604800 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD b30a75fbca82a37f09482861cfe129d3276359ae 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:01:39+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/curr` | 101096960000 | 101148155904 | -51195904 | 完成：HF archive/2026-10-24/worktrees/worktree_curr.MANIFEST.json @ 7d28d6efc21c60aa1c154ef11b018fbc247a4370；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:01:39+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/curr` | 101148160000 | 101089828864 | 58331136 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD f69ec11795704959f3deed8904f60624b7ac1dbb 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:01:53+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/curr39` | 101089828864 | 101106180096 | -16351232 | 完成：HF archive/2026-10-24/worktrees/worktree_curr39.MANIFEST.json @ 0ebd46e3d55c9bcc7ad46006859e5735b6b2ffaa；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:01:54+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/curr39` | 101106180096 | 101072064512 | 34115584 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD eec0d13a3378080e45492dff233c3ae40db9855a 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:01:54+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/dayearly` | 101072076800 | 101025452032 | 46624768 | 完成：C 类可再生产物；HEAD 1e7105efd818c219443c9eaa1fbcb9b80489ad75 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:01:54+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/deploy` | 101025452032 | 100911788032 | 113664000 | 完成：C 类可再生产物；HEAD 2a8dc97c8bb726457fb875a685438d5b8e955bcc 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:01:55+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/deploy2` | 100911665152 | 100797792256 | 113872896 | 完成：C 类可再生产物；HEAD f2ae755ab0fc2e4b617ebf56b5bcea491a89cf0d 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:04:15+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/fog` | 100797792256 | 100831576064 | -33783808 | 完成：HF archive/2026-10-24/worktrees/worktree_fog.MANIFEST.json @ 34c8df7f9cb8109634c7984ae9eae88dd9e672f1；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:04:15+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/fog` | 100831576064 | 100434640896 | 396935168 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 15b671aded5f5081bd9206d472953ba15c393518 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:05:08+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/forks` | 100434640896 | 100444631040 | -9990144 | 完成：HF archive/2026-10-24/worktrees/worktree_forks.MANIFEST.json @ fcae351a8a3feed9b52870b17adc65c51eea36af；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:05:09+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/forks` | 100444631040 | 99951419392 | 493211648 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 0e9738d78e76a2bbdf8fccfb4a811fa27d8ff4fc 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:05:44+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/identity` | 99951419392 | 99952304128 | -884736 | 完成：HF archive/2026-10-24/worktrees/worktree_identity.MANIFEST.json @ b860b84b75e56f829823dd84825cdcb15a00d244；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:05:44+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/identity` | 99952312320 | 99770105856 | 182206464 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD e8b2782e0ff8bbb0ad789f8c5f41f04f0ce4e2ef 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:06:13+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/keep` | 99770105856 | 99771478016 | -1372160 | 完成：HF archive/2026-10-24/worktrees/worktree_keep.MANIFEST.json @ 80ec6f9744e49fa3147efe2253ab2cd9a7d13847；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:06:13+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/keep` | 99771478016 | 99707736064 | 63741952 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 7d98a5fecce8eefe53d93e77548017809013c52b 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:06:31+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/l20diag` | 99707736064 | 99708608512 | -872448 | 完成：HF archive/2026-10-24/worktrees/worktree_l20diag.MANIFEST.json @ 045b451d95ead555609cc82ff16bf67dac89d52b；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:06:31+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/l20diag` | 99708608512 | 99661221888 | 47386624 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 86b82ac40a0f3da653eb1bae46f839c6b4a361bb 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:06:51+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/l20gap` | 99661221888 | 99661967360 | -745472 | 完成：HF archive/2026-10-24/worktrees/worktree_l20gap.MANIFEST.json @ a3ed1f1a14f944bd38a3df29175321b114e39541；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:06:51+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/l20gap` | 99661967360 | 99599290368 | 62676992 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 75f16e28de2df9aaf520562deb0471a8a963025b 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:07:07+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/macpool` | 99599290368 | 99599409152 | -118784 | 完成：HF archive/2026-10-24/worktrees/worktree_macpool.MANIFEST.json @ 7a43f56302a431cd8a5dc1f751f66ecc4ed63830；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:07:07+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/macpool` | 99599413248 | 99563163648 | 36249600 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD a4589db5f6b356acc9c84ae0f7845e87e7d84de1 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:07:25+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/n10p` | 99563163648 | 99563421696 | -258048 | 完成：HF archive/2026-10-24/worktrees/worktree_n10p.MANIFEST.json @ 1a437b2c2eeff7f8a4b3d5c69a695138e66c5b6a；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:07:25+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/n10p` | 99563421696 | 99512651776 | 50769920 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD d4caa1a86237b7a2ee3f2226135bd246a4c15557 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:07:47+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/night` | 99512655872 | 99513991168 | -1335296 | 完成：HF archive/2026-10-24/worktrees/worktree_night.MANIFEST.json @ 8f54f30fb196b50990d13c66c48f03d371940bca；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:07:47+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/night` | 99513991168 | 99462377472 | 51613696 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 81810b50b950817c74f756180ac6684696f88b0d 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:08:11+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/nightn10` | 99462377472 | 99463020544 | -643072 | 完成：HF archive/2026-10-24/worktrees/worktree_nightn10.MANIFEST.json @ b015518230e9f2bd3eb07b82cbd7632146b7812e；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:08:11+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/nightn10` | 99463020544 | 99413975040 | 49045504 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 615f35b6d41391f3e7e3e19febee7af064feeef3 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:08:11+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/oom` | 99413975040 | 99390226432 | 23748608 | 完成：C 类可再生产物；HEAD f4a3f17c07574c929637970a16e2b40523c7e3bf 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:08:32+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/pool` | 99390230528 | 99391778816 | -1548288 | 完成：HF archive/2026-10-24/worktrees/worktree_pool.MANIFEST.json @ 3a3208eef2238334ec373374e725129154fa737f；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:08:33+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/pool` | 99391778816 | 99342299136 | 49479680 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD c9d5066704975790b4486ae6f205a971d7dda15b 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:08:33+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/q101` | 99342299136 | 99295907840 | 46391296 | 完成：C 类可再生产物；HEAD 9ccf55373551102fa253b9873082e0d00f4c3a16 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:08:45+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/qcalloc` | 99295907840 | 99296182272 | -274432 | 完成：HF archive/2026-10-24/worktrees/worktree_qcalloc.MANIFEST.json @ 24001d3e6eead6c2cf7199d70ba211325a23d211；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:08:45+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/qcalloc` | 99296182272 | 99269693440 | 26488832 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 539cb2d8592e077994d7fe3c3c495d1f322b727d 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:10:34+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/qcbench` | 99269693440 | 99275304960 | -5611520 | 完成：HF archive/2026-10-24/worktrees/worktree_qcbench.MANIFEST.json @ efb777252643e418faab3390648b4c34ae8939db；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:10:34+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/qcbench` | 99275304960 | 98651410432 | 623894528 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 828c503e4c14f9aa9b4f084a8b9a2b23480e8f7a 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:10:34+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/qcpack` | 98651418624 | 98625101824 | 26316800 | 完成：C 类可再生产物；HEAD 25f1a529a640065dbaab547feb02061d86794b1a 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:11:39+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/qcpipe` | 98625126400 | 98626777088 | -1650688 | 完成：HF archive/2026-10-24/worktrees/worktree_qcpipe.MANIFEST.json @ ce6d6043bfb317af276f7f037c471557c6f0dc57；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:11:39+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/qcpipe` | 98626777088 | 98481057792 | 145719296 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD b48fac4fe60fcba8a9bbd21f953bf257177ea752 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:12:27+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/recipe` | 98481057792 | 98484637696 | -3579904 | 完成：HF archive/2026-10-24/worktrees/worktree_recipe.MANIFEST.json @ 5d0738c5eba4c2dc7b2ffacb13c8e0b5a29f8e9c；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:12:27+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/recipe` | 98484637696 | 98356260864 | 128376832 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 174b16b55109604705bf9284abe322b560b635d1 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:14:46+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/roof` | 98356260864 | 98363768832 | -7507968 | 完成：HF archive/2026-10-24/worktrees/worktree_roof.MANIFEST.json @ d671c02fc8e8f115e36909cd3f471eba3ec5ba88；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:14:46+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/roof` | 98363768832 | 97786949632 | 576819200 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 575d7da0bb76885f31941fb6282beb488615960f 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:14:47+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/rowpool` | 97786949632 | 97740832768 | 46116864 | 完成：C 类可再生产物；HEAD 4703cf11d230bb26ff9c437b23b8c9b81fe0e430 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:14:47+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/rowpool2` | 97740836864 | 97694830592 | 46006272 | 完成：C 类可再生产物；HEAD d6c8e5eb60b123e8c5fe880ee928685bc3c4920e 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:14:47+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/slimworker` | 97694830592 | 97648705536 | 46125056 | 完成：C 类可再生产物；HEAD 67c12e57f279d0e85522148d23a72658ae979f73 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:15:05+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/teacher` | 97648705536 | 97649913856 | -1208320 | 完成：HF archive/2026-10-24/worktrees/worktree_teacher.MANIFEST.json @ 602de53fd8994cf8921515e61a0a9f59e12695df；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:15:05+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/teacher` | 97649917952 | 97552912384 | 97005568 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD 2fdab81c90a62496f3cf5c2a70fed4d3c1d61ff7 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:15:21+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/termmerge` | 97553043456 | 97553547264 | -503808 | 完成：HF archive/2026-10-24/worktrees/worktree_termmerge.MANIFEST.json @ aee1892428775e211008f520c03b93c9afac8e52；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:15:21+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/termmerge` | 97553547264 | 97519259648 | 34287616 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD b5d29130aa683e6e6236a207e7c6c3f89e02884e 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:15:36+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/trainmem` | 97519259648 | 97520074752 | -815104 | 完成：HF archive/2026-10-24/worktrees/worktree_trainmem.MANIFEST.json @ 5f826d3eca0e1fffc1ae3bef028a33afa8397f82；全部部件强制下载、SHA256/字节数一致；源元数据稳定；上传/下载暂存已清除；详细清单在 HF MANIFEST |
| 2026-10-24T18:15:37+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/trainmem` | 97520074752 | 97487884288 | 32190464 | 完成：ignored/untracked 证据已在 HF 下载回验；HEAD ff597f6da9189ac090896deae48bc9c1b5765b0c 已被 origin/pvz-env 包含，branch -r --contains 有远端；git worktree remove 无 force；保留分支 |
| 2026-10-24T18:43:33+08:00 | 4-工作树证据-归档回验 | `~/PvZ-search/.claude/worktrees/vnet` | 93209493504 | 93069500416 | 139993088 | 完成：GNU sparse tar 本地逐文件 SHA256/字节数一致；HF 固定 revision e3b3b66bf77884e997574fad144e2e28ab532664 强制下载部件和 MANIFEST 回验一致；原始逻辑文件完整保留 |
| 2026-10-24T18:43:34+08:00 | 4-工作树退役 | `~/PvZ-search/.claude/worktrees/vnet` | 93069508608 | 91897368576 | 1172140032 | 完成：证据下载回验后 git worktree remove；原分支保留；HEAD e6c68f3ac18d4614f028e9430f80506dc105f562 已在 origin/pvz-env |
| 2026-10-24T18:57:01+08:00 | 4-prunable元数据 | `~/PvZ-Portable/.git/worktrees` | 91924418560 | 91923505152 | 913408 | 完成：git worktree prune --expire now；已验证全部原有分支仍在；仅清理不存在路径的过期元数据 |
| 2026-10-24T20:09:19+08:00 | 4-待传tar临时副本清理 | `~/task62-archive-stage/worktree_v5` | 92585099264 | 90036670464 | 2548428800 | 完成：HF私有存储额度不足；仅删除本任务生成、可由保留原证据重建的临时tar，未删除未回验的源目录；保留文件级MANIFEST与操作脚本便于恢复 |
| 2026-10-24T20:09:19+08:00 | 4-待传tar临时副本清理 | `~/task62-archive-stage/worktree_zones` | 90036670464 | 89768587264 | 268083200 | 完成：HF私有存储额度不足；仅删除本任务生成、可由保留原证据重建的临时tar，未删除未回验的源目录；保留文件级MANIFEST与操作脚本便于恢复 |

### 第二阶段进度检查点（2026-10-24T20:09:19+08:00，尚未完成）

- 步骤1–3完成：批准的 C 清单与追加项、6组B数据、旧frames整根及3592条离线入链均按先验后删执行；83,030个正式共享文件再次逐项核验 inode/字节数一致。
- 步骤4已完成 **33/35 棵工作树**，合计 **31组HF归档**全部下载回验；所有原有分支保留，4条prunable元数据已清。qcalloc 仅有控制器重建的100B `.keep` 占位目录，已不在Git工作树注册表，没有旧代码或实验数据。
- 剩余 **v5 / zones**：HF服务端明确返回 `Private storage limit reached`，分别约2.54GB / 267MB预备tar无法提交。无损稀疏格式对v5未减小，对vnet从约1.04GB降到422MB并已成功回验退役；未为腾配额删除任何已有HF备份，也未改写历史或公开私有数据。
- 待传源证据仍保留；仅清除了本任务生成的重复tar暂存。文件级MANIFEST及恢复操作脚本保留在 `~/.task62-cleanup-state/`，待可用私有空间恢复后从步骤4继续。待传表见[归档索引](task62-archive-index.md)。
- 原批准保留的12棵工作树全部存在；其他任务运行期间新建的 ['eaten', 'predeploy11', 'replace'] 不属本次授权，未处理。生产四模型/deploy3构建SHA256、主检出tracked diff指纹均与保护快照一致；task58和pipeline数据保留。
- **步骤5 fstrim 尚未执行**，未关闭WSL。最新主控指示停止上传、列出待传清单后等待；是否扩容HF或清理远端旧内容由用户决定，不再请求或自动执行。
- 当前df已用 **89768595456 B（83.604 GiB）**；相对本阶段首次删除前 168604545024 B，净减少 **73.422 GiB**，含生产并发增长及映射释放，不能以删除目录du简单相加。

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G   84G  873G   9% /
```

### 主控收口：停止上传并等待（2026-10-24T20:13:24+08:00）

按最新主控通知，HF `BadRequestError: Private storage limit reached` 为私有存储额度限制；已停止全部上传重试，当前没有本任务上传进程。主控告知仓库54.8 GB、当天archive占31.8 GB，此处作为通知记录，不作为重新上传依据。是否升级HF或清理远端旧内容由用户决定，本任务未执行这些操作。

不需要上传的获批C类清理，以及无ignored证据或证据已经下载回验的工作树退役均已完成。35棵批准退役工作树中已完成33棵；其余2棵因证据未回验而保留。31组已上传归档均在固定revision强制下载核对SHA256/字节数后才删除源数据。

**待上传未删**（下表为整棵源工作树当前 `du -s -B1` 实占，包含代码和证据；归档证据大小另见索引）：

| 保留的本地路径 | 实占 B | 实占 GiB | 状态 |
|---|---:|---:|---|
| `/home/newbiexvwu/PvZ-search/.claude/worktrees/v5` | 2,570,493,952 | 2.394 | 未上传成功、未下载回验；源证据与工作树均保留 |
| `/home/newbiexvwu/PvZ-search/.claude/worktrees/zones` | 293,486,592 | 0.273 | 未上传成功、未下载回验；源证据与工作树均保留 |

文件级MANIFEST保留于 `~/.task62-cleanup-state/worktree_v5.MANIFEST.json`、`worktree_zones.MANIFEST.json`。先前清除的仅是本任务生成的可重建tar临时副本，未回验的原始证据一律未删。

当前df（生产仍在运行，数值会变化）：

```text
Filesystem         1B-blocks        Used    Available Use% Mounted on
/dev/sdd       1081101176832 89895436288 936213385216   9% /

Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G   84G  872G   9% /
```

按主控要求在此暂停，等待主控；fstrim尚未执行，WSL未关闭。后续不自动恢复上传、不处理剩余两棵工作树。

### 用户20:20新指示执行结果（不再归档，直接清理）

**本节为当前结论，取代前面的待上传/等待配额状态。** 用户明确不要囤积，原先B类和旧帧等不再上传；HF旧归档已由主控删除并压缩历史，原归档索引现标明“已按用户要求删除，不再保留”。本轮没有HF上传操作。

保留qc流水线、frames_v2、dw4i_s1、value_net v4/v5、hard/value_branch1、round_*学生，以及deploy3/cq/dw4/conveyor生产链、memprobe/replace/fogvis等在跑任务；dw5保留工作树与会话，仅清理未在用的大文件及旧产物。

| 本机时间 | 步骤 | 明确路径 | df已用前 B | df已用后 B | 净减少 B | 状态与依据 |
|---|---|---|---:|---:|---:|---|
| 2026-10-24T20:49:37+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/v5` | 90014523392 | 87443828736 | 2570694656 | 完成：删除前实占2570493952 B；HEAD ced8ef48de953a6fe2cb8cd2841a22a09604943d 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:38+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/zones` | 87443832832 | 87150145536 | 293687296 | 完成：删除前实占293486592 B；HEAD 960737888c371d5a746369b8eb396f3ab4a9a3b4 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:38+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/cutoff` | 87150145536 | 87102750720 | 47394816 | 完成：删除前实占47218688 B；HEAD 69a8f0ed89f52a2a73a2ec2b8c281ae502969338 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:38+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/llmcand` | 87102750720 | 86935162880 | 167587840 | 完成：删除前实占167399424 B；HEAD cd4da073dc7d737632a308da5b215b60ba4d1325 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:39+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/perf` | 86935371776 | 86110072832 | 825298944 | 完成：删除前实占825106432 B；HEAD 84c02629d186f922f028af15fe72b007cecbea35 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:39+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/shovel` | 86110072832 | 86014128128 | 95944704 | 完成：删除前实占95694848 B；HEAD c7af988ac8839e66bf16ee6dcb5e2a16f3dda58f 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:39+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/vcpp` | 86014156800 | 85777301504 | 236855296 | 完成：删除前实占236691456 B；HEAD b73fd8cf979f0c240edcddfd7ad42b224f1c18b9 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:39+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/eaten` | 85777301504 | 85651533824 | 125767680 | 完成：删除前实占125562880 B；HEAD 3e59e53b5e1aa5b1160657933d4c8e11fbb950de 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:49:39+08:00 | 追加工作树退役 | `~/PvZ-search/.claude/worktrees/predeploy11` | 85651533824 | 85621694464 | 29839360 | 完成：删除前实占29638656 B；HEAD 4f4d217b1bcdd8244f8e9c45709cc0695eb07c89 已在GitHub远端引用中；无在用数据/生产入链，闲置agent仅cwd不算运行任务；git worktree remove无force，保留分支；ignored证据按20:20新指示直接删除，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/.npm/_cacache` | 85742411776 | 85739069440 | 3342336 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/.nv/ComputeCache` | 85739069440 | 85346463744 | 392605696 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/.cache/huggingface` | 85346467840 | 85284483072 | 61984768 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/.cache/pip` | 85284483072 | 85282230272 | 2252800 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/.cache/mesa_shader_cache` | 85282230272 | 85280038912 | 2191360 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/cleanup_20261015` | 85280038912 | 85221339136 | 58699776 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/PvZ-search/build` | 85221339136 | 85196787712 | 24551424 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/PvZ-Portable/build` | 85196787712 | 85172224000 | 24563712 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧缓存/构建/下载副本清理 | `~/PvZ-search/.claude/worktrees/dw5/build` | 85172224000 | 85148274688 | 23949312 | 完成：20:20批准直接清理；删除前确认无可见进程fd/maps/参数引用；构建未被Git跟踪，不上传 |
| 2026-10-24T20:51:18+08:00 | 旧解释器缓存清理 | `~/PvZ-search/.claude/worktrees/cleanup/.pytest_cache` | 85148274688 | 85148241920 | 32768 | 完成：可再生pyc/pytest缓存，无在用引用 |
| 2026-10-24T20:51:18+08:00 | 旧解释器缓存清理 | `~/PvZ-search/.claude/worktrees/cleanup/scripts/__pycache__` | 85148241920 | 85148073984 | 167936 | 完成：可再生pyc/pytest缓存，无在用引用 |
| 2026-10-24T20:51:18+08:00 | 旧解释器缓存清理 | `~/PvZ-search/.claude/worktrees/cleanup/python/__pycache__` | 85148073984 | 85147844608 | 229376 | 完成：可再生pyc/pytest缓存，无在用引用 |
| 2026-10-24T20:51:18+08:00 | 旧解释器缓存清理 | `~/PvZ-search/.claude/worktrees/dw5/scripts/__pycache__` | 85147844608 | 85147017216 | 827392 | 完成：可再生pyc/pytest缓存，无在用引用 |
| 2026-10-24T20:51:18+08:00 | 旧解释器缓存清理 | `~/PvZ-search/.claude/worktrees/dw5/python/__pycache__` | 85147017216 | 85146677248 | 339968 | 完成：可再生pyc/pytest缓存，无在用引用 |
| 2026-10-24T20:51:18+08:00 | APT下载缓存清理 | `/var/cache/apt/archives` | 85146677248 | 81499930624 | 3646746624 | 完成：删除已安装包的下载副本；未卸载软件、未改运行环境 |
| 2026-10-24T20:51:19+08:00 | 旧systemd日志清理 | `/var/log/journal` | 81499930624 | 80440045568 | 1059885056 | 完成：仅journalctl vacuum已归档日志；当前打开的活动journal保留；Vacuuming done, freed 1010.7M of archived journals from /var/log/journal/e292389aabe442bf942686b428795804. |
| 2026-10-24T20:52:08+08:00 | dw5 artifacts大文件清理 | `~/PvZ-search/.claude/worktrees/dw5/artifacts` | 80402137088 | 78132740096 | 2269396992 | 完成：逐文件删除199个≥1 MB文件，逻辑大小合计2478341055 B；无fd/maps/训练或打包进程引用；小报告、training_frames链接视图、unified_v2.json、工作树及qoder会话保留；未上传 |
| 2026-10-24T20:52:09+08:00 | 未打开的大日志清理 | `~/pvz-logs`；`~/.qoder/logs`；`~/.codebuddy/logs`；`~/.codex/log` | 78132744192 | 77548699648 | 584044544 | 完成：按明确文件清单删除213个≥1 MB旧.log/.out文件，逻辑大小合计583491589 B；4个被fd打开的日志保留，pipeline及会话JSONL/SQLite未动；完整删除清单见本节明细 |
| 2026-10-24T20:53:49+08:00 | 旧导入暂存清理 | `~/pvz-v2fill` | 77553176576 | 77474250752 | 78925824 | 完成：历史v2补帧导入输出，无进程引用或符号链接入链；正式frames_v2不动，按20:20直接清理，不上传 |
| 2026-10-24T20:53:49+08:00 | 旧安装下载副本清理 | `~/debs` | 77474250752 | 77464092672 | 10158080 | 完成：离线deb下载副本，不卸载已安装软件，无在用引用 |
| 2026-10-24T20:54:06+08:00 | 主检出未在用的旧实验产物清理 | `~/PvZ-search/artifacts` | 77464096768 | 76934660096 | 529436672 | 完成：按明确子路径删除1340个ignored旧实验目录/文件，删除前du合计530776064 B；保护frames_v2和dw4i_s1/value_net v4 v5/hard value_branch1，主检出tracked变动不动；无生产链接入链和运行引用；详单见task62-deleted-paths.tsv；保留额外项3 |
| 2026-10-24T20:54:06+08:00 | 生产源工作树的旧构建清理 | `~/PvZ-search/.claude/worktrees/dw4/build` | 76934774784 | 76910825472 | 23949312 | 完成：删除前实占23949312 B；仅清理未被使用的build输出，不触碰artifacts链接链、源代码或工作树；当前运行引擎为deploy3 |
| 2026-10-24T20:54:06+08:00 | 生产源工作树的旧构建清理 | `~/PvZ-search/.claude/worktrees/conveyor/build` | 76910825472 | 76886900736 | 23924736 | 完成：删除前实占23924736 B；仅清理未被使用的build输出，不触碰artifacts链接链、源代码或工作树；当前运行引擎为deploy3 |
| 2026-10-24T20:57:16+08:00 | 本任务归档暂存/清单清理 | `~/task62-archive-stage` | 76916670464 | 76916666368 | 4096 | 完成：所有旧HF归档已由主控删除；不再上传或恢复，清除本任务临时归档、文件级清单和旧上传脚本；结果与删除路径保留在Git文档 |
| 2026-10-24T20:57:16+08:00 | 本任务归档暂存/清单清理 | `~/.task62-cleanup-state` | 76916666368 | 76806270976 | 110395392 | 完成：所有旧HF归档已由主控删除；不再上传或恢复，清除本任务临时归档、文件级清单和旧上传脚本；结果与删除路径保留在Git文档 |
| 2026-10-24T20:57:16+08:00 | 本任务临时盘点副本清理 | `/tmp/task62-conveyor-broken.txt`；`/tmp/task62-production-broken-links.txt`；`/tmp/task62-production-target-roots.json`；`/tmp/task62-cache-log-du.txt`；`/tmp/task62-system-du.txt`；`/tmp/task62-dw5-big.json`；`/tmp/task62-artifacts-du.txt`；`/tmp/task62-wt-plan.json`；`/tmp/task62-live.json`；`/tmp/task62-current-du.txt`；`/tmp/task62-finalize.py`；`/tmp/task62-write-report.py`；`/tmp/task62-relations.json`；`/tmp/task62-link-groups.json`；`/tmp/task62-relations.py`；`/tmp/task62-files.json`；`/tmp/task62-code-refs.json`；`/tmp/task62-worktrees.json`；`/tmp/task62-remote.json`；`/tmp/task62-git.py`；`/tmp/task62-audit.py`；`/tmp/task62-state-paths.json`；`/tmp/task62-proc.json`；`/tmp/task62-du-inventory.json` | 76806270976 | 76806270976 | 0 | 完成：删除24个临时盘点/旧操作脚本，逻辑大小合计370686987 B；必要执行结果已写入任务书，避免留下重复清单 |
| 2026-10-24T21:01:02+08:00 | 最终fstrim | `/` | 76807639040 | 76912017408 | -104378368 | 完成：sudo -n fstrim -v / 成功：78.4 GiB (84223389696 bytes) trimmed；前df取自20:57:42采样，生产并发会影响后df；trim字节不等于本轮删除字节 |



#### 最终交付（2026-10-24T21:01:02+08:00）

- **已完成本轮可清理项，没有待上传项，没有进行任何新上传。** HF旧归档均以主控通知为准，已按用户要求删除、不再保留；原恢复命令已从[归档索引](task62-archive-index.md)撤下。
- 本轮追加退役 **9棵**：`v5`、`zones`、`cutoff`、`llmcand`、`perf`、`shovel`、`vcpp`、`eaten`、`predeploy11`。连同此前33棵累计 **42棵**，原35棵批准清单现全部完成。全部使用 `git worktree remove`，未删分支；删除前逐棵确认HEAD在GitHub远端引用中。`shovel` 原远端引用不存在，已先推送 `origin/shovel`（HEAD `c7af988ac8839e66bf16ee6dcb5e2a16f3dda58f`）再退役。
- 原“待上传未删”的v5/zones源工作树已直接删除；六组旧B目录与旧 `artifacts/student/frames` 也均确认不存在。旧归档副本、暂存、文件级大清单及旧上传脚本已清除。
- dw5保留工作树和qoder会话，删除artifacts内 **199个≥1 MB文件**（逻辑合计2,478,341,055 B；共享硬链接使实占释放量更小），最终扫描未剩≥1 MB普通文件；小报告、`training_frames`链接视图和`unified_v2.json`保留。
- 删除 **213个未打开的大日志**、未在用的旧构建/解释器缓存、CUDA/npm/HF/pip下载缓存、APT下载副本和已归档systemd日志；另清理主检出1340个无在用引用的ignored旧实验子路径。逐文件/子路径及删除前大小见[删除明细](task62-deleted-paths.tsv)，每批df见上表；未将任何数据、模型或日志内容提交到Git。
- 保护核验通过：生产4个模型与deploy3引擎SHA256/字节数未变；**83,030个正式共享文件**的device/inode/字节数逐项一致；dw4与conveyor训练链接检查均为0断链；主检出tracked diff指纹未变，所有原有本地分支保留。
- 保留工作树：`cleanup`、`cq`、`deploy3`、`dw4`、`conveyor`、`dw5`、`memprobe`、`replace`；本轮期间新建的 **redeploy** 已有3处tracked修改、新脚本和task70构建/测试产物，属于在跑任务，保留。两个主检出保留。没有触碰fogvis等其他在跑任务。
- 主检出 `artifacts/hard/c6roofl41r7m100_48` 当时有屋顶评估进程打开`games.jsonl`，保留；`artifacts/adventure2_level7`、`artifacts/multiterrain`含tracked结论，保留。打开的 `qc.out` 及3个qoder日志保留；qc流水线、round_*学生、frames_v2、生产模型、生产链接链、task58和replace均未清理。
- `sudo -n fstrim -v /` 成功返回 **`/: 78.4 GiB (84223389696 bytes) trimmed`**。本轮df已用净减少 **13,095,514,112 B（12.196 GiB）**；相对第二阶段首次删除前168,604,545,024 B，净减少 **85.395 GiB**。这些是采样间df净变化，含并发生产增长；trim范围也不等于删除量。未关闭WSL、未操作tmux会话。

当前df（2026-10-24T21:01:02+08:00，生产仍会继续改变数值）：

```text
Filesystem         1B-blocks        Used    Available Use% Mounted on
/dev/sdd       1081101176832 76912017408 949196804096   8% /

Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G   72G  885G   8% /
```

## 事故：221ae160误删在用stage4（恢复记录，2026-10-25补交）

### 责任、故障与当前状态

`221ae160` 把 `/home/newbiexvwu/PvZ-search/artifacts/stage4`（删除前22,876,160 B）列为 `main_unused_artifact` 整目录删除，这是执行者的错误。第一阶段已经识别state对stage4的引用，但后续实现只用白名单、进程fd/参数快照与符号链接入链判定通用ignored子目录，没有把 `cq/scripts/make_multiplier_tasks.py:source_paths()`、state和配方中的间接读取合并成保护集合。**“当时没有打开fd”不能证明路径不在用。** 前面“保护核验通过/清理完成”的结论不能作为生产依赖安全的证明。

2026-10-24 **20:58:37**，qc round 6的 `c6n18n10qm100` 读取 `night_panel_n10q.json` 报 `FileNotFoundError`，控制器退出。随后按主控指令从Git生成器恢复六份源清单并核验。主控独立复核后，已于 **21:35** 用14 worker重启qc；不是执行者启动。此后不再操作qc，不改deploy3/redeploy/replace/memprobe，也不再删除任何东西。journal及其余引用审计在本节后续追加。

### 恢复依据、命令与SHA256

恢复脚本来自cleanup的Git版本（恢复开始时HEAD为`221ae160`）；参数依据 task17/21/22/23/24/25/28/31 的任务书和生成器固定种子段。白天/夜间/泳池使用`--combined`，然后依次生成d7q、n10p、n10q，恢复到原主检出路径。d7q/n10p的整组检查使用cq现有引擎的独立副本：`~/pvz-logs/restore/task62-recovery/pvz-portable`，SHA256 `b642614f25789dd2ec503a56f40c7412157c8742071cd10b930cb83593be9e67`。没有修改或使用deploy3构建来重新生成。

Mac副本只作交叉核对：恢复的白天26项逐任务与Mac一致；恢复的夜间54项中，Mac幸存的36项逐任务一致，另外18项由task25记录的n10和L20原版任务补齐。没有直接用旧Mac清单覆盖台式机定义。

以下是实际执行命令（所有returncode=0）。最后一条是当时qc仍停机时的隔离dry-run：state使用副本，日志和out指向恢复目录，未启动采集。

```bash
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup/scripts/make_day_panel.py --combined --out /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup/scripts/make_night_panel.py --combined --out /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup/scripts/make_pool_panel.py --combined --out /home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup/scripts/make_day_d7q.py --manifest /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json --out /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel_d7q.json --executable /home/newbiexvwu/pvz-logs/restore/task62-recovery/pvz-portable --scan /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json /home/newbiexvwu/pvz-logs/restore/mac_day_panel_identity.json
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup/scripts/make_night_n10p.py --manifest /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json --out /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10p.json --executable /home/newbiexvwu/pvz-logs/restore/task62-recovery/pvz-portable --scan /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel_d7q.json /home/newbiexvwu/pvz-logs/restore/mac_day_panel_identity.json
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cleanup/scripts/make_night_n10q.py --source /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10p.json --out /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10q.json --scan /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json /home/newbiexvwu/PvZ-search/artifacts/stage4/day_panel_d7q.json /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10p.json /home/newbiexvwu/pvz-logs/restore/mac_day_panel_identity.json
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/scripts/curriculum_queue.py --dry-run --state /home/newbiexvwu/pvz-logs/restore/task62-recovery/dryrun-state.json --log /home/newbiexvwu/pvz-logs/restore/task62-recovery/dryrun-controller.log --out /home/newbiexvwu/pvz-logs/restore/task62-recovery/dryrun-output --executable /home/newbiexvwu/pvz-logs/restore/task62-recovery/pvz-portable --manifest /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/stage4/multiplier_panel.json --recipe /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/recipes/dw4i2.json --round-games 768 --games-per-group 48
```

六份文件均位于 `/home/newbiexvwu/PvZ-search/artifacts/stage4/`：

| 文件 | 任务数 | SHA256 |
|---|---:|---|
| `day_panel.json` | 26 | `861826360161c2762c66093c93b42c992b2395db8f36f2bdd199c1f14862c8dd` |
| `night_panel.json` | 54 | `f77536507e22544196ea048751629d8394bd7959f6dd44567bf0efde8e2a138f` |
| `pool_panel.json` | 35 | `10ebc1b404a15aa96e44e705747c99fdb6b3aaf90a9d930ac9f51f61cda1b0fb` |
| `day_panel_d7q.json` | 16 | `ea0f8c0348cafd6a897bc0defde239e4a15cf702126779edc76d3d2f236cfc8f` |
| `night_panel_n10p.json` | 24 | `af71030371c5cf7b0479139fe16d02909632ee6508887cbb8312cba175c35d87` |
| `night_panel_n10q.json` | 24 | `ab856b0e1213d64fd526574e4523ece8a68ee5fb032d3d3c9509559d9a662fae` |

### 验证结果与证据边界

1. **倍率逐字段核对：420/420通过，0不一致。** 每个变体去掉`_mXXX`后查对应源任务；排除`task_id`、`seeds`、`zombie_count_multiplier`后，字段集合和值均完全相同，包含完整`deck_perturb`和引擎检查表。
2. **state覆盖：**执行者核对47个基础任务（含offsets、cells、retired_cells和历史log引用）全部存在。主控独立核对state全部 **92个offsets**，对应任务均在完整面板中且未越界。
3. **历史种子：43格，0不一致。** 41格使用幸存的原始帧`meta.json`配合`curriculum.log`、`queue98.log`、`queue103.log`或state历史组记录，逐个核对选定组的实际种子与`seeds[offset+i]`一致。旧metadata没有task_id时，用独立采集日志里的group label和task_id绑定，未给原metadata补造字段。同名label经历多次采集时，按各次日志的offset/games区间核对，不误把合并归档当成一组。
   - 另2格为已退役的 `night_l14_train_n10p`、`night_l17_train_n10p` 历史评估，未保存到正式帧。使用Git中 `MEMORY/docs/night-panel.md` 的实际逐局表（各16条seed）与 `task28-night-coverage.md` 记录的独立offset=0核对，结果一致。**没有恢复它们的原始games.jsonl，也不声称原始档案仍在。**
4. **完整面板dry-run通过。** 实际`cq/scripts/curriculum_queue.py --dry-run`返回0，生成16组/768局计划；验证期间真实state与cq倍率清单SHA256保持不变。主控也独立确认dry-run正常。

完整六份哈希、精确argv及43格逐组证据索引见 [结构化恢复记录](task62-stage4-recovery.json)。主控据上述结果于21:35重启qc；执行者不再负责或操作qc运行。

### 今后删除判定

凡 `source_paths`、recipes、state或生产脚本默认路径出现的路径，均视为在用；包括惰性读取、输出父目录及其仍被引用的子路径。先解析这些权威引用及传递依赖，再谈处置；不得用“ignored”“暂时未打开fd”或“没有符号链接”替代引用核对。主控当前明确禁止继续删除任何东西。

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
