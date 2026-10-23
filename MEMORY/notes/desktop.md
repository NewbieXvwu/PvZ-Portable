# 台式机（WSL）连接与运维

速记：`ssh -p 22222 newbiexvwu@127.0.0.1 wsl -d Ubuntu -e bash -c "'cat > ~/pvz-logs/x.sh'" < 本地脚本`
先把脚本传上去，再 `... wsl -d Ubuntu -e bash ~/pvz-logs/x.sh`（多层引号直接写命令很容易炸）。WSL `Ubuntu`，venv `~/.venvs/ml`（torch 2.14.0+cu132，RTX 5080 16 GB，
`nproc=20`）；**venv 参数要加引号**。4 个 worktree（`~/PvZ-Portable` 主 + `-curriculum`/`-observation`/
`-reset-disposal`），**队列从主 worktree 跑**。远端 wrapper 带 `set -euo pipefail` → **无匹配的 grep 一律加
`|| true`**，否则吞掉后面所有输出。

**★ 踩过的坑：**

- **WSL 空闲就会关机，杀掉 tmux 里的长跑**（PID 1 换新 = 重启过；tmux server 一起死，**不留 EXIT 行**）
  → **没有 keeper 的长跑一定活不过一个 SSH 空档**。`PvZWSLKeeper`（Windows 计划任务，每 5 分钟）动作 =
  `wsl.exe -d Ubuntu -e /home/newbiexvwu/.pvz-keeper.sh`。**别动 `~/.pvz-keeper.sh`、`~/.pvz-chain.sh`、`PvZWSLKeeper`。**
- **`~/.pvz-chain.sh` 的三个关键设计**：① 用「进程存活」而非「日志新鲜度」判断 codex 死活（否则**误杀健康的运行**）；
  ② **「忙守卫」**：`n5_queue.sh` 在跑 / 有 `train_pvz_ppo` 进程 / GPU 上有进程 → 推迟接续，否则两个训练抢 GPU →
  `CUDA error: busy`；③ case A 要检查 `NEXT` 存在。改完必做 `bash -n` + 沙箱 8 分支测试。
- **GPU 独占**：起 CUDA 前查 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`；**`pkill` 后等 5 秒**。
- **隧道 = UURemote（UU远程）的端口转发，不是 ssh 隧道**。**Mac 侧代理抖动会让它拆掉端口映射且不自动重建**
  → 所有远端操作**重试 ≥4 次（间隔 6~8 秒）**。**用户指示：不要再开定时任务**（会跑空）。
- **PowerShell 经 SSH 调用**：多层引号必炸 → `powershell.exe -NoProfile -EncodedCommand <base64(UTF-16LE)>`；
  **反引号会被远端 bash 吃掉**；**`Register-ScheduledTask` 曾静默失败，`schtasks` 才可靠**；
  **`AllowStartIfOnBatteries` 不是真实属性名**（真的是 `DisallowStartIfOnBatteries`）。
- **★★ 台式机 C 盘满 → WSL 崩溃（10-15 凌晨实测）**：C 盘剩 0 字节时，WSL 的 VHDX（`C:\Users\NewbieXvwu\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx`，
  当时 124 GB）无法增长 → WSL 虚拟机重启（02:35，Windows 本身没重启）→ 之后 ext4 报 `Errno 5 I/O error`、`uptime` 段错误、`wc` 总线错误。
  **WSL 里 `df` 显示还有 840 GB 可用是假象，那是虚拟盘的容量；真正要看的是 `df -h /mnt/c`。**
  处置：删掉旧 WSL 实例遗留在 `%TEMP%\<GUID>\swap.vhdx` 的交换文件（8 GB；当前实例的 GUID 可以从 Hyper-V-VmSwitch 事件里对出来）→ `wsl --shutdown` → ext4 自动 recovery 后恢复正常。
  **WSL 重启会清空 `/tmp`**（这次 A1 的队列脚本就丢了）→ 长跑的脚本和日志放 `~/pvz-logs/`。
  已加看门狗 `~/pvz-logs/cdrive_guard.sh`：C 盘剩余低于 3 GB 就停掉训练和搜索。
  **C 盘本身被用户文件占满（约 820 GB）是根因，需要用户处理**（例如 `wsl --manage Ubuntu --move E:\WSL\Ubuntu`，E 盘当时还有 142 GB 可用）。
- **★★ `git clean -X` 加路径参数会按「字符串前缀」匹配（10-15 实测，误删了整个 `artifacts/research/`）**：
  `git clean -fdX artifacts/research_evidence` 会把 `artifacts/research/` 也删掉，因为后者是前者的字符串前缀，
  再加上 `artifacts/` 下默认全部忽略。**删东西只用对明确路径的 `rm`，删之前一律先 dry-run（`-n` / 列出清单）。**
  损失：A1（bc_seed0_guard002_1m_v1）的续跑状态（结论 json 在 HF 存档里，.pt 没了，A1 已放弃）；
  v3 的 4 个检查点已从 HF 恢复到 `artifacts/research/bc_finetune_level7_v3/output/runs/run_1/`；
  `gru_bc_converted.pt` 没有恢复（需要时从 `~/PvZAgent-gru-bc-level7-v1/` 用 `scripts/convert_bc_checkpoint.py` 转换）。
- **10-15 C 盘清理**：WSL 内从 117 GB 降到 31 GB；结论层（25.8 万个文件，2.45 GB）存档在 HF
  `archive/desktop_cleanup_20261015/conclusions.tar.gz`；旧 worktree 的代码都在 GitHub（`archive/worktrees/*`）；
  diskpart 压缩后 VHDX 从 126.4 GB 降到 34.2 GB，**C 盘可用 5.4 GB → 97.6 GB**。
  压缩前要先停用 PvZWSLKeeper 计划任务（`schtasks /Change /TN PvZWSLKeeper /DISABLE`，结束后再 `/ENABLE`），
  否则它每 5 分钟会把 WSL 拉起来，和 diskpart 冲突。

## WSL 的“空闲关机”（2026-10-15 下午）
- 15:13:55 和 17:28:11 两次 WSL 停机，journal 里都是 systemd **正常 poweroff**，不是崩溃。当时都有进程把 WSL 内存吃满（17:23 还有一次 OOM kill，16.6 GB 的价值网络训练）。推测内存打满时 `wsl.exe` 会话（守护任务 PvZWSLKeeper、ssh 进来的命令）连不上（报 `Wsl/Service/0x8007274c`），WSL 认为没人在用，就自动关机；setsid 起的后台任务不算“在用”。
- 已在 `C:\Users\newbiexvwu\.wslconfig` 加 `[wsl2] vmIdleTimeout=-1` 和 `[general] instanceIdleTimeout=-1`（备份为 `.wslconfig.bak_20261015`）。
- 教训：WSL 只有 24 GB，**同一时间只跑一个吃内存的任务**；队列脚本一律串行（`~/pvz-logs/queue3.sh`）。

## 内存打满时的防护（2026-10-23）

起因：10-22 21:18 内存打满后 `wsl.exe` 会话（`0x8007274c`）连不上，业务进程还在跑；
内核先杀掉了 WSL 的接入进程而不是吃内存的业务进程。上限已提到 28 GB（`.wslconfig`），
另加 `[experimental] autoMemoryReclaim=gradual`。防护分三层，**任何一层都不碰 init / sshd / tmux**。

### 1. 进程分（`/proc/self/oom_score_adj`，内核先杀分数最高的）

| 角色 | 分数 | 写在哪里 |
|---|---|---|
| 采集 worker（`search_play` 池进程） | **1000** | `search_play._init_worker` |
| 采集引擎子进程（`pvz-portable`） | **1000** | `pvz_env.PvZEnv._start` 的 `preexec_fn` |
| 课程控制器（`curriculum_queue`） | **500** | `main()` 开头 |
| 训练（`train_rollout_policy` / `train_value_net` / `train_value_branches`） | **300** | 各自 `main()` 开头 |
| WSL 接入 / init / sshd / tmux / 代理 | 0 或负（系统默认） | 不改 |

共享实现在 `python/pvz_common.py` 的 `set_oom_score_adj`：只写自己的分数，不需要 root；
没有 `/proc` 的平台（macOS）静默跳过。单测 `python/test_oom_guard.py`（Linux 子进程实测，Mac 跳过）。
后果：内存再紧时先死采集，其次训练；控制器比训练晚死，接入通道默认不会被选中。

### 2. 用户态守护 earlyoom（比内核 OOM 早一步，留出回收时间）

2026-10-22 装：系统 apt 源（bfsu）对 pool 返回 403，改从清华镜像下载同版本 deb：
`earlyoom 1.9.0-1`，`sudo dpkg -i` 安装；WSL 的 systemd 已开，服务 `enabled` + `active`。

`/etc/default/earlyoom`：
```
EARLYOOM_ARGS="-m 5 -s 50 -r 300 --prefer '(pvz-portable|search_play|python3)' --avoid '(init|sshd|tmux|codex|codebuddy|qodercli|node)'"
```
- `-m 5`：MemAvailable ≤ 5% 时发 SIGTERM（≤2.5% 时 SIGKILL）；`-s 50`：swap 空闲 ≤50% 才动手。
- earlyoom 只匹配 `/proc/pid/comm`（15 字节），`search_play` 的 comm 是 `python3`，所以 prefer 里
  补了 `python3`；worker 自身分数 1000 仍然是第一顺位，prefer 只是双保险。
- `--avoid` 保住 init / sshd / tmux / 各代理（codex、codebuddy、qodercli、node）。
- `-r 300` 只是把状态行改成 5 分钟一条，避免刷 journal。

查状态：`systemctl status earlyoom`、`journalctl -u earlyoom -n 50`。重启 WSL 后由 systemd 自动拉起，
无需 keeper；**不要**再用 tmux 起第二份，否则两个守护会抢着杀。

### 3. 采集并发按可用内存自动缩放（`scripts/curriculum_queue.py`）

- `--workers` 默认 16，语义是**上限**（不是实际进程数）。
- 每开一组前读 `/proc/meminfo` 的 MemAvailable：
  `workers = clamp(⌊(MemAvailable − 6 GiB) / 单 worker RSS⌋, 1, --workers)`。
  6 GiB 留给 WSL 接入、控制器和训练；训练在跑时自然降到个位数，训练结束自动加回去。
- 单 worker RSS 在控制器启动时实测：用 `--workers 1 --seeds 1` 跑一局真实任务（临时 out / 帧目录，
  不动状态偏移、不进档案），每 0.2 s 采样 `search_play` 的全部后代（Python worker + 引擎）取峰值；
  实测失败退回 1 GiB；没有 `/proc`（Mac）不缩放。每组日志先写
  `workers=k/16 (MemAvailable …, per-worker …)`，实际用了几个一眼可见。
- 压力测试脚本 `scripts/oom_stress_test.py`：默认拒绝，必须 `--i-know`；即使给了也再拒绝
  训练在跑 / 采集在跑 / 可用 < 12 GiB。它起 2 个分数 1000 的假 worker 把 MemAvailable 压到 5%，
  量 `tmux new-session -d` 和 `bash -c true` 的延迟（<2 s 为通过），并验证死的是假 worker、
  init/sshd/tmux/earlyoom 都活着。**待主控批准后运行，结果补在这一节。**

