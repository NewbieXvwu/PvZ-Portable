# 台式机（WSL）连接与运维

连接与日常操作见 `DESKTOP_HANDOFF.md`。速记：`export PVZ_DESKTOP_HOST=newbiexvwu@127.0.0.1
PVZ_DESKTOP_PORT=22222`，用 `scripts/win_ssh.py ps|wsl`；或 `ssh -p 22222 ... "wsl -d Ubuntu -e bash -s" < 脚本`
（**取大文件用 base64 走 stdin**）。WSL `Ubuntu`，venv `~/.venvs/ml`（torch 2.14.0+cu132，RTX 5080 16 GB，
`nproc=20`）；**venv 参数要加引号**。4 个 worktree（`~/PvZ-Portable` 主 + `-curriculum`/`-observation`/
`-reset-disposal`），**队列从主 worktree 跑**。远端 wrapper 带 `set -euo pipefail` → **无匹配的 grep 一律加
`|| true`**，否则吞掉后面所有输出。

**★ 踩过的坑（这些不在 `DESKTOP_HANDOFF.md` 里）：**

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
