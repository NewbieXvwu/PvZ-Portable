# task34（主控 2026-10-23 00:40 下发给台式机 Qoder，tmux qoder-oom）：内存打满时不再拖死 WSL

先读 `AGENTS.md`、`MEMORY/notes/desktop.md`（“WSL 的空闲关机”一节）、`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 22:00–23:55 的几条。

## 背景（事故经过，必须读懂再动手）
10-22 21:18 起台式机 WSL（当时上限 24 GB）被内存挤满：dw5i 训练（`train_rollout_policy.py`，常驻十几 GB，另有 2.4 GB 内存映射）+ 课程采集 8 个 worker（每个 Python `search_play` + 引擎 `pvz-portable`）+ 价值网络补采 6 个游戏进程。
结果：`wsl.exe` 再也连不上（`Wsl/Service/0x8007274c`，连 `wsl --system` 也不行），但里面的进程还在跑——说明内核在内存不足时杀掉或饿死了 WSL 自己的接入进程（init / 会话中继 / plan9），而吃内存的业务进程活着。只能 `wsl --shutdown`，训练白跑 3 小时。
现在上限 28 GB（`.wslconfig` 已改，另加了 `[experimental] autoMemoryReclaim=gradual`）。目标：**内存再紧也只牺牲业务进程（优先采集 worker，其次训练），绝不牺牲 WSL 接入通道**；并且尽量不走到那一步。

## 要做的（工作树 `git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/oom -b oom origin/pvz-env`）
1. **OOM 优先级（代码层，主干路径统一做）**：
   - 采集 worker：`scripts/search_play.py` 的每个 worker 进程启动时把自己的 `/proc/self/oom_score_adj` 写成 `1000`（最先被杀）；它拉起的引擎子进程（`python/pvz_env.py` 里 `subprocess.Popen`）用 `preexec_fn` 同样写 `1000`。
     写失败（非 Linux、Mac）静默跳过。课程控制器 `scripts/curriculum_queue.py` 自身写 `500`（比 worker 后死，比训练先死）。
   - 训练：`scripts/train_rollout_policy.py`、`scripts/train_value_net.py`、`scripts/train_value_branches.py` 启动时写 `300`。
   - 不碰任何 WSL 系统进程（init 等保持默认 0 或负值）；不需要 root（调高自己的分数不需要权限）。
   - 单元测试：Linux 上起一个子进程验证分数确实写进去；Mac 上跳过。
2. **用户态早期 OOM 守护**：先查 `apt list --installed 2>/dev/null | grep -i earlyoom` 和能否 `sudo -n true`（无密码 sudo）。
   - 能装：`sudo apt-get install -y earlyoom`，配置 `-m 5 -s 50 --prefer '(pvz-portable|search_play)' --avoid '(init|sshd|tmux|codex|codebuddy|qodercli|node)'`，systemd 服务开机启用（WSL 已开 systemd 的话；没开就写一个 tmux 常驻会话 `earlyoom` 并在 `~/pvz-logs/README-oom.md` 写清怎么起）。
   - 不能装（没网 / 没 sudo）：写一个等价的小守护 `scripts/mem_guard.py`（纯标准库）：每 2 秒读 `/proc/meminfo` 的 MemAvailable 与 `/proc/pressure/memory`，
     低于 5%（约 1.4 GB）时按 `oom_score_adj` 从高到低给最“该死”的一个进程发 SIGTERM（再等 10 秒不退就 SIGKILL），只对 `pvz-portable` / `search_play` / 训练进程动手，日志写 `~/pvz-logs/mem_guard.log`。在 tmux 会话 `memguard` 常驻。
3. **采集控制器按内存自动调并发**（`scripts/curriculum_queue.py`）：每开一个组之前读 MemAvailable，`workers = clamp(⌊(MemAvailable − 保留 6 GB) / 每 worker 估计值⌋, 1, --workers)`，每 worker 估计值启动时实测（一个 worker 的 Python+引擎 RSS 之和），日志里写实际用了几个 worker。
   训练在跑时自然就降下来，训练结束自动加回去，不再靠主控手动 4 ↔ 16。**不加新开关**：`--workers` 语义改成“上限”，在 help 里写清。
4. **WSL 侧验证**：在本工作树写一个可控的压力测试脚本（只在 `--i-know` 参数下运行，默认拒绝）：起 2 个假 worker（分数 1000）分配内存直到 MemAvailable < 5%，验证被杀的是假 worker、`wsl.exe` 新会话在压力期间仍可进入（用 `ssh 桌面 wsl -d Ubuntu -e true` 无法在 WSL 内部测，改为测 `tmux new-session -d` 与 `bash -c true` 的延迟 < 2 s）。**只在主控批准后跑，且跑之前确认没有训练在跑**（`pgrep -f train_rollout_policy` 为空）——先把脚本和计划写好提交，在“问题”里请示。
5. 报告写 `MEMORY/notes/desktop.md` 新的一节“内存打满时的防护（2026-10-23）”：分数表、守护怎么起、控制器怎么调并发、压力测试结果。

## 规则
- 不要动任何 tmux 队列（qc 等）与别的代理会话（cb-cuda、cb-vnet、codex-cuda）；不改别的工作树与主工作区。不要重启 WSL、不要 `wsl --shutdown`。
- 分支 `oom` 提交，可以 `git push origin oom`；不推 pvz-env、不合并（主控合并后再重启 qc 生效）。不写字面密钥；不 `git add -f`。只用绝对路径。
- **不加新开关**（主控 10-22 规则：重构直接替换旧实现）。每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续。

## 结果
- 10-22 23:51 OOM 分数落地：`python/pvz_common.py` 加 `set_oom_score_adj` 与三个角色常量（worker/引擎 1000、控制器 500、训练 300）；`search_play._init_worker`、`pvz_env` 引擎 Popen 的 `preexec_fn`、`curriculum_queue.main`、三个训练脚本 `main` 均已写入；`python/test_oom_guard.py` 3 项在 Linux 子进程里实测通过（Mac 跳过）。
- 10-22 23:53 用户态守护：系统 apt 镜像（bfsu）对 pools 返回 403，改从清华镜像取同版本 deb 用 dpkg 安装 earlyoom 1.9.0（systemd 已启用，`active` + `enabled`）。`/etc/default/earlyoom` = `-m 5 -s 50 -r 300 --prefer '(pvz-portable|search_play|python3)' --avoid '(init|sshd|tmux|codex|codebuddy|qodercli|node)'`。earlyoom 只匹配 comm（15 字节），`search_play` 的 comm 是 `python3`，故 prefer 里补 `python3`；worker 自身分数 1000 仍是第一顺位。当前 MemAvailable 约 31%，未触发。
- 10-22 23:55 课程控制器自动并发：`--workers` 默认 16、语义改为上限；每开一组前读 `/proc/meminfo` 的 MemAvailable，`workers = clamp(⌊(available − 6 GiB) / 单 worker 实测 RSS⌋, 1, --workers)`，每组日志先写 `workers=k/16 (MemAvailable …, per-worker …)`。单 worker 估计在控制器启动时实测：以 `--workers 1 --seeds 1` 跑一局真实任务（临时 out/帧目录，不动状态偏移、不进档案），0.2 s 轮询 search_play 的全部后代（worker + pvz-portable）取峰值 RSS；失败退回 1 GiB；没有 /proc（Mac）不缩放。停止标志在探测前也检查。`python/test_curriculum_queue.py` 新增 3 项（11 项全过）。
- 10-22 23:56 压力测试脚本 `scripts/oom_stress_test.py`（未运行）：默认拒绝（退出码 2）；`--i-know` 下仍先拒绝训练在跑 / 采集在跑 / 可用内存 < 12 GiB。起 2 个假 worker（`oom_score_adj=1000`，64 MiB 块 / 50 ms）把 MemAvailable 压到 5% 以下，期间量 `tmux new-session -d` 与 `bash -c true` 延迟；验证两个假 worker 被信号杀死、init/sshd/tmux/earlyoom 都存活、无强制 kill、延迟 < 2 s。当前 dw5i 训练在跑，已在“问题”请示。
- 10-22 23:57 报告写入 `MEMORY/notes/desktop.md` 新节“内存打满时的防护（2026-10-23）”：分数表、earlyoom 安装与参数、控制器并发公式与实测方法；压力测试结果待批准后补。
- 10-23 00:04 主控答复：批准压测，但由主控开时间窗——qc 在组间停下（`~/pvz-logs/curriculum.log` 出现 stop flag 行、`pgrep -f curriculum_queue` 为空）、确认无训练后，在 `~/pvz-logs/` 创建 `oom_window_open`。本会话已挂后台监视等该文件；出现即跑一次，结果写回本文与 `desktop.md` 并提交，然后删 `oom_window_open`、建 `oom_window_done`。等待期间补了压测预检（默认拒绝、按 cmdline 找进程、计时）3 项与控制器探测命令 / 启动估计 2 项单测（相关单测共 16 项全过），并用 qc 的 `multiplier_panel` + 状态副本跑 `--dry-run` 正常（5 组 / 240 局）；`python/test_env_protocol.py` + `python/test_shared_helpers.py` 42 项回归也通过。`mem_guard.py` 不写：earlyoom 已装好，任务书的等价守护只用于“不能装”分支，避免第二份守护抢杀。
- 10-23 08:07 修复压测预检误判（跑之前发现）：`matching_processes` 原先按 cmdline 子串匹配，earlyoom 的 `--prefer '(pvz-portable|search_play|python3)'` 会一直被当成“采集在跑”而拒绝。改为按 argv token 的 basename 精确匹配，新增 earlyoom 命令行回归测试；实测当前匹配列表只剩 qc 控制器 / `search_play` / 引擎，earlyoom 不再命中。压测单测 4 项、task34+回归共 61 项全过。
- 10-23 08:08 压测输出加 `pressure_observed`：按每 50 ms 采样的最小值判断是否确实低于 5%（杀进程可能恰好落在两次轮询之间），latency 仍只在轮询当场捕捉到 <5% 时测；62 项测试全过。
- 10-23 08:18 窗口内跑压测前收紧/校正预检：任务书的硬门禁只有“无训练”；主控 08:16 开窗时 cb-vnet / codex 的采集仍在跑（分数 0），假 worker 分数 1000。原“有采集就拒绝”会让已批准窗口无法使用，改为把其他采集进程记入输出（`other_collection`）后继续；训练在跑仍然拒绝。窗口前 qc 已在 08:15:25 组间停止（stop flag），无训练。
- 10-23 08:19 压测执行一次并通过（EXIT=0；窗口 08:16:29 开）。2 个假 worker（`oom_score_adj=1000`）：MemTotal 27.4 GiB，起压 MemAvailable 21.6 GiB，最低 **1.36 GiB = 4.95%**（`pressure_observed=true`），结束后 22.8 GiB。延迟：`tmux new-session -d` **0.084 s**（rc=0）、`bash -c true` **0.003 s**（rc=0），均 <2 s。两个假 worker（175719 RSS 22279 MiB ≈ 21.8 GiB、175720 RSS 11273 MiB ≈ 11.0 GiB）均被 earlyoom SIGTERM（rc=-15）；journal 记录 oom_score 2098 / 1861、`process_mrelease` 成功、0.1 s 退出；无强制 kill、无内核 OOM。init 存活；sshd 在 WSL 内未运行；tmux 接入由压力期间的 new-session 成功直接验证（`pgrep -x tmux` 不匹配 comm `tmux: server`，故 watched 列表为空，已如实记入输出）。同机其他代理未被误杀：cb-vnet task30 的控制器 168777 存活、游戏的 `games.jsonl` 12→16 继续推进；codex 的 task37 1-seed profiling 正常写完 `summary.json`（08:19）；运行前 12 个其他采集进程全部记入 `other_collection`。完整摘要另存 `~/pvz-logs/oom_stress_result.json`。

## 问题
- （已解决）压测已于 10-23 08:19 在主控时间窗内执行一次并通过，结果见“结果”；窗口按协议关闭（08:21 删 `oom_window_open`、建 `oom_window_done`）。原请求：在训练结束后跑 `python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/oom/scripts/oom_stress_test.py --i-know`；当前 dw5i 训练已结束、qc 正在跑第 2 轮，需主控在组间停 qc 并开窗。
