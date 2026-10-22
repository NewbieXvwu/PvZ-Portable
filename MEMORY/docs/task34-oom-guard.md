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

## 问题
