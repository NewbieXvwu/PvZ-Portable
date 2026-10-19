# 分布式采集：运维手册（+ 给朋友的一页说明）

主控在这台 Mac，朋友机器只负责跑局。**朋友机器上没有任何令牌，也不需要访问 HF**：
代码、模型、资源、任务都由控制端经 SSH 推过去，结果留在本地等控制端拉回。

> 交接书：`MEMORY/docs/distributed-worker-handoff.md`（设计权威）
> 台账：`~/.pvz-dist/ledger.json` —— 偏移 16384–65535 的唯一权威记录
> 代码：`scripts/dist/{bootstrap.sh,worker.py,controller.py,install_windows.ps1}`

## 当前状态

（每完成一步更新这里）

## 0. 一页纸（发给朋友）

> 我在你电脑上装了一个《植物大战僵尸》的对局采集程序。它只在我给你的那个窗口开着、
> 或者你说“可以跑”的时候才占 CPU；不需要你登录任何账号，也不会传走你的文件。
>
> - **看进度 / 控制**：双击桌面上的 **“PvZ progress”**（Windows）或 **“PvZ 采集”**（Mac），
>   会打开一个本地网页。
> - **三个按钮**：
>   - **暂停**：立刻让出 CPU（游戏进程被冻结），你随时可以点“继续”。
>   - **下完这局再停**：不再开新的一局，手上这局下完就停。
>   - **立即停止**：马上结束；已经下完的局不会丢，下次接着跑没下完的。
> - **彻底删除**：Windows 用安装脚本的 `-Uninstall`；Mac 删掉 `~/pvz-worker` 即可。
>   详见本文件第 7 节。
>
> 如果电脑变卡：点“暂停”就够了，不用关窗口。

## 1. 架构

| 组件 | 位置 | 作用 |
|---|---|---|
| `controller.py` | 本 Mac | 台账、推送、派活、回收、打包上传 HF |
| `bootstrap.sh` | 朋友机器 | 从零装环境（micromamba + torch + 编译 + 自检），可重复执行 |
| `worker.py` | 朋友机器 | 常驻采集：分段跑 `search_play.py`、断点续跑、本地网页 |

- 每台朋友机器在本 Mac 上有一个 SSH 隧道端口（`127.0.0.1:2222x`），控制端用 ssh / rsync。
- 隧道断了 worker 照常跑完手上的段，结果留在本地等下次拉取。
- 结果进训练数据：本 Mac 打包 `transfer/<label>.tgz` 上传 HF，台式机队列下载（`6）。

## 2. 接入一台新机器

### 2.1 前提

- 朋友机器能被本 Mac `ssh <别名>` 登录（隧道由用户配好，别名写在 `~/.ssh/config`）。
- 本 Mac 的 `~/.ssh/id_ed25519.pub` 已在对方 `authorized_keys` 里（公钥不是秘密）。

### 2.2 Mac

    ssh <别名> 'uname -m; sw_vers -productVersion'      # 确认架构与版本
    python3 scripts/dist/controller.py bootstrap <别名>
    python3 scripts/dist/controller.py push <别名> --all --student artifacts/student/v29/model.pt

`bootstrap.sh` 只往 `~/pvz-worker/` 写东西，不需要 sudo / brew。装完会打印耗时与磁盘占用。

### 2.3 Windows（WSL2）

把 `scripts/dist/install_windows.ps1` 发给朋友（或让他从仓库取），右键“以管理员身份运行”：

    powershell -ExecutionPolicy Bypass -File install_windows.ps1

首次要重启（脚本用 RunOnce 在下次登录后自动续跑）。它做四件事：启用 WSL2 → 装 Ubuntu →
在 Ubuntu 里装 sshd（端口 2222，只允许公钥，root 登录）→ 建登录自启任务 + 桌面网页快捷方式。

隧道：Windows 上把 `127.0.0.1:2222` 转发到本 Mac 的 `127.0.0.1:2222x` 即可
（WSL2 会把 Windows 的 localhost 自动转发进发行版，不用 portproxy）。
本机 `~/.ssh/config` 里对应 `Host pvz-w1 / Port 2222x`。

### 2.4 机器清单

`~/.pvz-dist/machines.json`：

    {
     "pvz-f1": {"host": "pvz-f1", "cores": 10, "gpu": false, "enabled": true, "note": "friend Mac M5"}
    }

`cores` 只用于默认进程数（`cores-2`，再按内存封顶）。禁用一台机器把 `enabled` 改 false。

## 3. 每天怎么派活 / 收数据

    # 1) 看所有机器
    python3 scripts/dist/controller.py status

    # 2) 派一段（自动取 16384–65535 里第一段空闲偏移）
    python3 scripts/dist/controller.py assign pvz-f1 artifacts/student/v29/model.pt 320 \
        --push --start

    # 3) 收回来（校验 → 打包 → 上传 HF transfer/<label>.tgz）
    python3 scripts/dist/controller.py collect --label z17dist16384

- 一个 label = 一段 = 一个 `transfer/<label>.tgz`；台式机按 label 收。
- 段与段不重叠；台账是唯一权威。机器掉线没跑完的段，**用同一个 label + 同一个 offset
  在另一台机器上再 `assign` 一次**即可改派（同 label 会覆盖台账里那条）。
- `--start` 会重启该机器上的 worker；不加的话，新任务要等 worker 下一轮启动才会被看到。

### 3.1 分段与断点续跑

- 任务 = `{label, student, task_id, seed_offset, seeds, 老师参数}`；worker 按 `chunk`（默认 =
  进程数）分段调用 `search_play.py`，一段 = 一波。
- `search_play.py` 自己会读 `out/<label>/games.jsonl` 跳过已完成的种子，所以“立即停止”后
  重新启动不会重复也不会漏。
- 连续失败的段会自动缩到单局重试；单局连续失败 3 次记进 `state/<label>.failed.json` 并跳过。

### 3.2 老师参数（必须和正式采集一致）

`controller.py` 从 `artifacts/hard_mac/round15_flags` 读 `--verify-samples 9`（以及两条价值网络
路径，改用 `models/` 下的副本），再补第 29b 轮的 `--dagger-beta 0 --student-candidates 2`。
`task.json` 里落的就是最终参数，worker 原样执行。

## 4. 常见故障

| 现象 | 处理 |
|---|---|
| 网页打不开 | worker 没在跑：`controller.py start <机器>`；或隧道没开：`controller.py web <机器>` |
| 状态一直 idle | `~/pvz-worker/tasks/` 下没有任务；`assign` 后加 `--start` |
| 每小时局数掉到 0 但状态 running | 机器睡眠了；让朋友在电源设置里关掉睡眠，或手动唤醒 |
| 段跑完但 `collect` 报种子缺失 | 有人动过 `out/<label>/games.jsonl`；用同一 label/offset 重新 assign 续跑 |
| 校验报 unreadable archive | 磁盘满 / 进程被强杀写了一半；删掉该 `seed_<n>` 目录后重跑该段 |
| 内存不够 | 调小 `--workers`（约 0.5 GB/进程） |

## 5. 实测数字

（跑完原型段后填：装机耗时、磁盘、每小时局数、每局 CPU 秒、峰值内存、胜率）

## 6. 台式机怎么接收

（原文照抄交接书 `4）

    L=z17distproto   # 换成要收的 label
    until L=$L $PY - <<'PY'
    import os, tarfile
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(os.environ["PVZ_HF_REPO"], f"transfer/{os.environ['L']}.tgz", token=os.environ["HF_TOKEN"], force_download=True)
    tarfile.open(p).extractall("artifacts/student/frames")
    PY
    do sleep 120; done
    for d in $F/seed_*/${L}_fast_search; do s=$(basename $(dirname $d)); mkdir -p $V2/$s; [ -d $V2/$s/${L}_fast_search ] || cp -al $d $V2/$s/; done

（`F=~/PvZ-search/artifacts/student/frames`，`V2=~/PvZ-search/artifacts/student/frames_v2`，
`PY=~/.venvs/ml/bin/python3`。）

## 7. 清理

**Mac 朋友机**：

    rm -rf ~/pvz-worker
    # 再从 ~/.ssh/authorized_keys 里删掉我们那一行（以 ssh-ed25519 ... newbiexvwu 开头）

**Windows 朋友机**：

    powershell -ExecutionPolicy Bypass -File install_windows.ps1 -Uninstall

它会删计划任务、桌面快捷方式、`C:\ProgramData\pvz-worker`，并 `wsl --unregister Ubuntu`
（**该发行版里的所有东西都会删掉**；WSL 本身还留着）。
