# 分布式采集：运维手册（+ 给朋友的一页说明）

主控在这台 Mac，朋友机器只负责跑局。**朋友机器上没有任何令牌，也不需要访问 HF**：
代码、模型、资源、任务都由控制端经 SSH 推过去，结果留在本地等控制端拉回。

> 交接书：`MEMORY/docs/distributed-worker-handoff.md`（设计权威）
> 台账：`~/.pvz-dist/ledger.json` —— 偏移 16384–65535 的唯一权威记录
> 代码：`scripts/dist/{bootstrap.sh,worker.py,controller.py,install_windows.ps1}`

## 明天开工的三条命令

    python3 scripts/dist/controller.py status
    python3 scripts/dist/controller.py assign pvz-f1 artifacts/student/v29/model.pt 320 --push --start
    python3 scripts/dist/controller.py collect --label <label>

（第一条看进度；第二条派 320 局并立刻开跑；第三条收回来、校验、上传 HF。）

## 当前状态（2026-10-19 01:55）

| 步骤 | 状态 |
|---|---|
| 1 bootstrap 在 pvz-f1 跑通 | ✅ 10 分 14 秒，1.3 GB；**自检 1 局通过**（1 个档案、1055 帧） |
| 2 worker + 网页 | ✅ 32 局原型**已在 pvz-f1 上跑完 32/32**（8 进程、偏移 14736–14767）；暂停/继续/下完这局再停/立即停止+续跑都在 pvz-f1 上实测 |
| 3 controller | ✅ status/push/assign/collect 全部跑通；`transfer/z17distproto.tgz` 已上传（本机演练版），**pvz-f1 版正在拉回覆盖** |
| 4 Windows 安装脚本 | ✅ 初版（**未实机验证**，见 §8） |
| 5 文档 | ✅ 本文件 |
| 6 GPU | 等主控通知 |

**卡点（2026-10-19 04:00）**：pvz-f1 的隧道仍然不通 —— 22223 一直停在 banner exchange，
22222 曾短暂通到一台 Windows OpenSSH（公钥被拒），随后也超时。bootstrap 装完后的自检与
8 进程复跑要等隧道恢复。恢复后一条命令即可补齐：

    python3 scripts/dist/controller.py push pvz-f1 --all --student artifacts/student/v29/model.pt \
      && ssh pvz-f1 "PVZ_COMMIT=$(git rev-parse HEAD) bash ~/bootstrap.sh" \
      && python3 scripts/dist/controller.py assign pvz-f1 artifacts/student/v29/model.pt 32 \
           --label z17distproto --offset 14736 --workers 8 --allow-reserved --start \
      && python3 scripts/dist/controller.py collect --label z17distproto

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
- 结果进训练数据：本 Mac 打包 `transfer/<label>.tgz` 上传 HF，台式机队列下载（见第 6 节）。

## 2. 接入一台新机器

### 2.1 前提

- 朋友机器能被本 Mac `ssh <别名>` 登录（隧道由用户配好，别名写在 `~/.ssh/config`）。
- 本 Mac 的 `~/.ssh/id_ed25519.pub` 已在对方 `authorized_keys` 里（公钥不是秘密）。

### 2.2 Mac

    ssh <别名> 'uname -m; sw_vers -productVersion'      # 确认架构与版本
    python3 scripts/dist/controller.py bootstrap <别名>
    python3 scripts/dist/controller.py push <别名> --all --student artifacts/student/v29/model.pt

`bootstrap.sh` 只往 `~/pvz-worker/` 写东西，不需要 sudo。依赖全部来自 micromamba（conda-forge），
**不依赖 brew**；不过 macOS 上它会先探测 `/opt/homebrew/bin`（Apple Silicon）与 `/usr/local/bin`（Intel）
并加进 PATH —— 非交互 ssh 登录的 PATH 里通常没有这两项，之前 `which cmake` 找不到就是这个原因，
不代表机器没装。装完会打印耗时与磁盘占用。

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
| `main.pak` 传到一半断了 / rsync 打印用法 | macOS 自带的是 **openrsync（2.6.9 兼容）**，有 `--partial` 但**没有 `--append-verify`**（用了只会打印用法并失败）。`controller push` 已用 `--partial`；大文件就多跑几次，每次断点保留，直到远端 `stat -f %z main.pak` 等于本地大小 |

## 5. 实测数字

### 5.1 装机（pvz-f1：Apple M5 / 10 核 / 32 GB / macOS 26.6）

| 项 | 数值 |
|---|---|
| 全程（micromamba → 编译 → 自检） | **10 分 14 秒**（13:16:20 → 13:26:34） |
| 其中 libopenmpt 源码编译 | 约 1 分钟 |
| 其中 cmake 配置 + 编译（-j10） | **15 秒** |
| 磁盘 | **1.3 GB**（env 1.0 GB，repo+build 95 MB） |
| 网络 | 下行约 7 MB/s（torch 127 MB 用 18 秒）；到本机上行只有约 100 KB/s |

### 5.2 采集（本机 2 进程验证，非朋友机器）

| 项 | 数值 |
|---|---|
| 每局墙钟 | 85–95 秒（10 核同时在跑正式采集） |
| 吞吐 | 约 52 局/小时（2 进程） |
| 单局档案 | 约 210 KB（npz，含逐决策帧） |
| 3 局胜率 | 1/3（样本太小，只用于确认管线通） |

### 5.3 原型段 z17distproto（32 局，偏移 14736–14767）

**本机演练（4 进程，2026-10-19 03:00–03:18；pvz-f1 当时在睡眠）**

| 项 | 数值 |
|---|---|
| 完成 | **32/32**，无失败、无重复 |
| 胜率 | **28/32 = 87.5%**（同学生同老师；32 局不做显著性） |
| 每局墙钟 | 103.0 秒 |
| 吞吐 | 约 99 局/小时（4 进程，10 核同时在跑正式采集） |
| CPU | 平均 322%，峰值 400%（4 进程 ≈ 4 核） |
| 内存 | 平均 1966 MB，峰值 2167 MB → **约 540 MB/进程**（与交接书 0.5 GB 的估计一致） |
| 档案 | 32 个，合计 9.6 MB（tgz） |
| 上传 | `transfer/z17distproto.tgz` |

### 5.4 pvz-f1 正式原型（8 进程，2026-10-19 05:36–05:46）

| 项 | 数值 |
|---|---|
| 自检 | 1 局通过（1 个档案、1055 帧），cmake 重编译 15 秒 |
| 完成 | **32/32**，无失败；32 个档案 |
| 吞吐 | 190.7 → 248.0 → 262.5 局/小时（8 进程，波次对齐后稳定在约 250） |
| 每局 CPU 秒 | 8 × 3600 / 248 ≈ **116 秒**（单线程） |
| 暂停 | 进程组全部 `T`、CPU 0%，9 秒后继续 |
| 立即停止 | 23/32 时点停止：进程清零、23 局保留；重启后从 23 续跑，种子无重复 |
| 下完这局再停 | 2/4 时点：本波跑完停在 2/4，60 秒后仍是 stopped、没有自己重启 |
| 内存 | 未在 pvz-f1 上采样；本机实测 540 MB/进程，8 进程约 4.3 GB（机器 32 GB） |

**注意**：pvz-f1 在国内，GitHub 不通。`bootstrap.sh` 检测 `repo/.pvz-pushed`（或 `PVZ_SKIP_GIT=1`）
就跳过所有 git 操作，代码只从控制端 rsync 推。朋友机器只依赖 conda-forge / PyPI。

**对照基线**：Mac 自己的第 29b 轮（同学生 v29、同老师参数）截至 01:52 已跑 378 局，
胜率 **75.7%**，每局 97.1 秒，平均波次 29.5。原型 32 局只看是否落在同一量级，不做显著性。

## 6. 台式机怎么接收

（原文照抄交接书第 4 节）

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

## 8. Windows 安装脚本：未实机验证的部分

本机没有 Windows，也没有 PowerShell（`which pwsh` 为空），所以
`scripts/dist/install_windows.ps1` **一行都没有真正执行过**。逐项列出风险：

| 步骤 | 状态 | 风险 |
|---|---|---|
| 提权（UAC 重新启动自己） | 未验证 | 低：标准写法 |
| 启用 WSL / VirtualMachinePlatform + RunOnce 重启续跑 | 未验证 | 中：RunOnce 以普通权限启动 → 会再弹一次 UAC |
| `wsl --install -d Ubuntu --no-launch` | 未验证 | 中：旧版 WSL 不认 `--no-launch`，脚本已回退到普通安装 |
| Ubuntu 首次初始化（`-u root` 跳过建用户） | 未验证 | 中：若发行版要求先建用户，需要朋友点一下 |
| 在 WSL 里装 sshd、改端口 2222、写公钥 | 未验证 | 低：纯 apt + sed |
| Windows localhost → WSL 的端口转发 | 未验证 | 中：WSL2 默认开 `localhostForwarding`，若被策略关掉需要 portproxy |
| 计划任务登录自启 | 未验证 | 中：任务在用户登录后跑，WSL 未启动时会拉起 |
| 卸载（`wsl --unregister`） | 未验证 | 低，但**会删掉整个发行版**，脚本里已写明 |

第一次在真机上跑时，按这个顺序看：`wsl -l -v`（发行版在不在）→
`wsl -d Ubuntu -u root -e service ssh status`（sshd 起没起）→
Windows 上 `ssh -p 2222 root@localhost`（端口通不通）→
`schtasks /Query /TN "PvZ Worker"`（自启任务）→ 桌面快捷方式。
