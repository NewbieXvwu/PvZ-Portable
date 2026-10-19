# 交接书：借朋友的闲置机器做分布式采集（主控 Claude 起草，2026-10-19）

写给本机（Mac）上的 CodeBuddy。先读仓库根目录 `AGENTS.md`（体积与交付规则，**优先级最高**），再读本文。
用户的 Claude 额度不多了，**之后由你和一个廉价模型操作**，所以每一步都要写成“照着做就能跑”的文档和脚本。

## 0. 背景（一段话）

我们在训练一个《植物大战僵尸》学生网络（DAgger：学生下棋，搜索老师在每个决策点给标签）。现在最缺的是**采集算力**：
每局完全独立（给定学生模型 + 一段种子 → 下完交回对局档案），每局约 0.3 MB，算力花在老师的续走上（每局约 150–180 CPU 秒）。
用户会借来若干朋友的机器（多数 Windows + WSL，少数 Mac，部分有 RTX 5070 Ti），明天开始正式分布式跑；**今天先在第一台朋友 Mac 上把整套流程跑通**。
训练仍只在台式机上做，分布式只管采集（以后可加评估）。

## 1. 开工前

- **工作区**：`cd /Users/newbiexvwu/PvZAgent && git fetch origin && git worktree add .claude/worktrees/distributed -b distributed origin/pvz-env`，
  只在这个 worktree 改代码、单独编译（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。
  **主仓库 `/Users/newbiexvwu/PvZAgent` 的工作区和 `build/` 不许碰**：Mac 自己的正式采集（`artifacts/hard_mac/mac_dagger*.sh`，10 个进程）在用。
  本机你同时最多 **4 个游戏进程**。
- **提交**：在 `distributed` 分支提交；可以 `git push origin distributed`（仓库是公开的，朋友机器要 clone 这个分支）。**不要推 `pvz-env`、不要合并**，主控 / 用户审过再合。
- **密钥**：不写任何字面值（HF 用 `$HF_TOKEN` / `$PVZ_HF_REPO`，本机 `~/.zshrc` 已有）。**朋友机器上不放任何令牌**（见 §2 的设计）。
  不打印 `~/.wb-switch/accounts.json`。
- **朋友 Mac**：`ssh pvz-f1`（`~/.ssh/config` 已配好，走用户的隧道，密钥登录；**不要用、不要记录任何密码**）。
  Apple M5、10 核、32 GB、macOS 26.6、有 Xcode 命令行工具、系统 Python 3.9、**没有 brew / cmake**。
  只在 `~/pvz-worker/` 里放东西；朋友可能随时用电脑。用完后的清理方法写进文档（删 `~/pvz-worker`、删 `~/.ssh/authorized_keys` 里我们那一行）。

## 2. 设计（主控已定，照此实现；有更好的想法写进报告再说）

**星形，控制端在本 Mac，一切经 SSH 隧道**：每台朋友机器在本 Mac 上有一个隧道端口（`127.0.0.1:2222x`），本 Mac 用 ssh / rsync 推代码、模型、任务，拉结果。
朋友机器**不需要任何令牌**，也不需要能访问 HF；隧道断了，worker 照常跑完手上已分配的任务，结果留在本地等下次拉取。
结果进训练数据走现有通道：本 Mac 打包上传 HF `transfer/<label>.tgz` → 台式机队列下载解包（台式机写法见 §4）。

| 组件 | 位置 | 作用 |
|---|---|---|
| `scripts/dist/bootstrap.sh` | 朋友机器（Linux / WSL / macOS） | 一键装环境：装 `uv` → Python 3.12 venv → CPU 版 torch、numpy、cmake、ninja（pip 装，不需要 brew / sudo）→ clone 指定提交 → 编译 → 自检（跑 1 局）。可重复执行 |
| `scripts/dist/worker.py` | 朋友机器 | 常驻进程：读本地任务目录、按段跑 `search_play.py`、**断点续跑**（跳过已完成的种子）、写状态、本地网页 |
| `scripts/dist/controller.py` | 本 Mac | 机器清单、推送（代码提交 / 学生模型 / 价值网络 / 清单 / 资源）、分段派活、拉结果、汇总打包上传 HF、台账 |
| 文档 | `MEMORY/docs/distributed-worker.md` | 给运维（廉价模型）的操作手册 + 给朋友的一页说明（中文、说人话） |

**worker 细节**
- 任务 = `{label, student 文件, task_id, seed_offset, seeds, 老师参数}`；一段 16–64 局。进程数默认 `核数 − 2`，内存按每进程约 0.5 GB 估算再封顶（实测后修正）。
- 老师参数**必须和正式采集一致**：`--verify-samples 9 --value <value_v3> --static-value <value_branch1> --dagger-beta 0 --student-candidates 2`，
  `--manifest day_l7_hard.json --task-id day_l7_z17_train`，`--frames <dir> --frames-label <label>`，`--resource-dir <含 main.pak 与 properties/ 的目录>`。
  参照 `artifacts/hard_mac/mac_dagger29b.sh` 与 `artifacts/hard_mac/round15_flags`。
- **本地网页**（Python 自带 `http.server`，只绑 `127.0.0.1`，WSL2 会自动转发到 Windows 的 localhost）：显示机器名、当前任务名、本段进度（如 12/32）、
  每小时局数、累计局数、状态；按钮：
  - **暂停**：冻结全部采集进程（SIGSTOP 进程组），CPU 立刻空出来；**继续**：SIGCONT。
  - **下完这局再停**：不再开新局，进行中的局下完退出。
  - **立即停止**：结束进程；进行中的局作废，下次启动从未完成的种子续跑。
  - 冻结超过 30 分钟自动转“立即停止”，把内存还给朋友。
  不打磨样式，能用就行。macOS / Windows 各放一个双击打开网页的快捷方式。
- **开机自启**：macOS 用 `launchd`（用户级 LaunchAgent），WSL 用 Windows 计划任务启动 `wsl -d <发行版> -e ...`（写进 Windows 安装脚本，§3 第 4 步）。

**controller 细节**
- 机器清单 `~/.pvz-dist/machines.json`（ssh 别名、核数、是否有 GPU、启用与否）；**台账** `~/.pvz-dist/ledger.json`（每段：机器、偏移、局数、学生、状态、完成时间）。
  台账是分布式偏移段的唯一权威记录：**偏移 16384–65535 归分布式**（DESIGN.md 已登记；`day_l7_z17_train` 已扩到 65536 个种子，两台机器的本地清单都已改），段与段不重叠，掉线机器的未完成段可以改派。
- 命令：`status`（所有机器进度一表）、`push <machine>`（同步代码 / 模型 / 资源）、`assign <machine> <student> <games>`、`collect`（拉回已完成段，校验后合并成一个 label 打包上传 HF）。
- **校验**：每局档案可被 `scripts/train_rollout_policy.py` 的加载器读出（obs v2）、种子在分配段内且不重复、`summary.json` 合理。

**资源文件**：`main.pak`（43 MB）与 `properties/` 在 `/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN/`，由 controller 经 scp 推到朋友机器（用户已同意，风险自担）。
价值网络：`artifacts/hard_mac/value_v3/model.pt`、`artifacts/hard_mac/value_branch1/model.pt`；清单：`artifacts/stage4/day_l7_hard.json`（不在 git 里，要推送）。
学生：`artifacts/student/v29/model.pt`（Mac 第 29b 轮开始后会从 HF 下载到这里；没有就从 HF `transfer/v29_model.pt` 取）。

## 3. 步骤与验收（每完成一步就提交一次）

1. **bootstrap 在 pvz-f1 跑通**：从零装好环境，自检 1 局能下完、档案可读。记录装机耗时与占用磁盘。
2. **worker + 网页**：在 pvz-f1 上跑原型段——**v29、偏移 14736–14767（32 局，已在 DESIGN 登记）、label `z17distproto`、8 个进程**。
   验收：32 局全部完成；暂停 / 继续 / 下完这局再停 / 立即停止 + 续跑都实测过（续跑后没有重复种子、没有缺失）；每小时局数、每局 CPU 秒、峰值内存。
   胜率与 Mac 自己的同学生轮次（第 29b 轮）同口径对比（只看是否合理，32 局不做显著性）。
3. **controller**：`status / push / assign / collect` 跑通；把原型 32 局打包上传 HF `transfer/z17distproto.tgz`；台账有这一段。
4. **Windows 安装脚本** `scripts/dist/install_windows.ps1`（朋友今天就要装，**这一步尽早给出初版**）：启用 WSL 并装 Ubuntu（首次需重启，脚本要能重启后续跑）、
   在 WSL 里装 OpenSSH 服务并加入我们的公钥（`~/.ssh/id_ed25519.pub` 的内容可以写进脚本，公钥不是秘密）、设置开机自启、桌面放网页快捷方式、卸载说明（`wsl --unregister`）。
   隧道由用户自己配，脚本里留出“隧道端口”的说明即可。没有 Windows 机器可测：在报告里写明“未实机验证”的部分。
5. **文档** `MEMORY/docs/distributed-worker.md`：
   - 运维手册：接入一台新机器的完整步骤、每天怎么派活 / 收数据、常见故障；
   - 给朋友的一页说明（安装、网页在哪、三个按钮什么意思、怎么彻底删除）；
   - 实测数字（每台每小时局数等）。
6. （之后，等主控通知）GPU：`cuda-rollout` 分支合进主线后，给 worker 加“GPU 采集”开关（5070 Ti 与台式机 5080 同为 cc 12.0，可直接用台式机编好的二进制）。

## 4. 台式机怎么接收（写进运维手册，台式机由主控 / 运维操作，你不用登录台式机）

台式机的队列脚本用这段把 Mac 的轮次收进训练数据（`~/pvz-logs/queue60.sh` 第 25–35 行）：
```bash
L=z17distproto   # 换成要收的 label
until L=$L $PY - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["PVZ_HF_REPO"], f"transfer/{os.environ['L']}.tgz", token=os.environ["HF_TOKEN"], force_download=True)
tarfile.open(p).extractall("artifacts/student/frames")
PY
do sleep 120; done
for d in $F/seed_*/${L}_fast_search; do s=$(basename $(dirname $d)); mkdir -p $V2/$s; [ -d $V2/$s/${L}_fast_search ] || cp -al $d $V2/$s/; done
```
（`F=~/PvZ-search/artifacts/student/frames`，`V2=~/PvZ-search/artifacts/student/frames_v2`，`PY=~/.venvs/ml/bin/python3`。）

## 5. 汇报

- 进度与数字追加到 `MEMORY/2026-10-19.md`（或当天日期），每条带时间。
- 有问题写 `MEMORY/docs/distributed-worker-questions.md`，然后按你的最佳判断继续，不要停下来等。
- 全部完成后在 `MEMORY/docs/distributed-worker.md` 首页写“当前状态 + 明天开工的三条命令”。
