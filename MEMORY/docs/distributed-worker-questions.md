# 分布式采集：待确认 / 卡住的事（CodeBuddy 记录）

> 按交接书 `5：有问题写这里，然后按最佳判断继续，不停下来等。

## 2026-10-19 01:45 朋友 Mac（pvz-f1）SSH 断在 banner exchange

**现象**：01:38 推完代码（1739 个文件）后，`ssh pvz-f1` 一律
`Connection timed out during banner exchange`。端口 22223 的 TCP 握手是通的（隧道还活着），
说明是本机（朋友 Mac）没有应答 —— 最可能是**合盖/空闲睡眠**。之前 10 分钟的 bootstrap
是正常的，正好在装完之后断的。

**我做的处理**：
- `controller.py start` 现在用 `caffeinate -i` 包住 worker（macOS 上采集期间不睡眠）。
- ssh / rsync 都加了 `ServerAliveInterval=30`、`rsync --timeout=180`，以后断线会快速失败
  而不是卡 10 分钟。

**需要用户/朋友做的一件事**（只有这一件）：
让那台 Mac 在采集期间别睡眠 —— 插上电源，系统设置 → 锁定屏幕 → “显示器关闭时防止自动进入睡眠”，
或者一直开着盖。装完之后第一次 `controller.py start pvz-f1 --label ...` 会带 `caffeinate`。

**未完成**：第 1 步的“自检 1 局”与第 2 步的原型段都要等机器醒来才能跑。

## 机器醒来之后要做的（照着抄）

    cd /Users/newbiexvwu/PvZAgent/.claude/worktrees/distributed
    python3 scripts/dist/controller.py status                     # 应该看到 pvz-f1
    python3 scripts/dist/controller.py assign pvz-f1 artifacts/student/v29/model.pt 32 \
        --label z17distproto --offset 14736 --workers 8 --allow-reserved --push --start
    python3 scripts/dist/controller.py status                     # 看 0/32 → 32/32
    python3 scripts/dist/controller.py collect --label z17distproto   # 校验 + 打包 + 上传 HF

如果后台的一次性脚本 `/tmp/pvz-dist-wait.sh` 已经把代码/模型/资源推过去并跑完自检，
上面第 3 条里的 `--push` 也安全（rsync 增量 + 重编译约 15 秒）。

原型段 32 局、8 进程，预计 **6–8 分钟**（Mac 上同参数每局 97 秒）。

## 2026-10-19 03:20 隧道端口观察（主控说"已重新连上"之后）

`lsof -nP -iTCP -sTCP:LISTEN` 里 UURemote 只开了两个相关端口：

| 端口 | 对端 | 状态 |
|---|---|---|
| 22223（~/.ssh/config 里的 pvz-f1） | 朋友 Mac | TCP 握手成功，**SSH banner 一直超时** |
| 22222 | **Windows OpenSSH 9.5** | banner 正常，但我们的公钥被拒（`Permission denied (publickey)`） |

22222 上的 host key 是 Windows 的，不是朋友 Mac —— 所以"重新连上"接的可能是另一台
Windows 机器，pvz-f1 那条转发仍然是死的。若 22222 就是新接入的 Windows 朋友机，需要先把
`~/.ssh/id_ed25519.pub` 放进它的 `administrators_authorized_keys`（Windows OpenSSH 的 root
等价物），或者按 §2.3 走 WSL 里的 sshd（端口 2222）。

**当前**：pvz-f1 仍不可达；32 局原型已用本机 4 进程跑完并上传 `transfer/z17distproto.tgz`，
台账登记为 local-mac。pvz-f1 恢复后用同一 label/偏移重跑即可覆盖。

## 2026-10-19 04:40 隧道抖动（04:17 通了一次，随后 20 分钟全超时）

04:17:10 有一次成功的 `ssh pvz-f1`（远端 16:17、load 1.11、repo 011acef、models/resources 为空）。
之后每 3 秒一次探测、连测 8 次全部 `banner exchange` 超时；rsync 传代码也在几秒内断。

已启动一个**会自己重试的 master 脚本** `/tmp/pvz-master.sh`（日志 `/tmp/pvz-master.log`）：
等 SSH → 分段推送（main.pak 用 `--append-verify` 续传）→ bootstrap 自检 → assign 32 局 8 进程
→ 等 32/32 → collect 上传。每一步都有 20–200 次重试，隧道恢复后会自己跑完。

**需要**：保持隧道在线、机器别合盖（`caffeinate` 只防空闲睡眠，合盖仍会睡）。

## 2026-10-19 06:30 隧道是谁提供的（查清了）

`lsof -nP -iTCP:22223 -sTCP:LISTEN` → **UURemote（UU 远程，PID 可变）**。

- 22223 不是我们起的 `ssh -L`（本机没有任何 ssh 转发进程），是 UU 远程桌面自带的端口转发。
  所以**隧道断了不能靠重启 ssh 恢复**，只能在 UU 里重连 / 唤醒对面机器；我们这侧只能重试。
- TCP 握手成功 + banner 超时 = 转发还在、对面不响应（睡眠或 UU 会话失效）。
- `collect` 现在自带重试与 `--wait <分钟>`，整轮失败不会回滚已收的 seed 目录：
  `python3 scripts/dist/controller.py collect --label z17distproto --wait 600 --retry-delay 20`

