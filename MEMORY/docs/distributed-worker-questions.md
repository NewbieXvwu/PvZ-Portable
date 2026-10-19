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
