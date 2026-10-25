# task80：录示范时的"操作暂停"——Esc 暂停游戏时间，但照样能种植物

> 主控 2026-10-25 写。执行者：Mac CodeBuddy（`cb-fog` 会话）。工作树自建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demopause`（分支 `demopause`，从 origin/pvz-env 拉出）。

## 为什么
用户要打 2.0 倍屋顶 L49 这类高压关。用户用触控板操作，速度跟不上，需要能暂停下来慢慢操作。

现状（`src/main.cpp` 约 340–361 行，`scripts/human_demo.py` record 循环）：
- 空格切换 `paused`，但暂停时鼠标点击被丢弃（`if (event.type != SDL_MOUSEBUTTONDOWN || paused) continue;`）；
- Esc 是结束录制。

## 要做的
1. **Esc 改为"操作暂停"开关**：
   - 暂停时游戏时间不走（录制循环不发 wait），但种植、铲子、收集照常接受。动作在当前 tick 执行，执行后棋盘立刻刷新显示。
   - 不弹原生暂停菜单，画面上不挡任何东西。可以在角落用很小的字或边框提示"已暂停"，不遮挡格子。
   - 再按一次 Esc 继续。
2. **空格**保持原来的暂停行为，或者改成与 Esc 相同，二选一并写清理由。**结束录制**改为关窗口，提示文字同步改。
3. **录制语义不变**：
   - 存档只记动作与 tick，暂停期间的多个动作就是同一 tick 上的连续动作；
   - `verify` / `rebaseline` / `export` / 示范起点恢复全部照常；
   - 旧存档逐位可重放。
4. **测试**：用 `human_demo_autoplay.py` 的办法，写一个有窗口的短局回归：暂停、种两株、继续、结束，然后 verify 通过。再加上原有测试。
5. 改动说明写进本文件"结果"，提交并 `git push origin demopause`。主控合并后告诉用户。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`（有窗口的测试除外）。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行。

## 结果

## 问题
