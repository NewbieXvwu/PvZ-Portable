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

### 实现（提交 b82c179b）

- **Esc = 操作暂停开关**（`src/main.cpp` `HumanInput::Poll`）：暂停时 poll 的默认动作从
  `wait ticks:1` 改为 `wait ticks:0`，录制循环对 wait 不 step，游戏时间不走；点击处理
  不再被 `paused` 拦截，种植/铲子/收集照常，动作在冻结 tick 执行，多个动作落在同一 tick。
  每次轮询仍走 poll 自带的呈现块，棋盘和光标持续刷新。
- **空格与 Esc 统一**：都切换同一个操作暂停。理由：旧空格暂停（冻结 + 丢弃所有点击）
  严格更差且是陷阱（暂停后点了没反应），录制循环对两个标志的处理本就相同；两种暂停
  语义并存只会让人混淆。
- **结束录制 = 关窗口**（SDL_QUIT / WINDOWEVENT_CLOSE），Esc 不再结束。提示文字改为
  「Esc 或空格暂停/继续（暂停时可以慢慢种、铲、收集），右键取消选卡，关闭窗口结束录制。」
- **暂停提示**走窗口标题（追加「已暂停（可继续操作）」，恢复时还原）。任务书允许「角落
  小字或边框」但不强制；标题零遮挡、不碰绘制管线，比画面叠加安全。
- **`DEMO_INPUT <escape|space|close>`**（main.cpp，仅 visible demo 会话接受）：把合成 SDL
  事件推进真实队列，脚本回归驱动的是人按键走的同一条路径。Python 侧 `PvZEnv.demo_key()`。

### 录制语义（不变）

- 存档仍只记动作与 tick；暂停期不产生任何 tape 条目，暂停期的多个动作就是同一 tick 上的
  连续动作（`Recorder.step` 对同 tick 动作本就不插 wait 帧）。
- 旧存档逐位重放：用新引擎 headless `replay()` 两局示范存档（demo_fog_32_snow_m200、
  demo_fog_34_cactus_snow_m200），逐操作 hash 全对；存档目录 sha256 前后一致（verify 会写
  meta.json，示范存档只读，所以这次用不落盘的 replay 验证）。

### 测试

- 新增 `scripts/human_demo_pause_regress.py`（有窗口，仿 autoplay）：Esc 暂停 → poll 返回
  `wait ticks:0` 且不推进时间 → 暂停中种两株（tape 上两笔同一 tick、时间未动）→ Esc 继续
  → `wait 90` → DEMO_INPUT close 结束 → verify 通过（20261031/demo_fog_31_fume，verified=true）。
- 原有测试：`test_human_demo` / `test_env_protocol` / `test_replay_task_spec` / `test_deck_warnings`
  共 54 passed, 47 subtests passed。

## 问题

- 暂停提示在窗口标题而不是画面角落：任务书的角落提示是「可以」项。若用户实际用下来还是
  想要画面内提示（比如全屏时标题看不见），再加画边框的方案——那要动绘制管线，这次没做。
