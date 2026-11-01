# task209：原版窗口选卡与人类录制

分支 `userpick`，基于 `origin/deploy26@adf0bcc6`。选卡使用原版 `ChooseSeedsOnCurrentLevel` / `SeedChooserScreen`，二周目、十槽、全部升级植物所有权，模仿者禁用。玩家确认后返回槽位顺序的真实卡组，沿用现有 reset、Recorder、npz 帧读写和自动 verify。用户的卡组按原版界面的确认结果记录。

选卡过程的过场耗时不应进入可重放的战斗状态。确认卡组后调用同一个 `EnvironmentReset` 的固定卡组路径生成初始局面，不复制开局收尾。存档保存 `loadout_mode=fixed`，重放直接用记录卡组。

Mac 仅单任务低并发构建和短录制自测。北极星及 bench_handoff 的冻结进程保持原状。

## 命令与结果

已构建窗口版引擎：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/userpick/build/pvz-portable`。`-env-visible` 在读取设置后强制窗口模式。以下录制命令会打开交互窗口，由用户准备玩游戏时亲自运行。本次不再启动窗口自测。

构建一次（本机已构建完成，更新代码后可再执行）：

```bash
cd /Users/newbiexvwu/PvZAgent/.claude/worktrees/userpick
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
nice -n 10 cmake --build build --target pvz-portable --parallel 1
```

每局一条命令，下面录制第7关，种子默认随机：

```bash
/Users/newbiexvwu/.local/share/mise/installs/python/3.14.6/bin/python3 /Users/newbiexvwu/PvZAgent/.claude/worktrees/userpick/scripts/human_demo.py --resource-dir /Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN --executable /Users/newbiexvwu/PvZAgent/.claude/worktrees/userpick/build/pvz-portable record --level 7
```

换关只改 `--level`。可追加 `--seed 209 --multiplier 1.0 --fog human_v2`；雾关默认 `human_v2`，第35关按原版夜间地形判断。资源目录也可改成 `/Users/newbiexvwu/.cache/pvz-research-resources`。旧 `record --task ...` 用法继续保留。

在原版界面选满10张卡，点击“开始战斗”，随后按现有录制方式操作。原版卡组警告由玩家自行确认，脚本保存实际卡槽顺序。Esc／空格暂停与继续，右键取消当前战斗选卡，关闭游戏窗口结束录制并自动 verify。模仿者不开放。原版没有选卡界面的传送带、砸罐子等关卡由 `ChooseSeedsOnCurrentLevel` 拒绝此入口，保留原有清单录制通路。

存档在：

```text
/Users/newbiexvwu/PvZAgent/.claude/worktrees/userpick/artifacts/human_demo/raw/seed_<种子>/user_l<关卡>_<排序卡组12位哈希>/
  meta.json
  ticks.jsonl
  frames.npz
```

终端会打印实际存档路径。meta 的 `label_source=human`，卡组按槽位顺序保存，`environment_task.loadout_mode=fixed`，verify／重放使用记录卡组并直接开局。

自测结果：相关 `-k` 用例4项通过，含14个子用例。按用户最新要求，仅用脚本模拟选卡返回并运行真实无窗口录制与 verify：卡组 `[7,0,1,2,3,4,5,6,17,27]`，任务名 `user_l7_945d6ccc9da3`，初始 tick=0，1次成功种植加3次单步等待，4个操作、3帧 npz，verify=True。临时数据和脚本已删除。

窗口中的实际选卡点击流程未验证，留给用户首次实际运行时验收。没有恢复、重启或终止被冻结的北极星及 bench_handoff 进程。
