# 人类示范：打一局，留下可重放的打法

初版计划共十局，交主控审阅。按自己的自然打法争取赢，一局保持一种思路。每局先按 **10–20 分钟**预留，实际时长取决于推进速度。所有任务为二周目、原速、1.0 倍僵尸量，同关各卡组共用种子。

## 开始和结束

在终端粘贴这一行即可打开游戏。把 `--task` 后的名称换成计划表里的任务名，`--note` 后可写本局想示范的一句话。

```bash
python3 /Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/scripts/human_demo.py record --task demo_fog_31_fume --note '这局用大喷菇配合窝瓜处理小丑'
```

鼠标左键选卡、种植、使用铲子，右键取消选卡，空格暂停，Esc 结束并保存。阳光按训练环境的既有口径自动收集，鼠标收集操作仍会录入。窗口关闭后会自动无头重放，只有显示 `verified: true` 的局才能转成训练数据。胜负与中途结束均会保存。

原始存档位于 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/human_demo/raw/seed_20261031/demo_fog_31_fume/`，关卡和任务名随所选任务变化。同名存档不会覆盖，重打时可用 `--out` 指定另一个绝对路径。卡组固定来自已登记任务族，变更卡组需要另建族名，保证历史局可比较。

## 十局示范计划

下列僵尸名单来自本引擎：L31 有小丑，L32 加入橄榄球，L33 出现气球，L34 为气球与海豚。各关同时有普通和路障僵尸。

| 任务名 | 卡组 | 本局想示范的思路 |
|---|---|---|
| `demo_fog_31_fume` | 阳光菇 / 小喷菇 / 大喷菇 / 睡莲 / 高坚果 / 窝瓜 / 缠绕海草 / 樱桃炸弹 / 坚果 / 路灯花 | 小丑：大喷菇建线，配合单点处理 |
| `demo_fog_31_burst` | 阳光菇 / 小喷菇 / 大喷菇 / 睡莲 / 高坚果 / 窝瓜 / 缠绕海草 / 樱桃炸弹 / 火爆辣椒 / 三叶草 | 小丑：展示爆炸植物的使用时机 |
| `demo_fog_32_magnet` | 阳光菇 / 小喷菇 / 大喷菇 / 睡莲 / 高坚果 / 窝瓜 / 缠绕海草 / 樱桃炸弹 / 磁力菇 / 坚果 | 橄榄球：磁力菇配合主火力 |
| `demo_fog_32_freeze` | 阳光菇 / 小喷菇 / 大喷菇 / 睡莲 / 高坚果 / 窝瓜 / 缠绕海草 / 樱桃炸弹 / 寒冰菇 / 火爆辣椒 | 橄榄球：控制与爆炸植物配合 |
| `demo_fog_33_cactus` | 阳光菇 / 小喷菇 / 大喷菇 / 睡莲 / 高坚果 / 窝瓜 / 缠绕海草 / 樱桃炸弹 / 仙人掌 / 路灯花 | 气球：仙人掌 |
| `demo_fog_33_blover` | 阳光菇 / 小喷菇 / 大喷菇 / 睡莲 / 高坚果 / 窝瓜 / 缠绕海草 / 樱桃炸弹 / 三叶草 / 双发射手 | 气球：三叶草 |
| `demo_fog_33_cattail` | 阳光菇 / 小喷菇 / 睡莲 / 香蒲 / 大喷菇 / 磁力菇 / 坚果 / 窝瓜 / 缠绕海草 / 路灯花 | 气球：香蒲 |
| `demo_fog_34_cactus` | 与 L33 仙人掌组相同 | 气球：仙人掌，兼顾海豚水路 |
| `demo_fog_34_blover` | 与 L33 三叶草组相同 | 气球：三叶草，兼顾海豚水路 |
| `demo_fog_34_cattail` | L33 香蒲组中的坚果换成高坚果 | 香蒲配合水路防守 |

这些思路用于区分示范局，实战操作由玩家决定。引擎已按对应玩家进度查询可用植物并做整组检查，香蒲的升级所有权随任务声明。备注写入每局 `meta.json` 的 `note`，用于分析，训练输入中没有备注文字。

## 数据交付

录制目录含 `meta.json`、决策帧 `frames.jsonl` 和逐 tick 动作带 `ticks.jsonl`。转换示例：

```bash
python3 /Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/scripts/human_demo.py export --archive /Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/human_demo/raw/seed_20261031/demo_fog_31_fume
```

转换结果在同工作树 `artifacts/human_demo/frames_v2/`，标签来源为 `human`。原始局与 npz 均走 HF 的 `transfer/`，代码仓库只保存工具和任务清单。

维护入口：`episode_query.py whatif --archive <原始局绝对路径> --decision N --try wait:60` 可替换一步，后续继续原动作带。`search_play.py` 的既有老师命令追加 `--demo-archive <原始局绝对路径> --demo-decision N --workers 1`，会恢复到第 N 步之前，再由老师接管。任务清单使用 `configs/human_demo_tasks.json`，任务名与种子须与示范一致，学生模型须支持该卡组。每次搜索加入后续示范中当前可执行的一个种植候选，再由推演评分。训练配方的 `train.source_weight` 可设为 `{"human":0.2,"teacher":1.0}`，与已有标签权重相乘。
