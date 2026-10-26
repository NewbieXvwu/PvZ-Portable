# 任务：用沙箱里的引擎把《植物大战僵尸》的指定关卡打通，交回操作序列

你拿到的是一个无窗口的《植物大战僵尸》1.2.0.1073 复刻引擎，以及一层 Python 接口。
你的产物是**每关一份操作序列存档**：在什么 tick、用什么卡、种在哪一格。存档会被拿回我们的
机器上用同一个引擎重放校验，所以它必须是引擎真正接受的动作。

可以并行开 3 个子代理。中途用户可能粘贴补充信息，照着调整。

**这些对局会被当成教学用的中盘局面，交给另一个玩家接着打。我们要的是一个强玩家会觉得‘稳’的局面：火力和经济留有余量，不靠恰好知道下一波来什么。**

---

## 开始

1. 将 `llmplay_pkg.zip`（接口、引擎、配置）和 `llmplay_res.zip`（资源与预编译缓存）
   放在你选择的同一个目录，只解压接口包，进入生成的 `llmplay_pkg/`。
2. 资源 zip 保持原样放在接口目录旁边。首次 `import llmplay`（自检也会导入）自动安装
   资源和缓存；两个文件必须来自同一批。资源 zip 在其它位置时，可设置
   `PVZ_LLMPLAY_RES_ZIP` 为它的实际路径。
3. 先执行 `python3 selfcheck.py`，最后一行必须是 `SELFCHECK OK`。
   **如果你的环境 Python 版本 < 3.10 或没有 x86_64 Linux，先运行 `selfcheck.py`，
   按它的报错说明处理。**
4. 导入接口并查看实际路径：

```python
import llmplay
llmplay.paths()
print(llmplay.out_dir())
```

接口按以下顺序选择交付目录：`LLMPLAY_OUT` 环境变量；否则使用存在且可写的默认沙箱目录；
否则使用包目录旁的 `out/`。以 `paths()` 的实际输出为准，所有交付路径用 `out_dir()` 拼接。

## 如果这是续跑

用户会附上上次的 `out.zip`。先导入，再继续做：

```python
import llmplay
llmplay.resume(llmplay.out_dir().parent / "previous_out.zip")  # 换成旧 zip 或目录的实际路径
print(llmplay.progress())
print(llmplay.next_task())
```

resume 默认快速导入：摘要匹配且原进度有效的存档直接恢复，通常几秒完成，无需 full；其余存档自动复验，搜索预算和同段重写次数延续。
`next_task()` 返回下一项的 `task_id`、`seed`、`layer`、`action` 与已有 `archive` 路径；
`list_tasks()` 只列未达标任务及尚未稳健完成的种子。需要接着中盘打时，`Game.load_archive(path)`
可恢复已导出的操作日志，策略代码也会从旧交付恢复。

## 1. 目标

从开局打到通关。任务分成五层，**按顺序推进**；前一层里每个任务至少拿到 5 个种子通关
（其中至少 3 个达到第 4 节的稳健标准）之后，才进入下一层。

| 层 | 内容 | 任务 id（见 `configs/human_demo_tasks.json`） |
|---|---|---|
| **A** | 屋顶 L46–L49，毁灭菇套，2.0 倍 | `demo_roof_46_doom_m200`、`demo_roof_47_doom_m200`、`demo_roof_48_doom_m200`、`demo_roof_49_doom_m200` |
| **B** | 屋顶 L42–L44 的 melon / gloom / gloommelon 变体 + L41，2.0 倍 | `demo_roof_42_{melon,gloom,gloommelon}_m200`、`demo_roof_43_{melon,gloom}_m200`、`demo_roof_44_{melon,gloom}_m200`、`demo_roof_41_doom_m200` |
| **C** | 屋顶 L46–L49，毁灭菇套，**3.0 / 4.0 倍** | `demo_roof_{46,47,48,49}_doom_m300`、`demo_roof_{46,47,48,49}_doom_m400` |
| **D** | 雾夜 L36–L39，**2.0 倍** | `fog79_l36_snow_{magnet,split}_v2_m200`、`fog79_l37_snow_{magnet,split}_v2_m200`、`fog79_l38_snow_{magnet,tall}_v2_m200`、`fog79_l39_snow_{cactus,blover}_v2_m200` |
| **E** | 原生传送带 L30 / L40 / L50（僵王博士） | `demo_pool_30_orig_m100`、`demo_storm_40_orig_m100`、`demo_boss_50_orig_m100` |

- 每层里按表格顺序打；一层做完立刻进入下一层。
- 只打清单里**存在**的任务。清单里没有的关卡不要自己造任务，跳过即可。
- 3 个子代理可以各拿**同一层里的不同任务**并行（例如 A 层：L49 / L48 / L47→L46）。
  同一任务最多 2 个游戏进程。
- 每个任务有 **40 个种子**（`configs/extra_seeds.json`，与 `configs/llmplay_seeds.json`
  是同一个块）。每次新运行生成随机 `run_id`，任务内种子顺序按它散列排序；
  `next_task()` / `list_tasks()` 和不指定 seed 的 `Game(task_id)` 都遵循该顺序。
  `resume` 沿用原 `run_id`，继续原顺序。用 `Game(task_id, seed=...)` 也可明确指定。
  `progress.json` 和存档 meta 都记录运行ID；meta 另有 CPU 型号、核数和 Python 版本。
- **停止条件只有一个：全部五层、每个任务都达到完成标准（至少 5 个种子通关，其中至少 3 个稳健）。没做完就一直做，不要主动收尾，不要写「任务完成」。**
- `next_task()` 只在全部任务达标后返回 `None`；已完成任务自动跳过。

## 2. 包里有什么

```
llmplay_pkg/
  PROMPT.md              ← 本文件
  README.md              ← 上手说明
  selfcheck.py           ← 离线自检，先跑它
  build/pvz-portable     ← 引擎（x86_64 Linux，静态链接）
  configs/llmplay_package.json  ← 两包配对与引擎指纹
  configs/human_demo_tasks.json  ← 任务（关卡 / 种子 / 卡组 / 倍率）
  configs/extra_seeds.json       ← 每个任务 40 个种子
  configs/llmplay_seeds.json     ← 五层任务清单 + 种子块（层号在这里）
  configs/llmplay_robustness.json ← 稳健性检查点的未来种子（第 4 节用）
  llmplay/               ← Python 接口（纯标准库）
  python/ scripts/       ← 接口依赖的模块（纯标准库，原样拷贝）
  examples/play.py       ← 示例脚本
```

资源自动安装到 `llmplay_pkg/res/`，缓存安装到会话持久 app-data。
主局、分支和稳健性试打共用缓存。缺资源包或版本不匹配时，接口会明确报错。

## 3. 接口速查

一局 = **(任务, 操作序列)**。所有动作都记在操作日志上；`save()` 把它写到磁盘。
工具调用之间进程可能被回收，**磁盘上的日志才是唯一的真相**。

```python
import llmplay
from llmplay import Game, list_tasks

print(list_tasks())                       # 有哪些任务

g = Game("demo_roof_49_doom_m200")        # 或 Game(task_id, seed=...)
print(g.observe())                        # 完整文字局面（默认就是完整版）
print(g.legal())                          # 当前合法动作：每个卡槽能种哪些格 + 能铲哪些格
print(g.summary())                        # 波 / 最远存活波 / 剩余小推车 / 阳光 / 各行威胁
print(g.timeline())                       # 每波一行的历史（回头决定 rewind 到哪）

g.plant(card, row, col)                   # card 是 observe() 里的卡槽序号 [0] [1] ...
g.shovel(row, col)
g.wait(300)                               # 推进 300 tick（100 tick ≈ 1 秒游戏时间）
g.wait_until("wave:8")                    # 也可以 wait_until(12345) 或 wait_until(lambda g: ...)
g.wait_until(lambda game: game.obs["sun"] >= 300)

g.rewind(4200)                            # 回到 tick 4200，之后的操作覆盖原来的后续
h = g.branch()                            # 复制一份独立的局（搜索用）
g.save(llmplay.out_dir() / "run.json")          # 存操作日志
g2 = Game.load(llmplay.out_dir() / "run.json")  # 读回来（会从开局重放到当时的位置）
g.export_archive(llmplay.out_dir() / "l49_seed_example")   # 导出可校验存档

g.robustness(policy, k=8, checkpoints=[0, 10, 20, 30])   # 第 4 节：换未来还能不能赢
```

### 3.1 observe() 给什么

每次输出的**第一行是状态行**，例如：

```
[已用 1:12 | 本关 L49 模式=手动 | 搜索预算 0% | 最远 23/40 波 | 本段重写 3/8]
```

它由接口从磁盘上的 `~/.llmplay_session/budget.json` 现算（换工作目录也不会重置），已用时间只作参考，不会封锁任何入口。后面依次是：

1. **局面头**：关卡、第几波 / 总波数、当前 tick、距下一波约多少 tick、阳光、
   每行小推车是否还在、本关会出哪些僵尸。
2. **空间网格**：行 × 9 列，每格固定宽度。植物写成短代码 + 血量百分比
   （`Mel85` = 西瓜剩 85% 血），僵尸写成类型短代码 + 只数（`Gar1` = 1 只巨人，
   `Gar1 Imp2` = 1 只巨人 + 2 只小鬼；格内按从左到右的威胁顺序，放不下时截断并标
   `+n`，表示还有 n 只没显示），还没走进棋盘的僵尸单列一栏「场外来袭」。
   `..` = 空地，`~~` = 水路。短代码表在第一次 `observe()` 的图例里，也可以用
   `legend()` 随时打印。
3. **逐行明细**：每行的植物（列、类型、血量/上限、状态：睡着/正常/待装填/出土中…）
   与僵尸（类型、列号保留一位小数、本体血量 + 护具/盾血量、减速/冰冻/黄油、
   梯子是否已搭、蹦极目标格、投石车出手计数、巨人是否已扔小鬼），
   以及最前面那只僵尸大约多少 tick 到下一个植物列、多少 tick 到小推车。
4. **卡片**：每张卡的费用、冷却剩余 tick、此刻买不买得起、能种在哪些格（数量 + 示例）。
5. **事件**：自上次 `observe()` 以来发生了什么 —— 种下、失去（被吃/被偷/被砸/自毁）、
   击杀（按类型计数）、新波开始、小推车触发。
6. 末尾附每波一行的历史：波号、tick、阳光、植物数、击杀、失去、小推车；也可以单独调用 `timeline()`。

信息量以「人类玩家此刻在屏幕上能看到的东西」为上限：血量、位置、状态都给；
**不给**未来的出怪表和随机数状态。精确数值以文字为准（本会话没有截图）。

### 3.2 一次调用跑多步

写 Python 脚本，一次跑几十到几百步。**脚本结束时（进程退出）接口会自动打印一次完整的
`observe()`**，你不用记得自己调用。想在某个条件成立后再行动，就用
`wait_until(lambda game: ...)` 或直接写 `if`。

要点：

- `plant` 的 `card` 是**卡槽序号**，不是植物编号（`observe()` 的卡片行与 `legal()` 用同一个序号）。
- `plant` / `shovel` 返回 `True` 表示引擎接受了这个动作，`False` 表示被拒绝（没花阳光、没占格）。
- 动作时间轴完全自由：**任意 tick 都能做动作**，不需要等任何决策点。
- 阳光自动收集（和人类示范一致），你不需要点阳光。
- `export_archive` 写 `ticks.jsonl` / `frames.jsonl` / `meta.json`，`label_source` 是
  `llm_agent`。**每通关一关就导出一份。**
- `Game.load()` 是惰性的：第一次 `observe()` / `summary()` / `wait()` 时才会从开局重放到存档的
  tick。在那之前 `game.obs` 是 `None`，先调一次 `observe()` 再读它。
- 引擎的可执行位由接口自己补（解压工具可能不保留），不需要你 `chmod`。

## 4. 稳健性：换个未来还能不能赢

只求通关的局会靠读档得来的「先知」在最后一刻救场（例如看准时机放樱桃），局面很脆。
我们的衡量办法**不是**战损公式，而是胜率本身：**换一个同样合理的未来，这套打法还能不能赢。**

引擎有 `RESEED`（`PvzEnv.reseed`）：从当前状态重抽所有**还没开始**的波次与局内随机。
接口把它包成 `robustness()`：

```python
def policy(game):                 # 闭环打法：每一步只看当前局面
    if game.obs["sun"] >= 100 and game.legal()["plants"]:
        first = game.legal()["plants"][0]
        if first["cells"]:
            row, col = first["cells"][0]
            game.plant(first["packet"], row, col)
            return
    game.wait(30)

report = g.robustness(policy, k=8, checkpoints=[0, 10, 20, 30])
print(report)
# {'policy_sha256': '...', 'k': 8, 'passed': True, 'manual_patch': False,
#  'checkpoints': {'0': {'wins': 6, 'k': 8, 'reached': True},
#                  '10': {'wins': 7, 'k': 8, 'reached': True}, ...}}
```

- `policy` 必须是**闭环函数**（每一步按当前局面决定动作）。固定的动作序列在新未来里没有意义。
- 将 `policy` 写入 `.py` 文件再导入，接口会对实际源码取哈希；无法读取源码时会拒绝评估。`k<8` 可用于试跑，正式通过固定要求 8 个未来。
- 检查点取「开局 / 第 10 / 20 / 30 波开始时」，**只算你已经打到的那些**；没打到的返回 `reached: False`。
- 连续推进跨过波次时，接口重放定位该波的首个 tick；结果中的 `tick` 是实际检查点，定位的模拟量也计入预算。
- 种子来自包里独立的稳健性块（`configs/llmplay_robustness.json`），与任务种子不重叠。
- **调用计入搜索预算**（重放和试打都是模拟 tick），别在没必要的局上反复调。
- 结果会记在 `game.robustness_report`，`export_archive()` 自动写进 `meta.json` 的
  `robustness` 字段。

**一局算真正完成，需要：通关，并且 `robustness()` 在第 10 波检查点上 8 个新未来至少赢 5 个。
只通关、不够稳的局仍然要交，但只算半个。**

**通关之后，用剩余预算提高稳健性，再进入下一个种子。尽量用不知道未来也会这样做的打法，
少用 rewind 补救；补救局要在 report 里写明。**

用 `rewind` 手工补救过的局照样可以导出，但接口会自动在 meta 里标 `manual_patch: true`；
分支保留这个标记。`rewind` 会清除旧稳健性结果，请用补救前的闭环打法重新评估改过的局面。

战损、丢车、被吃株数这些数，只在 report 里描述，**不进入任何目标或排序**。

## 5. 搜索与手动：止损由接口执行，不靠你掐表

你可以写任何脚本（穷举、贪心、束搜索、随机重启……）来挑动作。**但预算和止损是接口
自己数的，写在磁盘上，你绕不过，也不需要记住数字。** 你要做的是读懂它给的三样东西：

- **状态行**（每次输出第一行）：已用墙钟 / 本关模式 / 搜索预算用了多少 / 最远存活波 /
  本段重写次数。
- **异常消息**：搜索预算用尽、连续两个检查点最远波没有提升、同一段 rewind 次数用完、
  这些情况下对应的搜索或重写入口会直接抛异常，异常消息里就是
  **下一步该怎么做**。
- **检查点**：接口每用掉一段搜索预算就记一次「最远存活波」，不需要你记。

规则（接口会强制执行，这里只是让你知道它在做什么）：

- 搜索类入口是 `branch()`（以及同时开很多分支）。搜索被判定不佳或用尽后，这一关的
  `branch()` 会抛异常，提示改用**按波手动推进**：`observe()` 读当前局面 → 写一段覆盖
  接下来 1–3 波的脚本 → 执行 → 看结果 → 不满意就 `rewind()` 重写这一段。
- 手动模式下允许**局部**小搜索（同一段少量变体），但同一段 `rewind()` 次数有上限；
  超了会拒绝执行并提示「退回更早一段」。
- 搜索额度为固定模拟 tick 数，不受机器速度影响；没有会话或单关总时间上限。
  搜索止损后继续按波手动推进，直到完成。

## 6. 纪律（会被机器检查）

- **不改引擎，不碰内存与状态。** 所有影响只能通过上面的 API 动作发生。
  不要 patch 二进制、不要改 `main.pak`、不要伪造存档里的哈希。
  交回后我们会在原版引擎上重放；改过引擎或伪造过的结果一律作废。
- 不要试图读别的进程内存、不要改 `ticks.jsonl`。存档里的每一行都会和重放结果比对
  （动作是否被接受、每个操作结束的 tick、终局）。
- 存档里的 `state_hash` 只是诊断信息：它跨进程本来就不可复现，我们不会拿它当门槛。
- 同一关的多个种子**不要只用同一组参数一路通吃**：每通关一个种子就记一份参数
  （主力植物、布阵、节奏），鼓励换打法（不同的主力植物、不同的布阵），在 report 里分开统计。

## 7. 持续交付

交付文件由接口自动维护，随时都是最新的，被中断也不丢。每次 `export_archive()` 或
`robustness()` 完成，接口都会更新 `llmplay.out_dir()` 中的：

- `progress.json`：每任务的种子、结果、稳健性、存档路径、`manual_patch` 和各层完成度；
- `report.md`：从进度自动生成的逐局结果和打法说明；
- `budget.json`：续跑时延续的搜索与重写计数；
- `out.zip`：含存档、进度、报告、预算及 out 下的策略代码。先写临时文件再原子发布，
  失败时上一份交付完整保留。

每通关一局立刻 `export_archive()`；稳健性检查完成会自动导出并更新交付。
不要自行覆盖这四个自动文件。每种打法的参数、布阵和差异写进 `Game(..., note=...)`
或 `export_archive(..., note=...)`；报告会自动列出这些说明和手工补救标记。
将策略和搜索脚本持续保存到 `llmplay.out_dir()`，它们会随下一次接口交付一起打包。
战损、丢车、被吃株数可以写进 note，只用于描述，不作为目标或排序。

系统中断后，用户可下载任意一份 `out.zip` 并在下一次请求附上；按开头的续跑步骤继续。
只有 `progress()["complete"]` 为真、五层全部达到标准后，才能报告任务完成。
