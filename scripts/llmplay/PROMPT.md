# LLM Play 任务书（task267：238j）

你拿到的是无窗口的《植物大战僵尸》1.2.0.1073 复刻引擎及 Python 接口。
每局产物是一份真实操作序列存档：什么 tick、用什么卡、种在哪一格、注入了什么。

## 0. 这个任务没有终点

**任何阶段都不交班、不请示、不盘点，完成一局就立即 `next_task()` 领下一项。**
`progress()["complete"]` 永远是 `False`，这不是 bug。唯一停止条件是用户在对话里说停。

238j 按配置依次派发：回退子项 → 真实必刷题 drill → r160 八个重点格的同卡组败局 resume →
L30/L40/L45/L50 原生整局 native → 剩余 scenario → 其余旧 resume → 前沿倍率 → 其余原生支线 → 主线。
drill 从真实危急局面起手，scenario 从真实底局注入单一威胁，均由模拟器判分。
重点 resume 保持原局卡组与倍率，native 由引擎发牌。按 `next_task()` 返回的项推进。
最新交回进度已经放在包内 carry，首次导入自动接上。

## 1. 解包与输出目录

包内 `build/pvz-portable` 是已编译引擎，`python/`、`scripts/`、`llmplay/` 是接口，
`drill/`、`scenario/` 是题目存档，`configs/` 是题库。`llmplay_res.zip` 原样放在接口目录旁，
首次 `import llmplay` 自动安装资源和缓存。不需要联网、编译或自行运行自检。

```python
import os
os.environ["LLMPLAY_OUT"] = "/mnt/data/out"
import llmplay
item = llmplay.next_task()["item"]
print(item)
```

默认只开 1 片（不设 `LLMPLAY_SHARDS`）；一个会话覆盖全部地形。确实要多开时再显式分片，
每个会话固定自己的片号，不要中途改。

## 2. drill：真实必刷题

每个 drill 项 = 一份**存档前缀**（reset 后重放到起点决策的 tick 日志 + 起点帧）
+ 未来种子 + 视界 + 目标僵尸 id + 知识点标签。`item` 里直接带这些字段：

- `item["path"]`：存档前缀目录，用 `Game.load_archive()` 打开。
- `item["future_seed"]`：这一局要用的 RESEED 种子；同一个 drill 会按种子拆成多项。
- `item["horizon"]`：短视界（tick）。从起点算，不是从开局算。
- `item["targets"]`：要清掉的目标僵尸 id。
- `item["knowledge"]`：这道题考的知识点。
- `item["priority"]`：0 = 生产老师 0/3 全错（最优先），1 = 中间题，2 = 全对，3 = 未校准。

```python
import llmplay
from llmplay import Game

item = llmplay.next_task()["item"]
if item["layer"] == "drill":
    g = Game.load_archive(item["path"], strategy=item["knowledge"])
    g.reseed(item["future_seed"])          # 换一条同样合理的未来，不改当前棋盘
    print(g.observe())                     # 先看局面，再决定种什么
    # ... 用 plant / shovel / wait 解题 ...
    g.wait_until(g.tick + item["horizon"]) # 至少推到视界
    print(g.drill_result())                # 机械判据：目标清空、无割草机、无破屋
    g.export_archive(llmplay.out_dir() / ("drill_" + item["drill_id"]))
    g.close()
```

规则：

- 卡组、倍率、关卡、原始种子都照存档，**不能换卡**。
- 起点是残局，不要 `rewind()` 回开局；`rewind()` 会把 `manual_patch` 写进 meta。
- 首次加载若抛出 `llmplay.InvalidDrillError`，接口已把起点无目标的题记为无效并跳过其全部种子。
  关闭该 `g`，直接 `next_task()` 换下一题，不需要推进或导出。这项检查只读已加载的棋盘。
- `reason` / `strategy` 写一句“考的是什么、怎么应对”，例如
  “考小丑贴脸：先用窝瓜顶掉盒子，再补前排”。这是给训练和错题分析用的，不是判据。
- 判据只看模拟器：目标僵尸 id 全部离场、**或**进入引擎的死亡类阶段（DYING/BURNED/MOWERED，
  以 C++ `Zombie::IsDeadOrDying()` 为准），且没有割草机触发、没有僵尸进屋。
  死亡动画期间目标仍可能 `on_board=True`，这时已经算解决；不要为了等它消失而拖过视界。
  不要用“我觉得守住了”代替 `g.drill_result()`。
- 失败也有价值：照样导出。导入时失败局只记录、不进示范行；成功局带 `meta["drill"]`，
  训练时按 drill 行加权。
- 同一道题至少换一个 `future_seed` 再打一局（队列会把同题的下一个种子派给你），
  避免一次性运气。

## 3. scenario：GPT 自己出题

每个 scenario 项 = 一份真实中盘**底局**（同样是 tick 日志 + 起点帧）+ 地形 +
可用知识点清单 + 两个 future seed。底局没有预置“考题”，威胁由你注入。

```python
item = llmplay.next_task()["item"]
if item["layer"] == "scenario":
    g = Game.load_archive(item["path"])
    g.reseed(item["future_seed"])
    print(g.observe())
    # 只注入一个威胁：一只对策型僵尸；必要时可用 set_sun / set_cooldown 调整资源。
    g.spawn(16, row=2, x=760)              # 例：气球
    target = g.last_spawn_id               # 刚注入的那一只；也可用 g.zombies() 查
    g.declare_scenario("气球：三叶草/仙人掌/香蒲", horizon=1200, targets=[target])
    # declare_scenario 会顺手跑一次“什么都不做”的对照局；对照成功会被拒绝导出。
    # ... 用 plant / shovel / wait 解题 ...
    g.wait_until(g.tick + 1200)
    g.export_archive(llmplay.out_dir() / ("scenario_" + item["scenario_id"]))
    g.close()
```

规则：

- **一道题只考一个知识点**。不要同时注入气球+矿工+小丑；不要用 `set_wave` 把整波
  出怪表改掉来制造“多威胁”。
- 难度要真：不做任何事应当失败。`declare_scenario()` 自动跑的对照局若也成功，
  导出会被拒绝；换更强的威胁、换行、换视界，或换一道底局。
- 从 `item["knowledge_candidates"]` 里轮换选，不要反复只出同一类；同一个底局
  可以出不同知识点，但每局只留一个。
- 注入操作是普通 operation（`spawn` / `set_sun` / `set_cooldown`），会写进 ticks，
  导入时按同一路径重放；它们本身不是示范动作，不会进训练行。
- 冒险模式的巨人（23）和红眼巨人（32）只能注入第 1 行及以下，屋顶用 r1–r4。
  `spawn()` 返回 `False` 表示该次注入被引擎拒绝，改用合法行，成功后再取 `last_spawn_id` 声明目标。
- `reason` 写“考的是什么、怎么应对”；`declare_scenario` 的 `knowledge` 用同一个标签。
- 同一道题至少再试一个不同 future seed（队列会给同底局的第二个种子）。

### 3.1 知识点清单（对策型僵尸 × 地形）

按地形轮换，优先补本局地形还没考过的类型。地形由 `item["terrain"]` 给出。

| 僵尸 | type | 白天 | 夜间 | 泳池 | 雾 | 屋顶 |
|---|---:|---|---|---|---|---|
| 撑杆 | 3 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 橄榄球 | 7 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 鸭子泳圈 | 10 |  |  | ✓ | ✓ |  |
| 潜水 | 11 |  |  | ✓ | ✓ |  |
| 冰车 | 12 | ✓ | ✓ | ✓ | ✓ |  |
| 雪橇 | 13 |  |  |  | ✓ |  |
| 海豚 | 14 |  |  | ✓ | ✓ |  |
| 小丑 | 15 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 气球 | 16 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 矿工 | 17 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 跳跳 | 18 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 梯子 | 21 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 巨人 | 23 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 蹦极 | 20 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 投石车 | 22 |  |  |  |  | ✓ |

知识点标签示例（用一句“考什么 + 怎么应对”，不要写整套固定打法）：

- 气球：三叶草/仙人掌/香蒲覆盖，别让它越过前排。
- 小丑：磁力菇吸盒或窝瓜秒杀，别让它贴脸。
- 矿工：后排也要有火力/磁力菇，不能只守前排。
- 海豚：水路前段火力与睡莲覆盖。
- 橄榄球：高坚果/地刺/寒冰拖住，别让它贴脸。
- 跳跳：高坚果/地刺/磁力菇，别让它连续跳过。
- 冰车：地刺/窝瓜，冰道会加速。
- 投石车：后排输出，注意它砸后排。
- 蹦极：关键植物被偷前清掉或留冗余。

## 4. resume：学生败局中盘接回

drill 后先派 r160 重点格的 `layer=="resume"`，其余旧接回排在 scenario 之后。存档已截到起点，
`g.observe()` 就是学生当时的局面；卡组、倍率、种子照原局，不能再选卡。

```python
item = llmplay.next_task()["item"]
if item["layer"] == "resume":
    g = Game.load_archive(item["path"], resume_item=item.get("resume_item"),
                          strategy="根据局面补经济，保住下一波")
    print(g.observe())
    # ... 解题 ...
    # 判断当前切点已经救不回时，保存本次尝试并接着打更早的子局：
    g = g.rewind_resume(reason="第3波起只有2株向日葵，切点阳光0")
    # 不指定 to_tick 时默认退到当前切点的一半；目标必须早于切点且 >= 0。
    # g 为 None 表示已到开局或三次回退用尽，已标为结构性待复核，此时领下一题。
    if g is None:
        item = llmplay.next_task()["item"]
    else:
        print(g.observe())                # 当前子代理继续解题
        # ... 打到实际终局，再导出 ...
        g.export_archive(llmplay.out_dir() / ("resume_" + g.resume_id))
        g.close()
```

本包 `carry/` 已经接上 2026-11-11 00:53:23 HKT 的最新交回进度，
**不需要再附旧分卷**；`resume` 进度会从 carry 继续，不会重打已完成的条目。
导出时 `strategy` 必填：写清从这个局面先补什么、靠什么顶哪一波。

救不回时调用 `rewind_resume()`，诊断写一句“从哪一波开始走偏、缺什么”，可建议更早的
`to_tick`。不要因如实记录失败就直接跳过。接口会自动导出本次失败或放弃记录，用原局
前缀截出更早子项，放到队首并交给当前子代理继续。默认取开局与当前切点的中点，
让经济决策的可修改区间每次向前扩展一半。每条链最多回退三次，开局仍救不回或次数
用尽就标为结构性待复核，主控据此检查卡组与倍率。这里的经济诊断由你判断。

回退子项赢局后，仅新切点之后的 GPT 操作进入示范行，分档沿用普通 resume。
原题、回退次数、各次切点和诊断会随存档与进度一起保留。`rewind(tick)` 仍是局内重试，
回退题目请用 `rewind_resume()`。跨进程重领子项时，照示例把 `item.get("resume_item")`
传给 `load_archive()`；直接接住 `rewind_resume()` 返回的局则已完成认领。

## 5. 共用规则

- 先 `observe()`，合法动作以 `g.legal()` 的引擎结果为准；不要背固定种植表。
- `plant(card=...)` 可以传槽位号、英文卡名或中文名。英文名来自 C++ `SeedType` 枚举：
  `repeater` 只指双发射手(7)，另一张 `leftpeater` 才是 52；`observe()` 的卡槽行会打印
  该用的名字。以当前卡槽行为准，传送带换槽也不影响按名字种。
- `observe()` 卡槽行的“冷却剩 N tick”是**剩余**时间 = `refresh_time − 已过`。
  剩 0 的那一帧还不进 `legal()`，**下一帧才可种**；能不能种始终以 `legal()` 为准。
- 卡组 6–10 张、不重复；泳池/雾必须带睡莲(16)，屋顶必须带花盆(33)。
  升级植物只能种在底座上（香蒲→睡莲、忧郁菇→大喷菇、玉米加农炮→玉米投手）。
- 一局只做一件事：drill 解一道题；scenario 只注入一个威胁；resume 只救一个局面。
- 每局导出后立即 `next_task()`；`busy` 表示项被同目录的其他执行进程占用，
  先做自己手上的局，不要轮询等待。
- **约每小时调用一次 `llmplay.export_bundle()`**，把当前 out 打包交回；随后继续打，
  不要因为导出而停下来。`export_bundle()` 只打包已完成/已导出的存档，不重放、不校验。
- 导入由主控在台式机做：重放前缀和注入操作，只把 GPT 自己的 plant/wait 决策记成
  `llm_demo` 行；drill 成功局带 `meta["drill"]`，训练时 ×3；失败局记录但不进示范行。
- 不做全量复验：沙箱里不要逐份重放旧存档，主控导入时会在台式机做。
- 预算按 ≥10 小时设计：drill 2709 项、scenario 600 项，按每题几分钟计足够；
  这不是停止条件，做完继续领。

## 6. 快速参考

```python
import llmplay
llmplay.progress()                 # 当前进度（没有终点）
llmplay.next_task()                # 领下一项，已占位
llmplay.out_dir()                  # 本会话输出目录
llmplay.export_bundle()            # 约每小时一次，打成交回包
llmplay.roster(task_id, seed=...)  # 出怪表（drill/scenario 可跳过）
```

`Game` 常用：`observe()`、`legal()`、`plant()`、`shovel()`、`spawn()`、`set_sun()`、
`set_cooldown()`、`reseed()`、`zombies()`、`declare_scenario()`、`drill_result()`、
`wait()`、`wait_until()`、`rewind_resume()`、`export_archive()`、`close()`。
