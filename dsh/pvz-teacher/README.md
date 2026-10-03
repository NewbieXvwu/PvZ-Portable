# PvZ 教师 × DeepSeek Harness

把 PvZ 对局诊断工具挂进 DSH，让模型自己看败局、提改动、用精确反事实验证、
把结论写成可自演化的 skill。设计依据与分步计划在
[`RESEARCH_EXECUTION.md`](../../RESEARCH_EXECUTION.md) 的
「2026-10-03 LLM 教师方案」一节。

## 这条流水线和 RL 是什么关系（先读这个）

**它不产生行为，也不产生训练数据。** 那条路（教师蒸馏）已经判死，
见 `DESIGN.md` §5 与 `TODO.md` §1.2：策略由 RL 产生，行为来源有且只有 RL，
教师必须能被学生超越。本流水线产出的 skill 也**不允许**进奖励函数——
LLM 判断不当地写进奖励会直接污染优化目标。

它合法的产出是**判定与场景，不是行为**：

1. **失败模式判据（skill 的「判据」节）** → 变成 RL rollout 的**评估探针**。
   skill 判据全部是"从路的状态直接算出来"的可计算检查（例：火力=0 且
   僵尸压到前列时，策略是把阳光换成火力还是继续塞生产者？）。对 RL 策略
   的对局跑同一检查，就能回答"训练后的策略有没有同样的失败模式"——
   这比裸胜率细一个层级，且不干预学习。
2. **机制（skill 的「为什么错」节）** → **课程与任务族设计的依据**。
   例如泳池关的裸水路机制指向"需要在水路重武装下做决策"的任务变体。
3. **whatif 反事实** → 校准素材。脚本策略阶段的价值在这里：规则全部
   写在 `DECISION_RULES` 里、环境确定性回放，教师每一条声称都能对
   地面真值核对。S0–S5 证明的是**这条流水线能产出机制级、可证伪的
   结论**（S5 对照 S3/S4 的产出形态变化），不是脚本本身的修法。

**S6（待做）**：把被诊断对象从脚本策略换成 RL 策略的 rollout——加一个
policy 适配器让 `collect()` 支持 `policy=ppo`（加载 evaluated checkpoint
采样动作），其余流程（capture → 诊断 → whatif → skill）原样复用。届时
skill 描述的才是**学习者的失败模式**，评估探针与课程设计才有直接对象。
注意：训练按用户要求暂停（台式机占用），S6 只做推理与诊断，不动训练。


**当前状态：S0–S4 均已通过。** 工具面 12 个
（`ping / vocabulary / constants / policy / capture / index / frame / lane / actions /
whatif / narrative / lint_skills`），与 `episode_query.py` 的子命令一一对应，另发 3 个资源。
S3（2026-10-03）让模型真跑了一局败局诊断：它**找到了能通关的改动**（决策 96 的
土豆雷换列 → 30/30），但**一条 skill 都没写**。
S4（2026-10-03）换了场景（L21 泳池关）重跑，验证 skill 判据与闸门：

- **诊断正确**：模型发现 L21 是结构性输局 —— 卡组 0..5 里没有任何能种水路的植物，
  第 2、3 路全程零火力，割草机一耗尽必输。**"没有单点通关改动"是它自己的结论**
  （4 个决策点 × 13 次 whatif，全部独立复核吻合），不是答不出来。
- **地形字段用上了**：它主动调 `constants --section terrain --level 21`，
  从「场景：」行发现泳池关；S3 里那种"看不见地形"的困惑没有复现。
- **skill 恰好 1 条，不是 0 条也不是一筐**：`pvz-deck-terrain-mismatch`
  （kind: mechanism，三问全过、4 个证据点、falsifier 完整）。它考虑过把
  "孤睡莲反而更差"单独立条，自己否了 —— 判据起了作用。`lint_skills` 2/2 通过。
- **对照 S3**：推理轨迹里 "skill" 从 **0 次 → 208 次**（329k 字符推理）。
  根因确认：S3 的负结果是"没有判据"，不是"没有能力"。

**S4 之后的两处修正（2026-10-03 晚，S5 前置）：**

1. **卡组按关卡地形定**（约束写进 `AGENTS.md` §8）。S4 用的旧卡组
   `(0,1,2,3,4,5)` 在泳池关结构性输局 —— 那种局只能产出一条
   "卡组×地形不匹配"就到头了，产不出可教的政策缺陷。现在
   `scripted_baseline.deck_for_level` 按场景配卡（泳池 +睡莲+香蒲、
   夜间 +阳光菇、屋顶 +花盆），场景判定从 `Board::PickBackground`
   现解析（含第 35 关 ScaryPotter 特例）。实测：L21 五个 seed 从
   结构性卡死 14 波 → 14–19 波（最好 19/20），水路有睡莲+射手。
   香蒲语义也在这次查清：**它是睡莲的升级**（只能种在睡莲上，
   Board.cpp:2849/2866），此前"香蒲可直接种水上"是误读，constants
   与场景行的文案已改。L7 回归验证：`frames.jsonl` 逐字节一致。
2. **任务模板 reframe**（`prompts/diagnose_and_teach.md`）。S4 显示模型
   强烈倾向"单点通关改动"——把操作数压到最小。模板新增「你在产出什么」：
   目标是**教学数据**（策略缺陷模式 + 可迁移判断 + 多点证据），
   单点改动只是探针；产出评估看可迁移判断的质量，不看改动有多小。

**S5（2026-10-03 深夜）：reframe 生效的直接证据。** 同一关（L21/seed 40007）、
新卡组、教学数据导向，中途发现并修复了 6 路渲染 bug（见
`render_episode.py` 01219d1），修完同会话续跑：

- **诊断质量上了一个台阶**：结论是"两个规则的交互缺陷"（经济规则排在
  火力重建之前 × 落点选"生产者最少的路"，恰好把向日葵反复灌进被打成
  0 火力的水路），不是单点补丁。报告明确写"我特意不把 570 的通关
  作为答案本身"——探针/答案的区分被内化了。
- **证据 3 个决策点，全部独立复核吻合**：570 向日葵→豌豆射手 = 通关；
  602 同改 = +1 波；617 换坚果墙 = +1 波。基线动作、后果（向日葵活
  600/540/240 tick）与存档逐项对上。
- **skill 1 条**：`waterlane-naked-rearm`（kind: mechanism），判据
  "火力=0 + 僵尸压到前列 → 重火优先于补生产者"换局可算，反例三条
  （僵尸还远/已有射手扛/生产者≥10）完整。
- **教师模型反过来抓到了工具链 bug**：6 路渲染就是它在诊断中发现
  r5 隐形、lane 工具在 row 5 崩溃后钉死的——工具链被真实使用验证了。
- **环境信息补缺**（72f7649）：读它的思维链发现规则文档列偏好写反、
  规则命中无标注、全盘计数靠手数、x 单位不明四类困惑，全部修复。


## 目录

| 文件 | 作用 |
|---|---|
| `server/pvz_mcp_server.py` | MCP server 本体。stdio，官方 `mcp` SDK 低层 `Server`。**只做搬运**，不做任何游戏判断。工具表是声明式的（参数名→CLI 旗标→类型），加工具 = 加一行。 |
| `server/smoke_handshake.py` | 不经 DSH 的握手冒烟测试。有它才能区分"server 坏了"和"接线错了"。 |
| `bundle/package.json` | bundle 清单（`dsh.bundle.patch` 指向下一个文件）。 |
| `bundle/cordis.patch.yml` | 往组合树里插一行 `@deepseek-ai/dsh-mcp-client`，并注入 `PVZ_RESOURCE_DIR`。 |
| `verify_s0.sh` | S0 端到端验收（管道），**不需要 API key**。 |
| `verify_s1.py` | S1 端到端验收（工具面），**不需要 API key**。 |
| `verify_s2.sh` | S2 端到端验收（**资源通道**：`list_mcp_resources` 不再返回空），**不需要 API key**。 |
| `prompts/diagnose_and_teach.md` | **任务模板**：诊断一局 + 把判断写成 skill。判据就写在这份模板里 —— 模型读不到这个 README。 |
| `skills/pvz-episode-diagnosis/` | **种子 skill**（`kind: procedure`）。每次实验开始前拷进沙箱的 `.dsh/skills/`，让模型有个可参照的骨架。 |
| `patches/isolate.yml` | 实验隔离叠加层：关掉"能读到仓库"的工具，把观测面收敛到我们的工具面上。 |
| `tools/logging_proxy.py` | 记录型反向代理：看到 DSH 实际发出的请求体。可选修复 `thinking.budget_tokens`。 |
| `tools/test_repair.py` | 上面那个修复的回归测试（离线可跑）。 |
| `tools/trace_session.py` | 把一次会话的工具调用轨迹还原成可读证据（调了什么、参数、结果、token 花销）。 |

## 工具一览

| 工具 | 用途 |
|---|---|
| `vocabulary` | **动作写法速查**：`try` 怎么写、植物 id 表、行列范围。第一次用 `whatif` 前看一眼。 |
| `constants` | **游戏常量表**（从 C++ 源码现场解析）：tick↔秒、植物花费/冷却/血量、僵尸血量与出现关卡、源码常量清单。带 `level` 还现算该关波数与可能出现的僵尸。 |
| `policy` | **脚本策略的决策规则**（文字描述，不给源码）。想知道"它当时为什么这么走"就看这份。 |
| `capture` | 跑一局并**完整落盘**。所有其它工具的前提。 |
| `index` | 存档的**目录**：哪里值得看。带规则名，但**不是结论**。 |
| `frame` | 某一 tick 的完整状态（含相邻帧对比）。最细粒度。 |
| `lane` | 按行（路）看整局演变。 |
| `actions` | 列出做过的每个决策；`max_life` 过滤"种下很快就死"的。 |
| `whatif` | **反事实回放** —— 换掉某个动作**真的重放一整局**。模拟器确定性 ⇒ 精确结果，不是估计。 |
| `narrative` | 把一段时间叙述成一段话。压缩过，只能当线索。 |
| `lint_skills` | **skill 闸门**。写完 skill 必须调它；查证据够不够、以及证据是不是真的。 |
| `ping` | 连通性探针（默认只回"通没通"，不给绝对路径）。 |

### 关于 `whatif` 的 `try` 参数（踩过，值得单独说）

`try` 的写法曾经是个**静默陷阱**：工具**输出**用植物名字
（`种下 豌豆射手 @(3,6)`），而 `try` 只认**数字 packet id** —— 中间那道映射
没写在任何地方。2026-10-03 实测的代价：

- 模型写 `plant:peashooter:1:7` → 报错；写 `plant:豌豆射手:3:6` → 抛裸 traceback。
- 更糟的是 `plant:4:3:6`：**被静默当成土豆雷**跑完，返回一个像模像样的结果
  （"没有任何单个替换能多推一波，说明这一步不是瓶颈"）—— **错误输入推出了错误结论**。
- 为了反推 id，模型连着试了 2 → 1 → 0，还先去探了 4 次 `list_mcp_resources`
  （想找文档，但 MCP 没有资源）。

三处已修：**① `try` 现在同时接受数字、中文名、英文别名**；
**② `enumerate` 输出的左列就是可直接复制的 `--try` 字符串**，不用翻译；
**③ 参数写错返回可自我纠正的提示**（列出全部可用植物），不再是 traceback。
另加 `vocabulary` 工具补上"文档无处可查"。

## 资源：`pvz://vocabulary` / `pvz://constants` / `pvz://scripted-policy`

参考类文档（动作语法、植物 id 表、行列范围、游戏常量、策略规则）除了对应的工具，
还发布成 **MCP resource**。理由是实测的：

模型不知道语法时**第一反应不是猜，是找文档** —— 它在 S2 干净轮里连着调了 4 次
资源通道（`list_mcp_resources` / `list_mcp_resource_templates`，前两次还猜错了
工具名），拿到的是 `{"resources":[]}`。而这条通道**是 DSH 主动告诉它的**：
`packages/mcp/mcp-resources` 只要配了一个 server 就挂上那三个共享工具，
并且把 server 名写进 system prompt。**我们一个资源都不发布 = 给了它一个空房间。**

**单一事实源**：每个资源的正文由 `episode_query.py` 的对应子命令现场生成，
与同名工具走**同一条命令**，不手抄第二份。对应关系写在 server 的
`RESOURCE_COMMANDS` 里（一张表，不是三个 `if`）。

### 为什么 `constants` 必须**从源码解析**而不是手写一张表

2026-10-03 读 S3 会话的推理轨迹，发现模型把大段推理花在**猜游戏常量**上：

> I don't know the exact arm time. The emulator might model potato mine arm time
> as 15s = 900 ticks. … So maybe arm time is longer (e.g., 1500 ticks = 25s?).

它先猜 900，又反推到 ~1600，来回试了七八轮。正确答案是 **1500 tick 的倒计时
＋一段升起动画**。它猜错的**不是游戏常识**，而是 tick↔秒 的换算率 ——
它按 60 tick/s 算，这个模拟器是 100（`mSyncRefreshRate = 100`）。

这个错误的代价不是"多花几轮"：它写下的"可迁移结论"（要沉淀成 skill 的那条）
整条建立在那个常数上。**一个数错，结论就错，而且错得没有症状。**

所以 `scripts/pvz_constants.py` 全部**现解析** `Plant.cpp` / `Zombie.cpp` /
`Challenge.cpp` / `ConstEnums.h`，并且：

- **解析失败就抛**，不返回空表（空表比没有更坏：模型会转而相信自己的常识）；
- **交叉校验**：`gPlantDefs` 的顺序必须和 `enum SeedType` 的值一致，不一致就报错；
- 只给**事实**（`mStateCountdown = 1500`），不给结论（"所以该种在 c3"）；
  唯一一处推导（该关波数、可能出现的僵尸）**把规则也一起写出来**，
  因为结论脱离了条件就变成了另一种猜。

同理 `pvz://scripted-policy` 的描述写在 `scripted_baseline.py` 里、紧挨着它描述的
`choose()`，并自带一道校验：**描述里出现的每个数字都必须能在源码里找到**。
人写的描述会过期，而过期的描述比没有描述更坏。

放哪的三条判据（照此分类，别凭感觉）：

| 信息 | 放哪 | 为什么 |
|---|---|---|
| 第一次调用**之前**就必须知道 | 常驻（`instructions` / 工具描述） | 不知道自己缺知识的模型**不会去查** |
| 大而全的参考表 | **资源**（次选：查询工具） | 走它已经会走的通道；不读就不占上下文 |
| 兜底 | 错误信息里列出可用值 | 不查也不会**静默**拿错 |

一个反直觉点：MCP 工具描述是**常驻**的（11 个工具 schema 实测 4,842 字符，
每个请求都发；加这两个新工具只多了 358 字符，因为理由写在源码注释里、
不写在描述里），所以"放工具描述"和"放 system prompt"在 token 上**没有区别**，
区别只在**位置**。真正的选择是"放哪个位置、有没有第二份"。

## 什么时候写 skill

### 这条规则是被一个具体的失败逼出来的

S3 跑完，模型交出的最终回答里**有 4 条「可复用的判断」**，质量不差 ——
但它**一条 skill 都没写**。复盘它的推理轨迹（677 行），`skill` 这个词
**出现 0 次**：它把"可复用的判断"理解成"在回答里说清楚"，然后就停了。

所以问题不是"它没产出内容"，是**没有判据**。没有判据时只有两种结果：
写噪音，或者干脆不写。两种都坏。

**判据必须可核查。** 喊口号没用 —— 模型不会照口号做，也不会照口号不写。

### 判据：三问，全过才写

| # | 问 | 答不出的后果 |
|---|---|---|
| 1 | **反例问**：这条判断在什么情况下**不成立**？ | 它不是判断，是复述 |
| 2 | **证据问**：它在**至少 2 个不同的决策点**上被 `whatif` 验过吗？ | 一个点上的成功可能是运气 |
| 3 | **迁移问**：它的判据**换一局还能算出来**吗？ | 判据里带本局坐标 ⇒ 一次性答案，不是 skill |

第 2 问为什么是 2 个点而不是 1 个 —— S3 自己的数据就是证据：把土豆雷从
`(3,6)` 挪到 `(3,3)` 通关 30/30，但同一局的枚举显示 `(3,4)`、`(2,5)`、`(0,4)`
**反而更差**（15~22 波崩）。**一个点上的成功不足以支撑一条法则**，
它可能只是那个点恰好落在一个不连续的可行域里。

### 四类产出，四个去处

| 产出 | 特征 | 去处 |
|---|---|---|
| **机制** | 三问全过 | 写 skill（`kind: mechanism`） |
| **手法** | 第 1 问答不出（不可证伪），但顺序有理由 | 写 skill（`kind: procedure`），必须写 `rationale` |
| **假设** | 第 1 问过、第 2 问不过 | **不写**。留在回答里，或写进已有 skill 的「待验证」段 |
| **一次性答案 / 工具注记** | 第 3 问不过 | **不写**。该改的是工具或文档，不是 skill 库 |

**负结论同等对待。** 验证出「某类做法无效」也是知识，但只有在**说出机制**
（为什么无效）时才算；只说「改 X 没用」不算 —— 那仍然是一次性答案。

**默认不写。** 大多数局没有能过闸门的东西。「本局无新增 skill」是**正常结果**，
不是失败。但模型必须**显式说出来**并给出卡在哪一问 —— 沉默是这次要修的那个 bug。

### 闸门：`scripts/lint_skills.py`

判据里的第 1、3 问只能靠写的人自己回答（写进 `claim` / `falsifier`）。
**第 2 问真的去查**，这是这个脚本存在的全部理由：

```
frames.jsonl 的第 N 行 = 第 N 个决策，那一行记着原局当时真实做出的动作。
所以证据里的「改动前」必须与它一致 —— 对不上就是编的。
```

存档是原局跑出来的，模型改不了它。所以**它编不出一个决策点**。

实测（`/tmp/pvz-s3-demo/skills`，四份都是照 S3 的原始产出写的）：

| skill | S3 里的对应 | 判决 |
|---|---|---|
| `potato-mine-lead-time` | 「土豆雷法则」（4 个点，全部有 whatif） | ✅ 通过 |
| `bypassed-line-rule` | 「被绕过防线法则」（模型自己写的是「正确动作是 (a)…(b)…(c)…」——**没验过**） | ❌ 0 个决策点 |
| `one-point-only` | 只在决策 96 上验过 | ❌ 1 个点不够 |
| `fabricated-evidence` | 决策号是真的、`改动前` 写的不是那一帧的动作 | ❌ 与存档对不上 |

**4 条里只有 1 条够格** —— 这个比例本身就是判据在工作的证据。

**为什么它同时是一个 MCP 工具**（不只是个脚本）：隔离层
`patches/s3-allow-skill.yml` 里 **`tool-bash` 是关掉的**（那是 2026-10-03 答案泄漏
的通道），所以模型**没有命令行可用**。任务模板第一版写"写完自己跑
`python3 scripts/lint_skills.py`"，那句话在沙箱里执行不了 ——
一句执行不了的指令比不写更坏：它会让模型以为自己验过了。改成工具之后，
闸门是**当场**的：证据不足当场拒，模型还有机会补；事后才发现就只能作废整轮。

`scripts/lint_skills.py` 还查：frontmatter 的 `name` 与目录名一致、有
`description`、`kind` 合法、`point` 五行齐全、决策号不越界、`claim` 与
`falsifier` 不是同一句。`--no-check-archives` 用于存档不在本机的场合
（另一台机器写的 skill 只查结构）。

**它查不了什么**（别指望）：不重跑 `whatif`，所以**不核对 `result` 的真假**；
也不判 `falsifier` 写得像不像样。要复核结论得自己跑
`episode_query.py whatif --archive X --decision N --try <改动后>`。
之所以不做：重跑要秒级且输出靠文本解析，脆；而实测出问题的是
"决策点和动作对不上"，不是"结果数字被改小"。

### 放到模型面前的路径

| 层 | 内容 | 为什么在这 |
|---|---|---|
| `prompts/diagnose_and_teach.md` | 判据 + 四类去处 + 证据格式 + 「不写也要说理由」 | **模型读不到 README**；判据必须跟着任务走 |
| `skills/pvz-episode-diagnosis/` | 种子 skill（`kind: procedure`） | 给个可参照的骨架；也顺带示范证据块长什么样 |
| `scripts/lint_skills.py` | 闸门 | 判据的第 2 问由它强制，不靠自觉 |

## 已知的 DSH 兼容性问题：`thinking.budget_tokens` 缺失

**症状**（接严格校验 Anthropic 规范的网关时）：

```
INVALID_REQUEST: invalid Claude request:
  thinking: budget_tokens must be at least 1024 when type is enabled
```

**根因**在 DSH 里，不在网关：`packages/llm/llm-deepseek/src/serialize.ts:155`

```js
thinking: { type: effort === 'off' ? 'disabled' : 'enabled' },
...effort === 'off' ? {} : { output_config: { effort } },
```

**无条件不发 `budget_tokens`**。官方端点宽容接受，严格网关直接拒。
改 DSH 配置改不出来 —— 代码里根本没有这个字段。

**单变量对照（curl，不经过 DSH）**，把范围收窄到一个字段：

| 请求体 | 结果 |
|---|---|
| `thinking:{"type":"enabled"}`（DSH 的发法） | **400** |
| `thinking:{"type":"enabled","budget_tokens":1024}` | **200** |
| 上一行 + `output_config:{"effort":"max"}` | **200** |

只差一个字段，400 变 200 —— 根因确认，不必再推测。
`output_config.effort` 网关是认的，所以"思考拉满"本身没问题。

**两条出路**：

1. 走代理补上（不改上游）：
   ```sh
   python3 tools/logging_proxy.py --upstream <真端点>/v1 --port 8950 \
       --log /tmp/llm-req.jsonl --fix-thinking-budget 32768
   DEEPSEEK_BASE_URL=http://127.0.0.1:8950/v1 dsh --profile pvz-teacher "..."
   ```
2. 给 DSH 源码打补丁（运行时更干净，但会与上游分叉）。

`tools/test_repair.py` 钉住了补丁的边界行为，包括一个**已知无解的边界**：
`max_tokens <= 1024` 时，`budget_tokens < max_tokens` 与 `budget_tokens >= 1024`
无法同时满足，怎么改都过不去。

### 接真实端点跑一次

```sh
DSH_HOME="$HOME/.local/share/pvz-agent/dsh-home" \
DEEPSEEK_BASE_URL=http://127.0.0.1:8950/v1 \
DEEPSEEK_API_KEY=sk-... \
  dsh --profile pvz-teacher "调用一下 mcp__pvz__ping，把返回的 JSON 原样告诉我"
```

**实测通过（2026-10-03）**：模型先出推理、再发起 `mcp__pvz__ping`、原样回显 JSON，退出码 0。
代理录到的请求证实 `output_config.effort = "max"` 与 `max_tokens = 65536` 都落到了线上，
且 `tools=28`（含 `mcp__pvz__ping`）—— 该端点**支持 tool use**。
另有一个 `thinking.type = "disabled"` 的标题生成请求，代理**正确地没碰它**。

**注意 401 可能是瞬时的。** 当天出现过一段稳定 401 的窗口，未做任何改动就自行恢复。
遇到时先记录时间窗、隔一会儿用最小请求（`/v1/models`）复测，**不要直接判定 key 失效**。

## 怎么验

```sh
bash dsh/pvz-teacher/verify_s0.sh        # S0：管道，退出码 0 = 全通
```

它做四件事：起 mock LLM → 跑一次 headless 会话 → 解压会话日志 →
断言 `mcp__pvz__ping` 被调用且 `isError=false`。

```sh
export PVZ_ROOT=/Users/newbiexvwu/PvZAgent
export PVZ_PYTHON=~/.local/share/pvz-agent/mcp-venv/bin/python
export PVZ_RESOURCE_DIR=/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN
~/.local/share/pvz-agent/mcp-venv/bin/python dsh/pvz-teacher/verify_s1.py
```

S1 验收：用真 MCP 客户端走完 `capture → index → whatif`，断言
`index` 自称为目录、`whatif` 枚举出至少一个能通关的替代方案。
**为什么不用命令行跑 CLI 交差**：CLI 通只说明 `episode_query.py` 没问题，
那是**已经存在**的东西；S1 要验的是新增的那层参数翻译。

```sh
bash dsh/pvz-teacher/verify_s2.sh        # S2：资源通道，退出码 0 = 全通
```

S2 验收跑**两次**会话：一次让 mock 调 `list_mcp_resources`，断言列表里有
`pvz://vocabulary`；一次调 `read_mcp_resource`，断言正文里真有 `plant:` 语法。
**为什么不能只跑 `smoke_handshake.py`**：那只证明"server 自己发了资源"，
而 S2 要修的是"模型找不到文档"—— 中间还隔着 DSH 的 `mcp-resources` provider、
MCP client、以及 system prompt 有没有把 server 名告诉模型。这三层断任何一层，
server 端看起来都是好的。

**为什么都不需要 API key**：S0 用 DSH 自带的 `@deepseek-ai/dsh-llm-mock-server`，
它的 `tool_call_success` 行为能按指定工具名/参数假装模型发起调用。
证据取的是**会话日志**（`$DSH_HOME/sessions/**/session.v4.jsonl.zstd`），
不是 mock 的 stdout —— 前者才是"模型真的收到了什么"的权威记录。
S1 根本不经过模型，直接问 server。

skill 闸门（不需要 API key、不需要 DSH）：

```sh
python3 scripts/lint_skills.py dsh/pvz-teacher/skills   # 种子 skill 应当通过
cd python && python3 -m unittest test_lint_skills       # 闸门判据的边界（16 项）
```

只想验 server 本身（不碰 DSH）：

```sh
PVZ_RESOURCE_DIR=... ~/.local/share/pvz-agent/mcp-venv/bin/python \
    dsh/pvz-teacher/server/smoke_handshake.py
```

`smoke_handshake.py` 现在多查三件事（都是踩过坑之后加的）：

1. **每个资源必须能读到有信息量的正文** —— 只断言"资源存在"不够，
   一个空正文的资源会让模型以为"通道是空的"，比没有更坏。
2. **工具/资源描述里不能出现绝对路径** —— S3 那轮 `ping` 把仓库根回了出去，
   模型第 8 步就去读了那个目录。描述是**每个请求都发**的同一类"指路牌"。
3. 工具面清单里必须含 `constants` / `policy` —— 少一个就说明工具表被改坏了。

它会校验 8 个工具都在、每个 `inputSchema` 自洽、描述里没有零宽字符。

## 机器本地的前置（不进 git）

| 路径 | 是什么 |
|---|---|
| `~/.local/share/pvz-agent/mcp-venv` | server 的 Python venv，只装 `mcp` SDK |
| `~/.local/share/pvz-agent/dsh-home` | 隔离的 `DSH_HOME`，不碰用户真实的那个 |
| `~/deepseek-harness` | DSH 源码，tag `dsh-v0.2.0-rc.2`，需 `pnpm install` + `build:lib:host` |

重建 venv：

```sh
uv venv --python "$(mise which python)" ~/.local/share/pvz-agent/mcp-venv
uv pip install --python ~/.local/share/pvz-agent/mcp-venv/bin/python mcp
```

## 装进 profile

```sh
dsh plugin --profile pvz-teacher add file:/abs/path/to/dsh/pvz-teacher/bundle
dsh --profile pvz-teacher --dump-config | grep -A12 pvz-teacher
```

`add` 会**自动**把包名追加进 `dsh.profile.bundles`，不用手工改。

## 两个坑（都踩过，别再踩）

1. **bundle 里是绝对路径**，所以 `bundle/cordis.patch.yml` 是机器专属的。
   换机器要改 `command` / `args` / `cwd` / `env`。
2. **DSH 起 MCP 子进程时环境是被擦除的**（丢掉名字匹配 `/KEY|PASSWORD|SECRET|TOKEN/i`
   的和所有 `DSH_*`）。所以不要依赖 `PATH`，也不要指望 `HF_TOKEN` 之类的能自动传进去 ——
   需要就在 `config.env` 里显式写。`smoke_handshake.py` 默认就用擦除后的环境跑，
   正是为了让这个坑在本地暴露。
