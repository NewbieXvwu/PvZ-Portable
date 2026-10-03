# PvZ 教师 × DeepSeek Harness

把 PvZ 对局诊断工具挂进 DSH，让模型自己看败局、提改动、用精确反事实验证、
把结论写成可自演化的 skill。设计依据与分步计划在
[`RESEARCH_EXECUTION.md`](../../RESEARCH_EXECUTION.md) 的
「2026-10-03 LLM 教师方案」一节。

**当前状态：S0 与 S1 均已通过。** 工具面已铺开 —— 8 个工具
（`ping / capture / index / frame / lane / actions / whatif / narrative`），
与 `episode_query.py` 的子命令一一对应。S0 用 mock LLM（不需要 key）
和真实模型各验过一次；S1 用真 MCP 客户端走完一次完整诊断。
**S2（让模型自己找出一局败局的改法）未开始。**

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
| `patches/isolate.yml` | 实验隔离叠加层：关掉"能读到仓库"的工具，把观测面收敛到我们的工具面上。 |
| `tools/logging_proxy.py` | 记录型反向代理：看到 DSH 实际发出的请求体。可选修复 `thinking.budget_tokens`。 |
| `tools/test_repair.py` | 上面那个修复的回归测试（离线可跑）。 |
| `tools/trace_session.py` | 把一次会话的工具调用轨迹还原成可读证据（调了什么、参数、结果、token 花销）。 |

## 工具一览

| 工具 | 用途 |
|---|---|
| `vocabulary` | **动作写法速查**：`try` 怎么写、植物 id 表、行列范围。第一次用 `whatif` 前看一眼。 |
| `capture` | 跑一局并**完整落盘**。所有其它工具的前提。 |
| `index` | 存档的**目录**：哪里值得看。带规则名，但**不是结论**。 |
| `frame` | 某一 tick 的完整状态（含相邻帧对比）。最细粒度。 |
| `lane` | 按行（路）看整局演变。 |
| `actions` | 列出做过的每个决策；`max_life` 过滤"种下很快就死"的。 |
| `whatif` | **反事实回放** —— 换掉某个动作**真的重放一整局**。模拟器确定性 ⇒ 精确结果，不是估计。 |
| `narrative` | 把一段时间叙述成一段话。压缩过，只能当线索。 |
| `ping` | 连通性探针（会回报资源目录是否存在）。 |

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

## 资源：`pvz://vocabulary`

参考类文档（动作语法、植物 id 表、行列范围）除了 `vocabulary` 工具，
还发布成 **MCP resource**。理由是实测的：

模型不知道语法时**第一反应不是猜，是找文档** —— 它在 S2 干净轮里连着调了 4 次
资源通道（`list_mcp_resources` / `list_mcp_resource_templates`，前两次还猜错了
工具名），拿到的是 `{"resources":[]}`。而这条通道**是 DSH 主动告诉它的**：
`packages/mcp/mcp-resources` 只要配了一个 server 就挂上那三个共享工具，
并且把 server 名写进 system prompt。**我们一个资源都不发布 = 给了它一个空房间。**

**单一事实源**：资源正文由 `episode_query.py vocabulary` 现场生成，
与 `vocabulary` 工具走**同一条命令**，不手抄第二份。

放哪的三条判据（照此分类，别凭感觉）：

| 信息 | 放哪 | 为什么 |
|---|---|---|
| 第一次调用**之前**就必须知道 | 常驻（`instructions` / 工具描述） | 不知道自己缺知识的模型**不会去查** |
| 大而全的参考表 | **资源**（次选：查询工具） | 走它已经会走的通道；不读就不占上下文 |
| 兜底 | 错误信息里列出可用值 | 不查也不会**静默**拿错 |

一个反直觉点：MCP 工具描述是**常驻**的（9 个工具 schema 实测 4,484 字符，
每个请求都发），所以"放工具描述"和"放 system prompt"在 token 上**没有区别**，
区别只在**位置**。真正的选择是"放哪个位置、有没有第二份"。

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

只想验 server 本身（不碰 DSH）：

```sh
PVZ_RESOURCE_DIR=... ~/.local/share/pvz-agent/mcp-venv/bin/python \
    dsh/pvz-teacher/server/smoke_handshake.py
```

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
