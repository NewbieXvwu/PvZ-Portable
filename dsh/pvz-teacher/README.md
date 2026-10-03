# PvZ 教师 × DeepSeek Harness

把 PvZ 对局诊断工具挂进 DSH，让模型自己看败局、提改动、用精确反事实验证、
把结论写成可自演化的 skill。设计依据与分步计划在
[`RESEARCH_EXECUTION.md`](../../RESEARCH_EXECUTION.md) 的
「2026-10-03 LLM 教师方案」一节。

**当前状态：S0 通过。** 只剩一个 `ping` 工具，但整条管道（bundle → profile →
会话里可调用）已验证。

## 目录

| 文件 | 作用 |
|---|---|
| `server/pvz_mcp_server.py` | MCP server 本体。stdio，官方 `mcp` SDK 低层 `Server`。**只做搬运**，不做任何游戏判断。 |
| `server/smoke_handshake.py` | 不经 DSH 的握手冒烟测试。有它才能区分"server 坏了"和"接线错了"。 |
| `bundle/package.json` | bundle 清单（`dsh.bundle.patch` 指向下一个文件）。 |
| `bundle/cordis.patch.yml` | 往组合树里插一行 `@deepseek-ai/dsh-mcp-client`。 |
| `verify_s0.sh` | S0 端到端验收，**不需要 API key**。 |
| `tools/logging_proxy.py` | 记录型反向代理：看到 DSH 实际发出的请求体。可选修复 `thinking.budget_tokens`。 |
| `tools/test_repair.py` | 上面那个修复的回归测试（离线可跑）。 |

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

## 怎么验

```sh
bash dsh/pvz-teacher/verify_s0.sh      # 退出码 0 = 全通
```

它做四件事：起 mock LLM → 跑一次 headless 会话 → 解压会话日志 →
断言 `mcp__pvz__ping` 被调用且 `isError=false`。

**为什么不需要 API key**：DSH 自带 `@deepseek-ai/dsh-llm-mock-server`，
它的 `tool_call_success` 行为能按指定工具名/参数假装模型发起调用。
证据取的是**会话日志**（`$DSH_HOME/sessions/**/session.v4.jsonl.zstd`），
不是 mock 的 stdout —— 前者才是"模型真的收到了什么"的权威记录。

只想验 server 本身（不碰 DSH）：

```sh
~/.local/share/pvz-agent/mcp-venv/bin/python dsh/pvz-teacher/server/smoke_handshake.py
```

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
