#!/usr/bin/env bash
# S0 端到端验收：bundle → profile → 会话里真能调 mcp__pvz__ping。
#
# 为什么不用真 API key
# --------------------
# DSH 仓库自带 `@deepseek-ai/dsh-llm-mock-server`：一个可脚本化的
# Messages 兼容 HTTP/SSE 端点。它有个 `tool_call_success` 行为，能按我们指定的
# 工具名和参数"假装模型发起一次工具调用"。于是整条链路可以离线跑完：
#
#   mock LLM 发出 mcp__pvz__ping 调用
#     → DSH 路由到 MCP client
#     → MCP client 起 pvz_mcp_server.py 并转发
#     → server 返回 JSON
#     → 结果回灌进模型的上下文（写进会话日志）
#
# 我们读的是**会话日志**（`$DSH_HOME/sessions/**/session.v4.jsonl.zstd`），
# 不是 mock 的 stdout —— 因为前者才是"模型真的收到了什么"的权威记录。
#
# 用法
# ----
#     bash dsh/pvz-teacher/verify_s0.sh
#
# 前置：~/deepseek-harness 已 pnpm install + build:lib:host，且 dsh 在 PATH。
# 退出码 0 = 全通。

set -uo pipefail

HARNESS="${PVZ_DSH_SOURCE:-$HOME/deepseek-harness}"
DSH_HOME_DIR="${DSH_HOME:-$HOME/.local/share/pvz-agent/dsh-home}"
PROFILE="${PVZ_DSH_PROFILE:-pvz-teacher}"
VENV_PY="$HOME/.local/share/pvz-agent/mcp-venv/bin/python"
PORT="${PVZ_S0_PORT:-8934}"

MOCK_LOG="$(mktemp -t pvz-s0-mock)"
DSH_LOG="$(mktemp -t pvz-s0-dsh)"

cleanup() {
  [[ -n "${MOCK_PID:-}" ]] && kill "$MOCK_PID" 2>/dev/null
  pkill -f "llm-mock-server/src/bin.ts" 2>/dev/null
  return 0
}
trap cleanup EXIT

fail() { echo "✗ $*"; exit 1; }
ok()   { echo "✓ $*"; }

echo "── 前置检查"
[[ -x "$VENV_PY" ]] || fail "MCP venv 不存在：$VENV_PY"
[[ -d "$HARNESS" ]] || fail "DSH 源码目录不存在：$HARNESS"
[[ -d "$DSH_HOME_DIR/profiles/$PROFILE" ]] || fail "profile 不存在：$DSH_HOME_DIR/profiles/$PROFILE"
command -v dsh >/dev/null || fail "PATH 里没有 dsh"
ok "venv / 源码 / profile / dsh 都在"

echo
echo "── 起 mock LLM（端口 ${PORT}）"
# 序列要够长：主循环用掉 1~2 次，之后可能还有一次会话标题生成。
(
  cd "$HARNESS" && exec pnpm run mock:llm \
    --port "$PORT" --api-key mock-key \
    --sequence tool_call_success,success,success,success,success,success,success,success \
    --tool-name mcp__pvz__ping \
    --tool-arguments '{"note":"来自 verify_s0 的调用"}' \
    --success-text "工具已调用，收工。"
) > "$MOCK_LOG" 2>&1 &
MOCK_PID=$!

for _ in $(seq 1 40); do
  grep -q '"type":"ready"' "$MOCK_LOG" 2>/dev/null && break
  sleep 0.5
done
grep -q '"type":"ready"' "$MOCK_LOG" 2>/dev/null || { cat "$MOCK_LOG"; fail "mock LLM 没起来"; }
ok "mock 就绪：$(grep -o '"baseURL":"[^"]*"' "$MOCK_LOG" | head -1)"

echo
echo "── 跑 headless 会话"
BEFORE=$(find "$DSH_HOME_DIR/sessions" -name 'session.v4.jsonl.zstd' 2>/dev/null | wc -l | tr -d ' ')

(
  cd "$HARNESS" && \
  DSH_HOME="$DSH_HOME_DIR" \
  DEEPSEEK_BASE_URL="http://127.0.0.1:$PORT/v1" \
  DEEPSEEK_API_KEY="mock-key" \
  exec dsh --profile "$PROFILE" "调用一下 mcp__pvz__ping，把结果告诉我"
) > "$DSH_LOG" 2>&1
RC=$?

echo "  dsh 退出码 ${RC}，输出：$(tail -1 "$DSH_LOG")"
[[ $RC -eq 0 ]] || { tail -20 "$DSH_LOG"; fail "dsh 非零退出"; }

echo
echo "── 从会话日志取证据"
NEWEST=$(find "$DSH_HOME_DIR/sessions" -name 'session.v4.jsonl.zstd' -newermt '-3 minutes' 2>/dev/null | head -1)
[[ -n "$NEWEST" ]] || fail "找不到新会话日志"
echo "  日志：$NEWEST"

TMPJSONL=$(mktemp -t pvz-s0-session)
zstd -d -c "$NEWEST" > "$TMPJSONL" 2>/dev/null || fail "解压会话日志失败"

RESULT=$(DSH_LOG_JSONL="$TMPJSONL" "$VENV_PY" - <<'PY'
import json, os, sys

call = None
result = None
for line in open(os.environ["DSH_LOG_JSONL"]):
    line = line.strip()
    if not line:
        continue
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o.get("type") == "tool/call":
        call = o["data"]
    elif o.get("type") == "tool/result":
        result = o["data"]["message"]

if call is None or result is None:
    print("FAIL 会话日志里没有完整的 tool/call + tool/result")
    sys.exit(1)

name = call.get("name")
print(f"  工具名：{name}")
print(f"  参数  ：{call.get('arguments')}")

text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
print("  返回  ：")
for line in text.splitlines():
    print(f"      {line}")

if name != "mcp__pvz__ping":
    print(f"FAIL 工具名不是 mcp__pvz__ping，而是 {name}")
    sys.exit(1)
if result.get("isError"):
    print("FAIL 工具返回了 isError=true")
    sys.exit(1)
if '"pong": true' not in text:
    print("FAIL 返回内容里没有 pong:true")
    sys.exit(1)
if "来自 verify_s0 的调用" not in text:
    print("FAIL 参数没有被回显，参数传递链路有问题")
    sys.exit(1)
print("PASS")
PY
)
PYRC=$?
echo "$RESULT"

echo
if [[ $PYRC -eq 0 ]] && echo "$RESULT" | grep -q '^PASS$'; then
  ok "S0 通过：bundle 装进 profile 后，会话里 mcp__pvz__ping 可调用且返回正确"
  exit 0
else
  fail "S0 未通过"
fi
