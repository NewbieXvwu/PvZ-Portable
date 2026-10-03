#!/usr/bin/env bash
# S2 端到端验收：`pvz://vocabulary` 资源在 DSH 会话里真的能被**列到**、被**读到**。
#
# 为什么必须验这一层，而不是只跑 smoke_handshake.py
# -------------------------------------------------
# smoke_handshake.py 证明的是"server 自己发了资源"。
# 但 S2 要修的**不是 server** —— 是"模型找不到文档"这件事。
# 中间还隔着 DSH 的 `mcp-resources` provider、MCP client、以及
# **system prompt 里有没有把 server 名告诉模型**。
# 这三层任何一层断了，server 端看起来都是好的。
#
# 实测背景（2026-10-03）：模型在 S2 干净轮里连着调了 4 次资源通道：
#     [6] ✗ unknown tool "mcp__pvz__list_mcp_resources"
#     [7] ✗ unknown tool "mcp__pvz__list_mcp_resource_templates"
#     [8] · MCP server: pvz  {"resources":[]}
#     [9] · MCP server: pvz  {"resourceTemplates":[]}
# 前两次是它猜错了工具名（那三个是**共享**工具，不带 server 前缀），
# 后两次拿到了空列表。所以这个脚本要验两件事：
#   ① `list_mcp_resources` 返回的列表里有 pvz://vocabulary（通道不再空）
#   ② `read_mcp_resource` 能把正文取回来（不是只有一个名字）
#
# 用 mock LLM，不花 API key；读**会话日志**取证据，不读 mock 的 stdout。
#
# 用法：bash dsh/pvz-teacher/verify_s2.sh
# 退出码 0 = 全通。

set -uo pipefail

HARNESS="${PVZ_DSH_SOURCE:-$HOME/deepseek-harness}"
DSH_HOME_DIR="${DSH_HOME:-$HOME/.local/share/pvz-agent/dsh-home}"
PROFILE="${PVZ_DSH_PROFILE:-pvz-teacher}"
VENV_PY="$HOME/.local/share/pvz-agent/mcp-venv/bin/python"
PORT="${PVZ_S2_PORT:-8936}"

MOCK_LOG="$(mktemp -t pvz-s2-mock)"
DSH_LOG="$(mktemp -t pvz-s2-dsh)"

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

# 跑一次会话，让 mock 发一个指定工具调用，再从会话日志里把 tool/result 取出来。
# $1 工具名  $2 参数 JSON  $3 期望出现在返回里的子串  $4 人类可读标签
run_case() {
  local tool="$1" args="$2" needle="$3" label="$4"
  local before newest jsonl out rc

  echo
  echo "── ${label}"
  before=$(find "$DSH_HOME_DIR/sessions" -name 'session.v4.jsonl.zstd' 2>/dev/null | wc -l | tr -d ' ')

  # mock 每次重启，避免上一次的序列状态残留。
  pkill -f "llm-mock-server/src/bin.ts" 2>/dev/null
  sleep 0.3
  (
    cd "$HARNESS" && exec pnpm run mock:llm \
      --port "$PORT" --api-key mock-key \
      --sequence tool_call_success,success,success,success \
      --tool-name "$tool" \
      --tool-arguments "$args" \
      --success-text "收到，收工。"
  ) > "$MOCK_LOG" 2>&1 &
  MOCK_PID=$!

  for _ in $(seq 1 40); do
    grep -q '"type":"ready"' "$MOCK_LOG" 2>/dev/null && break
    sleep 0.5
  done
  grep -q '"type":"ready"' "$MOCK_LOG" 2>/dev/null || { cat "$MOCK_LOG"; fail "mock LLM 没起来"; }

  (
    cd "$HARNESS" && \
    DSH_HOME="$DSH_HOME_DIR" \
    DEEPSEEK_BASE_URL="http://127.0.0.1:$PORT/v1" \
    DEEPSEEK_API_KEY="mock-key" \
    exec dsh --profile "$PROFILE" "调用 ${tool}，把结果原样告诉我"
  ) > "$DSH_LOG" 2>&1
  rc=$?
  [[ $rc -eq 0 ]] || { tail -20 "$DSH_LOG"; fail "dsh 非零退出（$rc）"; }

  # 取**最新**那个会话日志。注意不能只 `find ... -newermt | head -1` ——
  # 那取到的是 find 的遍历顺序里第一个，不是时间上最新的；连跑两次时
  # 第二次会拿到第一次的日志，于是断言在一个**过期的会话**上做。
  # （第一版就踩了这个：case ② 报"工具名不是 read_mcp_resource"，
  #  实际是它读回了 case ① 的日志。）
  newest=$(find "$DSH_HOME_DIR/sessions" -name 'session.v4.jsonl.zstd' \
             -newermt '-3 minutes' -exec ls -t {} + 2>/dev/null | head -1)
  [[ -n "$newest" ]] || fail "找不到新会话日志"
  jsonl=$(mktemp -t pvz-s2-session)
  zstd -d -c "$newest" > "$jsonl" 2>/dev/null || fail "解压会话日志失败"

  out=$(DSH_LOG_JSONL="$jsonl" CASE_TOOL="$tool" CASE_NEEDLE="$needle" "$VENV_PY" - <<'PY'
import json, os, sys

call = result = None
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

want = os.environ["CASE_TOOL"]
needle = os.environ["CASE_NEEDLE"]

if call is None or result is None:
    print("FAIL 会话日志里没有完整的 tool/call + tool/result")
    sys.exit(1)

name = call.get("name")
print(f"  工具名：{name}")
print(f"  参数  ：{call.get('arguments')}")
text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")

if name != want:
    print(f"FAIL 工具名不是 {want}，而是 {name}")
    sys.exit(1)
if result.get("isError"):
    print(f"FAIL 工具返回 isError=true：\n{text}")
    sys.exit(1)
if needle not in text:
    print(f"FAIL 返回里没有 {needle!r}。实际返回：\n{text}")
    sys.exit(1)
print(f"  返回里含 {needle!r} ——")
for line in text.splitlines()[:12]:
    print(f"      {line}")
print("PASS")
PY
)
  rc=$?
  echo "$out"
  [[ $rc -eq 0 ]] && echo "$out" | grep -q '^PASS$' || fail "${label} 未通过"
  ok "${label} 通过"
}

run_case "list_mcp_resources" '{"server":"pvz"}' "pvz://vocabulary" \
  "① resources/list 通道不再为空"

run_case "read_mcp_resource" '{"server":"pvz","uri":"pvz://vocabulary"}' "plant:" \
  "② resources/read 能取回正文"

echo
ok "S2 通过：pvz://vocabulary 在 DSH 会话里可列、可读"
exit 0
