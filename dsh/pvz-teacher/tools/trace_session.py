#!/usr/bin/env python3
"""把一次 dsh 会话的**工具调用轨迹**抽出来 —— 给"模型到底做了什么"留证据。

为什么需要它
------------
S2 这类实验的结论是"模型几轮内找到了改法"。**这句话必须可核查**，
否则和"我觉得它找到了"没有区别。这个脚本把会话日志（zstd 压缩的 JSONL）
还原成一条可读的轨迹：它调了哪些工具、参数是什么、拿到了什么、
每一步花了多少 token、总共多少墙钟时间。

会话日志在哪
------------
    $DSH_HOME/sessions/<cwd-slug>/session-<id>/session.v4.jsonl.zstd

关键事件（实测确认）：
- `tool/call`     → `data.name`、`data.arguments`
- `tool/result`   → `data.message.isError`、`data.message.content[].text`
- `assistant/message` → `data.usage`（inputTokens/outputTokens/cacheReadTokens/
  cacheWriteTokens/totalTokens）—— **token 花销在这里，不在 step/end**

用法
----
    trace_session.py                      # 最新的一个会话
    trace_session.py --list               # 列出最近的会话
    trace_session.py --session <id 前缀>   # 指定会话
    trace_session.py --full               # 打印工具返回的完整内容
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DSH_HOME = Path(os.environ.get("DSH_HOME") or Path.home() / ".local/share/pvz-agent/dsh-home")
SESSIONS = DSH_HOME / "sessions"


def _sessions() -> list[Path]:
    if not SESSIONS.is_dir():
        return []
    found = list(SESSIONS.rglob("session.v4.jsonl.zstd"))
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _decompress(path: Path) -> list[dict]:
    """zstd 解压后逐行解析。用子进程调 zstd —— Python 标准库没有 zstd。"""
    proc = subprocess.run(
        ["zstd", "-d", "-c", str(path)], capture_output=True, check=True
    )
    events = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _text_of(message: dict) -> str:
    parts = []
    for c in message.get("content") or []:
        if isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text") or "")
    return "".join(parts)


def _compact(value, limit: int = 220) -> str:
    s = json.dumps(value, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="会话 id 前缀；默认最新一个")
    ap.add_argument("--list", action="store_true", help="列出最近会话")
    ap.add_argument("--full", action="store_true", help="打印工具返回的完整内容")
    ap.add_argument("--max-result", type=int, default=1200, help="单条结果最多打印多少字符")
    args = ap.parse_args()

    sessions = _sessions()
    if not sessions:
        print(f"没找到会话：{SESSIONS}")
        return 1

    if args.list:
        for p in sessions[:20]:
            print(f"  {p.parent.name}  {p.stat().st_mtime:.0f}  {p}")
        return 0

    target = sessions[0]
    if args.session:
        hit = [p for p in sessions if args.session in p.parent.name]
        if not hit:
            print(f"没有匹配 {args.session!r} 的会话")
            return 1
        target = hit[0]

    events = _decompress(target)
    print(f"会话：{target.parent.name}")
    print(f"事件：{len(events)} 条\n")

    # ---- 工具调用轨迹 ----------------------------------------------------
    calls: list[tuple[dict, dict | None]] = []
    pending: dict | None = None
    for ev in events:
        t = ev.get("type")
        if t == "tool/call":
            pending = ev.get("data") or {}
        elif t == "tool/result":
            calls.append((pending or {}, ev.get("data") or {}))
            pending = None

    print(f"── 工具调用 {len(calls)} 次 " + "─" * 30)
    for i, (call, result) in enumerate(calls, 1):
        name = call.get("name", "?")
        argv = call.get("arguments")
        msg = result.get("message") or {}
        is_err = msg.get("isError")
        text = _text_of(msg)
        flag = "✗" if is_err else "·"
        print(f"\n[{i:2}] {flag} {name}  {_compact(argv, 260)}")
        body = text if args.full else text[: args.max_result]
        for line in body.splitlines()[:40]:
            print(f"      {line}")
        if not args.full and len(text) > args.max_result:
            print(f"      …（结果共 {len(text)} 字符，用 --full 看全）")

    # ---- token 与时间 ----------------------------------------------------
    usage_total = {"inputTokens": 0, "outputTokens": 0,
                   "cacheReadTokens": 0, "cacheWriteTokens": 0, "totalTokens": 0}
    n_msg = 0
    for ev in events:
        if ev.get("type") == "assistant/message":
            u = (ev.get("data") or {}).get("usage") or {}
            if u:
                n_msg += 1
                for k in usage_total:
                    usage_total[k] += int(u.get(k) or 0)

    print("\n" + "─" * 46)
    print(f"assistant 消息数：{n_msg}")
    print("token 合计：")
    for k, v in usage_total.items():
        print(f"  {k:<18} {v:>10,}")

    # 工具返回的总字节数 —— 上下文膨胀主要来自这里
    tool_bytes = sum(len(_text_of((r.get("message") or {}))) for _, r in calls)
    print(f"工具返回总字符数：{tool_bytes:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
