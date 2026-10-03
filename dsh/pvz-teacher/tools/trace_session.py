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
- `assistant/message` → `data.message.content[]`，块类型有
  `reasoning`（**模型的推理轨迹**，字段是 `{type, text}`）、`text`、`tool-call`；
  以及 `data.usage`（inputTokens/outputTokens/cacheReadTokens/
  cacheWriteTokens/totalTokens）—— **token 花销在这里，不在 step/end**

用法
----
    trace_session.py                      # 最新的一个会话
    trace_session.py --list               # 列出最近的会话
    trace_session.py --session <id 前缀>   # 指定会话
    trace_session.py --reasoning          # 只打印推理轨迹（最常看这个）
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


def _reasoning_blocks(events: list[dict]) -> list[tuple[int, str]]:
    """抽出模型的推理轨迹。返回 [(turn, text)]。

    这是 headless 模式 stdout 里那些 `dsh: reasoning:` 行的**完整版** ——
    stdout 是流式截断过的，会话日志里才是全文。
    """
    out: list[tuple[int, str]] = []
    for ev in events:
        if ev.get("type") != "assistant/message":
            continue
        d = ev.get("data") or {}
        turn = d.get("turn") or 0
        for c in (d.get("message") or {}).get("content") or []:
            if isinstance(c, dict) and c.get("type") == "reasoning":
                text = (c.get("text") or "").strip()
                if text:
                    out.append((turn, text))
    return out


# 推理里"我觉得信息不够"的措辞。刻意收窄 —— "可能/应该" 这类词在正常推理里
# 太常见，放进来只会淹掉真信号。这里只找**明确指向缺失信息**的说法。
#
# **必须中英双语**：实测模型即使收到中文提示词，**推理轨迹仍然用英文写**
# （2026-10-03，S2 干净轮）。只写中文标记会得到 0 命中，然后你会误以为
# "模型没有困惑" —— 那是标记的问题，不是模型的问题。
_MISSING_INFO_MARKERS = (
    # 中文
    "不知道", "不确定", "搞不清", "不清楚", "没写", "找不到", "看不到",
    "没说明", "缺少", "猜一下", "猜是", "得猜", "试试看", "只能试",
    "没有说明", "没告诉我", "看不出来", "无法确定",
    # 英文
    "don't know", "do not know", "not sure", "unclear", "doesn't say",
    "does not say", "not specified", "not documented", "have to guess",
    "let me guess", "no way to know", "can't tell", "cannot tell",
    "unable to determine", "missing", "not given", "no indication",
    "let me try", "worth trying", "try a few",
)

# "原地打转"的措辞：说明上一步没解决问题。
# 注意排除**游戏语义**里的词：`wasted`（浪费阳光）、`useless`（这株没用）
# 在 PvZ 语境里是正常分析用词，放进来会淹掉真信号（实测 11 句里 10 句是误报）。
_BLOCKED_MARKERS = (
    # 中文
    "没用", "不行", "白试", "卡住", "没效果", "还是不知道",
    # 英文
    "didn't work", "did not work", "doesn't work", "does not work",
    "no effect", "that failed", "still don't know", "i'm stuck",
    "no improvement", "no progress",
)


def _sentences(text: str) -> list[str]:
    """按中英文句读切句。够用就行，不追求语言学正确。"""
    out, buf = [], []
    for ch in text:
        if ch in "。！？!?\n":
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _tool_calls(events: list[dict]) -> list[tuple[dict, dict | None]]:
    calls: list[tuple[dict, dict | None]] = []
    pending: dict | None = None
    for ev in events:
        t = ev.get("type")
        if t == "tool/call":
            pending = ev.get("data") or {}
        elif t == "tool/result":
            calls.append((pending or {}, ev.get("data") or {}))
            pending = None
    return calls


def _report_confusion(events: list[dict]) -> None:
    """把「模型在哪里卡了」变成一份可核查的清单。

    三类硬证据，按可信度从高到低：
      ① 工具**报错** —— 它撞到的墙，不需要解释
      ② **重复调用**（同工具同参数 ≥2 次）—— 原地打转，它自己没意识到
      ③ 推理里出现**指向信息缺失**的措辞 —— 最弱，但能指出该补什么
    """
    calls = _tool_calls(events)
    errs = []
    seen: dict[str, list[int]] = {}
    for i, (call, result) in enumerate(calls, 1):
        name = call.get("name", "?")
        args = json.dumps(call.get("arguments"), ensure_ascii=False, sort_keys=True)
        seen.setdefault(f"{name} {args}", []).append(i)
        if (result.get("message") or {}).get("isError"):
            errs.append((i, name, call.get("arguments"),
                         _text_of(result.get("message") or {})))

    repeats = {k: v for k, v in seen.items() if len(v) >= 2}

    print(f"── 卡点报告：{len(calls)} 次调用 " + "─" * 26)

    print(f"\n① 工具报错 {len(errs)} 次")
    for i, name, args, text in errs:
        first = next((l for l in text.splitlines() if l.strip()), "")
        print(f"   [{i:2}] {name}  {_compact(args, 120)}")
        print(f"         {first[:160]}")

    print(f"\n② 原地打转（同工具同参数重复） {len(repeats)} 组")
    for k, idxs in sorted(repeats.items(), key=lambda kv: -len(kv[1])):
        print(f"   {len(idxs)} 次：{_compact(k, 150)}")
        print(f"        出现在调用 {idxs}")

    print("\n③ 推理里指向「信息缺失」的句子")
    hits = 0
    for turn, text in _reasoning_blocks(events):
        for s in _sentences(text):
            if any(m in s for m in _MISSING_INFO_MARKERS):
                hits += 1
                print(f"   [turn {turn}] {s[:200]}")
    print(f"   共 {hits} 句")

    print("\n④ 推理里「上一步没解决」的句子")
    hits2 = 0
    for turn, text in _reasoning_blocks(events):
        for s in _sentences(text):
            if any(m in s for m in _BLOCKED_MARKERS):
                hits2 += 1
                print(f"   [turn {turn}] {s[:200]}")
    print(f"   共 {hits2} 句")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="会话 id 前缀；默认最新一个")
    ap.add_argument("--list", action="store_true", help="列出最近会话")
    ap.add_argument("--reasoning", action="store_true",
                    help="只打印推理轨迹（最常看这个）")
    ap.add_argument("--full", action="store_true", help="打印工具返回的完整内容")
    ap.add_argument("--confusion", action="store_true",
                    help="只打印卡点报告：报错 / 原地打转 / 推理里的信息缺失信号")
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

    # ---- 只打印推理轨迹 --------------------------------------------------
    if args.reasoning:
        blocks = _reasoning_blocks(events)
        print(f"── 推理轨迹 {len(blocks)} 段 " + "─" * 30)
        for i, (turn, text) in enumerate(blocks, 1):
            print(f"\n【第 {i} 段 · turn {turn}】")
            for line in text.splitlines():
                print(f"  {line}")
        if not blocks:
            print("（这个会话没有推理块 —— 可能用了 mock LLM，或 reasoningEffort 是 off）")
        return 0

    # ---- 只打印卡点报告 --------------------------------------------------
    if args.confusion:
        _report_confusion(events)
        return 0

    # ---- 工具调用轨迹 ----------------------------------------------------
    calls = _tool_calls(events)
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
