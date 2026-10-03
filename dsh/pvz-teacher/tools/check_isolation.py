#!/usr/bin/env python3
"""检查一次会话有没有"越界读到不该读的东西"。

为什么需要它
------------
S3 这类实验要的是"模型只看得到我们给它的观测面"。但 DSH 的 `fs-sandbox`
**只限制写，不限制读**（`packages/fs/fs-sandbox/README.md`：
「Reads, listings, metadata, and read-only watches work exactly as with
`fs-local`; the mutation fence does not restrict observation.」）。

所以"关掉 tool-fs-search / tool-bash"**不等于**读不到仓库 —— 只要模型知道
路径，`tool-fs` 的 read 就能读到任何地方。

2026-10-03 实测踩过：S3 第一轮里模型第 2 步调 `mcp__pvz__ping`，
**第 8 步就直接 `read /Users/newbiexvwu/PvZAgent`** —— 因为 `ping` 的返回里
带着 `root` 绝对路径。那一轮作废。（`ping` 已改成默认不回路径。）

**允许 + 事后核查，比假装锁死了更诚实。** 这个脚本就是"核查"那一步。

用法
----
    check_isolation.py --session <id 前缀> [--allow /tmp/pvz-s3] \
                       [--forbid /Users/newbiexvwu/PvZAgent]

退出码 0 = 干净；1 = 有越界（这次实验的结果不能用）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DSH_HOME = Path(os.environ.get("DSH_HOME") or Path.home() / ".local/share/pvz-agent/dsh-home")
SESSIONS = DSH_HOME / "sessions"

# 这些工具名里的 "read/write/ls/glob/grep" 说明它在碰文件系统。
FS_TOOL_HINT = re.compile(r"read|write|edit|ls|list|glob|grep|search|fs", re.IGNORECASE)
# 从任意文本里抓绝对路径。
ABS_PATH = re.compile(r"/(?:Users|home|root|opt|var|etc|private|Volumes)/[A-Za-z0-9_./-]*")


def _decompress(path: Path) -> list[dict]:
    import subprocess

    proc = subprocess.run(["zstd", "-d", "-c", str(path)], capture_output=True, check=True)
    events = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _text_of(message: dict) -> str:
    return "".join(
        c.get("text") or ""
        for c in (message.get("content") or [])
        if isinstance(c, dict) and c.get("type") == "text"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="会话 id 前缀；默认最新")
    ap.add_argument("--allow", action="append", default=[],
                    help="允许被触碰的目录前缀，可重复")
    ap.add_argument("--forbid", action="append", default=[],
                    help="一旦出现在参数或返回里就判为泄漏的字符串，可重复")
    args = ap.parse_args()

    found = sorted(SESSIONS.rglob("session.v4.jsonl.zstd"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        print(f"没找到会话：{SESSIONS}")
        return 1
    target = found[0]
    if args.session:
        hit = [p for p in found if args.session in p.parent.name]
        if not hit:
            print(f"没有匹配 {args.session!r} 的会话")
            return 1
        target = hit[0]

    events = _decompress(target)
    allow = tuple(args.allow) or ("/tmp/",)
    forbid = tuple(args.forbid)

    print(f"会话：{target.parent.name}")
    print(f"允许的前缀：{list(allow)}")
    print(f"禁词：{list(forbid) or '(未指定)'}\n")

    violations: list[str] = []
    calls = 0
    pending = None
    for ev in events:
        t = ev.get("type")
        if t == "tool/call":
            pending = ev.get("data") or {}
        elif t != "tool/result":
            continue
        else:
            call, result = pending or {}, ev.get("data") or {}
            pending = None
            calls += 1
            name = call.get("name") or "?"
            raw = json.dumps(call.get("arguments"), ensure_ascii=False)
            text = _text_of(result.get("message") or {})

            # ① 文件类工具的**参数**里出现允许范围之外的绝对路径
            if FS_TOOL_HINT.search(name):
                for p in set(ABS_PATH.findall(raw)):
                    if not p.startswith(allow):
                        violations.append(f"[{calls}] {name} 参数越界：{p}")

            # ② 任何返回里出现禁词
            for word in forbid:
                if word and word in text:
                    violations.append(f"[{calls}] {name} 返回里出现禁词 {word!r}")

    print(f"检查了 {calls} 次工具调用。")
    if violations:
        print(f"\n✗ 发现 {len(violations)} 处越界：")
        for v in violations:
            print(f"   {v}")
        print("\n这次实验的结果**不能用** —— 模型的观测面已经越界。")
        return 1
    print("\n✓ 干净：没有任何越界读。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
