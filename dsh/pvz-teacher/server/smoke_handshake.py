#!/usr/bin/env python3
"""不经过 DSH 的 MCP 冒烟测试。

为什么需要它
------------
如果只靠"装进 DSH 再试"，一旦调不通，你无法区分是
**server 本身坏了** 还是 **bundle/profile 接线错了**。
这个脚本把前半段单独隔离出来：用一个真的 MCP 客户端去握 server 的手。

为什么用官方客户端而不是手写 JSON-RPC
--------------------------------------
第一版手写了 JSON-RPC（写完 stdin 就关），结果 `tools/call` 稳定报
"Connection closed" —— 那是**测试自己的 bug**：stdin 一关，server 的写流就结束了，
还没处理的请求拿不到响应。这个坑不是 server 的，但会浪费半小时。
官方 `ClientSession` 正确管理生命周期，不再有这类假故障。

环境：刻意用**被擦除过的环境**
------------------------------
DSH 起 MCP 子进程时用的是 `scrubbedParentEnv()`（见
`packages/subprocess/subprocess/src/index.ts`）：丢掉名字匹配
`/KEY|PASSWORD|SECRET|TOKEN/i` 的、以及所有 `DSH_*` 前缀的，保留 PATH/HOME/locale。
这里照抄同一条规则，所以**这个测试通过 = DSH 里也不会因为环境被擦而挂**。
想用完整环境对比时加 `--full-env`。

用法
----
    <venv>/bin/python dsh/pvz-teacher/server/smoke_handshake.py [--full-env]

退出码 0 = initialize + tools/list + tools/call 全通。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = Path(__file__).resolve().parent
SERVER = HERE / "pvz_mcp_server.py"

# 与 DSH 侧 `@deepseek-ai/dsh-subprocess` 的 SENSITIVE_ENV_PATTERN 一致。
SENSITIVE_ENV_PATTERN = re.compile(r"KEY|PASSWORD|SECRET|TOKEN", re.IGNORECASE)
DSH_ENV_PREFIX = "DSH_"


def scrubbed_parent_env() -> dict[str, str]:
    """照抄 DSH 的擦除规则，让本地测试能预测 DSH 里的行为。"""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if SENSITIVE_ENV_PATTERN.search(key):
            continue
        if key.upper().startswith(DSH_ENV_PREFIX):
            continue
        env[key] = value
    return env


async def run(full_env: bool) -> int:
    env = dict(os.environ) if full_env else scrubbed_parent_env()
    label = "完整环境" if full_env else "已擦除环境（与 DSH 一致）"
    print(f"── 子进程环境：{label}，{len(env)} 个变量")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env=env,
        cwd=str(SERVER.parent),
    )

    ok = True
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = init.server_info
            print(
                f"✓ initialize —— 协商版本 {init.protocol_version}，"
                f"serverInfo {info.name}@{info.version}"
            )

            listed = await session.list_tools()
            names = [t.name for t in listed.tools]
            print(f"✓ tools/list —— {names}")
            if "ping" not in names:
                print("✗ tools/list 里没有 ping")
                ok = False

            called = await session.call_tool("ping", {"note": "冒烟测试"})
            if called.is_error:
                print(f"✗ ping 返回了错误：{called.content}")
                ok = False
            else:
                text = "".join(
                    c.text for c in called.content if getattr(c, "type", None) == "text"
                )
                print("✓ tools/call ping ——")
                for line in text.splitlines():
                    print(f"    {line}")
                if "冒烟测试" not in text:
                    print("✗ 参数没有被回显，参数传递链路有问题")
                    ok = False

    print("\n结果：" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full-env",
        action="store_true",
        help="用完整环境变量（默认用与 DSH 一致的已擦除环境）",
    )
    args = ap.parse_args()
    return asyncio.run(run(args.full_env))


if __name__ == "__main__":
    raise SystemExit(main())
