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
from mcp.shared.exceptions import MCPError

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
            print(f"✓ tools/list —— {len(names)} 个：{names}")

            # S1：工具面必须齐全，少一个就说明工具表被改坏了。
            expected = {"ping", "vocabulary", "capture", "index", "frame", "lane",
                        "actions", "whatif", "narrative"}
            missing = expected - set(names)
            if missing:
                print(f"✗ tools/list 缺少：{sorted(missing)}")
                ok = False

            # 每个工具的 inputSchema 必须能被解析成 object，且 required 里的字段
            # 都真的在 properties 里 —— 这类漂移（schema 声明了、handler 不认）
            # 在运行期表现为"模型填了参数但没生效"，很难查。
            for tool in listed.tools:
                schema = tool.input_schema or {}
                if schema.get("type") != "object":
                    print(f"✗ {tool.name} 的 inputSchema.type 不是 object")
                    ok = False
                    continue
                props = schema.get("properties") or {}
                undeclared = [r for r in (schema.get("required") or []) if r not in props]
                if undeclared:
                    print(f"✗ {tool.name} 的 required 里有未声明的字段：{undeclared}")
                    ok = False
                if not (tool.description or "").strip():
                    print(f"✗ {tool.name} 没有 description —— 模型只能靠猜")
                    ok = False

            # 描述文本里不能有零宽字符（曾经混进一个 U+200B，肉眼看不出来，
            # 却会让模型读到一个断掉的词）。
            invisible = {0x200B, 0x200C, 0x200D, 0xFEFF}
            for tool in listed.tools:
                blob = (tool.description or "")
                found = sorted({ord(c) for c in blob if ord(c) in invisible})
                if found:
                    print(f"✗ {tool.name} 的 description 含零宽字符："
                          f"{[hex(c) for c in found]}")
                    ok = False

            # 资源：S2 加的 `pvz://vocabulary`。它存在的理由是可实测的 ——
            # 模型找不到动作语法时会去调 `list_mcp_resources`（DSH 的 system prompt
            # 主动告诉了它这个通道），我们一个资源都不发布就等于给了它一个空房间。
            resources = await session.list_resources()
            uris = [str(r.uri) for r in resources.resources]
            print(f"✓ resources/list —— {len(uris)} 个：{uris}")
            if "pvz://vocabulary" not in uris:
                print("✗ 缺少资源 pvz://vocabulary —— 模型会拿到空列表")
                ok = False

            try:
                # 注意：客户端签名收的是 **str**，不是 AnyUrl ——
                # 传 AnyUrl 会被 pydantic 挡在**客户端**，服务端根本没被调到。
                # （第一版就踩了这个：负例"读不存在的 uri"因此**假通过**，
                #  它报的是客户端的 ValidationError，不是服务端的拒绝。）
                body = await session.read_resource("pvz://vocabulary")
                text = "".join(
                    getattr(c, "text", "") for c in body.contents
                )
                # 正文必须真的是"能照着写"的：有动作语法、有植物名、有行列范围。
                for needle in ("plant:", "豌豆射手", "row 0..4"):
                    if needle not in text:
                        print(f"✗ pvz://vocabulary 正文里没有 {needle!r}")
                        ok = False
                print(f"✓ resources/read pvz://vocabulary —— {len(text)} 字符")
            except Exception as exc:  # noqa: BLE001
                print(f"✗ 读 pvz://vocabulary 抛异常：{type(exc).__name__}: {exc}")
                ok = False

            # 不存在的 uri 必须**报错**，不能静默回空内容 ——
            # 静默空内容和"通道是空的"长得一模一样，模型分不出来。
            # 而且必须确认这是**服务端**的拒绝，不是客户端自己先挂了。
            try:
                await session.read_resource("pvz://nope")
                print("✗ 读不存在的资源居然成功了 —— 应该报错")
                ok = False
            except MCPError as exc:
                if "没有这个资源" in str(exc):
                    print("✓ resources/read 不存在的 uri —— 服务端正确拒绝")
                else:
                    print(f"✗ 拒绝原因不对：{exc}")
                    ok = False
            except Exception as exc:  # noqa: BLE001
                print(f"✗ 报的是客户端异常（服务端没被调到）："
                      f"{type(exc).__name__}: {exc}")
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
