#!/usr/bin/env python3
"""PvZ 教师 MCP server —— 把 episode_query.py 的能力暴露成模型可调的工具。

这个进程的职责边界
------------------
它**只做搬运**：把模型的一次工具调用翻译成一条 `episode_query.py` 子进程命令，
再把 stdout 原样回传。它自己**不做任何游戏判断** —— 没有阈值、没有规则、
没有"这一路危险"这种结论。所有判断留给模型。

为什么是子进程而不是 import
---------------------------
`episode_query.py` 需要 `scripts/` 与 `python/` 两个目录在 `sys.path` 上，
而且它依赖真实的模拟器资源目录。子进程隔离让这个 server 本身零重依赖
（只需要官方 `mcp` SDK），也让"模型看到的就是人在命令行看到的那一屏"
这件事成立 —— 同一个入口，同一份输出，不存在第二套实现。

协议
----
stdio，JSON-RPC。用官方 `mcp` SDK 的低层 `Server`（回调式，不是 FastMCP），
因为我们需要对 tools/list 的输出有完全控制。

阶段
----
S0（当前）：只暴露 `ping`。目的是验证"bundle → profile → 组合树 → 会话里能调到"
这条管道通不通，**不通就停下改方案，不往上堆功能**。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# ---------------------------------------------------------------------------
# 定位仓库与解释器
# ---------------------------------------------------------------------------
#
# 注意：DSH 起这个子进程时用的是**被擦除过的环境**（凭据形态的变量、过期的
# DSH_* 都被丢掉），所以 PATH 之类不能依赖。这里一律用绝对路径，或从本文件
# 的位置反推。`PVZ_ROOT` / `PVZ_PYTHON` 允许在 bundle 的 config.env 里显式覆盖。

_THIS = Path(__file__).resolve()


def _guess_root() -> Path:
    """从本文件位置反推仓库根：<root>/dsh/pvz-teacher/server/pvz_mcp_server.py"""
    return _THIS.parents[3]


ROOT = Path(os.environ.get("PVZ_ROOT") or _guess_root())
PYTHON = os.environ.get("PVZ_PYTHON") or sys.executable
QUERY_SCRIPT = ROOT / "scripts" / "episode_query.py"


def _run_query(args: list[str], timeout: float = 900.0) -> tuple[int, str, str]:
    """跑一次 episode_query.py，返回 (returncode, stdout, stderr)。

    超时按"跑一局"的量级给：实测 level 7 单局 0.23~1.39 s，但 `whatif --enumerate`
    会重放十几个候选，而且首次启动要载入模拟器资源。900 s 是上限不是预期。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts"), str(ROOT / "python"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [PYTHON, str(QUERY_SCRIPT), *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# 工具表
# ---------------------------------------------------------------------------
#
# 每项：name -> (description, inputSchema, handler)
# handler 收到 dict 参数，返回 (text, is_error)。

Handler = Callable[[dict[str, Any]], tuple[str, bool]]


def _tool_ping(args: dict[str, Any]) -> tuple[str, bool]:
    """S0 的连通性探针。回传它实际看到的仓库状态，好让"通了"这件事可证伪。"""
    info = {
        "pong": True,
        "root": str(ROOT),
        "root_exists": ROOT.is_dir(),
        "python": PYTHON,
        "python_exists": Path(PYTHON).is_file() or bool(_which(PYTHON)),
        "query_script": str(QUERY_SCRIPT),
        "query_script_exists": QUERY_SCRIPT.is_file(),
        "argv_echo": args,
    }
    return json.dumps(info, ensure_ascii=False, indent=2), False


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


TOOLS: dict[str, tuple[str, dict[str, Any], Handler]] = {
    "ping": (
        "连通性探针。回传本 server 看到的仓库根、解释器、工具脚本路径，"
        "以及它们是否真实存在。用它确认 MCP 管道通了、且指向的是正确的仓库。",
        {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "随便填点什么，会被原样回显，用来确认参数传递无损。",
                }
            },
            "additionalProperties": False,
        },
        _tool_ping,
    ),
}


# ---------------------------------------------------------------------------
# MCP 接线
# ---------------------------------------------------------------------------
#
# 工具名里的 `pvz` 前缀（`mcp__pvz__<tool>`）不在这里定义 —— 它是 bundle 的
# `config.serverName`，见 ../bundle/cordis.patch.yml。这里再写一份常量只会漂移。

SERVER_VERSION = "0.1.0-s0"


async def _on_list_tools(
    ctx: Any, params: Any
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=desc, input_schema=schema)
            for name, (desc, schema, _handler) in TOOLS.items()
        ]
    )


async def _on_call_tool(
    ctx: Any, params: types.CallToolRequestParams
) -> types.CallToolResult:
    name = params.name
    raw_args = params.arguments or {}

    entry = TOOLS.get(name)
    if entry is None:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"没有这个工具：{name!r}。可用：{sorted(TOOLS)}")],
            is_error=True,
        )

    _desc, _schema, handler = entry
    try:
        text, is_error = await asyncio.to_thread(handler, dict(raw_args))
    except Exception as exc:  # noqa: BLE001 —— 错误要回给模型看，不是打死 server
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
            is_error=True,
        )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        is_error=is_error,
    )


server: Server = Server(
    "pvz-teacher",
    version=SERVER_VERSION,
    instructions=(
        "PvZ 对局诊断工具。这一阶段只有 ping —— 管道验证用的探针。"
    ),
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
