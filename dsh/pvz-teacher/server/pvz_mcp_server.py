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
- S0（2026-10-03 通过，两重验收）：只暴露 `ping`，验证"bundle → profile → 组合树 →
  会话里能调到"这条管道。管道不通就停下改方案，不往上堆功能。
- S1（2026-10-03 通过）：把 `capture / index / frame / lane / actions / whatif /
  narrative` 暴露出来。它们与 CLI 子命令一一对应，`ping` 保留作探针。
- S2（2026-10-03 通过）：词汇表问题修复（`vocabulary` 工具、别名输入、
  可复制的枚举列、自纠正报错），并把 `vocabulary` 发布成资源 `pvz://vocabulary`
  —— 见下面「资源」一节的实测依据。
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
#
# 设计立场：**工具与 `episode_query.py` 的子命令一一对应，不做二次加工。**
# 模型看到的就是人在命令行看到的那一屏 —— 同一个入口，同一份输出，
# 不存在第二套实现，也就不存在"工具的结论和 CLI 的结论对不上"这种问题。
# 唯一的例外是超长输出会被截断（见 MAX_OUTPUT_CHARS），且截断处有明确标注。

Handler = Callable[[dict[str, Any]], tuple[str, bool]]

# 工具输出上限。CLI 本身就是按"一次一屏"设计的，但 `whatif --enumerate`
# 在没收敛参数时可能很长。超了截断并**明确告知**，不静默丢内容 ——
# 静默截断会让模型以为自己看到了全部，那比报错更坏。
MAX_OUTPUT_CHARS = int(os.environ.get("PVZ_MCP_MAX_OUTPUT_CHARS") or 60000)

# 模拟器资源目录。CLI 的默认值是 `~/.cache/pvz-research-resources`，
# 本机并不存在（真目录在 Downloads 下），所以这里用环境变量显式给，
# 由 bundle 的 `config.env` 注入。
RESOURCE_DIR = os.environ.get("PVZ_RESOURCE_DIR") or ""


def _tool_ping(args: dict[str, Any]) -> tuple[str, bool]:
    """S0 的连通性探针。回传它实际看到的仓库状态，好让"通了"这件事可证伪。

    ⚠️ 默认**只回最小信息**，绝对路径要显式开 `PVZ_MCP_VERBOSE_PING=1` 才给。
    为什么改（2026-10-03 实测）：这个工具原先把 `root` / `query_script` 的绝对
    路径一起回了。S3 那轮里模型第 2 步调它，**第 8 步就直接去读
    `/Users/newbiexvwu/PvZAgent`** —— 而当时 `tool-fs` 是开着的（S3 要让它写
    skill），`fs-sandbox` 只限制**写**、不限制读。也就是说：
    **我们自己的探针把仓库地址告诉了模型，等于亲手拆掉了隔离层。**
    探针的价值在于"通没通"，不在于"仓库在哪"；后者是给做实验的人看的，
    所以放进显式开关。
    """
    if os.environ.get("PVZ_MCP_VERBOSE_PING") != "1":
        return json.dumps(
            {"pong": True, "tools": sorted(TOOLS), "argv_echo": args},
            ensure_ascii=False,
            indent=2,
        ), False

    info = {
        "pong": True,
        "root": str(ROOT),
        "root_exists": ROOT.is_dir(),
        "python": PYTHON,
        "python_exists": Path(PYTHON).is_file() or bool(_which(PYTHON)),
        "query_script": str(QUERY_SCRIPT),
        "query_script_exists": QUERY_SCRIPT.is_file(),
        "resource_dir": RESOURCE_DIR or "(未设置)",
        "resource_dir_exists": Path(RESOURCE_DIR).is_dir() if RESOURCE_DIR else False,
        "max_output_chars": MAX_OUTPUT_CHARS,
        "tools": sorted(TOOLS),
        "argv_echo": args,
    }
    return json.dumps(info, ensure_ascii=False, indent=2), False


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _run_query_tool(argv: list[str], timeout: float = 900.0) -> tuple[str, bool]:
    """跑一次 CLI，把结果整理成 (text, is_error)。

    失败**不抛异常**：把 stdout/stderr 原样交给模型，让它自己判断是参数写错了
    还是环境有问题。工具把错误吞掉或包装成友好文案，模型就失去了纠错依据。
    """
    try:
        rc, out, err = _run_query(argv, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (
            f"超时（{timeout:.0f}s 未返回）：episode_query.py {' '.join(argv)}\n"
            f"如果这是 whatif --enumerate，用 row / limit 收敛枚举范围。",
            True,
        )

    if rc != 0:
        detail = (err or "").strip() or (out or "").strip() or "(没有任何输出)"
        return f"episode_query.py 退出码 {rc}：\n{detail}", True

    text = out.rstrip("\n")
    if not text:
        return "(命令成功，但没有任何输出)", False
    if len(text) > MAX_OUTPUT_CHARS:
        return (
            text[:MAX_OUTPUT_CHARS]
            + f"\n\n[输出被截断：共 {len(text)} 字符，只回了前 {MAX_OUTPUT_CHARS} 个。"
            f"用更窄的参数（every / wave / limit / row）重看一遍，别当成已看全。]",
            False,
        )
    return text, False


def _cli_argv(cmd: str, args: dict[str, Any], spec: list[tuple[str, str, str]]) -> list[str]:
    """把工具参数翻译成 CLI argv。

    spec 里每项是 (参数名, CLI 旗标, 类型)，类型 ∈ str / int / flag / list / env。
    `env` 用于资源目录这类由部署环境注入、不该让模型填的参数。
    """
    argv = [cmd]
    for key, flag, kind in spec:
        if kind == "env":
            if RESOURCE_DIR:
                argv += [flag, RESOURCE_DIR]
            continue
        if key not in args or args[key] is None:
            continue
        val = args[key]
        if kind == "flag":
            if val:
                argv.append(flag)
        elif kind == "list":
            for item in val:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(val)]
    return argv


def _make_cli_handler(cmd: str, spec: list[tuple[str, str, str]]) -> Handler:
    def handler(args: dict[str, Any]) -> tuple[str, bool]:
        return _run_query_tool(_cli_argv(cmd, args, spec))

    return handler


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_A_ARCHIVE = {
    "type": "string",
    "description": "存档目录路径（capture 的 out 返回的那个）。",
}

TOOLS: dict[str, tuple[str, dict[str, Any], Handler]] = {
    "vocabulary": (
        "**动作写法速查**。whatif 的 try 参数怎么写、每个植物对应哪个 packet 数字、"
        "有哪些合法动作类型。第一次用 whatif 前看一眼，能省掉试错。\n"
        "（加这个工具的原因：实测模型为了搞清楚 packet id 浪费了 6 次调用，"
        "还因为猜错数字拿到了**另一个植物的结果**却没察觉。）",
        _schema({}, []),
        _make_cli_handler("vocabulary", []),
    ),
    "ping": (
        "连通性探针。回传本 server 看到的仓库根、解释器、工具脚本路径、资源目录，"
        "以及它们是否真实存在，还有当前可用的工具列表。"
        "用它确认 MCP 管道通了、且指向的是正确的仓库。",
        _schema(
            {
                "note": {
                    "type": "string",
                    "description": "随便填点什么，会被原样回显，用来确认参数传递无损。",
                }
            },
            [],
        ),
        _tool_ping,
    ),
    "capture": (
        "跑一局并**完整落盘**成存档。存档是无损的：每一帧都在盘上，之后可以用 "
        "frame / lane / strip / trace 查任意 tick、任意路、任意一只僵尸。"
        "这是所有其它工具的前提 —— 没有存档，其它工具无从查起。"
        "一局 level 7 大约 0.2~1.4 s。",
        _schema(
            {
                "seed": {"type": "integer", "description": "随机种子。同一个 seed 结果完全可复现。"},
                "level": {"type": "integer", "description": "关卡号，默认 7。"},
                "policy": {
                    "type": "string",
                    "enum": ["scripted", "donothing"],
                    "description": "用哪套策略打这一局：scripted（默认）或 donothing（对照用）。",
                },
                "deck": {"type": "string", "description": "卡组，逗号分隔的卡片 id。不给就用该关默认卡组。"},
                "max_actions": {"type": "integer", "description": "动作数上限，默认 4000。"},
                "out": {"type": "string", "description": "存档写到哪个目录（必需）。"},
            },
            ["seed", "out"],
        ),
        _make_cli_handler(
            "capture",
            [
                ("seed", "--seed", "int"),
                ("level", "--level", "int"),
                ("policy", "--policy", "str"),
                ("deck", "--deck", "str"),
                ("max_actions", "--max-actions", "int"),
                ("out", "--out", "str"),
                ("_res", "--resource-dir", "env"),
            ],
        ),
    ),
    "index": (
        "存档的**目录**：哪里值得看。它列出各条信号（每条都带产生它的规则名），"
        "但**不构成结论** —— 规则能覆盖的失败方式有限，这些只是候选，"
        "每一条都可以被后续的 frame / lane 查询推翻。建议第一眼先看它。",
        _schema({"archive": _A_ARCHIVE}, ["archive"]),
        _make_cli_handler("index", [("archive", "--archive", "str")]),
    ),
    "frame": (
        "看某一 tick 的完整状态（含相邻帧做对比）。tick 会被吸附到最近的已存帧。"
        "这是最细的粒度 —— 当你已经知道要盯哪一步时用它。",
        _schema(
            {
                "archive": _A_ARCHIVE,
                "tick": {"type": "integer", "description": "目标 tick，会吸附到最近的已存帧。"},
            },
            ["archive", "tick"],
        ),
        _make_cli_handler("frame", [("archive", "--archive", "str"), ("tick", "--tick", "int")]),
    ),
    "lane": (
        "按**行（路）**看整局的演变。想知道“第 3 路是什么时候崩的”就用它。",
        _schema(
            {
                "archive": _A_ARCHIVE,
                "row": {"type": "integer", "description": "行号（路）。"},
            },
            ["archive", "row"],
        ),
        _make_cli_handler("lane", [("archive", "--archive", "str"), ("row", "--row", "int")]),
    ),
    "actions": (
        "列出这局**做过的每个决策**（种了什么、种在哪、活了多久）。"
        "`max_life` 能过滤出“种下去很快就死”的决策 —— 这类决策最值得复盘，"
        "因为它的失败是局部的、可归因的。",
        _schema(
            {
                "archive": _A_ARCHIVE,
                "from": {"type": "integer", "description": "起始决策序号，默认 1。"},
                "to": {"type": "integer", "description": "结束决策序号，默认到最后一个。"},
                "max_life": {
                    "type": "integer",
                    "description": "只显示种下去活不过这么多 tick 的决策。",
                },
            },
            ["archive"],
        ),
        _make_cli_handler(
            "actions",
            [
                ("archive", "--archive", "str"),
                ("from", "--from", "int"),
                ("to", "--to", "int"),
                ("max_life", "--max-life", "int"),
            ],
        ),
    ),
    "whatif": (
        "**反事实回放** —— 这是整条线上最关键的工具。在指定的决策点上，"
        "把当时的动作换成别的动作，然后**真的重放一整局**，告诉你结果如何。"
        "因为模拟器是确定性的（同一 task+seed+动作序列结果逐位相同），"
        "这里得到的不是估计、不是启发式，而是**精确的**结果。\n"
        "两种用法：① 给 `try` 指定想试的动作；② `enumerate=true` 枚举该点上"
        "所有合法动作（用 row / limit 收敛）。\n"
        "`save_best` 填一个目录路径时，会把最好的那个反事实整局存成新存档，"
        "之后可以用 lane / narrative / frame 复查它 —— 也就是“改完之后这局长什么样”。",
        _schema(
            {
                "archive": _A_ARCHIVE,
                "decision": {"type": "integer", "description": "要改的决策序号（见 actions）。"},
                "try": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "想试的替代动作，可给多个。写法："
                    "plant:packet:row:col / wait:ticks / shovel:row:col，或原始 JSON。",
                },
                "enumerate": {
                    "type": "boolean",
                    "description": "枚举该决策点上所有合法动作（用 row / limit 收敛）。",
                },
                "row": {"type": "integer", "description": "枚举时只试这一行。"},
                "limit": {"type": "integer", "description": "枚举上限，默认 12。"},
                "save_best": {
                    "type": "string",
                    "description": "把最好的反事实整局存成一个**新存档目录**（填目录路径），"
                    "之后可以用 lane / narrative / frame 复查它 —— 也就是"
                    "\u201c改完之后这局长什么样\u201d。",
                },
            },
            ["archive", "decision"],
        ),
        _make_cli_handler(
            "whatif",
            [
                ("archive", "--archive", "str"),
                ("decision", "--decision", "int"),
                ("try", "--try", "list"),
                ("enumerate", "--enumerate", "flag"),
                ("row", "--row", "int"),
                ("limit", "--limit", "int"),
                ("save_best", "--save-best", "str"),
                ("_res", "--resource-dir", "env"),
            ],
        ),
    ),
    "narrative": (
        "把一段时间**叙述成一段话**（按波次或全局限定范围）。"
        "适合快速建立“这局大概怎么输的”的印象，代价是细节被压缩 —— "
        "压缩时哪些留哪些丢是规则决定的，所以它**只能当线索，不能当结论**，"
        "要下判断请回到 frame / lane / whatif。",
        _schema(
            {
                "archive": _A_ARCHIVE,
                "wave": {"type": "integer", "description": "起始波次。不给就从头。"},
                "to_wave": {"type": "integer", "description": "结束波次。"},
                "detail": {
                    "type": "integer",
                    "enum": [1, 2, 3],
                    "description": "详细程度 1~3，默认 2。",
                },
            },
            ["archive"],
        ),
        _make_cli_handler(
            "narrative",
            [
                ("archive", "--archive", "str"),
                ("wave", "--wave", "int"),
                ("to_wave", "--to-wave", "int"),
                ("detail", "--detail", "int"),
            ],
        ),
    ),
}


# ---------------------------------------------------------------------------
# 资源（MCP resources）—— 参考类文档走这里，不走 prompt
# ---------------------------------------------------------------------------
#
# 为什么除了 `vocabulary` 工具之外还要发一份资源
# ------------------------------------------------
# 2026-10-03 实测：模型不知道动作语法时，**第一反应不是猜，是找文档**。
# 它在 S2 干净轮里连着调了 4 次：
#
#   [6] ✗ unknown tool "mcp__pvz__list_mcp_resources"
#   [7] ✗ unknown tool "mcp__pvz__list_mcp_resource_templates"
#   [8] · MCP server: pvz  {"resources":[]}
#   [9] · MCP server: pvz  {"resourceTemplates":[]}
#
# 而这 4 次是**它唯一知道的文档通道**：DSH 的 `mcp-resources` 只要配了
# 一个 MCP server 就会挂上那三个共享工具，并且 **system prompt 里会列出
# server 名** —— 也就是它被明确告知"有 pvz，去问它"。我们一个资源都没发布，
# 于是它拿到的是一句"空的"。
#
# 所以：参考类文档（语法、词表、枚举表）**发布成资源**，
# 让模型走它已经会走的通道；不读就不占上下文。
#
# 单一事实源
# ----------
# 资源正文由 `episode_query.py vocabulary` 现场生成，与 `vocabulary` 工具
# 走的是**同一条命令**。这里绝不手抄一份 —— 两份必然漂移。

RESOURCES: dict[str, tuple[str, str, str]] = {
    # uri -> (name, description, mimeType)
    "pvz://vocabulary": (
        "动作写法速查",
        "whatif 的 try 参数怎么写、每个植物对应哪个 packet 数字、"
        "合法动作类型、行列范围。不确定语法时先读这一份。",
        "text/markdown",
    ),
}


def _resource_body(uri: str) -> str:
    """按 uri 现场生成资源正文。不认识就抛，让调用方看到可用清单。"""
    if uri == "pvz://vocabulary":
        rc, out, err = _run_query(["vocabulary"], timeout=120.0)
        if rc != 0:
            return (
                f"[读取 {uri} 失败] episode_query.py vocabulary 退出码 {rc}\n"
                f"--- stdout ---\n{out}\n--- stderr ---\n{err}"
            )
        return out
    raise ValueError(f"没有这个资源：{uri!r}。可用：{sorted(RESOURCES)}")


# ---------------------------------------------------------------------------
# MCP 接线
# ---------------------------------------------------------------------------
#
# 工具名里的 `pvz` 前缀（`mcp__pvz__<tool>`）不在这里定义 —— 它是 bundle 的
# `config.serverName`，见 ../bundle/cordis.patch.yml。这里再写一份常量只会漂移。

SERVER_VERSION = "0.3.0-s2"


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


async def _on_list_resources(
    ctx: Any, params: Any
) -> types.ListResourcesResult:
    return types.ListResourcesResult(
        resources=[
            types.Resource(
                uri=uri, name=name, description=desc, mimeType=mime
            )
            for uri, (name, desc, mime) in RESOURCES.items()
        ]
    )


async def _on_read_resource(
    ctx: Any, params: types.ReadResourceRequestParams
) -> types.ReadResourceResult:
    uri = str(params.uri)
    entry = RESOURCES.get(uri)
    if entry is None:
        # 抛出去 → JSON-RPC error → 模型看到失败原因与可用清单，而不是空内容。
        raise ValueError(f"没有这个资源：{uri!r}。可用：{sorted(RESOURCES)}")
    text = await asyncio.to_thread(_resource_body, uri)
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(uri=uri, mimeType=entry[2], text=text)
        ]
    )


server: Server = Server(
    "pvz-teacher",
    version=SERVER_VERSION,
    instructions=(
        "PvZ 对局诊断工具。工作流通常是：先用 capture 把一局完整落盘，"
        "再用 index 看哪里值得看，然后用 frame / lane / actions 细看，"
        "最后用 whatif 做**精确的**反事实回放来验证你的改动到底有没有用。\n"
        "所有工具都只是搬运 episode_query.py 的输出，不含任何游戏判断 —— "
        "结论要你自己下。模拟器是确定性的，所以 whatif 给的是精确结果而不是估计。\n"
        "**不确定动作怎么写**（plant 的 packet 是什么、行列范围、有哪些动作类型）"
        "时，读资源 `pvz://vocabulary`，或调 `vocabulary` 工具 —— "
        "两者是同一份内容，别靠猜 id。"
    ),
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
    on_list_resources=_on_list_resources,
    on_read_resource=_on_read_resource,
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
