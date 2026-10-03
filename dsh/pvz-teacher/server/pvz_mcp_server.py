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
- S3（2026-10-03）：常量与规则补全 —— `constants` 工具 + `pvz://constants`、
  `policy` 工具 + `pvz://scripted-policy`。触发点是 S3 会话的推理轨迹：
  模型在**猜**游戏常量（土豆雷引爆时间猜了 900 → ~1600，实际是 1500 tick 倒计时
  ＋升起动画），并反复反推脚本策略的规则而反推不出来。它猜错的不是游戏常识，
  而是 tick↔秒 的换算率（60 vs 100）。一个常数错，写下的"可迁移结论"整条跟着错。
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


def _run_script(script: str, argv: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
    """跑 `scripts/` 下的另一个脚本（不是 episode_query.py）。

    `lint_skills.py` 是独立脚本 —— 它不归 `episode_query.py` 管，
    硬塞成它的子命令只会让两个入口互相污染。
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts"), str(ROOT / "python"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / script), *argv],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _tool_lint_skills(args: dict[str, Any]) -> tuple[str, bool]:
    """skill 闸门。**写完 skill 一定要跑它** —— 见工具描述里的理由。

    为什么做成工具而不是让模型跑命令行（2026-10-03 实测发现的问题）：
    隔离层 `patches/s3-allow-skill.yml` 里 **`tool-bash` 是关掉的**
    （正是 2026-10-03 那次答案泄漏的通道），所以模型**没有命令行可用**。
    任务模板第一版里写"写完自己跑 `python3 scripts/lint_skills.py`"，
    那句话在沙箱里根本执行不了 —— 一句执行不了的指令比不写更坏：
    它会让模型以为自己验过了。
    """
    d = args.get("dir") or os.environ.get("PVZ_SKILLS_DIR") or ""
    if not d:
        return ("没有给 `dir`，环境变量 PVZ_SKILLS_DIR 也没设。\n"
                "填你写 skill 的那个目录（它的每个子目录是一个 skill），"
                "例如 /tmp/pvz-s4/.dsh/skills 。"), True
    argv = [d]
    if args.get("no_check_archives"):
        argv.append("--no-check-archives")
    try:
        rc, out, err = _run_script("lint_skills.py", argv)
    except subprocess.TimeoutExpired:
        return f"超时（60s）：lint_skills.py {d}", True
    text = (out or "").rstrip("\n") or "(没有输出)"
    if rc != 0 and not text:
        text = err.strip() or "(退出码非 0，且没有任何输出)"
    return text, rc != 0


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


def _make_cli_handler_probe() -> Handler:
    """probe 工具要按给的参数选子命令（eval / check），不是固定一个。"""
    def handler(args: dict[str, Any]) -> tuple[str, bool]:
        if args.get("skill"):
            return _run_query_tool(["probe", "check", "--skill", args["skill"]])
        if args.get("archive") and args.get("probe"):
            return _run_query_tool(["probe", "eval", "--probe", args["probe"],
                                    "--archive", args["archive"]])
        return ("probe 需要成对给参数：eval 要 archive + probe，check 要 skill。"
                "不确定能写哪些字段就读资源 pvz://probe-fields。"), True
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
    # `constants` 与 `policy` 两个工具：**为什么值得常驻**（2026-10-03 实测）
    #
    # 这两个是"调用**之前**就得知道的东西"这一类 —— 与 `vocabulary` 同理：
    # 模型不会去查一个它不知道自己需要查的表。
    #
    # · 常量：模型花了大段推理去**猜**土豆雷引爆时间，先猜 900 tick、又反推到
    #   ~1600。正确答案是 1500 tick 倒计时 + 一段升起动画。它猜错的不是游戏常识，
    #   而是 tick↔秒 的换算率（按 60 tick/s 算，这里是 100）。一个常数错，
    #   它写下的「可迁移结论」整条跟着错 —— 而那条结论是要沉淀成 skill 的。
    #   所以描述里必须明说"别用通用 PvZ 常识推，这是另一个实现"。
    #
    # · 策略规则：模型多轮在推理里反推"脚本策略到底按什么规则走"，
    #   原文如「maybe the policy adds a shooter when the row is at ⚠…but row 3's
    #   turn never came」。规则本身不在它能看到的任何数据里 —— 数据是行为的
    #   结果，不是行为的规则。
    #
    # 描述写短、理由写这里：工具描述**每个请求都发**，注释不发。
    "constants": (
        "**游戏常量表**，从 C++ 源码现场解析（不是抄本，不会漂移）。\n"
        "时间基准（多少 tick 算一秒）、战斗常量、植物表（花费/冷却/攻击间隔/血量）、"
        "僵尸表（血量/首次出现关卡与波/抽样权重）、**场景与地形**（哪几路种得下）、"
        "源码常量清单。\n"
        "给 level 还会现算该关的波数与可能出现的僵尸。\n"
        "**不确定某个数值时来这里查，别用通用 PvZ 常识推** —— 这是另一个实现。",
        _schema(
            {
                "section": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["time", "combat", "plants", "zombies", "level", "terrain",
                                 "literals"],
                    },
                    "description": "只取这几节（省 token）。不给就全给。",
                },
                "level": {
                    "type": "integer",
                    "description": "关卡号 1..50。给了就额外算这一关的波数与可能出现的僵尸。",
                },
            },
            [],
        ),
        _make_cli_handler("constants", [("section", "--section", "list"),
                                        ("level", "--level", "int")]),
    ),
    "policy": (
        "**这一局是谁在打、按什么规则。**\n"
        "给了 --archive 就按那一份存档回答：RL 存档会明确告诉你它是训练出来的模型、"
        "**没有可读的规则表**（别拿脚本规则去解释它）。不给存档就是脚本策略的规则表。",
        _schema({"archive": {"type": "string",
                             "description": "存档目录。强烈建议给：RL 局和脚本局的答案完全不同。"}},
                []),
        _make_cli_handler("policy", [("archive", "--archive", "str")]),
    ),
    # `lint_skills`：**写完 skill 一定要调它**。
    #
    # 它的判据（三问）见任务模板，这里只说为什么它必须存在：
    # 判据喊口号没用 —— 2026-10-03 的 S3 里模型产出了 4 条判断却没写任何 skill，
    # 推理里 `skill` 一词出现 0 次。所以闸门要能**当场**告诉它对不对：
    # 证据不足就当场拒，它才有机会补；事后才发现就只能作废。
    #
    # 描述写短、理由写注释：工具描述每个请求都发。
    "lint_skills": (
        "**skill 闸门** —— 写完 skill 之后必须跑一次。\n"
        "它查：证据有没有、够不够（至少 2 个不同决策点）、"
        "以及每条证据里的「改动前」是否与存档里那一帧**真实做出的动作**一致"
        "（存档是跑出来的，改不了 —— 对不上就是这条证据编的）。\n"
        "返回 ✗ 时按它说的补证据；**别把没过闸门的 skill 留在库里。**",
        _schema(
            {
                "dir": {
                    "type": "string",
                    "description": "skill 所在目录（每个子目录是一个 skill）。"
                                   "就是任务里让你写 SKILL.md 的那个目录。",
                },
                "no_check_archives": {
                    "type": "boolean",
                    "description": "只查结构，不去存档里核对（存档不在本机时）。默认 false。",
                },
            },
            ["dir"],
        ),
        _tool_lint_skills,
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
                    "enum": ["scripted", "donothing", "ppo"],
                    "description": "用哪套策略打这一局：scripted（默认）、donothing（对照用）、"
                                   "ppo（训练出来的 RL 模型，必须同时给 checkpoint）。",
                },
                "checkpoint": {"type": "string",
                               "description": "policy=ppo 时的检查点 .pt 路径。只做推理，不训练。"},
                "sampled": {"type": "boolean",
                            "description": "policy=ppo 时按概率采样；默认贪心（可复现）。"},
                "deck": {"type": "string", "description": "卡组，逗号分隔的卡片 id。不给就用该关默认卡组。"},
                "wave_cap": {"type": "integer",
                             "description": "只打前 N 波。复现训练任务族时常用（它们多半设了上限）。"},
                "zombie_mult": {"type": "number", "description": "僵尸数量倍率（复现训练任务）。"},
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
                ("checkpoint", "--checkpoint", "str"),
                ("sampled", "--sampled", "flag"),
                ("deck", "--deck", "str"),
                ("wave_cap", "--wave-cap", "int"),
                ("zombie_mult", "--zombie-mult", "float"),
                ("max_actions", "--max-actions", "int"),
                ("out", "--out", "str"),
                ("_res", "--resource-dir", "env"),
            ],
        ),
    ),
    # `probe`：**把"什么算这个失败模式"写成能自动跑的条件**（可选）。
    #
    # 为什么它值得存在：教师看一局要几分钟，训练一晚上打几万局 —— 没有这一层，
    # 教师永远只能看它碰巧看到的那一局。有了这一层，它写下的判据能被自动跑在
    # 每一局上，把可疑的局挑出来。
    #
    # 为什么条件是封闭词表：它只能引用工具本来就在显示的量。**不许发明新逻辑**
    # —— 一条能自由发挥的"条件"迟早会长成第二个策略脚本，那就是把硬编码塞回主线。
    "probe": (
        "把失败模式写成**能自动跑在每一局上的条件**，或验收一条写好的条件。\n"
        "给 archive + probe 就是在那一局上跑（列出命中的决策）；给 skill 就是验收\n"
        "（派生局必须命中、反例必须不命中，过不了就不入库）。\n"
        "条件只能引用 `pvz://probe-fields` 里那几个字段 —— 不许自己发明判据。",
        _schema({
            "archive": {"type": "string", "description": "存档目录（eval 模式）。"},
            "probe": {"type": "string",
                      "description": "条件，JSON：如 {\"action_type\":\"plant\","
                                     "\"plant_role\":[\"producer\"],"
                                     "\"lane_front_zombie_x_max\":260}"},
            "skill": {"type": "string", "description": "skill 目录（check 模式）。"},
        }, []),
        _make_cli_handler_probe(),
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
        "**重放语义**：用**同一套策略从头重跑**，只把那一步的动作换掉；"
        "之后的决策由策略看着新局面**重新做出**，不是照抄原局录下的动作序列。"
        "所以结果回答的是「改了这一步之后，策略会怎么接着打」。\n"
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
    #
    # 判据（三条，按顺序问）：
    #   1. 调用**之前**就必须知道的 → 常驻（放进工具描述）
    #   2. 大而全的参考表，且模型知道自己需要它 → **资源**（读之前零成本）
    #   3. 兜底：模型连"自己缺什么"都不知道 → 只能在错误信息里列可用值
    #
    # `pvz://constants` 落在 2 和 3 之间：它比 `vocabulary` 大得多（1.2 万字符），
    # 常驻不划算；但"我不知道引爆时间"这件事模型**意识得到**（它确实去猜了），
    # 所以资源 + 一个可查询的工具是合适的组合。
    "pvz://vocabulary": (
        "动作写法速查",
        "whatif 的 try 参数怎么写、每个植物对应哪个 packet 数字、"
        "合法动作类型、行列范围。不确定语法时先读这一份。",
        "text/markdown",
    ),
    "pvz://constants": (
        "游戏常量表",
        "时间基准（tick↔秒）、战斗常量、植物表（花费/冷却/攻击间隔/血量）、"
        "僵尸表（血量/首次关卡与波/抽样权重）。从 C++ 源码现场解析。"
        "任何「这个数是多少」的问题都在这里，别用通用 PvZ 常识推。",
        "text/markdown",
    ),
    "pvz://scripted-policy": (
        "脚本策略的决策规则（**只适用于 scripted 存档**）",
        "脚本策略按什么顺序做决策、阈值是多少。想知道「它当时为什么这么走」就读这份。\n"
        "⚠ 如果手上那局是 RL 模型打的（capture 时 policy=ppo），这份规则描述的"
        "**不是**它 —— 那时用 policy 工具并带上 --archive，它会告诉你那局是谁在打。",
        "text/markdown",
    ),
    "pvz://probe-fields": (
        "可自动跑的「失败模式条件」能写哪些字段（封闭词表）",
        "把发现写成能跑在每一局上的条件时用。只能引用这里的字段 —— "
        "每个都是工具本来就在显示的量（火力、最前僵尸距离、植物类别、这株活了多久）。",
        "text/markdown",
    ),
}

# uri -> 生成正文的 CLI 子命令（**不在这里手写正文**）。
# 正文一律由 episode_query.py 现场生成，与对应工具走同一条命令 ——
# 这里再抄一份必然漂移，而漂移的参考文档会让模型照着不存在的规则推理。
RESOURCE_COMMANDS: dict[str, list[str]] = {
    "pvz://vocabulary": ["vocabulary"],
    "pvz://constants": ["constants"],
    "pvz://scripted-policy": ["policy"],
    "pvz://probe-fields": ["probe", "fields"],
}


def _resource_body(uri: str) -> str:
    """按 uri 现场生成资源正文。不认识就抛，让调用方看到可用清单。"""
    argv = RESOURCE_COMMANDS.get(uri)
    if argv is None:
        raise ValueError(f"没有这个资源：{uri!r}。可用：{sorted(RESOURCES)}")
    rc, out, err = _run_query(argv, timeout=120.0)
    if rc != 0:
        return (
            f"[读取 {uri} 失败] episode_query.py {' '.join(argv)} 退出码 {rc}\n"
            f"--- stdout ---\n{out}\n--- stderr ---\n{err}"
        )
    return out


# ---------------------------------------------------------------------------
# MCP 接线
# ---------------------------------------------------------------------------
#
# 工具名里的 `pvz` 前缀（`mcp__pvz__<tool>`）不在这里定义 —— 它是 bundle 的
# `config.serverName`，见 ../bundle/cordis.patch.yml。这里再写一份常量只会漂移。

SERVER_VERSION = "0.5.0-s4"


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
        "两者是同一份内容，别靠猜 id。\n"
        "**不确定某个数值**（血量、花费、冷却、引爆时间、tick 与秒的换算、"
        "这一关有多少波、会出现哪些僵尸）时，读资源 `pvz://constants`，"
        "或调 `constants` 工具。**这是另一个实现，不要用通用 PvZ 常识替代** —— "
        "实测有模型按 60 tick/s 推算引爆时间，而这里是 100 tick/s。\n"
        "**想知道打这一局的是谁、按什么规则走**时，调 `policy` 工具并带上存档路径。\n"
        "注意：如果那局是 RL 模型打的（policy=ppo），它**没有可读的规则表** —— "
        "那时别拿 `pvz://scripted-policy` 里的规则去解释它的每一步，那套规则描述的"
        "是另一个策略。想弄清它某一步为什么这么选，用 `whatif --enumerate` 把它当时"
        "所有合法动作的结果都算出来。\n"
        "**场景与地形**（这一关是白天/夜间/泳池/屋顶、哪几路种不下）在每个 "
        "`frame` / `index` / `narrative` 输出的「场景：…」那一行，不用另外查；"
        "棋盘上 `~` = 水路。要查规则本身看 `constants` 的 terrain 一节。\n"
        "**如果任务要求你把判断沉淀成 skill**：写完之后调 `lint_skills` 过闸门。"
        "这是唯一的检查入口 —— 隔离层里没有命令行可用。"
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
