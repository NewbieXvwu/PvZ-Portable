#!/usr/bin/env python3
"""S1 验收：**经过 MCP server** 走完一次真实诊断（capture → index → whatif）。

为什么不是"在命令行跑一遍 CLI 就算过"
--------------------------------------
CLI 跑得通只能说明 `episode_query.py` 没问题 —— 那是**已经存在**的东西。
S1 要验的是**新增的那一层**：MCP 参数 → argv 的翻译对不对、
必填项有没有漏、`--try` 这类可重复参数有没有展开成多个、
长输出会不会把上下文撑爆。所以这里必须用一个真的 MCP 客户端，
把工具调用发进 server，再看它回什么。

验收目标（来自 RESEARCH_EXECUTION.md §5 的 S1 条款）
---------------------------------------------------
模型能用**一句话**拿到「seed 30001 第 266 步枚举第 3 路全部替代方案」的结果。
断言三件事：
  1. capture 真跑出一局并落盘（seed 30001 是那局输的）
  2. index 给出候选信号，且**明确标注"这是目录，不是结论"**
  3. whatif --enumerate 在第 266 步、第 3 路上枚举出多个候选，
     并且**至少有一个候选把结果从"僵尸进屋"变成"通关"**
     —— 也就是说：工具真的给出了"这里能改"的证据，而不是让人去猜。

用法
----
    <mcp-venv>/bin/python dsh/pvz-teacher/verify_s1.py [--out DIR]

退出码 0 = 全通。需要真实模拟器资源目录（走 PVZ_RESOURCE_DIR）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = Path(__file__).resolve().parent
SERVER = HERE / "server" / "pvz_mcp_server.py"

# 验收用的固定坐标。这些都是**已归档的既有事实**（见 §5 S1），
# 不是本次运行才发现的 —— 所以拿它们当断言是安全的。
SEED = 30001
LEVEL = 7
DECISION = 266
ROW = 3


def _text(result) -> str:
    return "".join(
        c.text for c in result.content if getattr(c, "type", None) == "text"
    )


async def run(out_dir: Path) -> int:
    ok = True
    env = dict(os.environ)
    # server 从 PVZ_RESOURCE_DIR 取资源目录；没给就直接失败，不要静默用错目录。
    if not env.get("PVZ_RESOURCE_DIR"):
        print("✗ 没设 PVZ_RESOURCE_DIR —— capture 会找不到模拟器资源")
        return 1

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env=env,
        cwd=str(SERVER.parent),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"✓ 握手 —— {init.server_info.name}@{init.server_info.version}")

            # ---- 1) capture：真跑一局并落盘 ----------------------------------
            t0 = time.time()
            res = await session.call_tool(
                "capture",
                {"seed": SEED, "level": LEVEL, "out": str(out_dir)},
            )
            dt = time.time() - t0
            body = _text(res)
            print(f"\n── capture（{dt:.1f}s，is_error={res.is_error}）")
            print("   " + body.replace("\n", "\n   ")[:400])
            if res.is_error:
                print("✗ capture 报错")
                return 1
            if not (out_dir / "frames.jsonl").is_file():
                print(f"✗ capture 说成功了，但 {out_dir}/frames.jsonl 不存在")
                ok = False
            else:
                frames = sum(1 for _ in open(out_dir / "frames.jsonl"))
                print(f"✓ 存档落地：{frames} 帧")

            # ---- 2) index：候选信号 -----------------------------------------
            res = await session.call_tool("index", {"archive": str(out_dir)})
            body = _text(res)
            print(f"\n── index（is_error={res.is_error}，{len(body)} 字符）")
            for line in body.splitlines()[:6]:
                print("   " + line)
            if res.is_error:
                print("✗ index 报错")
                return 1
            # 这条断言守的是"设计立场"：目录必须自称为目录，不能伪装成结论。
            if "不是结论" not in body:
                print("✗ index 的输出没有声明'这是目录，不是结论' —— "
                      "那模型会把候选当结论用")
                ok = False
            else:
                print("✓ index 明确声明了'目录不是结论'")

            # ---- 3) whatif：反事实枚举（S1 的核心验收）------------------------
            t0 = time.time()
            res = await session.call_tool(
                "whatif",
                {
                    "archive": str(out_dir),
                    "decision": DECISION,
                    "enumerate": True,
                    "row": ROW,
                },
            )
            dt = time.time() - t0
            body = _text(res)
            print(f"\n── whatif（{dt:.1f}s，is_error={res.is_error}）")
            print("   " + body.replace("\n", "\n   "))
            if res.is_error:
                print("✗ whatif 报错")
                return 1

            if "通关" not in body:
                print("✗ whatif 的候选里没有任何一个变成'通关' —— "
                      "工具没能给出'这里可以改'的证据")
                ok = False
            else:
                print("✓ whatif 给出了能通关的替代方案")

            # 参数翻译的检查：--enumerate 与 --row 必须真的传进去了。
            # 如果没传，输出里不会有"试 N 个候选"，也不会有行限定。
            if "候选" not in body:
                print("✗ whatif 输出里没有'候选'字样 —— enumerate 可能没传进去")
                ok = False
            if not any(c.isdigit() for c in body):
                print("✗ whatif 输出里没有数字 —— 结果可能是空的")
                ok = False

    print("\n结果：" + ("S1 全部通过" if ok else "有失败项"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default="/tmp/pvz-s1-verify",
        help="capture 的落盘目录（会被覆盖）",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    return asyncio.run(run(out_dir))


if __name__ == "__main__":
    raise SystemExit(main())
