#!/usr/bin/env python3
"""记录型反向代理：把 DSH 发给模型的请求原样转发，同时落盘。

两个用途
--------
1. **观察**（主要用途）。调不通的时候，报错信息只告诉你"哪里不对"，不告诉你
   "DSH 到底发了什么"。猜配置字段是低效的。这个代理让你直接看到请求体：
   `max_tokens` 是多少、`thinking` 块长什么样、system prompt 里塞了什么、
   工具 schema 有多大。对"LLM 当教师"这条线尤其重要 —— 整个设计的前提就是
   **我们要知道模型到底看到了什么**。
2. **修复**（可选，默认关）。`--fix-thinking-budget` 会补上 DSH 漏发的
   `thinking.budget_tokens`，让严格校验 Anthropic 规范的网关也能用。
   原因见 `_repair` 的注释。**默认关闭**：观察工具不该偷偷改东西，
   记录里 `body` 记的始终是**修复前**的内容（合法 JSON 会解析成对象落盘，
   字段不变、空白归一化），改了什么单独记在 `repaired`。

只依赖标准库（`http.server` + `urllib`），不需要装任何东西。

用法
----
    python3 logging_proxy.py --upstream https://api.example.com/v1 \
        --port 8950 --log /tmp/llm-requests.jsonl

    DEEPSEEK_BASE_URL=http://127.0.0.1:8950/v1 DEEPSEEK_API_KEY=... dsh ...

需要修复时再加：

    ... --fix-thinking-budget 32768

每个请求写一行 JSONL（含 headers、body、响应状态、耗时、修复说明）。
SSE 流式响应按块转发，不会被缓冲成一次性返回。

安全
----
默认**不记录**请求头里的凭据（`authorization` / `x-api-key` / `api-key` 会被
替换成 `<redacted>`）。要连凭据一起记，加 `--log-headers-raw`，但那会把 key
写进磁盘文件 —— 只在排查认证问题时用，用完删掉。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REDACT = {"authorization", "x-api-key", "api-key", "proxy-authorization"}

UPSTREAM = ""
LOG_PATH = ""
LOG_HEADERS_RAW = False
VERBOSE = True
FIX_THINKING_BUDGET = 0  # 0 = 关闭；>0 = 目标预算（另受 max_tokens-1024 上限约束）
COUNTER = 0


def _log(record: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _repair(body: bytes) -> tuple[bytes, str | None]:
    """修 DSH 与严格 Anthropic 网关之间的一个不兼容。返回 (新body, 说明或 None)。

    问题：`packages/llm/llm-deepseek/src/serialize.ts:155` 无条件发

        thinking: { type: effort === 'off' ? 'disabled' : 'enabled' }
        output_config: { effort }

    **不带 `budget_tokens`**。Anthropic Messages 规范里 `type: "enabled"` 必须带
    `budget_tokens`；官方端点宽容接受，严格校验的网关直接拒：

        INVALID_REQUEST: invalid Claude request:
          thinking: budget_tokens must be at least 1024 when type is enabled

    改 DSH 的配置改不出来（代码里根本没这个字段）。这里补上：
    `budget_tokens = min(目标值, max_tokens - 1024)`，下限 1024。
    """
    if FIX_THINKING_BUDGET <= 0 or not body:
        return body, None
    try:
        doc = json.loads(body.decode("utf-8"))
    except Exception:
        return body, None
    if not isinstance(doc, dict):
        return body, None
    thinking = doc.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return body, None
    if thinking.get("budget_tokens"):
        return body, None

    max_tokens = doc.get("max_tokens")
    ceiling = (max_tokens - 1024) if isinstance(max_tokens, int) else FIX_THINKING_BUDGET
    budget = max(1024, min(FIX_THINKING_BUDGET, ceiling))
    thinking["budget_tokens"] = budget
    return json.dumps(doc, ensure_ascii=False).encode("utf-8"), f"thinking.budget_tokens={budget}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - 基类签名
        if VERBOSE:
            sys.stderr.write("[proxy] " + (fmt % args) + "\n")

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        return self.rfile.read(int(length))

    def _forward(self, method: str) -> None:
        global COUNTER
        COUNTER += 1
        n = COUNTER

        body = self._read_body()
        started = time.time()

        headers = {}
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in ("host", "content-length", "connection", "accept-encoding"):
                continue
            if lk in REDACT and not LOG_HEADERS_RAW:
                headers[k] = "<redacted>"
            else:
                headers[k] = v

        record: dict = {
            "n": n,
            "ts": started,
            "method": method,
            "path": self.path,
            "headers": headers,
        }
        try:
            # 记录 DSH 发出的请求体 —— 诊断要看的就是这个。
            # 注意落盘形式：合法 JSON 会被解析成对象再写回（不是字节级原样），
            # 字段内容不变，但 key 顺序与空白会归一化。要字节级比对就别用这个。
            record["body"] = json.loads(body.decode("utf-8")) if body else None
        except Exception:
            record["body"] = body.decode("utf-8", "replace")

        # 修复放在记录之后：record["body"] 记的始终是**修复前**的内容，
        # 改了什么单独记在 record["repaired"]。
        body, repaired = _repair(body)
        if repaired is not None:
            record["repaired"] = repaired

        # 转发时用原始凭据，不用脱敏后的。
        fwd_headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "content-length", "connection")}
        # 去重版本前缀：DSH 会把 baseURL 当根、再拼 `/v1/messages`，
        # 而 --upstream 通常已经带了 `/v1`。不去重就会变成 `/v1/v1/messages` → 404。
        base = urlsplit(UPSTREAM)
        base_path = base.path.rstrip("/")
        path = self.path
        if base_path and (path == base_path or path.startswith(base_path + "/")):
            path = path[len(base_path):] or "/"
        url = f"{base.scheme}://{base.netloc}{base_path}{path}"
        req = urllib.request.Request(url, data=body or None, headers=fwd_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                record["status"] = resp.status
                self.send_response(resp.status)
                passthrough = ("content-type", "cache-control", "request-id", "x-request-id")
                for k, v in resp.headers.items():
                    if k.lower() in passthrough:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                total = 0
                first = None
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    if first is None:
                        first = time.time() - started
                    total += len(chunk)
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                record["bytes"] = total
                record["ttfb_s"] = None if first is None else round(first, 3)
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            record["status"] = exc.code
            record["error_body"] = err_body.decode("utf-8", "replace")[:4000]
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as exc:  # noqa: BLE001
            record["status"] = None
            record["error"] = f"{type(exc).__name__}: {exc}"
            payload = json.dumps({"error": {"message": str(exc)}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        record["elapsed_s"] = round(time.time() - started, 3)
        _log(record)
        if VERBOSE:
            sys.stderr.write(f"[proxy] #{n} {method} {self.path} -> {record.get('status')} "
                             f"({record['elapsed_s']}s)\n")

    def do_POST(self):  # noqa: N802
        self._forward("POST")

    def do_GET(self):  # noqa: N802
        self._forward("GET")


def main() -> int:
    global UPSTREAM, LOG_PATH, LOG_HEADERS_RAW, VERBOSE, FIX_THINKING_BUDGET
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="真实端点，如 https://host/v1")
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--log", required=True, help="JSONL 落盘路径（追加写）")
    ap.add_argument("--log-headers-raw", action="store_true",
                    help="连凭据一起记（会写进磁盘，仅排查认证问题时用）")
    ap.add_argument("--fix-thinking-budget", type=int, default=0, metavar="N",
                    help="补上 DSH 缺失的 thinking.budget_tokens（N=目标值；"
                         "实际取 min(N, max_tokens-1024)，下限 1024）。0=关闭。"
                         "用于严格校验 Anthropic 规范的网关，详见 _repair 的注释。")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    UPSTREAM = args.upstream
    LOG_PATH = args.log
    LOG_HEADERS_RAW = args.log_headers_raw
    VERBOSE = not args.quiet
    FIX_THINKING_BUDGET = args.fix_thinking_budget

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f"[proxy] 监听 http://{args.host}:{args.port} → {UPSTREAM}\n")
    sys.stderr.write(f"[proxy] 请求落盘：{LOG_PATH}\n")
    if FIX_THINKING_BUDGET > 0:
        sys.stderr.write(f"[proxy] 已开启修复：thinking.budget_tokens 目标 {FIX_THINKING_BUDGET}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
