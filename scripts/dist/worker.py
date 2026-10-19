"""Distributed collection worker: run one segment of teacher games, resumably, with a local web page.

The controller (on the Mac) pushes code, models and a task.json into ~/pvz-worker/tasks/<label>/.
This process runs scripts/search_play.py over that segment in chunks, keeps every finished game
(search_play itself skips seeds already present in out/<label>/games.jsonl), and serves a small
page on 127.0.0.1:8765 with pause / resume / stop-after-this-game / stop-now.

    python worker.py run [--label L] [--root ~/pvz-worker] [--port 8765]
    python worker.py status [--json]
    python worker.py serve            # page only (no collection)

Nothing here talks to the network: results stay in out/ and frames/ until the controller pulls them.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                 # repo/scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))      # repo/python

PAUSE_AUTO_STOP = 30 * 60          # 冻结超过这么久 → 立即停止（把内存还给朋友）
DEFAULT_PORT = 8765


def now() -> str:
    return datetime.now().strftime("%F %T")


def log(msg: str) -> None:
    print(f"{now()} {msg}", flush=True)


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))          # Linux
    except AttributeError:
        return os.cpu_count() or 4


def memory_gb() -> float:
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2 ** 30
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2 ** 20
    except Exception:
        pass
    return 8.0


def default_workers() -> int:
    # 每个采集进程约 0.5 GB（实测值，见 MEMORY/docs/distributed-worker.md）；留 20% 给系统。
    by_mem = max(1, int(memory_gb() * 0.8 / 0.5))
    return max(1, min(cpu_count() - 2, by_mem, 16))


class Task:
    _seeds = None

    def __init__(self, root: Path, spec: dict):
        self.root = root
        self.spec = spec
        self.label = spec["label"]
        self.seeds = int(spec["seeds"])
        self.offset = int(spec["seed_offset"])
        self.workers = int(spec.get("workers") or default_workers())
        # 一段 = 一波：这样"下完这局再停"正好是"不再开下一段"。
        self.chunk = int(spec.get("chunk") or self.workers)
        self.out = root / "out" / self.label
        self.frames = root / "frames" / self.label
        self.state_file = root / "state" / f"{self.label}.json"
        self.failed_file = root / "state" / f"{self.label}.failed.json"

    def path(self, key: str) -> Path:
        value = Path(self.spec[key])
        return value if value.is_absolute() else self.root / value

    def done_seeds(self) -> set:
        path = self.out / "games.jsonl"
        if not path.is_file():
            return set()
        seeds = set()
        for line in path.read_text().splitlines():
            try:
                seeds.add(int(json.loads(line)["seed"]))
            except Exception:
                continue
        return seeds

    def failed_seeds(self) -> set:
        if not self.failed_file.is_file():
            return set()
        try:
            return {int(s) for s in json.loads(self.failed_file.read_text())}
        except Exception:
            return set()

    def add_failed(self, seed: int) -> None:
        failed = self.failed_seeds() | {seed}
        self.failed_file.write_text(json.dumps(sorted(failed)))

    def seed_at(self, index: int):
        """第 index 个种子（相对段起点）。只读清单里的种子列表。"""
        if self._seeds is None:
            from render_episode import load_training_task
            self._seeds = load_training_task(self.path("manifest"), self.spec["task_id"])["seeds"]
        if index < 0 or index >= self.seeds:
            return None
        return int(self._seeds[self.offset + index])


class Worker:
    def __init__(self, root: Path, port: int, only_label, repo: Path | None = None):
        self.root = root
        self.repo = repo or (root / "repo")
        self.env_bin = root / "env" / "bin"
        self.port = port
        self.only_label = only_label
        self.lock = threading.Lock()
        self.proc = None
        self.task = None
        self.status = "idle"            # idle running paused stopping stopped done error
        self.message = ""
        self.chunk_text = ""
        self.started_at = None
        self.done_at_start = 0
        self.paused_since = None
        self.paused_total = 0.0
        self.stop_after = False
        self.stop_now = False
        self.history = []
        self.current_seeds = []

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict:
        with self.lock:
            task = self.task
            done = task.done_seeds() if task else set()
            total = task.seeds if task else 0
            failed = task.failed_seeds() if task else set()
            elapsed = (time.time() - self.started_at) if self.started_at else 0.0
            paused = self.paused_total + ((time.time() - self.paused_since) if self.paused_since else 0.0)
            active = max(1e-9, elapsed - paused)
            rate = (len(done) - self.done_at_start) / active * 3600 if self.started_at else 0.0
            return {
                "machine": socket.gethostname(),
                "label": task.label if task else None,
                "task_id": task.spec.get("task_id") if task else None,
                "status": self.status,
                "message": self.message,
                "done": len(done),
                "failed": len(failed),
                "total": total,
                "workers": task.workers if task else None,
                "chunk": self.chunk_text,
                "current_seeds": self.current_seeds,
                "games_per_hour": round(rate, 2),
                "elapsed_seconds": round(elapsed),
                "paused_seconds": round(paused),
                "eta_seconds": round((total - len(done)) / rate * 3600) if rate > 0 else None,
                "pid": self.proc.pid if self.proc else None,
                "updated": now(),
                "python": platform.python_version(),
                "cores": cpu_count(),
                "memory_gb": round(memory_gb(), 1),
            }

    def write_progress(self) -> None:
        task = self.task
        if not task:
            return
        task.state_file.parent.mkdir(parents=True, exist_ok=True)
        task.state_file.write_text(json.dumps(self.snapshot(), indent=1))

    # ------------------------------------------------------------------ control
    def request(self, action: str) -> str:
        with self.lock:
            if action == "pause":
                if self.status == "running":
                    self._signal_group(signal.SIGSTOP)
                    self.status = "paused"
                    self.paused_since = time.time()
                    return "已暂停（CPU 已让出）"
                return "当前不在运行"
            if action == "resume":
                if self.status == "paused":
                    self._signal_group(signal.SIGCONT)
                    self.paused_total += time.time() - (self.paused_since or time.time())
                    self.paused_since = None
                    self.status = "running"
                    return "已继续"
                return "当前没有暂停"
            if action == "stop_after":
                self.stop_after = True
                if self.status == "paused":
                    self._signal_group(signal.SIGCONT)
                    self.paused_total += time.time() - (self.paused_since or time.time())
                    self.paused_since = None
                    self.status = "running"
                return "本段跑完就停（不再开新段）"
            if action == "stop_now":
                self.stop_now = True
                if self.status == "paused":
                    self._signal_group(signal.SIGCONT)
                    self.paused_since = None
                self._signal_group(signal.SIGTERM)
                self.status = "stopping"
                return "正在停止"
        return "未知操作"

    def _signal_group(self, sig: int) -> None:
        proc = self.proc
        if not proc or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass

    # ------------------------------------------------------------------ runner
    def _command(self, task: Task, index: int, count: int) -> list:
        spec = task.spec
        cmd = [
            str(self.env_bin / "python"), str(self.repo / "scripts" / "search_play.py"),
            "--rollout-policy", str(task.path("student")),
            "--value", str(task.path("value")),
            "--static-value", str(task.path("static_value")),
            "--manifest", str(task.path("manifest")),
            "--task-id", spec["task_id"],
            "--seed-offset", str(task.offset + index),
            "--seeds", str(count),
            "--arms", spec.get("arms", "fast_search"),
            "--workers", str(task.workers),
            "--frames", str(task.frames),
            "--frames-label", task.label,
            "--resource-dir", str(task.path("resource_dir")),
            "--out", str(task.out),
        ] + [str(x) for x in spec.get("teacher_flags") or []]
        return cmd

    def _env(self) -> dict:
        env = dict(os.environ)
        env["PATH"] = f"{self.env_bin}:{env.get('PATH', '')}"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _launch(self, task: Task, index: int, count: int):
        task.out.mkdir(parents=True, exist_ok=True)
        task.frames.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        logfile = self.root / "logs" / f"{task.label}.log"
        cmd = self._command(task, index, count)
        self.chunk_text = f"seeds {index + 1}-{index + count} / {task.seeds}"
        self.current_seeds = [s for s in (task.seed_at(i) for i in range(index, index + count)) if s is not None]
        log("run " + " ".join(cmd))
        with logfile.open("a") as fh:
            fh.write(f"\n===== {now()} {self.chunk_text} =====\n")
            fh.flush()
            return subprocess.Popen(cmd, cwd=self.repo, env=self._env(), stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)

    def _wait(self, proc) -> int:
        while proc.poll() is None:
            with self.lock:
                if self.stop_now:
                    self._signal_group(signal.SIGTERM)
                if (self.status == "paused" and self.paused_since
                        and time.time() - self.paused_since > PAUSE_AUTO_STOP):
                    self.message = "暂停超过 30 分钟，自动停止（把内存还给朋友）"
                    self.stop_now = True
                    self._signal_group(signal.SIGTERM)
                    self.status = "stopping"
            time.sleep(1.0)
        # 进程组里可能还有残留（mp 子进程），收干净
        with self.lock:
            self._signal_group(signal.SIGKILL)
        return proc.wait()

    def _run_task(self, task: Task) -> bool:
        (self.root / "state" / "_idle.json").unlink(missing_ok=True)
        self.task = task
        self.status = "running"
        self.started_at = time.time()
        self.done_at_start = len(task.done_seeds())
        self.stop_after = False
        self.stop_now = False
        self.paused_total = 0.0
        self.paused_since = None
        self.history = [(time.time(), len(task.done_seeds()))]
        self.message = ""
        log(f"task {task.label}: {task.seeds} seeds from offset {task.offset}, {task.workers} workers")
        attempts = {}
        index = 0
        while index < task.seeds:
            if self.stop_now or self.stop_after:
                break
            if self.status == "paused":
                time.sleep(1)
                continue
            count = min(task.chunk, task.seeds - index)
            proc = self._launch(task, index, count)
            with self.lock:
                self.proc = proc
            self.status = "running"
            self._wait(proc)
            with self.lock:
                self.proc = None
            done = task.done_seeds()
            self.history.append((time.time(), len(done)))
            self.write_progress()
            if self.stop_now or self.stop_after:
                break
            # 这一段里还有没完成的种子 → 重试；连续失败就缩小到单局定位。
            pending = [i for i in range(index, index + count) if task.seed_at(i) not in done]
            if not pending:
                index += count
                attempts.clear()
                continue
            attempts[count] = attempts.get(count, 0) + 1
            if attempts[count] >= 2 and count > 1:
                log(f"chunk of {count} failed twice, retrying one seed at a time")
                task.chunk = 1
            elif count == 1:
                failures = attempts.get(1, 0)
                if failures >= 3:
                    seed = task.seed_at(index)
                    log(f"seed {seed} failed {failures} times — recorded as failed, moving on")
                    task.add_failed(seed)
                    index += 1
                    attempts.clear()
        done = task.done_seeds()
        self.write_progress()
        complete = len(done) + len(task.failed_seeds()) >= task.seeds
        if complete and not self.stop_now:
            self.status = "done"
            self.message = f"完成 {len(done)}/{task.seeds}"
            self._write_done(task)
        elif self.stop_now:
            self.status = "stopped"
            self.message = f"已停止（完成 {len(done)}/{task.seeds}）"
        else:
            self.status = "stopped"
            self.message = f"本段结束，不再开新段（完成 {len(done)}/{task.seeds}）"
        log(self.message)
        return complete

    def _write_done(self, task: Task) -> None:
        rows = []
        path = task.out / "games.jsonl"
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        summary = {
            "label": task.label, "seeds": task.seeds, "seed_offset": task.offset,
            "done": len(rows), "won": sum(1 for r in rows if r.get("won")),
            "failed": sorted(task.failed_seeds()),
            "mean_seconds": round(sum(r.get("seconds", 0) for r in rows) / len(rows), 1) if rows else None,
            "finished_at": now(),
        }
        (task.out / "worker_done.json").write_text(json.dumps(summary, indent=1))
        log(f"done: {json.dumps(summary, ensure_ascii=False)}")

    def run(self, stay: bool = True) -> int:
        """Collect every pending task; with stay=True keep the page up and poll for new ones."""
        while True:
            pending = []
            for path in sorted((self.root / "tasks").glob("*/task.json")):
                spec = json.loads(path.read_text())
                if self.only_label and spec.get("label") != self.only_label:
                    continue
                task = Task(self.root, spec)
                if len(task.done_seeds()) + len(task.failed_seeds()) < task.seeds:
                    pending.append(task)
            if not pending:
                if not stay:
                    log("no pending tasks")
                    return 0
                with self.lock:
                    self.status = "idle"
                    self.message = "没有待跑的任务，等控制端派活"
                    self.task = None
                self.write_idle()
                time.sleep(30)
                continue
            for task in pending:
                try:
                    self._run_task(task)
                except Exception as exc:                   # noqa: BLE001 - keep the page alive
                    self.status = "error"
                    self.message = f"{type(exc).__name__}: {exc}"
                    log("ERROR " + self.message)
                    self.write_progress()
                    if not stay:
                        raise
                if self.stop_now:
                    self.write_progress()
                    return self._halt(stay)
            if not stay:
                return 0
            time.sleep(5)

    def _halt(self, stay: bool) -> int:
        """After an explicit stop: exit, or keep the page up showing "stopped" until killed."""
        if not stay:
            return 0
        while True:
            time.sleep(60)

    def write_idle(self) -> None:
        path = self.root / "state" / "_idle.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=1))

    # ------------------------------------------------------------------ web
    def serve_forever(self) -> None:
        handler = make_handler(self)
        server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        log(f"web: http://127.0.0.1:{self.port}/")
        server.serve_forever()


def _fmt_seconds(value) -> str:
    if value is None:
        return "—"
    value = int(value)
    return f"{value // 3600}h{value % 3600 // 60:02d}m" if value >= 3600 else f"{value // 60}m{value % 60:02d}s"


def make_handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _page(self) -> str:
            s = worker.snapshot()
            pct = (s["done"] / s["total"] * 100) if s["total"] else 0
            status = {"running": "采集中", "paused": "已暂停", "idle": "空闲", "done": "已完成",
                      "stopped": "已停止", "stopping": "正在停止", "error": "出错"}.get(s["status"], s["status"])
            seeds = ", ".join(str(x) for x in s["current_seeds"]) or "—"
            return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>PvZ 采集 · {html.escape(str(s['machine']))}</title>
<style>
body{{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;color:#222}}
h1{{font-size:20px;margin:0 0 4px}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td{{padding:5px 8px;border-bottom:1px solid #eee}} td:first-child{{color:#666;width:34%}}
.bar{{height:14px;background:#eee;border-radius:7px;overflow:hidden}} .bar>div{{height:100%;background:#2e7d32}}
form{{display:inline}} button{{font-size:15px;padding:8px 14px;margin:4px 6px 4px 0;border:1px solid #bbb;
background:#fafafa;border-radius:6px;cursor:pointer}} button:hover{{background:#eee}}
.note{{color:#666;font-size:13px}}
</style></head><body>
<h1>{html.escape(str(s['machine']))} · 分布式采集</h1>
<div class="note">任务 {html.escape(str(s['label']))} · {html.escape(str(s['task_id']))} · {s['cores']} 核 / {s['memory_gb']} GB</div>
<div class="bar" style="margin-top:12px"><div style="width:{pct:.1f}%"></div></div>
<table>
<tr><td>状态</td><td><b>{status}</b> {html.escape(str(s['message']))}</td></tr>
<tr><td>本段进度</td><td>{s['done']} / {s['total']} 局（{pct:.0f}%）{(' · 失败 ' + str(s['failed'])) if s['failed'] else ''}</td></tr>
<tr><td>当前段</td><td>{html.escape(str(s['chunk'] or '—'))}</td></tr>
<tr><td>正在跑的种子</td><td>{seeds}</td></tr>
<tr><td>每小时局数</td><td>{s['games_per_hour']}</td></tr>
<tr><td>已运行</td><td>{_fmt_seconds(s['elapsed_seconds'])}（暂停 {_fmt_seconds(s['paused_seconds'])}）</td></tr>
<tr><td>预计剩余</td><td>{_fmt_seconds(s['eta_seconds'])}</td></tr>
<tr><td>进程数</td><td>{s['workers'] or '—'}</td></tr>
<tr><td>更新时间</td><td>{s['updated']}</td></tr>
</table>
<form method="post" action="/pause"><button>暂停（让出 CPU）</button></form>
<form method="post" action="/resume"><button>继续</button></form>
<form method="post" action="/stop_after"><button>下完这局再停</button></form>
<form method="post" action="/stop_now"><button>立即停止</button></form>
<p class="note">暂停超过 30 分钟会自动转成“立即停止”，把内存还给电脑主人。<br>
关掉这个网页不影响采集；“立即停止”之后重新打开采集会从未完成的种子接着跑。</p>
</body></html>"""

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = self._page().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/state.json":
                body = json.dumps(worker.snapshot(), ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            action = self.path.strip("/")
            if action in ("pause", "resume", "stop_after", "stop_now"):
                log("control: " + worker.request(action))
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "status", "serve"])
    p.add_argument("--root", default=str(Path.home() / "pvz-worker"))
    p.add_argument("--label", help="only this task label (default: every pending task, in name order)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--repo", help="code checkout (default: <root>/repo)")
    p.add_argument("--no-stay", dest="stay", action="store_false",
                   help="exit when every task is done instead of polling for new ones")
    p.add_argument("--json", action="store_true", help="status: machine-readable")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    if args.command == "status":
        rows = []
        for path in sorted((root / "state").glob("*.json")):
            try:
                rows.append(json.loads(path.read_text()))
            except Exception:
                continue
        if args.json:
            print(json.dumps(rows, ensure_ascii=False))
        else:
            for row in rows:
                print(f"{row.get('machine')} {row.get('label')} {row.get('status')} "
                      f"{row.get('done')}/{row.get('total')} {row.get('games_per_hour')}/h")
        return 0

    worker = Worker(root, args.port, args.label,
                    Path(args.repo).expanduser() if args.repo else None)
    if args.command == "serve":
        worker.serve_forever()
        return 0

    thread = threading.Thread(target=worker.serve_forever, daemon=True)
    thread.start()

    def stop(_sig, _frame):
        worker.request("stop_now")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = worker.run(args.stay)
    worker.write_progress()
    return code


if __name__ == "__main__":
    sys.exit(main())
