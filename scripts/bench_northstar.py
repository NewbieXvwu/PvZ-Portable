#!/usr/bin/env python3
"""Northstar benchmark: poll the desktop curriculum queue and chart each new student.

task179 §5 asks for a trend line that never takes part in promotion:

* completion -- adventure 1..50 at the original difficulty with the standard deck
  (``pvz_tasks.deck_for_level``; the engine's own conveyor bank on 5/10/20/25/30/40/45/50),
  16 seeds per level, pure student;
* generalization -- the 54 unfamiliar v5 decks of task129 (``configs/task129_decks_v5.json``),
  2 seeds each, taken from the registry's own seed block;
* difficulty -- the 0.5..3.0 multiplier ladder on L7/L14/L25/L33/L45, 16 seeds per rung,
  reported as the multiplier where the win rate crosses 50 % (logit-linear interpolation).

Every model plays exactly the same (task, seed) pairs, so rounds are paired across time.
Results are appended to one CSV keyed by student round; ``northstar.png`` plots the trend;
a short paragraph goes to ``MEMORY/docs/task181-northstar.md``. CSV, PNG, logs and the
pulled checkpoints live under ``artifacts/northstar/`` and ``~/.cache/pvz-northstar/``
and are never committed.

The desktop is a production machine: the poller only runs ``cat`` and ``stat`` there and
never writes, starts or stops anything. Benchmarks are capped at 4 game processes (3 by
default) because this Mac runs other experiments as well.

Usage:
  python3 scripts/bench_northstar.py --once --dry-run          # show the active student
  python3 scripts/bench_northstar.py --once                    # one poll, bench if new
  python3 scripts/bench_northstar.py --model m.pt --round 52   # one explicit student
  nohup nice -n 10 python3 scripts/bench_northstar.py \\
      >> artifacts/northstar/poller.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_play  # noqa: E402

REMOTE_STATE = "/home/newbiexvwu/pvz-logs/curriculum_state.json"
DEFAULT_MODEL_DIR = Path("~/.cache/pvz-northstar/models")
DEFAULT_OUT = ROOT / "artifacts/northstar"
DEFAULT_DOC = ROOT / "MEMORY/docs/task181-northstar.md"
UNSEEN_DECKS = ROOT / "configs/task129_decks_v5.json"
DEFAULT_EXECUTABLE = ROOT / "build/pvz-portable"
DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    "PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))
MAX_WORKERS = 4
LEVELS = tuple(range(1, 51))
SEEDS_PER_LEVEL = 16
UNSEEN_SEEDS = 2
LADDER_LEVELS = (7, 14, 25, 33, 45)
LADDER = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
LADDER_SEEDS = 16
COMPLETION_SEED_START = 932_000_000
LADDER_SEED_START = 934_000_000
ZONE_BY_BACKGROUND = {"BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night",
                      "BACKGROUND_3_POOL": "pool", "BACKGROUND_4_FOG": "fog",
                      "BACKGROUND_5_ROOF": "roof", "BACKGROUND_6_BOSS": "roof"}
FIELDS = ("round", "evaluated_at", "model_sha256", "model_bytes", "backend", "workers",
          "seconds", "completion_wins", "completion_games", "completion_rate",
          "day_wins", "day_games", "night_wins", "night_games", "pool_wins", "pool_games",
          "fog_wins", "fog_games", "roof_wins", "roof_games",
          "unseen_wins", "unseen_games", "unseen_rate",
          *[f"m50_l{level}" for level in LADDER_LEVELS])


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- suites

def zone_of(level: int) -> str:
    import pvz_constants
    return ZONE_BY_BACKGROUND[pvz_constants.background_for_level(level)]


def standard_task(task_id: str, level: int, multiplier: float = 1.0) -> dict:
    import pvz_tasks
    conveyor = pvz_tasks.conveyor_belt_level(level)
    task = {"task_id": task_id, "zone": zone_of(level), "level": level,
            "deck": list(pvz_tasks.deck_for_level(
                level, conveyor_start="original" if conveyor else None)),
            "zombie_count_multiplier": multiplier, "playthrough": 2}
    if conveyor:
        task["conveyor_start"] = "original"
    return task


def completion_suite() -> dict:
    tasks, plan = {}, {}
    for index, level in enumerate(LEVELS):
        task = standard_task(f"north_l{level:02d}", level)
        tasks[task["task_id"]] = task
        first = COMPLETION_SEED_START + index * SEEDS_PER_LEVEL
        plan[task["task_id"]] = list(range(first, first + SEEDS_PER_LEVEL))
    return tasks, plan


def unseen_suite() -> dict:
    document = json.loads(UNSEEN_DECKS.read_text())
    tasks, plan = {}, {}
    for entry in document["tasks"]:
        task_id = f"unseen_{entry['task_id']}"
        tasks[task_id] = {"task_id": task_id, "zone": zone_of(entry["level"]),
                          "level": entry["level"], "deck": list(entry["deck"]),
                          "zombie_count_multiplier": 1.0, "playthrough": 2}
        plan[task_id] = [int(seed) for seed in entry["seeds"][:UNSEEN_SEEDS]]
    return tasks, plan


def ladder_suite() -> dict:
    tasks, plan = {}, {}
    for level_index, level in enumerate(LADDER_LEVELS):
        for rung, multiplier in enumerate(LADDER):
            task_id = f"ladder_l{level:02d}_m{multiplier:.2f}"
            tasks[task_id] = standard_task(task_id, level, multiplier)
            first = LADDER_SEED_START + level_index * 1000 + rung * LADDER_SEEDS
            plan[task_id] = list(range(first, first + LADDER_SEEDS))
    return tasks, plan


SUITES = {"completion": completion_suite, "unseen": unseen_suite, "ladder": ladder_suite}


# --------------------------------------------------------------------------- bench

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_suite(name: str, model: Path, native: Path | None, backend: str, workers: int,
              resource_dir: Path, executable: Path) -> list[dict]:
    tasks, plan = SUITES[name]()
    jobs = [{"task_id": task_id, "seed": seed}
            for task_id, seeds in plan.items() for seed in seeds]
    cfg = {"backend": backend, "model": str(model),
           "native": str(native) if native else None, "policy_module": None,
           "resource_dir": str(resource_dir), "executable": str(executable), "tasks": tasks}
    start_method = "spawn" if sys.platform == "darwin" else (
        "fork" if "fork" in mp.get_all_start_methods() else "spawn")
    rows: list[dict] = []
    failure: Exception | None = None
    context = mp.get_context(start_method)
    pool = context.Pool(workers, initializer=bench_play._init_worker, initargs=(cfg,))
    try:
        pending = jobs
        for attempt in (1, 2):
            failed = []
            try:
                for index, row in enumerate(pool.imap_unordered(
                        bench_play._play_one, pending, chunksize=1), 1):
                    rows.append(row)
                    if index % 100 == 0 or index == len(pending):
                        log(f"{name}: {index}/{len(pending)} games")
            except Exception as exc:  # a game crashed the worker; retry the rest once
                failure = exc
                log(f"{name}: game failed ({exc}); retrying the unfinished jobs")
                done = {(row["task_id"], row["seed"]) for row in rows}
                failed = [job for job in pending if (job["task_id"], job["seed"]) not in done]
                pending = failed or pending
                continue
            failure = None
            break
    finally:
        pool.close()
        pool.join()
    if failure is not None:
        raise failure
    return rows


def wins(rows: list[dict]) -> tuple[int, int]:
    return sum(bool(row["won"]) for row in rows), len(rows)


def rate(block: tuple[int, int]) -> float | None:
    won, games = block
    return round(won / games, 4) if games else None


def crossing(rows: list[dict], level: int) -> float:
    """Multiplier where the win rate crosses 50% (logit-linear in log multiplier)."""
    points = []
    for multiplier in LADDER:
        task_id = f"ladder_l{level:02d}_m{multiplier:.2f}"
        won, games = wins([row for row in rows if row["task_id"] == task_id])
        if not games:
            continue
        p = min(max((won + 0.5) / (games + 1), 1e-6), 1 - 1e-6)
        points.append((multiplier, math.log(p / (1 - p))))
    if not points:
        return float("nan")
    if all(logit >= 0 for _, logit in points):
        return LADDER[-1]
    if all(logit < 0 for _, logit in points):
        return LADDER[0]
    for (mult_a, logit_a), (mult_b, logit_b) in zip(points, points[1:]):
        if logit_a >= 0 > logit_b:
            fraction = logit_a / (logit_a - logit_b)
            return round(math.exp(math.log(mult_a) +
                                  fraction * (math.log(mult_b) - math.log(mult_a))), 2)
    return LADDER[-1]


def evaluate(round_number: int, model: Path, backend: str, workers: int,
             resource_dir: Path, executable: Path) -> dict:
    started = time.time()
    native = None
    if backend in ("auto", "cpp"):
        try:
            native = bench_play._export(model, None)
            backend = "cpp"
        except Exception as exc:
            if backend == "cpp":
                raise
            log(f"export unavailable ({exc}); using the Python decision loop")
            backend = "python"
    log(f"round {round_number}: backend={backend} workers={workers} model={model}")

    completion = run_suite("completion", model, native, backend, workers, resource_dir, executable)
    unseen = run_suite("unseen", model, native, backend, workers, resource_dir, executable)
    ladder = run_suite("ladder", model, native, backend, workers, resource_dir, executable)
    zones = {}
    for zone in ("day", "night", "pool", "fog", "roof"):
        zones[zone] = wins([row for row in completion if row["zone"] == zone])
    completion_block, unseen_block = wins(completion), wins(unseen)
    record = {"round": round_number,
              "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "model_sha256": sha256_file(model), "model_bytes": model.stat().st_size,
              "backend": backend, "workers": workers,
              "seconds": round(time.time() - started, 1),
              "completion_wins": completion_block[0], "completion_games": completion_block[1],
              "completion_rate": rate(completion_block),
              **{f"{zone}_{name}": value for zone, block in zones.items()
                 for name, value in zip(("wins", "games"), block)},
              "unseen_wins": unseen_block[0], "unseen_games": unseen_block[1],
              "unseen_rate": rate(unseen_block),
              **{f"m50_l{level}": crossing(ladder, level) for level in LADDER_LEVELS}}
    return record


# --------------------------------------------------------------------------- csv/plot/doc

def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({name: record.get(name) for name in FIELDS})


def plot(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds = [int(row["round"]) for row in rows]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2]})
    top.plot(rounds, [float(row["completion_rate"]) for row in rows], "o-", label="adventure 1-50")
    top.plot(rounds, [float(row["unseen_rate"]) for row in rows], "s-", label="54 unseen decks")
    top.set_ylabel("win rate")
    top.set_ylim(0, 1)
    top.grid(alpha=0.3)
    top.legend(loc="lower right")
    for level in LADDER_LEVELS:
        bottom.plot(rounds, [float(row[f"m50_l{level}"]) for row in rows], "o-", label=f"L{level}")
    bottom.set_xlabel("student round")
    bottom.set_ylabel("multiplier at 50% win rate")
    bottom.set_ylim(0.4, 3.1)
    bottom.grid(alpha=0.3)
    bottom.legend(loc="upper left", ncol=5)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".building.png")
    figure.savefig(temporary, dpi=120)
    plt.close(figure)
    temporary.replace(path)


def percent(value) -> str:
    return "—" if value in (None, "") else f"{float(value) * 100:.1f}%"


def append_doc(doc: Path, record: dict) -> None:
    doc.parent.mkdir(parents=True, exist_ok=True)
    zones = "、".join(f"{zone} {record[f'{zone}_wins']}/{record[f'{zone}_games']}"
                     for zone in ("day", "night", "pool", "fog", "roof"))
    ladder = "、".join(f"L{level} {record[f'm50_l{level}']}" for level in LADDER_LEVELS)
    text = (f"\n### r{record['round']}（{record['evaluated_at']}）\n\n"
            f"- 模型 `{record['model_sha256'][:12]}`（{int(record['model_bytes']) / 1e6:.1f} MB），"
            f"{record['backend']} 后端，{record['workers']} 进程，耗时 {record['seconds']} s。\n"
            f"- 冒险 1–50：{record['completion_wins']}/{record['completion_games']} = "
            f"{percent(record['completion_rate'])}（{zones}）。\n"
            f"- 54 套陌生卡组：{record['unseen_wins']}/{record['unseen_games']} = "
            f"{percent(record['unseen_rate'])}。\n"
            f"- 倍率 50%（0.5–3.0 阶梯，端点表示未跨过）：{ladder}。\n")
    with doc.open("a", encoding="utf-8") as handle:
        handle.write(text)


DOC_HEADER = """# task181 北极星基准

不参与晋升，只看走势（task179 §5）。每出现一个已激活学生跑一次：

- **通关**：冒险 1–50 关，原难度、`pvz_tasks.deck_for_level` 标准卡组（5/10/20/25/30/40/45/50
  用引擎自己的传送带），纯学生，每关 16 个种子。
- **泛化**：task129 的 54 套 v5 陌生卡组（`configs/task129_decks_v5.json`），每套 2 个种子，
  用登记表自带的种子段。
- **难度**：L7/L14/L25/L33/L45 的 9 档倍率（0.5–3.0），每档 16 个种子；报胜率 50% 对应的倍率
  （相邻两档在 log(倍率) × logit 上线性插值，落在端点表示没跨过 50%）。

种子固定（通关 932M、倍率 934M、陌生卡组 129M），所有学生逐局配对。CSV、曲线图、日志与拉取的
检查点都在 `artifacts/northstar/` 与 `~/.cache/pvz-northstar/`，不入 git。L35 是砸罐子关，
学生没有砸罐动作，通关率结构性为 0，看走势时忽略这一关。
"""


# --------------------------------------------------------------------------- remote

class Remote:
    def __init__(self, host: str, distro: str):
        self.host, self.distro = host, distro

    def _wsl(self, argv: list[str], *, stdin: bytes | None = None, stdout=None,
             timeout: float = 120):
        command = ["ssh", "-o", "ConnectTimeout=40", self.host,
                   "wsl", "-d", self.distro, "-e", *argv]
        return subprocess.run(command, input=stdin, stdout=stdout or subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout)

    def read_state(self) -> tuple[int, str]:
        result = self._wsl(["cat", REMOTE_STATE])
        if result.returncode != 0:
            raise RuntimeError(f"remote state read failed: "
                               f"{result.stderr.decode(errors='replace').strip()}")
        document = json.loads(result.stdout)
        return int(document["student_round"]), str(document["student"])

    def size(self, path: str) -> int:
        script = f'stat -c "%s" "{path}"\n'
        result = self._wsl(["bash", "-s"], stdin=script.encode())
        if result.returncode != 0:
            raise RuntimeError(f"remote stat failed for {path}: "
                               f"{result.stderr.decode(errors='replace').strip()}")
        return int(result.stdout.decode().strip())

    def copy(self, path: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with temporary.open("wb") as handle:
            result = self._wsl(["cat", path], stdout=handle, timeout=1800)
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"remote cat failed for {path}: "
                               f"{result.stderr.decode(errors='replace').strip()}")
        temporary.replace(destination)


def pull_model(remote: Remote, round_number: int, remote_path: str, model_dir: Path) -> Path:
    expected = remote.size(remote_path)
    target = model_dir / f"r{round_number:06d}.pt"
    for attempt in (1, 2):
        remote.copy(remote_path, target)
        if target.stat().st_size == expected:
            log(f"round {round_number}: model {expected} bytes from {remote_path}")
            return target
        log(f"round {round_number}: size mismatch (local {target.stat().st_size}, "
            f"remote {expected}), retrying")
    raise RuntimeError(f"round {round_number}: checkpoint copy is truncated after two attempts")


# --------------------------------------------------------------------------- main

def run_round(args, remote: Remote | None, round_number: int, remote_path: str | None,
              model: Path | None) -> dict:
    if model is None:
        model = pull_model(remote, round_number, remote_path, args.model_dir)
    record = evaluate(round_number, model, args.backend, args.workers,
                      args.resource_dir, args.executable)
    append_row(args.out / "northstar.csv", record)
    plot(args.out / "northstar.png", read_rows(args.out / "northstar.csv"))
    if not args.doc.is_file():
        args.doc.write_text(DOC_HEADER, encoding="utf-8")
    append_doc(args.doc, record)
    log(f"round {round_number}: completion {record['completion_rate']}, unseen "
        f"{record['unseen_rate']}, m50 "
        + ", ".join(f"L{level}={record[f'm50_l{level}']}" for level in LADDER_LEVELS))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="newbiexvwu@100.120.155.85")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--model", type=Path, help="bench this checkpoint instead of the active one")
    parser.add_argument("--round", type=int, help="student round of --model")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--interval", type=int, default=1800,
                        help="seconds between polls (default 1800 = 30 minutes)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what the next poll would do and exit")
    args = parser.parse_args()

    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be from 1 to {MAX_WORKERS} on this Mac")
    if bool(args.model) != bool(args.round):
        parser.error("--model and --round go together")
    args.model_dir = args.model_dir.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.executable = args.executable.expanduser().resolve()
    args.resource_dir = args.resource_dir.expanduser().resolve()

    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another northstar run is active ({args.out / '.lock'})", file=sys.stderr)
        return 2

    if args.model:
        run_round(args, None, args.round, None, args.model.expanduser().resolve())
        return 0

    remote = Remote(args.host, args.distro)
    while True:
        try:
            round_number, remote_path = remote.read_state()
            done = {int(row["round"]) for row in read_rows(args.out / "northstar.csv")}
            if args.dry_run:
                print(f"active student round {round_number} at {remote_path}; "
                      f"benchmarked rounds {sorted(done)}")
                return 0
            if round_number in done:
                log(f"student round {round_number} already benchmarked")
            else:
                log(f"new student round {round_number}: {remote_path}")
                run_round(args, remote, round_number, remote_path, None)
        except Exception as exc:
            log(f"poll failed: {exc}")
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
