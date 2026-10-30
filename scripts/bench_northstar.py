#!/usr/bin/env python3
"""Northstar benchmark: poll the desktop curriculum state and chart each student.

task179 §5 asks for a trend line that never takes part in promotion:

* completion -- adventure 1..50 at the original difficulty with the standard deck
  (``pvz_tasks.deck_for_level``; the engine's own conveyor bank on 5/10/20/25/30/40/45/50),
  16 seeds per level, pure student;
* generalization -- decks a teacher can actually win with: the 15 task129 v5 decks on which
  the task177 teacher won at least one game, plus the 38 GPT Pro verified winning decks of
  ``configs/deploy17_random_deck_pool.json``; 2 seeds each;
* difficulty -- the 0.5..3.0 multiplier ladder on L7/L14/L25/L33/L45, 16 seeds per rung,
  reported as the multiplier where the win rate crosses 50 % (logit-linear interpolation).

Every model plays exactly the same (task, seed) pairs, so rounds are paired across time.
Results are appended to one CSV keyed by student round; ``northstar.png`` plots the trend;
a short paragraph goes to ``MEMORY/docs/task181-northstar.md``. Everything except the note
lives under ``~/pvz-logs/northstar/`` and is never committed.

This runs on the desktop, so the curriculum state is read in place and nothing is written
there: the poller only reads ``curriculum_state.json`` and the checkpoint it points at.
The systemd user unit caps it at two game processes, nice 19 and 3 GiB so it yields to the
production queue.

Usage:
  python3 scripts/bench_northstar.py --once --dry-run          # show the active student
  python3 scripts/bench_northstar.py --once                    # one poll, bench if new
  python3 scripts/bench_northstar.py --model m.pt --round 52   # one explicit student
  systemctl --user start pvz-northstar                         # the installed poller
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
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_play  # noqa: E402

STATE = Path.home() / "pvz-logs/curriculum_state.json"
DEFAULT_OUT = Path.home() / "pvz-logs/northstar"
DEFAULT_DOC = ROOT / "MEMORY/docs/task181-northstar.md"
TASK129_DECKS = ROOT / "configs/task129_decks_v5.json"
DEPLOY17_DECKS = ROOT / "configs/deploy17_random_deck_pool.json"
DEFAULT_EXECUTABLE = DEFAULT_OUT / "engine/pvz-portable"
DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    "PVZ_RESOURCE_DIR", str(Path.home() / ".cache/pvz-research-resources")))
MAX_WORKERS = 2
LEVELS = tuple(range(1, 51))
SEEDS_PER_LEVEL = 16
UNSEEN_SEEDS = 2
LADDER_LEVELS = (7, 14, 25, 33, 45)
LADDER = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
LADDER_SEEDS = 16
COMPLETION_SEED_START = 932_000_000
LADDER_SEED_START = 934_000_000
UNSEEN_SEED_START = 936_000_000
ZONE_BY_BACKGROUND = {"BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night",
                      "BACKGROUND_3_POOL": "pool", "BACKGROUND_4_FOG": "fog",
                      "BACKGROUND_5_ROOF": "roof", "BACKGROUND_6_BOSS": "roof"}
# task177 base+deep arms: these are the only task129 decks on which the teacher won >= 1 game.
TASK177_WINS = {(7, "d02"), (7, "d04"), (7, "d05"), (9, "d05"), (9, "d06"),
                (14, "d01"), (14, "d02"), (14, "d03"), (24, "d02"), (24, "d05"),
                (26, "d01"), (26, "d02"), (26, "d03"), (37, "d01"), (37, "d05")}
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
    tasks, plan = {}, {}
    registry = json.loads(TASK129_DECKS.read_text())
    for entry in registry["tasks"]:
        # deck129_v5_l07_d02 -> level 7, deck d02
        _, _, _, deck_tag = entry["task_id"].split("_")
        if (entry["level"], deck_tag) not in TASK177_WINS:
            continue
        task_id = f"unseen_{entry['task_id']}"
        tasks[task_id] = {"task_id": task_id, "zone": zone_of(entry["level"]),
                          "level": entry["level"], "deck": list(entry["deck"]),
                          "zombie_count_multiplier": 1.0, "playthrough": 2}
        plan[task_id] = [int(seed) for seed in entry["seeds"][:UNSEEN_SEEDS]]
    pool = json.loads(DEPLOY17_DECKS.read_text())
    for index, entry in enumerate(pool["tasks"]):
        task_id = f"unseen_{entry['task_id']}"
        tasks[task_id] = {"task_id": task_id, "zone": zone_of(entry["level"]),
                          "level": entry["level"], "deck": list(entry["deck"]),
                          "zombie_count_multiplier": 1.0, "playthrough": 2}
        first = UNSEEN_SEED_START + index * UNSEEN_SEEDS
        plan[task_id] = list(range(first, first + UNSEEN_SEEDS))
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
    remaining = jobs
    context = mp.get_context(start_method)
    for attempt in (1, 2):
        if not remaining:
            break
        pool = context.Pool(workers, initializer=bench_play._init_worker, initargs=(cfg,))
        try:
            for index, row in enumerate(pool.imap_unordered(
                    bench_play._play_one, remaining, chunksize=1), 1):
                rows.append(row)
                if index % 100 == 0 or index == len(remaining):
                    log(f"{name}: {index}/{len(remaining)} games")
            remaining = []
        except Exception as exc:  # a game crashed the worker; retry the rest once
            log(f"{name}: game failed ({exc}) on attempt {attempt}")
            if attempt == 2:
                raise
            done = {(row["task_id"], row["seed"]) for row in rows}
            remaining = [job for job in jobs if (job["task_id"], job["seed"]) not in done]
        finally:
            pool.close()
            pool.join()
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
    from matplotlib.ticker import MaxNLocator
    rounds = [int(row["round"]) for row in rows]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2]})
    top.plot(rounds, [float(row["completion_rate"]) for row in rows], "o-", label="adventure 1-50")
    top.plot(rounds, [float(row["unseen_rate"]) for row in rows], "s-", label="winnable unseen decks")
    top.set_ylabel("win rate")
    top.set_ylim(0, 1)
    top.grid(alpha=0.3)
    top.legend(loc="lower right")
    for level in LADDER_LEVELS:
        bottom.plot(rounds, [float(row[f"m50_l{level}"]) for row in rows], "o-", label=f"L{level}")
    bottom.set_xlabel("student round")
    bottom.set_ylabel("multiplier at 50% win rate")
    bottom.set_ylim(0.4, 3.1)
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
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
            f"- 可赢陌生卡组：{record['unseen_wins']}/{record['unseen_games']} = "
            f"{percent(record['unseen_rate'])}。\n"
            f"- 倍率 50%（0.5–3.0 阶梯，端点表示未跨过）：{ladder}。\n")
    with doc.open("a", encoding="utf-8") as handle:
        handle.write(text)


DOC_HEADER = """# task181 北极星基准

不参与晋升，只看走势（task179 §5）。台式机用户服务 `pvz-northstar` 每 30 分钟读
`~/pvz-logs/curriculum_state.json` 的 `student` / `student_round`，出现新的现任学生就跑一遍，
只读生产状态。结果 CSV 与曲线图在 `~/pvz-logs/northstar/`，不入 git；本文件只记关键数字。

- **通关**：冒险 1–50 关，原难度、`pvz_tasks.deck_for_level` 标准卡组（5/10/20/25/30/40/45/50
  用引擎自己的传送带），纯学生，每关 16 个种子。
- **泛化**：用老师能赢的陌生卡组 —— task177 里老师至少赢过 1 局的 15 套 deck129 v5 卡组
  （L7 d02/d04/d05、L9 d05/d06、L14 d01/d02/d03、L24 d02/d05、L26 d01/d02/d03、L37 d01/d05），
  加 `configs/deploy17_random_deck_pool.json` 里 38 套 GPT Pro 胜局卡组，每套 2 个种子。
- **难度**：L7/L14/L25/L33/L45 的 9 档倍率（0.5–3.0），每档 16 个种子；报胜率 50% 对应的倍率
  （相邻两档在 log(倍率) × logit 上线性插值，落在端点表示没跨过 50%）。

引擎固定在 `~/pvz-logs/northstar/engine/pvz-portable`（deploy19 生产二进制，SHA256 `41bb2a36…`），
C++ 原生整局，最多 2 个游戏进程。种子固定（通关 932M、倍率 934M、deploy17 卡组 936M、
task129 卡组用登记表自带种子），所有学生逐局配对。L35 是砸罐子关，学生没有砸罐动作，
通关率结构性为 0，看走势时忽略这一关。
"""


# --------------------------------------------------------------------------- main

def active_student(path: Path) -> tuple[int, Path]:
    document = json.loads(path.read_text())
    return int(document["student_round"]), Path(document["student"])


def acquire_lock(path: Path):
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def run_round(args, round_number: int, model: Path) -> dict:
    if not model.is_file():
        raise RuntimeError(f"checkpoint not found: {model}")
    record = evaluate(round_number, model, args.backend, args.workers,
                      args.resource_dir, args.executable)
    append_row(args.out / "results.csv", record)
    finalize(args)
    log(f"round {round_number}: completion {record['completion_rate']}, unseen "
        f"{record['unseen_rate']}, m50 "
        + ", ".join(f"L{level}={record[f'm50_l{level}']}" for level in LADDER_LEVELS))
    return record


def finalize(args) -> None:
    """Rewrite the plot and append any CSV round missing from the note."""
    rows = read_rows(args.out / "results.csv")
    if not rows:
        return
    plot(args.out / "northstar.png", rows)
    if not args.doc.is_file():
        args.doc.write_text(DOC_HEADER, encoding="utf-8")
    text = args.doc.read_text(encoding="utf-8")
    for row in rows:
        marker = f"### r{row['round']}（"
        if marker not in text:
            append_doc(args.doc, row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="bench this checkpoint instead of the active one")
    parser.add_argument("--round", type=int, help="student round of --model")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--interval", type=int, default=1800,
                        help="seconds between polls (default 1800 = 30 minutes)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what the next poll would do and exit")
    args = parser.parse_args()

    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be from 1 to {MAX_WORKERS} on the desktop")
    if bool(args.model) != bool(args.round):
        parser.error("--model and --round go together")
    args.state = args.state.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.executable = args.executable.expanduser().resolve()
    args.resource_dir = args.resource_dir.expanduser().resolve()

    args.out.mkdir(parents=True, exist_ok=True)
    lock_path = args.out / ".lock"

    if args.model:
        lock = acquire_lock(lock_path)
        if lock is None:
            print(f"another northstar run is active ({lock_path})", file=sys.stderr)
            return 2
        try:
            finalize(args)
            run_round(args, args.round, args.model.expanduser().resolve())
        finally:
            lock.close()
        return 0

    while True:
        lock = acquire_lock(lock_path)
        if lock is None:
            log(f"another northstar run is active ({lock_path}); skipping this poll")
        else:
            try:
                finalize(args)
                round_number, model = active_student(args.state)
                done = {int(row["round"]) for row in read_rows(args.out / "results.csv")}
                if args.dry_run:
                    print(f"active student r{round_number} at {model}; "
                          f"benchmarked rounds {sorted(done)}")
                    return 0
                if round_number in done:
                    log(f"student round {round_number} already benchmarked")
                else:
                    log(f"new student round {round_number}: {model}")
                    run_round(args, round_number, model)
            except Exception as exc:
                log(f"poll failed: {exc}")
                if args.once:
                    return 1
            finally:
                lock.close()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
