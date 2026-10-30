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
* GPT -- task105 wins from the two GPT Pro runs, original 1.0x, two fixed seeds per
  (level, deck); fog requires human_v2, matching task194's clean source filter.
* teacher -- production search on the control suite, once per student, idle Mac only.

Every model plays exactly the same (task, seed) pairs, so rounds are paired across time.
Results are appended to one CSV keyed by student round; ``northstar.png`` plots the trend;
a short paragraph goes to ``MEMORY/docs/task181-northstar.md``. Everything except the note
lives under ``~/pvz-logs/northstar/`` and is never committed.

Runs on the Mac at nice 10. Four workers plus their simulators and the driver stay below
the twelve-process budget. Checkpoints travel through HF transfer/ and are deleted there
immediately after downloading. Teacher games only run while the Mac is idle.

Usage:
  python3 scripts/bench_northstar.py --once --dry-run          # show the active student
  python3 scripts/bench_northstar.py --once                    # one poll, bench if new
  python3 scripts/bench_northstar.py --model m.pt --round 52   # one explicit student
  nice -n 10 python3 scripts/bench_northstar.py --workers 4
"""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_play  # noqa: E402

STATE = Path('/home/newbiexvwu/pvz-logs/curriculum_state.json')
DEFAULT_OUT = Path.home() / "pvz-logs/northstar"
DEFAULT_DOC = ROOT / "MEMORY/docs/task181-northstar.md"
TASK129_DECKS = ROOT / "configs/task129_decks_v5.json"
DEPLOY17_DECKS = ROOT / "configs/deploy17_random_deck_pool.json"
CONTROL_PANEL = ROOT / "configs/gate_v5_panel.json"
DEFAULT_EXECUTABLE = ROOT / "build/pvz-portable"
DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    "PVZ_RESOURCE_DIR", str(Path.home() / ".cache/pvz-research-resources")))
MAX_WORKERS = 4  # ten game/driver processes, leaving room for SSH polling
MODEL_CACHE = Path.home() / '.cache/pvz-northstar/models'
REMOTE = 'newbiexvwu@fd7a:115c:a1e0::ab2a:9b56'
IMPORT_ROOT = '/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport'
GPT_SEED_START = 938_000_000
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
          "control_completion_wins", "control_completion_games", "control_completion_rate",
          *[f"control_{zone}_{name}" for zone in ("day", "night", "pool", "fog", "roof")
            for name in ("wins", "games")],
          "unseen_wins", "unseen_games", "unseen_rate",
          "gpt_wins", "gpt_games", "gpt_rate",
          "teacher_wins", "teacher_games", "teacher_rate",
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


def control_deck_for_level(level: int) -> tuple[int, ...]:
    """Deck the training queue and gate actually use on this level (task184 control).

    Read from the gate v5 panel rather than a second hand-written deck table: its
    per-level decks are the d7/n10/p9k/fog_selected/roof_demo families, and the first
    deck of a zone covers levels the panel leaves out (L15 night, L42-45/47-49 roof).
    Conveyor levels keep the engine's own bank, exactly like the standard suite.
    """
    import pvz_tasks
    if pvz_tasks.conveyor_belt_level(level):
        return ()
    document = json.loads(CONTROL_PANEL.read_text())
    by_level, by_zone = {}, {}
    for task in document["tasks"]:
        zone = document["groups"][task["task_id"]]
        if zone == "heldout":
            continue
        by_level[int(task["level"])] = tuple(task["deck"])
        by_zone.setdefault(zone, tuple(task["deck"]))
    return by_level.get(level) or by_zone[zone_of(level)]


def control_suite() -> dict:
    """Adventure 1-50 on the training/gate decks, paired with the completion seeds."""
    tasks, plan = {}, {}
    for index, level in enumerate(LEVELS):
        task = standard_task(f"control_l{level:02d}", level)
        deck = control_deck_for_level(level)
        task["deck"] = list(deck)
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


SUITES = {"completion": completion_suite, "control": control_suite,
          "unseen": unseen_suite, "ladder": ladder_suite}


def remote_python(remote: str, code: str) -> str:
    options = ['-o', 'HostKeyAlias=100.120.155.85'] if remote == REMOTE else []
    encoded = base64.b64encode(code.encode()).decode()
    command = f'"exec(__import__(\'base64\').b64decode(\'{encoded}\'))"'
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=60', '-o', 'ServerAliveInterval=10', *options,
                             remote, 'wsl', '-d', 'Ubuntu',
                             '-e', '/home/newbiexvwu/.venvs/ml/bin/python3', '-c', command],
                            stdin=subprocess.DEVNULL, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def gpt_suite(remote: str) -> tuple[dict, dict]:
    # Read task105's existing import ledger and metadata, never frames or replay.
    code = f'''
import json, sys
from pathlib import Path
root = Path({IMPORT_ROOT!r})
sys.path.insert(0, str(root/'scripts'))
from import_llm_run import family_entry
from pvz_tasks import terrain_covered
from pvz_constants import background_for_level
rows = {{}}
for name in ('merged_progress.json', 'import_progress.json'):
    p = root/'artifacts/task105/import'/name
    if p.exists():
        for r in json.loads(p.read_text()).get('archives', []):
            rows[r.get('archive_sha256') or r.get('archive')] = r
tasks = []
for _, r in sorted(rows.items()):
    if r.get('verify') != '通过' or r.get('result') != 1 or r.get('manual_patch'):
        continue
    p = Path(r['archive'])
    if not p.is_absolute(): p = root/p
    if not (p/'meta.json').is_file(): continue
    meta = json.loads((p/'meta.json').read_text())
    if not str(meta.get('run_id', '')).startswith(('438cb30a', '8024414b')):
        continue
    if not meta.get('verified') or meta.get('manual_patch') or meta.get('outcome', {{}}).get('result') != 1:
        continue
    t = family_entry('gpt', meta, None)
    if background_for_level(t['level']) == 'BACKGROUND_4_FOG' and t.get('fog_observation') != 'human_v2':
        continue
    if (t['zombie_count_multiplier'] == 1 and t['deck'] and terrain_covered(t['level'], t['deck'])
            and not any(t.get(k) for k in ('wave_cap', 'preplanted', 'conveyor_start'))):
        tasks.append(t)
print(json.dumps(tasks))
'''
    entries = json.loads(remote_python(remote, code))
    tasks, plan = {}, {}
    for entry in entries:
        level, deck = int(entry['level']), tuple(sorted(entry['deck']))
        identity = hashlib.sha1(f'{level}:{deck}'.encode()).hexdigest()[:10]
        task_id = f'gpt_l{level:02d}_{identity}'
        if task_id in tasks:
            continue
        task = dict(entry, task_id=task_id, zone=zone_of(level), playthrough=2)
        task.pop('seeds', None)
        tasks[task_id] = task
        # Identity-based assignment stays paired when another winning deck is imported.
        first = GPT_SEED_START + 2 * (int(identity, 16) % 100_000_000)
        plan[task_id] = [first, first + 1]
    return tasks, plan


def cached_model(remote: str, round_number: int, source: Path) -> Path:
    target = MODEL_CACHE / f'r{round_number:06d}.pt'
    if target.is_file():
        return target
    from huggingface_hub import HfApi, hf_hub_download
    name = f'transfer/northstar/r{round_number:06d}.pt'
    # The desktop only reads its model; upload credentials stay on that machine.
    code = f'''
import os, re, shlex
from pathlib import Path
from huggingface_hub import HfApi
for line in (Path.home()/'.bashrc').read_text().splitlines():
    if re.match(r'^export (HF_TOKEN|PVZ_HF_REPO)=', line):
        key, value = line.removeprefix('export ').split('=', 1)
        os.environ[key] = shlex.split(value)[0]
commit = HfApi(token=os.environ['HF_TOKEN']).upload_file(
    repo_id=os.environ['PVZ_HF_REPO'], repo_type='model', path_or_fileobj={str(source)!r},
    path_in_repo={name!r})
print(commit.oid)
'''
    revision = remote_python(remote, code).strip()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pull-', dir=MODEL_CACHE) as temporary:
        pulled = hf_hub_download(repo_id=os.environ['PVZ_HF_REPO'], repo_type='model',
            filename=name, revision=revision, local_dir=temporary, token=os.environ['HF_TOKEN'])
        shutil.move(pulled, target)
    HfApi(token=os.environ['HF_TOKEN']).delete_file(
        repo_id=os.environ['PVZ_HF_REPO'], repo_type='model', path_in_repo=name)
    return target


def _init_bench_worker(cfg):
    bench_play._init_worker(cfg)
    from multiprocessing.util import Finalize
    Finalize(None, bench_play._W['env'].close, exitpriority=10)


# --------------------------------------------------------------------------- bench

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_suite(name: str, model: Path, native: Path | None, backend: str, workers: int,
              resource_dir: Path, executable: Path, suite=None) -> list[dict]:
    tasks, plan = suite if suite is not None else SUITES[name]()
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
        pool = context.Pool(workers, initializer=_init_bench_worker, initargs=(cfg,))
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
             resource_dir: Path, executable: Path, gpt_tasks, previous=None) -> dict:
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

    try:
        if previous is None:
            completion = run_suite("completion", model, native, backend, workers, resource_dir, executable)
            control = run_suite("control", model, native, backend, workers, resource_dir, executable)
            unseen = run_suite("unseen", model, native, backend, workers, resource_dir, executable)
            ladder = run_suite("ladder", model, native, backend, workers, resource_dir, executable)
        gpt = run_suite('gpt', model, native, backend, workers, resource_dir, executable, gpt_tasks)
    finally:
        if native is not None:
            shutil.rmtree(native.parent)
    gpt_block = wins(gpt)
    gpt_record = dict(gpt_wins=gpt_block[0], gpt_games=gpt_block[1], gpt_rate=rate(gpt_block))
    if previous is not None:
        return dict(previous, **gpt_record, seconds=round(float(previous['seconds']) + time.time()-started, 1))
    zones, control_zones = {}, {}
    for zone in ("day", "night", "pool", "fog", "roof"):
        zones[zone] = wins([row for row in completion if row["zone"] == zone])
        control_zones[zone] = wins([row for row in control if row["zone"] == zone])
    completion_block = wins(completion)
    control_block, unseen_block = wins(control), wins(unseen)
    record = {"round": round_number,
              "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "model_sha256": sha256_file(model), "model_bytes": model.stat().st_size,
              "backend": backend, "workers": workers,
              "seconds": round(time.time() - started, 1),
              "completion_wins": completion_block[0], "completion_games": completion_block[1],
              "completion_rate": rate(completion_block),
              **{f"{zone}_{name}": value for zone, block in zones.items()
                 for name, value in zip(("wins", "games"), block)},
              "control_completion_wins": control_block[0],
              "control_completion_games": control_block[1],
              "control_completion_rate": rate(control_block),
              **{f"control_{zone}_{name}": value
                 for zone, block in control_zones.items()
                 for name, value in zip(("wins", "games"), block)},
              "unseen_wins": unseen_block[0], "unseen_games": unseen_block[1],
              "unseen_rate": rate(unseen_block),
              **gpt_record,
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
    rows = {int(row['round']): row for row in read_rows(path)}
    rows[int(record['round'])] = record
    temporary = path.with_suffix('.building')
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for _, row in sorted(rows.items()):
            writer.writerow({name: row.get(name) for name in FIELDS})
    temporary.replace(path)


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
    top.plot(rounds, [float(row["completion_rate"]) for row in rows], "o-", label="adventure 1-50 (standard deck)")
    control = [(int(row["round"]), float(row["control_completion_rate"])) for row in rows
               if row.get("control_completion_rate")]
    if control:
        top.plot(*zip(*control), "^-", label="adventure 1-50 (training/gate decks)")
    top.plot(rounds, [float(row["unseen_rate"]) for row in rows], "s-", label="winnable unseen decks")
    for field, label in [('gpt_rate', 'GPT won (level, deck), 1.0x'),
                         ('teacher_rate', 'teacher, training/gate decks')]:
        values = [(int(row['round']), float(row[field])) for row in rows if row.get(field)]
        if values:
            top.plot(*zip(*values), 'o--' if field == 'teacher_rate' else 'D-', label=label)
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
    control_zones = "、".join(f"{zone} {record[f'control_{zone}_wins']}/{record[f'control_{zone}_games']}"
                              for zone in ("day", "night", "pool", "fog", "roof"))
    ladder = "、".join(f"L{level} {record[f'm50_l{level}']}" for level in LADDER_LEVELS)
    text = (f"\n### r{record['round']}（{record['evaluated_at']}）\n\n"
            f"- 模型 `{record['model_sha256'][:12]}`（{int(record['model_bytes']) / 1e6:.1f} MB），"
            f"{record['backend']} 后端，{record['workers']} 进程，耗时 {record['seconds']} s。\n"
            f"- 冒险 1–50：{record['completion_wins']}/{record['completion_games']} = "
            f"{percent(record['completion_rate'])}（{zones}）。\n"
            f"- 对照：训练/门禁卡组同一批种子 {record['control_completion_wins']}/"
            f"{record['control_completion_games']} = {percent(record['control_completion_rate'])}"
            f"（{control_zones}）。\n"
            f"- 可赢陌生卡组：{record['unseen_wins']}/{record['unseen_games']} = "
            f"{percent(record['unseen_rate'])}。\n"
            f"- GPT 原版胜局卡组：{record.get('gpt_wins', '—')}/{record.get('gpt_games', '—')} = "
            f"{percent(record.get('gpt_rate'))}。老师同训练/门禁任务：{percent(record.get('teacher_rate'))}。\n"
            f"- 倍率 50%（0.5–3.0 阶梯，端点表示未跨过）：{ladder}。\n")
    with doc.open("a", encoding="utf-8") as handle:
        handle.write(text)


DOC_HEADER = """# task181 北极星基准

不参与晋升，只看走势（task179 §5）。台式机用户服务 `pvz-northstar` 每 30 分钟读
`~/pvz-logs/curriculum_state.json` 的 `student` / `student_round`，出现新的现任学生就跑一遍，
只读生产状态。结果 CSV 与曲线图在 `~/pvz-logs/northstar/`，不入 git；本文件只记关键数字。

- **通关**：冒险 1–50 关，原难度、`pvz_tasks.deck_for_level` 标准卡组（5/10/20/25/30/40/45/50
  用引擎自己的传送带），纯学生，每关 16 个种子。
- **对照**：同一批关卡/种子、换成训练与门禁实际在用的卡组（d7 / n10 / p9k /
  fog_selected / roof_demo，从 `configs/gate_v5_panel.json` 现读；面板没列的关卡用同区
  第一套卡组，传送带关仍用引擎牌组）。标准卡组与它的差距是卡组口径差，不是能力差。
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

def active_student(path: Path, remote: str) -> tuple[int, Path]:
    document = json.loads(remote_python(remote, f'from pathlib import Path\nprint(Path({str(path)!r}).expanduser().read_text())'))
    return int(document["student_round"]), Path(document["student"])


def idle(args) -> bool:
    """Yield to other compute jobs; exclude this poller and all its descendants."""
    listing = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,%cpu='], text=True)
    rows = [line.split() for line in listing.splitlines()]
    ours = {os.getpid()}
    for _ in range(4):
        ours.update(int(pid) for pid, parent, _cpu in rows if int(parent) in ours)
    return sum(float(cpu) for pid, _parent, cpu in rows if int(pid) not in ours) < args.idle_cpu_percent


def _teacher_worker(cfg):
    import search_play
    global _TEACHER
    _TEACHER = cfg
    search_play._init_worker(cfg['values'][0], cfg['args'].rollout_policy,
        cfg['args'].rollout_sim_skip, cfg['values'][1])


def _teacher_game(job):
    import search_play
    task, seed = job
    args = argparse.Namespace(**{**vars(_TEACHER['args']), 'task_id': task['task_id']})
    row = search_play._play(dict(args=args, arm='fast_search', seed=seed, task_definition=task))
    return dict(task_id=task['task_id'], seed=seed, zone=task['zone'],
                won=int(row['won']), final_wave=row['final_wave'], seconds=row['seconds'])


def teacher(args, record, model):
    """Bounded in-flight games, durable numerical results, student-first idle scheduling."""
    from concurrent.futures import ThreadPoolExecutor
    import search_play
    tasks, plan = control_suite()
    path = args.out / f'teacher_r{int(record["round"]):06d}.csv'
    rows = read_rows(path)
    done = {(row['task_id'], int(row['seed'])) for row in rows}
    jobs = [(tasks[key], seed) for key, seeds in plan.items() for seed in seeds
            if (key, seed) not in done]
    if jobs and idle(args):
        params = search_play.parser().parse_args([
            '--rollout-policy', str(model), '--value', str(args.value),
            '--static-value', str(args.static_value), '--manifest', 'builtin:northstar',
            '--task-id', jobs[0][0]['task_id'], '--out', str(args.out),
            '--resource-dir', str(args.resource_dir), '--executable', str(args.executable),
            '--workers', '1', '--verify-samples', '9', '--student-candidates', '2',
            '--native-games', '--dagger-beta', '1'])
        keep, values = search_play._export_values(str(args.value), str(args.static_value))
        pool = mp.get_context('spawn').Pool(args.workers, initializer=_teacher_worker,
                                          initargs=(dict(args=params, values=values),))
        pending, cursor, admit, next_poll = [], 0, True, time.monotonic()+60
        polls = ThreadPoolExecutor(max_workers=1)
        poll = None
        try:
            with path.open('a', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=('task_id','seed','zone','won','final_wave','seconds'))
                if handle.tell() == 0:
                    writer.writeheader()
                while pending or (admit and cursor < len(jobs)):
                    if time.monotonic() >= next_poll:
                        admit = idle(args)
                        if poll is None:
                            poll = polls.submit(active_student, args.state, args.remote)
                        next_poll = time.monotonic() + 60
                    if poll is not None and poll.done():
                        try:
                            current, _ = poll.result()
                            ready = {int(r['round']) for r in read_rows(args.out/'results.csv') if r.get('gpt_rate')}
                            admit = admit and current in ready
                        except RuntimeError:
                            pass
                        poll = None
                    while admit and len(pending) < args.workers and cursor < len(jobs):
                        pending.append(pool.apply_async(_teacher_game, (jobs[cursor],)))
                        cursor += 1
                    ready_jobs = [job for job in pending if job.ready()]
                    for job in ready_jobs:
                        row = job.get()
                        rows.append(row)
                        writer.writerow(row)
                        handle.flush()
                        pending.remove(job)
                        if len(rows) % 16 == 0:
                            log(f'teacher r{record["round"]}: {len(rows)}/800 games')
                    if not ready_jobs:
                        time.sleep(.2)
        finally:
            pool.close()
            pool.join()
            polls.shutdown(wait=True)
            keep.cleanup()
    if len(rows) == sum(map(len, plan.values())):
        won = sum(int(row['won']) for row in rows)
        record.update(teacher_wins=won, teacher_games=len(rows), teacher_rate=round(won/len(rows), 4))
        append_row(args.out/'results.csv', record)
        finalize(args)
        if record.get('gpt_rate') not in (None, '') and model.parent.resolve() == MODEL_CACHE.resolve():
            model.unlink(missing_ok=True)
        log(f'teacher r{record["round"]}: {won}/{len(rows)} complete')
        return True
    return False


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
    previous = next((row for row in read_rows(args.out/'results.csv')
                     if int(row['round']) == round_number), None)
    if previous and previous.get('gpt_rate'):
        return previous
    if args.gpt_tasks is None:
        args.gpt_tasks = gpt_suite(args.remote)
        log(f'GPT original winning decks: {len(args.gpt_tasks[0])} level/deck combinations')
    record = evaluate(round_number, model, args.backend, args.workers,
                      args.resource_dir, args.executable, args.gpt_tasks, previous)
    append_row(args.out / "results.csv", record)
    finalize(args)
    if record.get('teacher_rate') not in (None, '') and model.parent.resolve() == MODEL_CACHE.resolve():
        model.unlink(missing_ok=True)
    log(f"round {round_number}: completion {record['completion_rate']}, control "
        f"{record['control_completion_rate']}, unseen {record['unseen_rate']}, m50 "
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
            text = args.doc.read_text(encoding='utf-8')
        elif row.get('gpt_rate') or row.get('teacher_rate'):
            start = text.index(marker)
            end = text.find('\n### ', start+len(marker))
            end = len(text) if end < 0 else end
            note = (f"- GPT 原版胜局卡组：{row.get('gpt_wins') or '—'}/{row.get('gpt_games') or '—'} = {percent(row.get('gpt_rate'))}。"
                    f"老师同训练/门禁任务：{row.get('teacher_wins') or '—'}/"
                    f"{row.get('teacher_games') or '—'} = {percent(row.get('teacher_rate'))}。\n")
            block = text[start:end]
            pattern = r'(?m)^- GPT 原版胜局卡组：[^\n]*\n?'
            block = re.sub(pattern, note, block) if re.search(pattern, block) else block.rstrip()+'\n'+note+'\n'
            text = text[:start]+block+text[end:]
    if text != args.doc.read_text(encoding='utf-8'):
        args.doc.write_text(text, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="bench this checkpoint instead of the active one")
    parser.add_argument("--round", type=int, help="student round of --model")
    parser.add_argument("--state", type=Path, default=STATE)
    parser.add_argument('--remote', default=REMOTE, help='desktop SSH host; state and task105 metadata only')
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument('--value', type=Path, default=Path.home()/'pvz-bench-data/task171/teacher/value.pt')
    parser.add_argument('--static-value', type=Path, default=Path.home()/'pvz-bench-data/task171/teacher/static.pt')
    parser.add_argument('--idle-cpu-percent', type=float, default=150,
                        help='pause teacher admission when other compute uses this %%CPU')
    parser.add_argument("--interval", type=int, default=1800,
                        help="seconds between polls (default 1800 = 30 minutes)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what the next poll would do and exit")
    args = parser.parse_args()

    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be from 1 to {MAX_WORKERS} (at most twelve processes)")
    if bool(args.model) != bool(args.round):
        parser.error("--model and --round go together")
    args.out = args.out.expanduser().resolve()
    args.executable = args.executable.expanduser().resolve()
    args.resource_dir = args.resource_dir.expanduser().resolve()
    args.value = args.value.expanduser().resolve()
    args.static_value = args.static_value.expanduser().resolve()
    args.gpt_tasks = None
    os.nice(max(0, 10-os.getpriority(os.PRIO_PROCESS, 0)))

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
                round_number, source = active_student(args.state, args.remote)
                records = read_rows(args.out/'results.csv')
                done = {int(row['round']) for row in records if row.get('gpt_rate')}
                if args.dry_run:
                    print(f"active student r{round_number} at {source}; "
                          f"benchmarked rounds {sorted(done)}")
                    return 0
                if round_number in done:
                    log(f"student round {round_number} already benchmarked")
                else:
                    model = cached_model(args.remote, round_number, source)
                    log(f"new student round {round_number}: {model}")
                    run_round(args, round_number, model)
                for row in read_rows(args.out/'results.csv'):
                    number = int(row['round'])
                    if row.get('gpt_rate'):
                        continue
                    model = cached_model(args.remote, number, source.parents[2]/f'round_{number:06d}/student/model.pt')
                    run_round(args, number, model)
                for row in read_rows(args.out/'results.csv'):
                    if row.get('teacher_rate'):
                        continue
                    if not idle(args):
                        log('Mac compute is busy; teacher remains queued')
                        break
                    number = int(row['round'])
                    model = cached_model(args.remote, number, source.parents[2]/f'round_{number:06d}/student/model.pt')
                    if not teacher(args, row, model):
                        break
            except Exception as exc:
                log(f"poll failed: {exc}")
                if args.once:
                    return 1
                # Existing models need no network to advance the teacher column.
                for row in read_rows(args.out/'results.csv'):
                    model = MODEL_CACHE / f'r{int(row["round"]):06d}.pt'
                    if not row.get('teacher_rate') and model.is_file() and idle(args):
                        log(f'continuing cached teacher r{row["round"]} while remote input waits')
                        teacher(args, row, model)
                        break
            finally:
                lock.close()
        if args.once:
            return 0
        time.sleep(min(args.interval, 60) if any(not row.get('teacher_rate') or not row.get('gpt_rate')
            for row in read_rows(args.out/'results.csv')) else args.interval)


if __name__ == "__main__":
    sys.exit(main())
