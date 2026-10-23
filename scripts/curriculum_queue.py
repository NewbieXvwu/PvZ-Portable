#!/usr/bin/env python3
"""task29 第 2 步：自动课程采集控制器（取代手写的 queue101–103）。

分组 = 关卡 × 卡组族 × 倍率（1.0 档用原任务 id，其余用 `<base>_m125` … `<base>_m300`）。
每轮按 DESIGN §3.2 的元规则分配局数，只硬编码“优先学边缘任务”这一条：

1. **升级 / 降级**：某格当前倍率的滑动窗口胜率 > 0.85 且样本够 → 倍率升一档；
   < 0.5 → 降一档（不低于 1.0）。
2. **分配**：预留 `--guard-ratio`（默认 12%）给 1.0 档防遗忘；其余优先给当前倍率
   胜率落在 [0.5, 0.85] 的格；没有边缘格就退回全体。1.0 档也 < 0.5 的难关照样采
   （那正是老师标注有价值的地方）。
3. **落盘**：状态 json（默认 `~/pvz-logs/curriculum_state.json`），每轮一行日志；
   偏移在**跑之前**写进状态，崩溃/中断不会重采同一批种子。
4. **停止标志** `~/pvz-logs/stop_curr`：只在两组之间检查，绝不 SIGTERM 子进程。
5. **打包**：每轮结束把“本队列写过的全部标签 + `--recipe` 的来源标签”打成训练包（task33 第 2 项），
   并删掉被它取代的旧包。训练端只读包，行选项统一从训练配方解析（它们是包键的一部分）。

采集参数照 `~/pvz-logs/queue103.sh` 的 `run()`（beta=0，学生自走、搜索打标签），
帧默认直接写 `~/PvZ-search/artifacts/student/frames_v2`（只增不改）。并发不再手动定：
每开一组前读 MemAvailable，`--workers` 只是上限，实际 worker 数按
`(MemAvailable − 6 GiB) / 一个 worker 的实测 RSS` 取整，至少 1。每 worker 的 RSS 在
控制器启动时用一个单 worker 的真实对局实测（Python + 引擎峰值；对局写临时目录，不占偏移）。
非 Linux（没有 /proc）时不做缩放，直接用 `--workers`。

用法：
  python3 scripts/curriculum_queue.py --dry-run                 # 第一轮分配表（用已有结果初始化）
  python3 scripts/curriculum_queue.py --round-games 4 --games-per-group 2 \
      --state artifacts/task29/smoke_state.json --out artifacts/task29/smoke_runs \
      --frames artifacts/task29/smoke_frames                     # 冒烟（每组 2 局）
  python3 scripts/curriculum_queue.py --rounds 8                # 正式跑（主控决定何时替换 q103）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from make_multiplier_tasks import FAMILIES, SUPERSEDED, load_tasks, source_paths  # noqa: E402
from pvz_recipe import load_recipe, training_row_options
from pvz_training_pack import pack_flags, pack_options
from pvz_metrics import Metrics  # noqa: E402
from pvz_common import OOM_SCORE_CONTROLLER, set_oom_score_adj  # noqa: E402

FRONTIER_LOW = 0.5
FRONTIER_HIGH = 0.85
PROMOTE_ABOVE = 0.85
DEMOTE_BELOW = 0.5
MIN_SAMPLES = 8            # 少于这个局数不做升降级，只继续采
WINDOW = 256               # 滑动窗口：每组只记最近 256 局的结果（0/1）
GUARD_RATIO = 0.12
DEFAULT_STATE = Path.home() / 'pvz-logs/curriculum_state.json'
DEFAULT_LOG = Path.home() / 'pvz-logs/curriculum.log'
STOP_FLAG = Path.home() / 'pvz-logs/stop_curr'
DEFAULT_FRAMES = MAIN / 'artifacts/student/frames_v2'
DEFAULT_MODEL = MAIN / 'artifacts/student/dw4i_s1/model.pt'
DEFAULT_STATIC_VALUE = MAIN / 'artifacts/hard/value_branch1/model.pt'
DEFAULT_VALUE = MAIN / 'artifacts/value_net/v4/model.pt'
DEFAULT_RESOURCES = Path.home() / '.cache/pvz-research-resources'
DEFAULT_PYTHON = Path('/home/newbiexvwu/.venvs/ml/bin/python3')   # torch / numpy 在这里；系统 python3 没有

GIB = 1024 ** 3
MEMORY_RESERVE_BYTES = 6 * GIB          # 留给 WSL 接入通道、控制器与训练，不交给采集
DEFAULT_WORKER_BYTES = GIB              # 实测失败时的保守值（一个 worker 的 Python + 引擎）
PROBE_TIMEOUT_SECONDS = 900.0
RSS_POLL_SECONDS = 0.2


def mem_available_bytes() -> int | None:
    """Linux 的 MemAvailable（字节）；没有 /proc 的平台返回 None。"""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def worker_count(available: int | None, per_worker: int, upper: int) -> int:
    """clamp(⌊(MemAvailable − 6 GiB) / per_worker⌋, 1, upper)；未知内存或未实测时用上限。"""
    if available is None or per_worker <= 0:
        return upper
    return max(1, min(upper, (available - MEMORY_RESERVE_BYTES) // per_worker))


def _proc_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, 常驻字节)，一次扫完 /proc；没有 /proc 时为空。"""
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        entries = list(os.scandir('/proc'))
    except (OSError, ValueError):
        return {}
    table = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = Path(entry.path, 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(entry.name)] = (int(fields[1]), int(fields[21]) * page)
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendant_rss(root: int, table: dict[int, tuple[int, int]]) -> int:
    """root 的所有后代（不含 root）的常驻字节之和。"""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss) in table.items():
        children.setdefault(ppid, []).append(pid)
    total, stack = 0, list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        total += table[pid][1]
        stack.extend(children.get(pid, ()))
    return total


def measure_worker_bytes(args, item: dict) -> int:
    """用一个单 worker 的真实对局测一个 worker 的 Python + 引擎峰值 RSS。

    对局写在临时目录，不动状态偏移，也不进帧档案；正式第一组会用同一个种子重跑。
    """
    with tempfile.TemporaryDirectory(prefix='pvz-oom-probe-') as tmp:
        tmp_path = Path(tmp)
        command = command_for(args, Path(item['manifest']), item['task_id'], item['offset'], 1,
                              'oomprobe', workers=1, frames=tmp_path / 'frames', out=tmp_path)
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   text=True)
        peak = 0
        deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
        while process.poll() is None:
            peak = max(peak, _descendant_rss(process.pid, _proc_table()))
            if time.monotonic() > deadline:
                process.kill()
                process.communicate()
                raise RuntimeError(f'probe timed out after {PROBE_TIMEOUT_SECONDS:.0f}s')
            time.sleep(RSS_POLL_SECONDS)
        _stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f'probe exited {process.returncode}: {stderr.strip()[-500:]}')
        return peak


def startup_worker_estimate(args, plan: list[dict]) -> int:
    """控制器启动时实测一次；没有 /proc 返回 0（worker_count 会退回 --workers）。"""
    if mem_available_bytes() is None:
        return 0
    if not plan:
        return DEFAULT_WORKER_BYTES
    started = time.time()
    try:
        measured = measure_worker_bytes(args, plan[0])
    except (OSError, RuntimeError) as error:
        log_line(args.log, f'worker probe failed ({error}); assuming {DEFAULT_WORKER_BYTES / GIB:.1f} GiB/worker')
        return DEFAULT_WORKER_BYTES
    log_line(args.log, f'worker estimate: {max(measured, 1) / GIB:.2f} GiB peak RSS (python+engine), '
                       f'{time.time() - started:.0f}s probe')
    return max(measured, 1)


def load_panel(path: Path) -> tuple[dict[str, dict], list[float], dict[str, float], dict[str, Path]]:
    """返回 (任务表, 倍率档, 后缀->倍率, 任务->它所在的清单)。

    1.0 档不是新任务：它们就是各源清单里的原任务，所以每个任务的清单要单独记，
    `search_play --manifest` 必须拿到含这个 task_id 的那一份。
    """
    document = json.loads(path.read_text())
    tasks = {task['task_id']: task for task in document['tasks']}
    manifests = {task['task_id']: path for task in document['tasks']}
    names: dict[str, float] = {}
    for task in document['tasks']:
        if '_m' in task['task_id']:
            suffix = 'm' + task['task_id'].rsplit('_m', 1)[1]
            names[suffix] = float(task['zombie_count_multiplier'])
    for path_ in source_paths({}).values():
        for task_id, task in load_tasks(path_).items():
            if task_id not in tasks:
                tasks[task_id] = task
                manifests[task_id] = path_
    values = sorted({float(task['zombie_count_multiplier']) for task in document['tasks']})
    values = [1.0] + [value for value in values if value != 1.0]
    return tasks, values, names, manifests


def parse_task_id(task_id: str, names: dict[str, float]) -> tuple[str, float]:
    """`day_l7_train_d7_m150` -> (`day_l7_train_d7`, 1.5)；原任务 id -> 1.0。"""
    for suffix, value in names.items():
        if task_id.endswith('_' + suffix):
            return task_id[: -len(suffix) - 1], value
    return task_id, 1.0


def training_base_ids() -> set[str]:
    ids = set()
    for family, _key, pattern, levels, _terrain in FAMILIES:
        if family in SUPERSEDED:
            continue
        for level in levels:
            ids.add(pattern.format(level=level))
    return ids


def cell_of(task_id: str, names: dict[str, float]) -> str:
    return parse_task_id(task_id, names)[0]


def init_from_runs(patterns: list[str], names: dict[str, float]) -> tuple[dict, dict]:
    """从已有 run 的 config.json / games.jsonl 初始化窗口胜率与已用偏移。"""
    windows: dict[str, dict[str, dict]] = {}
    offsets: dict[str, int] = {}
    base_ids = training_base_ids()
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            config_path = Path(path)
            if config_path.name != 'config.json':
                continue
            try:
                config = json.loads(config_path.read_text())
                games_path = config_path.parent / 'games.jsonl'
                if not games_path.exists():
                    continue
                rows = [json.loads(line) for line in games_path.read_text().splitlines() if line.strip()]
            except (json.JSONDecodeError, OSError):
                continue
            task_id = config.get('task_id')
            if not task_id:
                continue
            multiplier = float(config.get('zombie_count_multiplier') or 1.0)
            base = cell_of(task_id, names)
            if base not in base_ids:
                continue
            bucket = windows.setdefault(base, {}).setdefault(
                str(multiplier), {'recent': [], 'wins': 0, 'games': 0})
            outcomes = [1 if row.get('won') else 0 for row in rows]
            bucket['recent'] = (bucket['recent'] + outcomes)[-WINDOW:]
            bucket['wins'] += sum(outcomes)
            bucket['games'] += len(outcomes)
            try:
                end = int(config.get('seed_offset', 0)) + int(config.get('seeds', 0))
            except (TypeError, ValueError):
                end = 0
            offsets[task_id] = max(offsets.get(task_id, 0), end)
    return windows, offsets


def family_of(base: str) -> str | None:
    """训练 base id -> FAMILIES 里的族短名（`night_l12_train_n10p` -> `night_n10p`）。"""
    for family, _key, pattern, levels, _terrain in FAMILIES:
        if any(base == pattern.format(level=level) for level in levels):
            return family
    return None


def reconcile_cells(state: dict, tasks: dict[str, dict], names: dict[str, float],
                    windows_from_runs: dict) -> list[str]:
    """把状态里的格子对齐到当前训练集合；返回日志行，由调用方逐行写 curriculum.log。"""
    lines = []
    training = training_base_ids()
    cells = state.setdefault('cells', {})
    retired = state.setdefault('retired_cells', {})
    wanted = {}
    for task_id, task in tasks.items():
        base, multiplier = parse_task_id(task_id, names)
        if multiplier != 1.0 or base in wanted or base not in training:
            continue
        wanted[base] = task
    for base in sorted(cells):
        if base in training:
            continue
        retired[base] = cells.pop(base)
        family = family_of(base)
        reason = (f'superseded by {SUPERSEDED[family]}' if family in SUPERSEDED
                  else 'not in the training set')
        lines.append(f'cell retired: {base} ({reason})')
    for base in sorted(retired):
        if base not in training:
            continue
        cells[base] = retired.pop(base)
        lines.append(f'cell restored: {base}')
    for base, task in wanted.items():
        if base in cells:
            continue
        cells[base] = dict(family=base.rsplit('_train', 1)[0], level=int(task['level']),
                           mult_index=0, last_round=0, windows=dict(windows_from_runs.get(base, {})))
        lines.append(f'cell added: {base}')
    return lines


def make_state(tasks: dict[str, dict], names: dict[str, float], windows: dict, offsets: dict,
               panel: str) -> dict:
    state = dict(schema_version=1, panel=panel, round=0, cells={}, offsets=offsets, log=[])
    reconcile_cells(state, tasks, names, windows)
    return state


def rate(cell: dict, multiplier: float) -> float | None:
    bucket = cell['windows'].get(str(multiplier))
    if not bucket or not bucket.get('recent'):
        return None
    return sum(bucket['recent']) / len(bucket['recent'])


def adjust(cells: dict, values: list[float]) -> list[str]:
    """按窗口胜率升降档；返回变更描述（用于日志）。"""
    changes = []
    for base, cell in sorted(cells.items()):
        current = values[cell['mult_index']]
        bucket = cell['windows'].get(str(current)) or {}
        if len(bucket.get('recent') or []) < MIN_SAMPLES:
            continue
        win_rate = sum(bucket['recent']) / len(bucket['recent'])
        if win_rate > PROMOTE_ABOVE and cell['mult_index'] < len(values) - 1:
            cell['mult_index'] += 1
            changes.append(f'{base}: {current:g} -> {values[cell["mult_index"]]:g} (win {win_rate:.2f})')
        elif win_rate < DEMOTE_BELOW and cell['mult_index'] > 0:
            cell['mult_index'] -= 1
            changes.append(f'{base}: {current:g} -> {values[cell["mult_index"]]:g} (win {win_rate:.2f})')
    return changes


def allocate(cells: dict, values: list[float], budget: int, per_group: int,
             guard_ratio: float, round_index: int) -> list[tuple[str, float, int]]:
    """返回 [(base, multiplier, games)]，总局数不超过 budget。"""
    if not cells or budget < 1 or per_group < 1:
        return []
    picks: list[tuple[str, float, int]] = []
    spent = 0
    # 1) 防遗忘：1.0 档，按最久没采过的格轮转。
    guard_budget = max(per_group, int(budget * guard_ratio))
    guard_candidates = sorted(cells, key=lambda base: (cells[base]['last_round'], base))
    for base in guard_candidates:
        if spent + per_group > guard_budget or spent + per_group > budget:
            break
        picks.append((base, 1.0, per_group))
        spent += per_group
        cells[base]['last_round'] = round_index
    # 2) 余下优先给边缘格；没有边缘格就全体按“离 0.675 最近”排。
    remaining = budget - spent
    frontier = []
    others = []
    for base, cell in cells.items():
        current = values[cell['mult_index']]
        win_rate = rate(cell, current)
        if win_rate is not None and FRONTIER_LOW <= win_rate <= FRONTIER_HIGH:
            frontier.append((abs(win_rate - 0.675), base, current))
        else:
            others.append((0.0 if win_rate is None else abs(win_rate - 0.675), base, current))
    for _distance, base, multiplier in sorted(frontier) + sorted(others):
        if remaining < per_group:
            break
        if any(pick[0] == base and pick[1] == multiplier for pick in picks):
            continue
        picks.append((base, multiplier, per_group))
        remaining -= per_group
        cells[base]['last_round'] = round_index
    return picks


def task_id_for(base: str, multiplier: float, tasks: dict[str, dict], names: dict[str, float]) -> str:
    for task_id in tasks:
        candidate_base, candidate_mult = parse_task_id(task_id, names)
        if candidate_base == base and candidate_mult == multiplier:
            return task_id
    raise KeyError(f'no task for {base} at {multiplier:g}')


def label_for(round_index: int, base: str, multiplier: float) -> str:
    short = base.replace('_train', '').replace('day_l', 'd').replace('night_l', 'n').replace('pool_l', 'p')
    short = short.replace('_', '')
    return f'c{round_index}{short}m%03d' % round(multiplier * 100)


def command_for(args, manifest: Path, task_id: str, offset: int, games: int, label: str,
                workers: int | None = None, frames: Path | None = None,
                out: Path | None = None) -> list[str]:
    python = str(args.python) if args.python.exists() else sys.executable
    return [python, str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(args.rollout_policy),
            '--verify-samples', '9', '--student-candidates', '2',
            '--static-value', str(args.static_value), '--value', str(args.value),
            '--manifest', str(manifest), '--task-id', task_id,
            '--seed-offset', str(offset), '--seeds', str(games),
            '--executable', str(args.executable), '--dagger-beta', '0',
            '--workers', str(args.workers if workers is None else workers),
            '--resource-dir', str(args.resource_dir),
            '--frames', str(args.frames if frames is None else frames), '--frames-label', label,
            '--out', str((args.out if out is None else out) / f'{label}_{games}')]


def pack_labels(state: dict, extra: str = '') -> list[str]:
    """Every label this queue has written so far, plus ``extra`` (the base corpus training also uses)."""
    labels = {group['label'] for entry in state['log'] for group in entry['groups']}
    labels.update(name for name in extra.split(',') if name)
    return sorted(labels)


def pack_command(args, labels: list[str]) -> list[str]:
    """Pack the same row options the trainer resolves from this recipe."""
    recipe = load_recipe(args.recipe, 'training')
    options = pack_options(**training_row_options(recipe, labels))
    python = str(args.python) if args.python.exists() else sys.executable
    return [python, str(ROOT / 'scripts/pack_training_rows.py'),
            '--data', str(args.frames), '--labels', ','.join(sorted(labels)),
            *pack_flags(options), '--prune']


def write_pack(args, state: dict) -> None:
    """Pack every archive this queue has written into one sequential file, at the end of a round.

    The trainer never reads per-game archives (task33): this is the collector's half of that
    contract. Packing is incremental in content, not in cost: each round rewrites the union of the
    labels so far and prunes the packs it supersedes. A failed pack is fatal -- it is raised after
    the round is saved, so the collected games are never lost.
    """
    recipe = load_recipe(args.recipe, 'training')
    labels = pack_labels(state, ','.join(recipe['labels']))
    if not labels:
        return
    started = time.time()
    result = subprocess.run(pack_command(args, labels), cwd=ROOT, capture_output=True, text=True)
    summary = ' | '.join(result.stdout.strip().splitlines())
    log_line(args.log, f'pack {len(labels)} labels {time.time() - started:.0f}s rc={result.returncode} {summary}')
    if result.returncode != 0:
        log_line(args.log, f'pack failed: {result.stderr.strip()[-2000:]}')
        raise SystemExit(f'pack_training_rows exited {result.returncode}')


def log_line(path: Path, message: str) -> None:
    stamp = datetime.now().isoformat(timespec='seconds')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(f'{stamp} {message}\n')
    print(f'{stamp} {message}', flush=True)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state['updated'] = datetime.now().isoformat(timespec='seconds')
    path.write_text(json.dumps(state, indent=1) + '\n')


def plan_round(state: dict, tasks: dict[str, dict], names: dict[str, float], values: list[float],
               manifests: dict[str, Path], budget: int, per_group: int,
               guard_ratio: float) -> tuple[list[dict], list[str]]:
    changes = adjust(state['cells'], values)
    picks = allocate(state['cells'], values, budget, per_group, guard_ratio, state['round'] + 1)
    plan = []
    for base, multiplier, games in picks:
        task_id = task_id_for(base, multiplier, tasks, names)
        task = tasks[task_id]
        offset = state['offsets'].get(task_id, 0)
        if offset + games > len(task['seeds']):
            raise SystemExit(f'{task_id}: seeds exhausted (offset {offset} + {games} > {len(task["seeds"])})')
        plan.append(dict(base=base, multiplier=multiplier, task_id=task_id, offset=offset,
                         games=games, manifest=str(manifests[task_id]),
                         label=label_for(state['round'] + 1, base, multiplier)))
    return plan, changes


def run_round(args, state: dict, plan: list[dict], changes: list[str], per_worker: int = 0) -> bool:
    """跑一轮；返回 False 表示看到停止标志。"""
    round_index = state['round'] + 1
    for item in plan:
        if STOP_FLAG.exists():
            log_line(args.log, f'round {round_index}: stop flag seen before {item["label"]}; stopping at group boundary')
            return False
        state['offsets'][item['task_id']] = item['offset'] + item['games']
        save_state(args.state, state)          # 先占偏移，崩溃也不会重采
        available = mem_available_bytes()
        workers = worker_count(available, per_worker, args.workers)
        per_worker_shown = f'{per_worker / GIB:.2f} GiB' if per_worker > 0 else 'unmeasured'
        log_line(args.log, f'round {round_index} {item["label"]}: workers={workers}/{args.workers} '
                           f'(MemAvailable {"n/a" if available is None else f"{available / GIB:.1f} GiB"}, '
                           f'per-worker {per_worker_shown})')
        command = command_for(args, Path(item['manifest']), item['task_id'], item['offset'],
                              item['games'], item['label'], workers=workers)
        started = time.time()
        result = subprocess.run(command, cwd=ROOT)
        seconds = time.time() - started
        folder = args.out / f'{item["label"]}_{item["games"]}'
        wins = games = 0
        rows = []
        if (folder / 'games.jsonl').exists():
            rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines() if line.strip()]
            games = len(rows)
            wins = sum(1 for row in rows if row.get('won'))
        metrics = getattr(args, 'metrics', None)
        if metrics is not None:
            args.metrics_games += games
            args.metrics_decisions += sum(row.get('decisions', 0) for row in rows)
            metrics.update(games=args.metrics_games, decisions=args.metrics_decisions, round=round_index)
        bucket = state['cells'][item['base']]['windows'].setdefault(
            str(item['multiplier']), {'recent': [], 'wins': 0, 'games': 0})
        bucket['recent'] = (bucket['recent'] + [1 if row.get('won') else 0
                                                for row in rows])[-WINDOW:]
        bucket['wins'] += wins
        bucket['games'] += games
        bucket['last_round'] = round_index
        log_line(args.log, f'round {round_index} {item["label"]} {item["task_id"]} mult={item["multiplier"]:g} '
                           f'offset={item["offset"]} games={games} wins={wins} '
                           f'rate={(wins / games if games else 0):.2f} rc={result.returncode} {seconds:.0f}s')
        if result.returncode != 0:
            raise SystemExit(f'{item["label"]}: search_play exited {result.returncode}')
    state['round'] = round_index
    state['log'].append(dict(round=round_index,
                             groups=[{k: item[k] for k in ('base', 'multiplier', 'task_id', 'offset', 'games',
                                                            'manifest', 'label')}
                                     for item in plan],
                             changes=changes))
    save_state(args.state, state)
    log_line(args.log, f'round {round_index} done: {len(plan)} groups, '
                       f'{sum(item["games"] for item in plan)} games, changes={len(changes)}'
                       + ((' [' + '; '.join(changes) + ']') if changes else ''))
    write_pack(args, state)          # after the state is saved: a failed pack must not lose the round
    return True


def print_plan(state: dict, plan: list[dict], changes: list[str]) -> None:
    print(f"{'cell':<34}{'mult':>6}{'win%':>7}{'n':>6}{'games':>7}  label")
    for item in plan:
        cell = state['cells'][item['base']]
        win_rate = rate(cell, item['multiplier'])
        shown = f'{win_rate * 100:.1f}' if win_rate is not None else '-'
        games = len(cell['windows'].get(str(item['multiplier']), {}).get('recent') or [])
        print(f'{item["base"]:<34}{item["multiplier"]:>6g}{shown:>7}{games:>6}{item["games"]:>7}  {item["label"]}')
    print(f'total groups {len(plan)}, games {sum(item["games"] for item in plan)}')
    if changes:
        print('multiplier changes:')
        for change in changes:
            print('  ' + change)


def main() -> None:
    # task34: die after the workers and training, but before the WSL access path.
    set_oom_score_adj(OOM_SCORE_CONTROLLER)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/stage4/multiplier_panel.json')
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--out', type=Path, default=MAIN / 'artifacts/hard')
    parser.add_argument('--frames', type=Path, default=DEFAULT_FRAMES)
    parser.add_argument('--rollout-policy', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--static-value', type=Path, default=DEFAULT_STATIC_VALUE)
    parser.add_argument('--value', type=Path, default=DEFAULT_VALUE)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument('--workers', type=int, default=16,
                        help='上限；每组实际进程数 = (MemAvailable − 6 GiB) / 实测单 worker RSS，至少 1')
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON,
                        help='跑 search_play 的解释器（默认 ~/.venvs/ml/bin/python3）')
    parser.add_argument('--round-games', type=int, default=240)
    parser.add_argument('--games-per-group', type=int, default=48)
    parser.add_argument('--guard-ratio', type=float, default=GUARD_RATIO)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--init-glob', action='append', default=None,
                        help='用于初始化胜率/已用偏移的 config.json glob（可重复）')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--recipe', type=Path, default=ROOT / 'recipes/dw4i.json',
                        help='training recipe for the collection student; supplies every pack row option')
    args = parser.parse_args()
    try:
        load_recipe(args.recipe, 'training')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.workers < 1:
        parser.error('--workers must be positive')
    for name in ('manifest', 'state', 'log', 'out', 'frames', 'rollout_policy', 'static_value',
                 'value', 'executable', 'resource_dir'):
        value = getattr(args, name)
        if not value.is_absolute():
            parser.error(f'all paths must be absolute: {name}')
    tasks, values, names, manifests = load_panel(args.manifest)
    patterns = args.init_glob or [str(MAIN / 'artifacts/hard/*/config.json'),
                                  str(MAIN / 'artifacts/task28/*/config.json'),
                                  str(MAIN / '.claude/worktrees/*/artifacts/**/config.json')]
    if args.state.exists():
        state = json.loads(args.state.read_text())
        # 已有状态按当前训练集合增删格子；n10q 没有旧局，除非显式给 --init-glob 否则从空窗口起步。
        windows = init_from_runs(args.init_glob, names)[0] if args.init_glob else {}
        for line in reconcile_cells(state, tasks, names, windows):
            log_line(args.log, line)
    else:
        windows, offsets = init_from_runs(patterns, names)
        state = make_state(tasks, names, windows, offsets, str(args.manifest))
    if args.dry_run:
        plan, changes = plan_round(state, tasks, names, values, manifests, args.round_games,
                                   args.games_per_group, args.guard_ratio)
        print_plan(state, plan, changes)
        import shlex
        for item in plan:
            print('collect: ' + shlex.join(command_for(args, Path(item['manifest']), item['task_id'],
                  item['offset'], item['games'], item['label'])))
        # The pack command this controller would run at the end of the round: reviewable before
        # a round is spent, and the only place the recipe's row options become a real command.
        labels = pack_labels(state, ','.join(load_recipe(args.recipe, 'training')['labels']))
        if labels:
            print('pack: ' + ' '.join(pack_command(args, labels)))
        return
    args.metrics = Metrics("curriculum_queue").start()
    args.metrics.update(games=0, decisions=0, round=state["round"])
    args.metrics_games = args.metrics_decisions = 0
    per_worker = -1          # -1 = 还没测；0 = 平台无 /proc，按 --workers 上限跑
    for _ in range(args.rounds):
        plan, changes = plan_round(state, tasks, names, values, manifests, args.round_games,
                                   args.games_per_group, args.guard_ratio)
        print_plan(state, plan, changes)
        if per_worker < 0:
            if STOP_FLAG.exists():
                log_line(args.log, f'round {state["round"] + 1}: stop flag seen before the worker probe; stopping')
                break
            per_worker = startup_worker_estimate(args, plan)
        if not run_round(args, state, plan, changes, per_worker):
            break

    args.metrics.close()


if __name__ == '__main__':
    main()
