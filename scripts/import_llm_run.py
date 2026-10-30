#!/usr/bin/env python3
"""把 GPT Pro / llmplay 交回的一整包存档一键入库（task89）。

    python3 scripts/import_llm_run.py <out.zip 或解压目录> --dest artifacts/llm_demo/raw

流程：安全解压（拒绝绝对路径与 ..）→ 找出全部 episode_archive_v1 存档 →
逐份用本机引擎 human_demo.verify 校验（操作数、每步 tick/ok、终局一致；不要求
逐位 state_hash，理由见 task82）→ 读 meta 分档（full / half / loss）→ 种子不属于
原 demo 族时按 task87 的命名规则登记 <base>_s<seed> 新族 → 为 full/half 的局
直接更新既有导入索引，供生产课程下轮读取；不再生成接手扫描作业。

输入只读：存档先复制进 --dest，校验写的是副本的 meta。

规则（AGENTS.md）：存档本身不进 git；不写死密钥；不手写打法规则。
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

ARCHIVE_SCHEMA = 'episode_archive_v1'
DEFAULT_DEST = ROOT / 'artifacts/llm_demo/raw'
DEFAULT_MANIFEST = ROOT / 'configs/human_demo_tasks.json'
DEFAULT_LAYERS = ROOT / 'configs/llmplay_seeds.json'
DEFAULT_RESOURCES = Path.home() / '.cache/pvz-research-resources'
DEFAULT_SEED_LEDGER = ROOT / 'artifacts/llm_demo/scan_seed_ledger.json'

SCAN_POINTS = 12
SCAN_FUTURES = 8
FUTURE_BLOCK = 96          # 每份存档预留的种子块；实际只用前 8 个
FUTURE_STRIDE = 1000
DEFAULT_FUTURE_BASE = 84_000_000
SCAN_TAIL = 0.94           # 12 个点覆盖决策序列的前 94%，避开终局读档点
ROBUST_WAVE = '10'
ROBUST_MIN = 5

LEVEL_IN_ID = re.compile(r'_(?:l|roof|fog|pool|conv|night|day)_(\d{1,2})(?:_|$)')


# ---------------------------------------------------------------- 输入与解压

def _zip_target(root: Path, name: str) -> Path:
    """zip 成员 -> 落盘路径；绝对路径、盘符、.. 一律拒绝。"""
    if not name or name.startswith(('/', '\\')) or re.match(r'^[A-Za-z]:', name):
        raise ValueError(f'zip 里有绝对路径：{name!r}')
    name = name.replace('\\', '/')
    if '..' in Path(name).parts:
        raise ValueError(f'zip 里有路径穿越：{name!r}')
    target = (root / name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f'zip 成员逃出解压目录：{name!r}') from error
    return target


def safe_extract(archive: Path, destination: Path) -> None:
    """只解普通文件；符号链接、绝对路径、.. 都拒绝。"""
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                _zip_target(destination, info.filename)
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'zip 里有符号链接：{info.filename!r}')
            target = _zip_target(destination, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open('wb') as sink:
                shutil.copyfileobj(source, sink)


def discover(root: Path) -> list[Path]:
    """按 llmplay 的导出格式找存档：meta.json + ticks.jsonl + episode_archive_v1。"""
    found = []
    for meta_path in sorted(root.rglob('meta.json')):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if meta.get('schema') != ARCHIVE_SCHEMA:
            continue
        if not (meta_path.parent / 'ticks.jsonl').is_file():
            continue
        found.append(meta_path.parent)
    return found


# ---------------------------------------------------------------- meta 检查

def _tick_lines(archive: Path) -> int:
    with (archive / 'ticks.jsonl').open(encoding='utf-8') as stream:
        return sum(1 for line in stream if line.strip())


def archive_problems(meta: dict, archive: Path) -> list[str]:
    """meta 与存档本身是否自洽。文件名不参与判断（任务书：不信任文件名）。"""
    problems = []
    task = meta.get('task') or {}
    env = meta.get('environment_task') or {}
    task_id = str(task.get('id') or '')
    if not task_id:
        problems.append('meta 缺 task.id')
    if task_id and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', task_id):
        problems.append('meta.task.id 不是安全的任务名')
    for key in ('level', 'seed', 'zombie_count_multiplier'):
        if task.get(key) is not None and env.get(key) is not None:
            if task[key] != env[key]:
                problems.append(f'meta 自相矛盾：task.{key}={task[key]} vs environment_task.{key}={env[key]}')
    if task_id and task.get('level') is not None:
        match = LEVEL_IN_ID.search(task_id)
        if match and int(match.group(1)) != int(task['level']):
            problems.append(f'task.id={task_id} 编码的是 L{match.group(1)}，meta 写 L{task["level"]}')
    deck = task.get('deck')
    conveyor = task.get('conveyor_start') or env.get('conveyor_start')
    if not isinstance(deck, list) or (not deck and conveyor != 'original'):
        problems.append('meta 的 task.deck 为空或不是列表')
    if task.get('seed') is None:
        problems.append('meta 缺 task.seed')
    if not isinstance(meta.get('outcome'), dict):
        problems.append('meta 缺 outcome')
    else:
        if 'won' not in meta['outcome'] or 'final_wave' not in meta['outcome']:
            problems.append('meta.outcome 缺 won / final_wave')
        if 'result' in meta['outcome'] and bool(meta['outcome'].get('won')) != (meta['outcome']['result'] == 1):
            problems.append('meta.outcome.won 与 result 不一致')
    try:
        lines = _tick_lines(archive)
    except OSError as error:
        problems.append(f'ticks.jsonl 读不了：{error}')
        lines = None
    if lines is not None and meta.get('operations') is not None and int(meta['operations']) != lines:
        problems.append(f'meta.operations={meta["operations"]} 与 ticks.jsonl 的 {lines} 行不一致')
    frames = archive / 'frames.jsonl'
    if frames.is_file() and meta.get('frame_count') is not None:
        with frames.open(encoding='utf-8') as stream:
            count = sum(1 for line in stream if line.strip())
        if int(meta['frame_count']) != count:
            problems.append(f'meta.frame_count={meta["frame_count"]} 与 frames.jsonl 的 {count} 行不一致')
    return problems


def grade(meta: dict) -> str:
    """task81d 的分档：full / half / loss。"""
    outcome = meta.get('outcome') or {}
    won = bool(outcome.get('won')) or int(outcome.get('result') or 0) == 1
    if not won:
        return 'loss'
    checkpoint = ((meta.get('robustness') or {}).get('checkpoints') or {}).get(ROBUST_WAVE) or {}
    k = int(checkpoint.get('k') or (meta.get('robustness') or {}).get('k') or 0)
    if checkpoint.get('reached') and k == SCAN_FUTURES and ROBUST_MIN <= int(checkpoint.get('wins') or 0) <= k:
        return 'full'
    return 'half'


def robustness_text(meta: dict) -> str:
    report = meta.get('robustness') or {}
    checkpoint = (report.get('checkpoints') or {}).get(ROBUST_WAVE) or {}
    if not report:
        return '无记录'
    if not checkpoint.get('reached'):
        return '未到第10波'
    return f'{int(checkpoint.get("wins") or 0)}/{int(checkpoint.get("k") or SCAN_FUTURES)}'


# ---------------------------------------------------------------- 任务族

def load_manifest(path: Path) -> dict:
    document = json.loads(path.read_text(encoding='utf-8'))
    return {task['task_id']: task for task in document.get('tasks') or []}


def resolve_family(meta: dict, tasks: dict) -> tuple[str, bool, list[str]]:
    """返回 (最终 task_id, 是否新族, 问题列表)。

    - 种子在原族里 -> 沿用原 task_id；
    - 种子不在原族里 -> <base>_s<seed>（task87 的命名规则）；
    - 原族不存在 -> 直接登记 meta 里的 task.id（卡组照 meta）。
    """
    task = meta['task']
    base = str(task['id'])
    seed = int(task['seed'])
    deck = [int(value) for value in task['deck']]
    entry = tasks.get(base)
    if entry is None:
        return base, True, []
    family_deck = [int(value) for value in entry.get('deck') or []]
    if deck != family_deck:
        return base, False, [f'卡组与已登记的 {base} 不同：meta {deck} vs 清单 {family_deck}']
    for key, default in (('level', None), ('zombie_count_multiplier', 1.0)):
        if task.get(key, default) != entry.get(key, default):
            return base, False, [f'{base} 的 {key} 与已登记任务不同']
    if seed in {int(value) for value in entry.get('seeds') or []}:
        return base, False, []
    final = f'{base}_s{seed}'
    existing = tasks.get(final)
    if existing is not None:
        if [int(value) for value in existing.get('deck') or []] != deck:
            return final, False, [f'{final} 已登记但卡组不同']
        if seed not in {int(value) for value in existing.get('seeds') or []}:
            return final, False, [f'{final} 已登记但没有种子 {seed}']
        return final, False, []   # 族和种子都已登记过，不算新族
    return final, True, []


def family_entry(final: str, meta: dict, base_entry: dict | None) -> dict:
    task = meta['task']
    env = meta.get('environment_task') or {}
    entry = {'task_id': final, 'level': int(task['level']), 'seeds': [int(task['seed'])],
             'zombie_count_multiplier': float(task.get('zombie_count_multiplier') or 1.0),
             'deck': [int(value) for value in task['deck']]}
    for key in ('wave_cap', 'preplanted', 'conveyor_start', 'fog_observation'):
        value = (base_entry or {}).get(key)
        if value is None:
            value = env.get(key)
        if value is None:
            value = task.get(key)
        if value not in (None, [], ''):
            entry[key] = value
    return entry


def register_tasks(path: Path, entries: list[dict]) -> None:
    """把新族按现有排版（一行一个任务）插到清单末尾，其余字节不动。"""
    if not entries:
        return
    lines = path.read_text(encoding='utf-8').splitlines()
    closing = [index for index, line in enumerate(lines) if line.strip() == ']']
    if not closing:
        # 单行 JSON 也接受：整份重排成仓库惯用的一行一任务。
        document = json.loads(path.read_text(encoding='utf-8'))
        document.setdefault('tasks', []).extend(entries)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        load_manifest(path)
        return
    index = closing[-1]
    if index == 0 or not lines[index - 1].strip():
        raise ValueError(f'{path} 的 tasks 数组是空的，拒绝自动登记')
    if lines[index - 1].strip() != '"tasks": [' and not lines[index - 1].rstrip().endswith(','):
        lines[index - 1] = lines[index - 1].rstrip() + ','
    new_lines = ['    ' + json.dumps(entry, ensure_ascii=False, separators=(',', ':'))
                 + (',' if i < len(entries) - 1 else '') for i, entry in enumerate(entries)]
    lines[index:index] = new_lines
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    load_manifest(path)  # 写坏了立刻报


def register_design(path: Path, entries: list[dict]) -> None:
    """与清单同步登记新族；重复调用不追加已有 task_id。"""
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    missing = [entry for entry in entries if f'`{entry["task_id"]}`' not in text]
    if not missing:
        return
    lines = ['\n### task89：LLM 存档自动登记的新族\n',
             '| task_id | 关卡 | 种子 | 倍率 | 卡组（存档 meta） |',
             '|---|---|---|---|---|']
    for entry in missing:
        lines.append(f'| `{entry["task_id"]}` | {entry["level"]} | {entry["seeds"]} | '
                     f'{entry["zombie_count_multiplier"]} | {entry["deck"]} |')
    path.write_text(text.rstrip() + '\n' + '\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- 扫描作业

def scan_decisions(operations: int, points: int = SCAN_POINTS) -> list[int]:
    """12 个等距决策号，覆盖前 94% 的决策（终局附近的读档点没有教学意义）。"""
    last = max(1, int(operations * SCAN_TAIL))
    if points <= 1:
        return [1]
    values = {max(1, round(1 + (last - 1) * index / (points - 1))) for index in range(points)}
    return sorted(values)


def _config_seed_points(document, points: set) -> None:
    if isinstance(document, dict):
        start, count = document.get('future_seed_start'), document.get('future_seed_count')
        if type(start) is int and type(count) is int:
            points.update(range(start, start + count))
        for key, value in document.items():
            if key in ('seeds', 'future_seeds') and isinstance(value, list):
                points.update(int(item) for item in value if isinstance(item, int))
                _config_seed_points(value, points)
            elif key == 'tasks' and isinstance(value, dict):
                for task in value.values():
                    if isinstance(task, list):
                        points.update(item for item in task if type(item) is int)
                    else:
                        _config_seed_points(task, points)
            elif key in ('seed_block', 'seed_window') and isinstance(value, list) and len(value) == 2:
                points.update(range(*value))
            elif key in ('future_seed_start', 'seed') and isinstance(value, int):
                points.add(value)
            elif key == 'block' and isinstance(value, dict):
                base = value.get('base')
                stride = int(value.get('stride') or 1)
                count = int(value.get('per_task') or 0)
                tasks = value.get('tasks') or document.get('tasks') or 1
                tasks = len(tasks) if isinstance(tasks, (dict, list)) else int(tasks)
                if isinstance(base, int) and count:
                    for index in range(max(tasks, 1)):
                        points.update(range(base + index * stride, base + index * stride + count))
            else:
                _config_seed_points(value, points)
    elif isinstance(document, list):
        for item in document:
            _config_seed_points(item, points)


def occupied_seed_ranges(root: Path, gap: int = 100_000) -> list:
    """仓库里所有登记过的种子段（configs/*.json + DESIGN.md 的大数）。机械求，不手抄。"""
    points = set()
    for path in sorted((root / 'configs').glob('*.json')):
        try:
            _config_seed_points(json.loads(path.read_text(encoding='utf-8')), points)
        except (OSError, ValueError):
            continue
    design = root / 'DESIGN.md'
    if design.is_file():
        for match in re.finditer(r'\b\d{1,3}(?:,\d{3}){2,}\b', design.read_text(encoding='utf-8')):
            value = int(match.group(0).replace(',', ''))
            if value >= 1_000_000:
                points.add(value)
    ranges = []
    for value in sorted(points):
        if ranges and value - ranges[-1][1] <= gap:
            ranges[-1] = (ranges[-1][0], value)
        else:
            ranges.append((value, value))
    snapshot = root / 'MEMORY/docs/llm-import-seed-inventory.json'
    if snapshot.is_file():
        document = json.loads(snapshot.read_text(encoding='utf-8'))
        ranges += [(low, high - 1) for low, high in document['occupied'] + document['reserved_pools']]
    return ranges


def allocate_future_block(ranges: list, base: int, count: int) -> int:
    while True:
        if base < 0 or count < 1 or base + count > 2**32:
            raise ValueError('未来种子块超出 uint32 范围')
        end = base + count - 1
        if not any(start <= end and base <= stop for start, stop in ranges):
            return base
        base += 1_000_000


def scan_plan(rows: list, layers: dict, base: int, root: Path = ROOT,
              ledger_path: Path | None = None) -> dict:
    """锁住本目标目录的分配账本，跨导入不复用新未来，同档重跑保持分配。"""
    if ledger_path is None:
        return _scan_plan(rows, layers, base, root, {})
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = (json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists()
                  else dict(schema='llm_scan_seed_ledger_v1', reservations={}))
        if ledger.get('schema') != 'llm_scan_seed_ledger_v1':
            raise ValueError('未知扫描种子账本格式')
        result = _scan_plan(rows, layers, base, root, ledger['reservations'])
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=ledger_path.parent,
                                         delete=False) as temporary:
            temporary.write(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n')
        Path(temporary.name).replace(ledger_path)
        return result


def _scan_plan(rows: list, layers: dict, base: int, root: Path, reservations: dict) -> dict:
    """为 full / half 的局生成 task71b 作业清单；只生成，不运行。"""
    registered_ranges = occupied_seed_ranges(root)
    ranges = list(registered_ranges)
    usable = [row for row in rows if row['grade'] != 'loss' and row.get('archive')]
    claimed = []
    for entry in reservations.values():
        start, count = entry['start'], entry['count']
        if not (isinstance(start, int) and 0 <= start < 2**32 and count == FUTURE_BLOCK
                and start + count <= 2**32):
            raise ValueError('扫描种子账本范围无效')
        interval = (start, start + count - 1)
        if any(a <= interval[1] and interval[0] <= b for a, b in claimed):
            raise ValueError('扫描种子账本区间重叠')
        claimed.append(interval)
    ranges += claimed
    cursor = base
    archives = []
    for row in usable:
        key = json.dumps([row['task_id'], row['seed'], row['archive']], ensure_ascii=False)
        existing = reservations.get(key)
        start = existing['start'] if existing else allocate_future_block(ranges, cursor, FUTURE_BLOCK)
        if not existing:
            reservations[key] = dict(start=start, count=FUTURE_BLOCK)
            ranges.append((start, start + FUTURE_BLOCK - 1))
        cursor = start + FUTURE_STRIDE
        futures = list(range(start, start + SCAN_FUTURES))
        decisions = scan_decisions(row['frames'])
        jobs = []
        for decision in decisions:
            for arm, beta in (('teacher', 1.0), ('student', 0.0)):
                jobs.append(dict(task_id=row['task_id'], archive=row['archive'], level=row['level'],
                                 seed=row['seed'], decision=decision, arm=arm,
                                 dagger_beta=beta, future_seeds=futures))
        archives.append(dict(task_id=row['task_id'], archive=row['archive'], level=row['level'],
                             seed=row['seed'], grade=row['grade'], decisions=decisions,
                             reused_reservation=existing is not None,
                             future_seed_start=start, future_seed_count=FUTURE_BLOCK,
                             future_seeds=futures, layer=layers.get(row['base_task_id'], '其他'),
                             jobs=jobs))
    return dict(schema='llm_scan_jobs_v1',
                generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                future_block=dict(base=archives[0]['future_seed_start'] if archives else base,
                                  requested_base=base, stride=FUTURE_STRIDE, count=FUTURE_BLOCK),
                seed_audit=dict(existing_ranges=[[a, b] for a, b in registered_ranges],
                                previous_allocations=[[a, b] for a, b in claimed],
                                allocated=[[entry['future_seed_start'],
                                            entry['future_seed_start'] + FUTURE_BLOCK - 1]
                                           for entry in archives]),
                archives=archives)


# ---------------------------------------------------------------- 入库

def _same_archive(left: Path, right: Path) -> bool:
    try:
        for filename in ('ticks.jsonl', 'frames.jsonl', 'frames.npz'):
            a, b = left / filename, right / filename
            if a.exists() != b.exists() or (a.exists() and a.read_bytes() != b.read_bytes()):
                return False
        def metadata(path):
            return {k: v for k, v in json.loads((path / 'meta.json').read_text()).items()
                    if k not in ('verified', 'hash_verification')}
        return metadata(left) == metadata(right)
    except (OSError, ValueError):
        return False


def import_archive(archive: Path, dest: Path, task_id: str, seed: int) -> tuple:
    target = dest / f'seed_{seed}' / task_id
    if target.exists():
        return target, 'duplicate' if _same_archive(archive, target) else 'conflict'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(archive, target)
    return target, 'copied'


def reject_archive(target: Path, dest: Path) -> Path:
    rejected = dest / '_rejected' / target.parent.name / target.name
    rejected.parent.mkdir(parents=True, exist_ok=True)
    if rejected.exists():
        shutil.rmtree(rejected)
    shutil.move(str(target), str(rejected))
    return rejected


def verify_copy(archive: Path, resource_dir: Path, executable: Path | None) -> dict:
    from human_demo import verify
    return verify(archive, resource_dir, executable)


# ---------------------------------------------------------------- 报告

def render_report(rows: list, layers: dict, plan: dict) -> str:
    lines = ['| 任务 | 种子 | 倍率 | 结果 | 波 | 操作 | verify | 分档 | 稳健性（第10波） | 补救 | 新族 | 备注 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        patch = row.get('manual_patch')
        lines.append('| {task_id} | {seed} | {multiplier} | {outcome} | {wave} | {operations} | {verify} | {grade} | '
                     '{robust} | {patch} | {new} | {note} |'.format(
                         task_id=row['task_id'], seed=row['seed'], wave=row['wave'],
                         multiplier=row.get('multiplier', ''), outcome=row.get('result', ''),
                         operations=row['operations'], verify=row['verify'],
                         grade=row['grade'], robust=row['robustness'],
                         patch='未知' if patch is None else ('是' if patch else '否'),
                         new='是' if row['new_family'] else '否', note=row.get('note', '').replace('|', '\\|').replace('\n', ' ')))
    summary = {layer: dict(total=0, won=0, full=0, half=0, loss=0) for layer in 'ABCDE'}
    for row in rows:
        layer = layers.get(row['base_task_id'], '其他')
        cell = summary.setdefault(layer, dict(total=0, won=0, full=0, half=0, loss=0))
        cell['total'] += 1
        cell['won'] += row['grade'] != 'loss'
        cell[row['grade']] += 1
    lines += ['', '| 层 | 份数 | 通关 | full | half | loss |', '|---|---|---|---|---|---|']
    for layer in sorted(summary):
        cell = summary[layer]
        lines.append('| {layer} | {total} | {won} | {full} | {half} | {loss} |'.format(
            layer=layer, **cell))
    jobs = sum(len(entry['jobs']) for entry in plan['archives'])
    trajectories = sum(len(job['future_seeds']) for entry in plan['archives'] for job in entry['jobs'])
    lines += ['', '扫描作业：{count} 组（每组 8 个未来），合计 {trajectories} 条续走；{archives} 份存档。'.format(
                  count=jobs, trajectories=trajectories,
                  archives=len(plan['archives'])),
              '未来种子块：{base} 起，每份存档 {stride} 步长、{count} 个。'.format(
                  **plan['future_block'])]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- 主流程

def run(input_path: Path, dest: Path, manifest: Path, layers_path: Path,
        resource_dir: Path, executable: Path | None, do_verify: bool,
        future_base: int, design_path: Path | None = None,
        seed_ledger: Path | None = None) -> dict:
    if input_path.is_dir():
        source_path, output_path = input_path.resolve(), dest.resolve()
        if source_path == output_path or source_path in output_path.parents or output_path in source_path.parents:
            raise ValueError('输入目录与 dest 不得互相包含（输入只读）')
    if design_path is None:
        design_path = (manifest.parent.parent if manifest.parent.name == 'configs' else manifest.parent) / 'DESIGN.md'
    if input_path.is_dir() and any(input_path.resolve() in path.resolve().parents
                                   for path in (manifest, design_path)):
        raise ValueError('任务清单与 DESIGN 不得写进只读输入目录')
    layers = (json.loads(layers_path.read_text(encoding='utf-8')).get('layers') or {})
    layer_of = {task_id: name for name, ids in layers.items() for task_id in ids}
    tasks = load_manifest(manifest)

    workdir = None
    try:
        if input_path.is_dir():
            source = input_path
        elif input_path.is_file() and zipfile.is_zipfile(input_path):
            workdir = Path(tempfile.mkdtemp(prefix='llm_import_'))
            safe_extract(input_path, workdir)
            source = workdir
        else:
            raise ValueError(f'输入既不是目录也不是 zip：{input_path}')

        archives = discover(source)
        if not archives:
            raise ValueError(f'{input_path} 里没有找到 episode_archive_v1 存档')

        rows, pending = [], []
        seen = {}
        for archive in archives:
            meta = json.loads((archive / 'meta.json').read_text(encoding='utf-8'))
            problems = archive_problems(meta, archive)
            task = meta.get('task') or {}
            seed = int(task.get('seed') or 0)
            base_id = str(task.get('id') or archive.name)
            final_id, new_family, family_problems = resolve_family(meta, tasks)
            problems += family_problems
            key = (base_id, seed, int(meta.get('operations') or 0))
            if key in seen:
                problems.append(f'包内重复存档（与 {seen[key]} 相同）')
            else:
                seen[key] = str(archive)
            row = dict(source=str(archive), task_id=final_id, base_task_id=base_id, seed=seed,
                       level=int(task.get('level') or 0),
                       multiplier=float(task.get('zombie_count_multiplier') or 1.0),
                       result=(meta.get('outcome') or {}).get('result'),
                       wave=(meta.get('outcome') or {}).get('final_wave'),
                       operations=int(meta.get('operations') or 0),
                       frames=int(meta.get('frame_count') or meta.get('operations') or 0),
                       verify='未做', grade='loss', robustness=robustness_text(meta),
                       new_family=new_family, archive='', note='',
                       manual_patch=meta.get('manual_patch'))
            if problems:
                row['verify'] = '拒绝'
                row['note'] = '；'.join(problems)
                rows.append(row)
                continue
            pending.append((row, meta, archive))

        # 每份先校验；只有通关的局进 dest（任务书：loss 只入报告）。
        # 校验在 staging 副本上做，输入目录始终只读。
        entries = []
        staging = dest / '_staging'
        for row, meta, archive in pending:
            rows.append(row)
            copy = staging / archive.name
            if copy.exists():
                shutil.rmtree(copy)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(archive, copy)
            won = grade(meta) != 'loss'
            if do_verify:
                try:
                    verify_copy(copy, resource_dir, executable)
                    row['verify'] = '通过'
                except Exception as error:  # noqa: BLE001 - 引擎的报错要原样进报告
                    row['verify'] = '失败'
                    row['note'] = str(error)
            else:
                row['verify'] = '跳过'
            row['grade'] = grade(meta) if row['verify'] in ('通过', '跳过') else 'loss'
            if not won or row['verify'] == '失败':
                if won:
                    reject_archive(copy, dest)
                    row['archive'] = ''
                else:
                    shutil.rmtree(copy, ignore_errors=True)
                continue
            target, state = import_archive(copy, dest, row['task_id'], row['seed'])
            shutil.rmtree(copy, ignore_errors=True)
            try:
                row['archive'] = str(target.relative_to(ROOT))
            except ValueError:
                row['archive'] = str(target)
            if state == 'conflict':
                row['verify'] = '拒绝'
                row['note'] = f'目标已存在且内容不同：{target}'
                row['grade'] = 'loss'
                continue
            if state == 'duplicate':
                row['note'] = '同一份存档此前已入库'
            if row['new_family'] and row['task_id'] not in tasks:
                entry = family_entry(row['task_id'], meta, tasks.get(row['base_task_id']))
                tasks[row['task_id']] = entry
                entries.append(entry)
        shutil.rmtree(staging, ignore_errors=True)

        register_tasks(manifest, entries)
        register_design(design_path, entries)
        rows.sort(key=lambda item: (item['base_task_id'], item['seed']))
        plan = scan_plan(rows, layer_of, future_base,
                         ledger_path=seed_ledger or dest / 'scan_seed_ledger.json')
        return dict(rows=rows, layers=layer_of, plan=plan, registered=entries)
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


def main_with(args) -> int:
    from pvz_live_import import main_with as live_import
    return live_import(args, sys.modules[__name__])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='交回的 out.zip 或解压目录（只读）')
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--layers', type=Path, default=DEFAULT_LAYERS)
    parser.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--no-verify', action='store_true', help='只入库不校验（测试用）')
    parser.add_argument('--report', type=Path, help='把 markdown 报告写到这里')
    parser.add_argument('--scan-out', type=Path, help='扫描作业清单 json')
    parser.add_argument('--future-base', type=int, default=DEFAULT_FUTURE_BASE)
    parser.add_argument('--seed-ledger', type=Path, default=DEFAULT_SEED_LEDGER,
                        help='扫描种子分配账本；默认各次导入共用，不随 dest 改变')
    args = parser.parse_args()
    return main_with(args)


if __name__ == '__main__':
    raise SystemExit(main())
