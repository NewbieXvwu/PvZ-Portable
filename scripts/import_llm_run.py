#!/usr/bin/env python3
"""Import GPT/human archives once by SHA, convert to NPZ, and publish one live index."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_episode_archive import FILES, READ_SCHEMAS, copy_archive, convert_archive, write_meta
from pvz_frame_store import frame_path, frame_bytes
DEFAULT_DEST = Path(__file__).absolute().parents[1] / 'artifacts/task105/import'
DEFAULT_MANIFEST = ROOT / 'configs/human_demo_tasks.json'
DEFAULT_RESOURCES = Path.home() / '.cache/pvz-research-resources'
SCAN_FUTURES = 8
ROBUST_WAVE = '10'
ROBUST_MIN = 5
LEVEL_IN_ID = re.compile(r'_(?:l|roof|fog|pool|conv|night|day)_(\d{1,2})(?:_|$)')


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
    """Find old and current numerical archives with metadata and an operation tape."""
    found = []
    for meta_path in sorted(root.rglob('meta.json')):
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if meta.get('schema') not in READ_SCHEMAS:
            continue
        if not (meta_path.parent / 'ticks.jsonl').is_file():
            continue
        found.append(meta_path.parent)
    return found


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
    frames = frame_path(archive)
    if not frames.is_file():
        problems.append('存档缺帧文件')
    elif meta.get('frame_count') is not None:
        count = sum(bool(line.strip()) for line in frame_bytes(frames).splitlines())
        if int(meta['frame_count']) != count:
            problems.append(f'meta.frame_count={meta["frame_count"]} 与帧数 {count} 不一致')
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
    if entry.get('deck_selection') != 'player' and deck != family_deck:
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




def verify_copy(archive: Path, resource_dir: Path, executable: Path | None) -> dict:
    from human_demo import verify
    return verify(archive, resource_dir, executable)


def render_report(rows: list, layers: dict) -> str:
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
    return '\n'.join(lines) + '\n'


def archive_sha(archive: Path) -> str:
    digest = hashlib.sha256()
    for name in ('meta.json', 'ticks.jsonl', frame_path(archive).name):
        path = archive / name
        if not path.exists():
            continue
        data = path.read_bytes()
        if name == 'meta.json':
            meta = json.loads(data)
            for key in ('verified', 'hash_verification'):
                meta.pop(key, None)
            data = json.dumps(meta, sort_keys=True, ensure_ascii=False,
                              separators=(',', ':')).encode()
        digest.update(name.encode() + b'\0' + len(data).to_bytes(8, 'big') + data)
    return digest.hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=1)
        stream.write('\n')
    Path(stream.name).replace(path)


def verification_context(executable: Path, resources: Path) -> dict:
    return dict(engine=file_sha(executable), resources={
        name: file_sha(resources / name) for name in ('main.pak', 'properties/partner.xml')
        if (resources / name).exists()}, verifier={
        name: file_sha(ROOT / name) for name in
        ('scripts/human_demo.py', 'python/pvz_env.py', 'python/pvz_event_env.py')})




def verify_once(target, resources, executable, receipt_path, sha, context, problems):
    """One worker owns one game; persist its result before another archive starts."""
    try:
        result = verify_copy(target, resources, executable)
        receipt = dict(verified=True, result=result, reason='')
    except Exception as error:
        receipt = dict(verified=False, reason=str(error))
    receipt.update(archive_sha256=sha, context=context, input_problems=problems)
    save(receipt_path, receipt)
    return receipt


def run(input_path, dest, manifest, resources, executable, *, executor_factory=None, verify_workers=2):
    if verify_workers not in (1, 2):
        raise ValueError('verify_workers must be 1 or 2')
    if input_path.is_dir() and (input_path.resolve() == dest.resolve()
                               or input_path.resolve() in dest.resolve().parents
                               or dest.resolve() in input_path.resolve().parents):
        raise ValueError('输入与dest不得互相包含')
    dest.mkdir(parents=True, exist_ok=True)
    # Shared dest lock covers receipts and the incremental live index.
    with (dest / 'import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='llm_sha_') as directory:
            source = input_path
            if input_path.is_file():
                source = Path(directory)
                safe_extract(input_path, source)
            archives = discover(source)
            if not archives:
                raise ValueError('没有可导入的对局存档')
            context = None
            tasks = load_manifest(manifest)
            layer_of = {}
            previous = {}
            for name in ('import_progress.json', 'merged_progress.json'):
                path = dest / name
                if path.exists():
                    for row in json.loads(path.read_text()).get('archives', []):
                        key = row.get('archive_sha256') or row.get('archive') or row['source']
                        if row.get('archive') or key not in previous:
                            previous[key] = row
            rows, seen, pending, futures = [], set(), [], {}
            pool = None
            try:
                for archive in archives:
                    sha = archive_sha(archive)
                    meta = json.loads((archive / 'meta.json').read_text())
                    task = meta['task']
                    base, seed = task['id'], int(task['seed'])
                    final = f'{base}_s{seed}_a{sha[:12]}'
                    target = dest / 'sha256' / sha
                    receipt_path = dest / 'receipts' / f'{sha}.json'
                    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                    copied = not target.exists()
                    if copied:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(prefix='.staging_', dir=target.parent) as copy_directory:
                            staged = Path(copy_directory) / 'archive'
                            copy_archive(archive, staged)
                            staged.rename(target)
                    else:
                        source_tape, stored_tape = archive / 'ticks.jsonl', target / 'ticks.jsonl'
                        if source_tape.stat().st_size != stored_tape.stat().st_size:
                            raise ValueError(f'库内存档大小不符，禁止自动复放：{stored_tape}')
                        source_frame, stored_frame = frame_path(archive), frame_path(target)
                        converted = source_frame.suffix == '.jsonl' and stored_frame.suffix == '.npz'
                        if not converted and source_frame.stat().st_size != stored_frame.stat().st_size:
                            raise ValueError(f'库内存档大小不符，禁止自动复放：{stored_frame}')
                        convert_archive(target)
                    if receipt is not None:
                        if receipt.get('archive_sha256') != sha or type(receipt.get('verified')) is not bool:
                            raise ValueError(f'已有SHA回执损坏，禁止自动复放：{receipt_path}')
                        problems = receipt.get('input_problems')
                        if problems is None:
                            old = previous.get(sha)
                            problems = ([old['note']] if old and old['verify'] in ('失败', '拒绝')
                                        and old.get('note') else [])
                        if receipt['verified']:
                            meta['verified'] = True
                            write_meta(target, meta)
                        pending.append((archive, sha, meta, final, target, receipt, True, problems))
                        continue
                    if sha not in futures:
                        problems = archive_problems(meta, archive)
                        if context is None:
                            context = verification_context(executable, resources)
                        if pool is None:
                            pool = (executor_factory(max_workers=verify_workers) if executor_factory else
                                    ProcessPoolExecutor(max_workers=verify_workers,
                                        mp_context=multiprocessing.get_context('spawn')))
                        futures[sha] = (pool.submit(verify_once, target, resources, executable,
                                                   receipt_path, sha, context, problems), problems)
                    pending.append((archive, sha, meta, final, target, None, False, futures[sha][1]))

                for archive, sha, meta, final, target, receipt, cached, problems in pending:
                    if receipt is None:
                        receipt = futures[sha][0].result()
                    task = meta['task']
                    base, seed = task['id'], int(task['seed'])
                    _, _, family_problems = resolve_family(meta, tasks)
                    problems = problems + family_problems
                    verified = receipt['verified'] and not problems
                    reason = '；'.join(problems + ([receipt['reason']] if receipt['reason'] else []))
                    relative = str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)
                    row = dict(source=str(archive.relative_to(source)), archive_sha256=sha,
                               task_id=final, base_task_id=base, seed=seed, level=int(task['level']),
                               multiplier=task.get('zombie_count_multiplier', 1),
                               result=meta['outcome']['result'], wave=meta['outcome']['final_wave'],
                               operations=meta['operations'], frames=meta['frame_count'],
                               verify='通过' if verified else '失败',
                               grade=grade(meta) if verified else 'loss',
                               robustness=robustness_text(meta), manual_patch=meta.get('manual_patch'),
                               new_family=sha not in previous, archive=relative if verified else '',
                               note=reason, verification_reused=cached)
                    if sha in seen:
                        row.update(note='包内相同SHA，仅入库一次', archive='')
                    seen.add(sha)
                    rows.append(row)
                    layer_of[base] = meta.get('workflow_layer') or '其他'
                    if row.get('archive') or sha not in previous:
                        previous[sha] = row
                    save(dest / 'merged_progress.json', dict(cache='sha_once', archives=list(previous.values())))
                    (dest / 'import_progress.json').unlink(missing_ok=True)
                    print(f'[{len(rows)}/{len(archives)}] {base} s{seed} {sha[:12]} '
                          f'{row["verify"]} {reason}', flush=True)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
            return dict(rows=rows, layers=layer_of)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--verify-workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    rows, layers = [], {}
    for source in args.inputs:
        result = run(source, args.dest, args.manifest, args.resource_dir, args.executable,
                     verify_workers=args.verify_workers)
        rows.extend(result['rows'])
        layers.update(result['layers'])
    report = render_report(rows, layers)
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    return int(any(row['verify'] in ('失败', '拒绝') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
