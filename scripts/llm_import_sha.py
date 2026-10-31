"""Content-addressed import: two first-time verifiers, permanent SHA receipts.

Only numerical archive files are copied. Policy source stays in the original local zip.
Each distinct archive version gets its own family so a later zip cannot replace a
previously scanned trajectory or overwrite its curriculum start cell.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import re
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import import_llm_run as legacy

FILES = ('meta.json', 'ticks.jsonl', 'frames.jsonl', 'frames.npz')


def archive_sha(archive: Path) -> str:
    digest = hashlib.sha256()
    for name in FILES:
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
        name: file_sha(legacy.ROOT / name) for name in
        ('scripts/human_demo.py', 'python/pvz_env.py', 'python/pvz_event_env.py')})


def compact_design(path: Path, tasks: dict) -> None:
    """Keep SHA families in one table instead of one generated section per archive."""
    text = path.read_text() if path.exists() else ''
    blocks = re.split(r'(?=^### )', text, flags=re.M)
    blocks = [block for block in blocks if not (
        block.startswith('### task89：LLM 存档自动登记的新族')
        and re.search(r'_a[0-9a-f]{12}`', block))]
    text = ''.join(blocks)
    begin, end = '<!-- task105 SHA families -->', '<!-- /task105 SHA families -->'
    entries = [t for t in tasks.values() if re.search(r'_a[0-9a-f]{12}$', t['task_id'])]
    if not entries:
        return
    lines = [begin, '| task_id | 关卡 | 种子 | 倍率 | 卡组（存档 meta） |',
             '|---|---|---|---|---|']
    for entry in sorted(entries, key=lambda t: t['task_id']):
        lines.append(f'| `{entry["task_id"]}` | {entry["level"]} | {entry["seeds"]} | '
                     f'{entry["zombie_count_multiplier"]} | {entry["deck"]} |')
    block = '\n'.join(lines + [end])
    if begin in text:
        text = re.sub(re.escape(begin) + r'.*?' + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        numbers = [int(n) for n in re.findall(r'^### 14\.(\d+)', text, re.M)]
        text = text.rstrip() + f'\n\n### 14.{max(numbers, default=35) + 1} GPT Pro SHA 存档族（task105）\n\n' + block + '\n'
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def verify_once(target, resources, executable, receipt_path, sha, context, problems):
    """One worker owns one game; persist its result before another archive starts."""
    try:
        result = legacy.verify_copy(target, resources, executable)
        receipt = dict(verified=True, result=result, reason='')
    except Exception as error:
        receipt = dict(verified=False, reason=str(error))
    receipt.update(archive_sha256=sha, context=context, input_problems=problems)
    save(receipt_path, receipt)
    return receipt


def run_sha(input_path, dest, manifest, layers_path, resources, executable,
            future_base, seed_ledger, *, executor_factory=None, verify_workers=2):
    if verify_workers not in (1, 2):
        raise ValueError('verify_workers must be 1 or 2')
    if input_path.is_dir() and (input_path.resolve() == dest.resolve()
                               or input_path.resolve() in dest.resolve().parents
                               or dest.resolve() in input_path.resolve().parents):
        raise ValueError('输入与dest不得互相包含')
    dest.mkdir(parents=True, exist_ok=True)
    # Shared dest lock covers receipts and incremental family/plan mutations.
    with (dest / 'import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='llm_sha_') as directory:
            source = input_path
            if input_path.is_file():
                source = Path(directory)
                legacy.safe_extract(input_path, source)
            archives = legacy.discover(source)
            if not archives:
                raise ValueError('没有episode_archive_v1存档')
            context = None
            tasks = legacy.load_manifest(manifest)
            layers = json.loads(layers_path.read_text()).get('layers', {})
            layer_of = {task: layer for layer, ids in layers.items() for task in ids}
            previous_path = dest / 'import_progress.json'
            previous = ({r['archive_sha256']: r for r in json.loads(previous_path.read_text())['archives']}
                        if previous_path.exists() else {})
            rows, entries, seen, pending, futures = [], [], set(), [], {}
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
                            staged.mkdir()
                            for name in FILES:
                                if (archive / name).is_file():
                                    shutil.copyfile(archive / name, staged / name)
                            staged.rename(target)
                    else:
                        for name in FILES[1:]:
                            source_file, stored_file = archive / name, target / name
                            if name in ('frames.jsonl', 'frames.npz') and (archive / 'frames.jsonl').is_file() \
                                    and not (archive / 'frames.npz').exists() and (target / 'frames.npz').is_file():
                                # task32's verified in-place conversion preserves this SHA's receipt.
                                continue
                            if (source_file.exists() != stored_file.exists()
                                    or (source_file.exists() and source_file.stat().st_size != stored_file.stat().st_size)):
                                raise ValueError(f'库内存档大小不符，禁止自动复放：{stored_file}')
                    if receipt is not None:
                        if receipt.get('archive_sha256') != sha or type(receipt.get('verified')) is not bool:
                            raise ValueError(f'已有SHA回执损坏，禁止自动复放：{receipt_path}')
                        problems = receipt.get('input_problems')
                        if problems is None:
                            old = previous.get(sha)
                            problems = ([old['note']] if old and old['verify'] in ('失败', '拒绝')
                                        and old.get('note') else [])
                        if copied and receipt['verified']:
                            meta['verified'] = True
                            save(target / 'meta.json', meta)
                        pending.append((archive, sha, meta, final, target, receipt, True, problems))
                        continue
                    if sha not in futures:
                        problems = legacy.archive_problems(meta, archive)
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
                    _, _, family_problems = legacy.resolve_family(meta, tasks)
                    problems = problems + family_problems
                    verified = receipt['verified'] and not problems
                    reason = '；'.join(problems + ([receipt['reason']] if receipt['reason'] else []))
                    relative = str(target.relative_to(legacy.ROOT)) if target.is_relative_to(legacy.ROOT) else str(target)
                    row = dict(source=str(archive.relative_to(source)), archive_sha256=sha,
                               task_id=final, base_task_id=base, seed=seed, level=int(task['level']),
                               multiplier=task.get('zombie_count_multiplier', 1),
                               result=meta['outcome']['result'], wave=meta['outcome']['final_wave'],
                               operations=meta['operations'], frames=meta['frame_count'],
                               verify='通过' if verified else '失败',
                               grade=legacy.grade(meta) if verified else 'loss',
                               robustness=legacy.robustness_text(meta), manual_patch=meta.get('manual_patch'),
                               new_family=final not in tasks, archive=relative if verified else '',
                               note=reason, verification_reused=cached)
                    if sha in seen:
                        row.update(note='包内相同SHA，仅入库和扫描一次', archive='')
                    elif verified and row['grade'] != 'loss' and final not in tasks:
                        entry = legacy.family_entry(final, meta, tasks.get(base))
                        tasks[final] = entry
                        entries.append(entry)
                        # Persist each verified family immediately, making interruption resumable.
                        legacy.register_tasks(manifest, [entry])
                        legacy.register_design(manifest.parent.parent / 'DESIGN.md', [entry])
                    seen.add(sha)
                    rows.append(row)
                    save(dest / 'import_progress.json', dict(cache='sha_once', archives=rows))
                    print(f'[{len(rows)}/{len(archives)}] {base} s{seed} {sha[:12]} '
                          f'{row["verify"]} {reason}', flush=True)
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)
            plan = legacy.scan_plan(rows, layer_of, future_base, ledger_path=seed_ledger)
            compact_design(manifest.parent.parent / 'DESIGN.md', tasks)
            return dict(rows=rows, layers=layer_of, plan=plan, registered=entries)
