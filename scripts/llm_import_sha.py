"""Content-addressed import with resumable, engine-bound verification receipts (task105).

Only numerical archive files are copied. Policy source stays in the original local zip.
Each distinct archive version gets its own family so a later zip cannot replace a
previously scanned trajectory or overwrite its curriculum start cell.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

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


def run_sha(input_path, dest, manifest, layers_path, resources, executable,
            future_base, seed_ledger):
    dest.mkdir(parents=True, exist_ok=True)
    # Shared dest lock covers receipts and incremental family/plan mutations.
    with (dest / 'import.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix='llm_sha_') as directory:
            source = input_path
            if input_path.is_file():
                source = Path(directory)
                legacy.safe_extract(input_path, source)
            elif (input_path.resolve() == dest.resolve() or input_path.resolve() in dest.resolve().parents
                  or dest.resolve() in input_path.resolve().parents):
                raise ValueError('输入与dest不得互相包含')
            archives = legacy.discover(source)
            if not archives:
                raise ValueError('没有episode_archive_v1存档')
            context = verification_context(executable, resources)
            tasks = legacy.load_manifest(manifest)
            layers = json.loads(layers_path.read_text()).get('layers', {})
            layer_of = {task: layer for layer, ids in layers.items() for task in ids}
            rows, entries, seen = [], [], set()
            for archive in archives:
                sha = archive_sha(archive)
                meta = json.loads((archive / 'meta.json').read_text())
                task = meta['task']
                base, seed = task['id'], int(task['seed'])
                final = f'{base}_s{seed}_a{sha[:12]}'
                target = dest / 'sha256' / sha
                receipt_path = dest / 'receipts' / f'{sha}.json'
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                if not target.exists():
                    target.mkdir(parents=True)
                    for name in FILES:
                        if (archive / name).is_file():
                            shutil.copyfile(archive / name, target / name)
                if archive_sha(target) != sha:
                    raise ValueError(f'库内存档SHA不符：{target}')
                problems = legacy.archive_problems(meta, archive)
                _, _, family_problems = legacy.resolve_family(meta, tasks)
                problems += family_problems
                cached = bool(receipt and receipt.get('context') == context
                              and receipt.get('archive_sha256') == sha)
                if not cached:
                    # Verify every archive, including structurally rejected input.
                    try:
                        result = legacy.verify_copy(target, resources, executable)
                        receipt = dict(verified=True, result=result, reason='')
                    except Exception as error:
                        receipt = dict(verified=False, reason=str(error))
                    receipt.update(archive_sha256=sha, context=context)
                    save(receipt_path, receipt)
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
                save(dest / 'import_progress.json', dict(context=context, archives=rows))
                print(f'[{len(rows)}/{len(archives)}] {base} s{seed} {sha[:12]} '
                      f'{row["verify"]} {reason}', flush=True)
            plan = legacy.scan_plan(rows, layer_of, future_base, ledger_path=seed_ledger)
            return dict(rows=rows, layers=layer_of, plan=plan, registered=entries)
