#!/usr/bin/env python3
"""Serial SHA imports, persistent package snapshots, then resume the union scan.

The baseline snapshots in artifacts/task105 preserve the pre-batch rows/jobs.
Each package is extracted only until its size/mtime receipt is complete; archive
verification continues to use llm_import_sha's permanent SHA receipts.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from llm_import_sha import compact_design, save
from import_llm_run import load_manifest, register_tasks
from task105_pipeline import ENGINE, ROOT

OUT = ROOT / 'artifacts/task105'


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def merge(previous_rows, previous_plan, packages):
    rows = {r['archive_sha256']: r for r in previous_rows}
    jobs = {r['task_id']: r for r in previous_plan['archives']}
    for package in packages:
        rows.update((r['archive_sha256'], r) for r in package['archives'])
        for entry in package['plan']['archives']:
            old = jobs.get(entry['task_id'])
            if old and (old['archive'] != entry['archive']
                        or old['future_seeds'] != entry['future_seeds']):
                raise ValueError(f'conflicting saved scan context: {entry["task_id"]}')
            jobs.setdefault(entry['task_id'], entry)
    return (dict(cache='sha_once', archives=list(rows.values())),
            dict(schema=previous_plan['schema'], archives=list(jobs.values())))


def append_completed_import(source, index_path, staged_manifest):
    """Publish a completed staged import while the caller owns pipeline.lock.

    Import can run beside scanning with one verifier and an isolated manifest.
    Publication keeps the current manifest's ladders and all old scan contexts.
    No archive verification or replay happens here.
    """
    imported = json.loads(index_path.read_text())
    if Path(imported['input']).resolve() != source.resolve():
        raise ValueError('completed import index does not match input')
    rows = {}
    for row in imported['archives']:
        sha = row['archive_sha256']
        # A working copy may duplicate an exported archive. Keep its usable row.
        if sha not in rows or (row.get('archive') and not rows[sha].get('archive')):
            rows[sha] = row
    package = dict(identity=identity(source), archives=list(rows.values()),
                   plan=json.loads(Path(imported['scan_plan']).read_text()))
    previous = json.loads((OUT / 'import/merged_progress.json').read_text())['archives']
    previous_plan = json.loads((OUT / 'scan_jobs.json').read_text())
    progress, plan = merge(previous, previous_plan, [package])
    manifest = ROOT / 'configs/human_demo_tasks.json'
    current = load_manifest(manifest)
    staged = load_manifest(staged_manifest)
    additions = []
    for entry in package['plan']['archives']:
        task_id = entry['task_id']
        task = staged[task_id]
        if task_id in current:
            # Existing registration owns its live demo_start/rung state.
            if any(current[task_id].get(k) != v for k, v in task.items() if k != 'demo_start'):
                raise ValueError(f'conflicting staged family: {task_id}')
        else:
            additions.append(task)
    register_tasks(manifest, additions)
    compact_design(ROOT / 'DESIGN.md', load_manifest(manifest))
    package_name = source.parent.name if source.is_dir() else source.stem
    save(OUT / 'import_packages' / f'{package_name}.json', package)
    save(OUT / 'import/merged_progress.json', progress)
    save(OUT / 'scan_jobs.json', plan)
    packages_path = OUT / 'input_packages.json'
    packages = json.loads(packages_path.read_text())['packages']
    packages = [p for p in packages if p['identity']['path'] != package['identity']['path']]
    packages.append(dict(identity=package['identity'], archives=len(rows),
                         source_copies=len(imported['archives']),
                         reused=sum(r.get('verification_reused', False) for r in rows.values()),
                         verified=sum(r['verify'] == '通过' for r in rows.values()),
                         rejected=sum(r['verify'] in ('失败', '拒绝') for r in rows.values())))
    save(packages_path, dict(packages=packages))
    # Previous state=complete must not hide newly appended scan work.
    save(OUT / 'pipeline_status.json', dict(stage='merged_import', state='incomplete',
         imported=len(progress['archives']), scan_expected=len(plan['archives']), deployed=False))
    return progress, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--first-import-index', type=Path,
                        help='复用已正常完成的首包导入索引，不重新解压或校验')
    parser.add_argument('--completed-index', type=Path,
                        help='仅合并已完成的单包暂存导入；需先停止本任务扫描')
    parser.add_argument('--staged-manifest', type=Path,
                        help='与--completed-index配套的独立暂存清单')
    args = parser.parse_args()
    if args.completed_index and (len(args.inputs) != 1 or not args.staged_manifest):
        parser.error('--completed-index requires one input and --staged-manifest')
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.completed_index:
            progress, plan = append_completed_import(args.inputs[0], args.completed_index,
                                                      args.staged_manifest)
            print(f'merged {len(progress["archives"])} SHA, {len(plan["archives"])} scans', flush=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/task105_results.py')],
                           cwd=ROOT, env=env, check=True)
            return
        baseline = json.loads((OUT / 'import/progress_resume00099.json').read_text())['archives']
        baseline_plan = json.loads((OUT / 'scan_jobs_resume00099.json').read_text())
        packages = []
        for index, source in enumerate(args.inputs):
            receipt = OUT / 'import_packages' / f'{source.stem}.json'
            source_identity = identity(source)
            save(OUT / 'batch_status.json', dict(state='importing', input=str(source),
                 completed=len(packages), total=len(args.inputs), pid=os.getpid()))
            package = json.loads(receipt.read_text()) if receipt.exists() else {}
            if package.get('identity') != source_identity:
                plan_path = OUT / f'scan_jobs_{source.stem}.json'
                if index == 0 and args.first_import_index:
                    imported = json.loads(args.first_import_index.read_text())
                    if Path(imported['input']).resolve() != source.resolve():
                        raise ValueError('首包完成索引与输入不一致')
                    plan_path = Path(imported['scan_plan'])
                    package_rows = imported['archives']
                else:
                    command = [sys.executable, str(ROOT / 'scripts/import_llm_run.py'), str(source),
                               '--content-addressed', '--dest', str(OUT / 'import'),
                               '--executable', str(ENGINE), '--report', str(OUT / f'import_{source.stem}.md'),
                               '--scan-out', str(plan_path)]
                    with (OUT / f'import_{source.stem}.out').open('a') as stream:
                        code = subprocess.run(command, cwd=ROOT, env=env,
                                              stdout=stream, stderr=subprocess.STDOUT).returncode
                    if code not in (0, 1):
                        raise RuntimeError(f'import failed ({code}): {source}')
                    package_rows = json.loads((OUT / 'import/import_progress.json').read_text())['archives']
                package = dict(identity=source_identity,
                               archives=package_rows,
                               plan=json.loads(plan_path.read_text()))
                save(receipt, package)
            packages.append(package)
            progress, plan = merge(baseline, baseline_plan, packages)
            save(OUT / 'import/merged_progress.json', progress)
            save(OUT / 'scan_jobs.json', plan)
            save(OUT / 'input_packages.json', dict(packages=[dict(identity=p['identity'],
                 archives=len(p['archives']), reused=sum(r.get('verification_reused', False) for r in p['archives']),
                 verified=sum(r['verify'] == '通过' for r in p['archives']),
                 rejected=sum(r['verify'] in ('失败', '拒绝') for r in p['archives'])) for p in packages]))
            print(f'imported {len(packages)}/{len(args.inputs)} packages; '
                  f'union {len(progress["archives"])} SHA, {len(plan["archives"])} scans', flush=True)
        save(OUT / 'batch_status.json', dict(state='imported', completed=len(packages),
             total=len(args.inputs), pid=os.getpid()))
        subprocess.run([sys.executable, str(ROOT / 'scripts/task105_results.py')],
                       cwd=ROOT, env=env, check=True)
    # Release the pipeline lock before the supervisor starts its child.
    os.execve(sys.executable, [sys.executable, str(ROOT / 'scripts/task105_supervise.py'),
              str(args.inputs[0]), '--skip-import'], env)


if __name__ == '__main__':
    main()
