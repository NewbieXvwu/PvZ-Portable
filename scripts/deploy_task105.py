#!/usr/bin/env python3
"""Stage task105 curriculum additions; controller applies after qc drains.

Default is read-only. The plan is produced by audit_task105.py after checking all
archives and scans. --apply refuses while the production queue is alive. It adds
only the audited start cells and teacher-rate evidence to current production
files, preserving later changes and keeping a local rollback copy. Restarting
qc remains the controller's responsibility.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from llm_import_sha import archive_sha, file_sha, save

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq')


def merge_tasks(current, additions):
    merged = json.loads(json.dumps(current))
    by_id = {task['task_id']: task for task in merged['tasks']}
    for task in additions:
        existing = by_id.get(task['task_id'])
        if existing is not None and existing != task:
            raise ValueError(f'production task conflict: {task["task_id"]}')
        if existing is None:
            merged['tasks'].append(task)
            by_id[task['task_id']] = task
    return merged


def queue_running(target):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(target / 'scripts/curriculum_queue.py').encode() in args:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'artifacts/task105/deployment_plan.json')
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get('schema') != 'task105_deployment_v1' or not plan.get('audit_passed'):
        raise ValueError('deployment plan has no passing audit')
    if file_sha(args.plan.parent / 'audit.json') != plan['audit_sha256']:
        raise ValueError('audit receipt changed since deployment plan preparation')
    if file_sha(args.target / 'build5/pvz-portable') != plan['engine_sha256']:
        raise ValueError('production engine differs from the audited deploy14 engine')
    manifest = args.target / 'configs/human_demo_tasks.json'
    current = json.loads(manifest.read_text())
    merged = merge_tasks(current, plan['tasks'])
    for task in plan['tasks']:
        relative = Path(task['demo_start']['archive'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('deployment archive must be repository-relative')
        source = ROOT / relative
        expected = plan['archive_sha256'][task['task_id']]
        if archive_sha(source) != expected:
            raise ValueError(f'source archive SHA mismatch: {task["task_id"]}')
        target = args.target / relative
        if target.exists() and archive_sha(target) != expected:
            raise ValueError(f'production archive conflict: {target}')
    added = len(merged['tasks']) - len(current['tasks'])
    print(f'{len(plan["tasks"])} audited cells; {added} new production tasks; apply={args.apply}')
    if not args.apply:
        return
    if queue_running(args.target):
        raise ValueError('qc is still running; controller must drain it at a group boundary first')
    rates_path = Path.home() / 'pvz-logs/teacher_rates.json'
    rates = json.loads(rates_path.read_text()) if rates_path.exists() else {'bases': {}}
    before = {str(path): file_sha(path) for path in (manifest, rates_path) if path.exists()}
    backup = Path(tempfile.mkdtemp(prefix='task105_deploy_', dir=Path.home() / 'pvz-logs'))
    for path in (manifest, rates_path):
        if path.exists():
            shutil.copyfile(path, backup / path.name)
    for task in plan['tasks']:
        relative = Path(task['demo_start']['archive'])
        target = args.target / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix='task105_', dir=target.parent))
            for name in ('meta.json', 'ticks.jsonl', 'frames.jsonl', 'frames.npz'):
                if (ROOT / relative / name).exists():
                    shutil.copyfile(ROOT / relative / name, stage / name)
            if archive_sha(stage) != plan['archive_sha256'][task['task_id']]:
                raise ValueError('archive copy corrupted')
            stage.rename(target)
    # Validate the actual production loader against a staged manifest before switching.
    staged_manifest = backup / 'manifest.staged.json'
    save(staged_manifest, merged)
    command = [os.sys.executable, '-c',
               'import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
               'from curriculum_queue import load_demo_start_cells;'
               'c,_=load_demo_start_cells(Path(sys.argv[2]));print(len(c))',
               str(args.target / 'scripts'), str(staged_manifest)]
    subprocess.run(command, check=True)
    if queue_running(args.target) or any(file_sha(Path(path)) != sha for path, sha in before.items()):
        raise ValueError('production queue or config changed during staging; no config switched')
    bases = rates.setdefault('bases', {})
    for task, rate in plan['teacher_rates'].items():
        if task not in bases:
            bases[task] = rate
    try:
        save(rates_path, rates)
        save(manifest, merged)
    except BaseException:
        for path in (manifest, rates_path):
            if (backup / path.name).exists():
                shutil.copyfile(backup / path.name, path)
        raise
    save(backup / 'receipt.json', dict(plan_sha256=file_sha(args.plan), added=added,
                                      target=str(args.target), before=before,
                                      after={str(p): file_sha(p) for p in (manifest, rates_path)}))
    print(f'Applied. Rollback copies: {backup}. Controller can now dry-run and restart qc.')


if __name__ == '__main__':
    main()
