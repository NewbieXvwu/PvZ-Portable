#!/usr/bin/env python3
"""Use task105's third game slot on the union queue, starting with new imports.

The original two-game pipeline keeps its forward order. This single-game
worker visits the same immutable jobs in reverse; per-archive locks prevent
duplicate games, and shared metadata locks serialize registration/reporting.
"""
import json
import os
import subprocess
import sys

from import_llm_run import load_manifest
from llm_import_sha import save
from task105_pipeline import ENGINE, ROOT, scan_finished
from task105_state import file_lock

OUT = ROOT / 'artifacts/task105'


def pending(plan, tasks):
    return [entry for entry in reversed(plan['archives']) if not scan_finished(entry, OUT, tasks)]


def main():
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    OUT.mkdir(parents=True, exist_ok=True)
    with file_lock(OUT / 'secondary.lock'):
        plan_path = OUT / 'scan_jobs.json'
        plan = json.loads(plan_path.read_text())
        tasks = load_manifest(ROOT / 'configs/human_demo_tasks.json')
        failures = []
        for entry in pending(plan, tasks):
            task = entry['task_id']
            # The forward pipeline may have completed this entry since startup.
            tasks = load_manifest(ROOT / 'configs/human_demo_tasks.json')
            if scan_finished(entry, OUT, tasks):
                continue
            result = OUT / 'scan/results' / f'{task}_s{entry["seed"]}.json'
            commands = [
                ('adaptive', [sys.executable, str(ROOT / 'scripts/takeover_scan.py'), 'adaptive',
                              str(plan_path), '--out', str(OUT / 'scan'), '--task-id', task,
                              '--executable', str(ENGINE), '--workers', '1', '--skip-complete']),
                ('register', [sys.executable, str(ROOT / 'scripts/takeover_scan.py'), 'register',
                              str(result), '--out', str(OUT / 'registration' / task)]),
                ('report', [sys.executable, str(ROOT / 'scripts/task105_results.py')]),
            ]
            for stage, command in commands:
                save(OUT / 'secondary_status.json', dict(state='running', stage=stage,
                     task_id=task, pid=os.getpid(), workers=1, command=command))
                with (OUT / f'secondary_{task}.out').open('a') as stream:
                    code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT).returncode
                if code:
                    failures.append(dict(task_id=task, stage=stage, returncode=code))
                    save(OUT / 'secondary_failures.json', dict(failures=failures))
                    print(f'{task}: {stage} failed ({code}); continuing', flush=True)
                    break
            else:
                tasks = load_manifest(ROOT / 'configs/human_demo_tasks.json')
                if not scan_finished(entry, OUT, tasks):
                    failures.append(dict(task_id=task, stage='registration', returncode=1))
                else:
                    print(f'{task}: scan and registration complete', flush=True)
        save(OUT / 'secondary_failures.json', dict(failures=failures))
        save(OUT / 'secondary_status.json', dict(state='incomplete' if failures else 'complete',
             stage='done', pid=os.getpid(), workers=1, failed=len(failures)))
        if failures:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
