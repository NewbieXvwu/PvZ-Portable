#!/usr/bin/env python3
"""Resume task105 import → adaptive → register, without touching production qc.

Run with the ML venv. --wait-import-pid can attach to an existing import instead
of launching a second copy. Registration is local to this worktree; deployment
is a separate, explicitly invoked operation for the controller.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from llm_import_sha import file_sha, save

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build5/pvz-portable')
ENGINE_SHA = '76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a'


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] if stat[0] != 'Z' else None
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--wait-import-pid', type=int)
    args = parser.parse_args()
    out = ROOT / 'artifacts/task105'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = out / 'pipeline_status.json'
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(stage, command, accepted=(0,)):
        save(status, dict(stage=stage, state='running', command=command, pid=os.getpid()))
        with (out / f'{stage}.out').open('a') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode not in accepted:
            save(status, dict(stage=stage, state='failed', returncode=result.returncode))
            raise SystemExit(result.returncode)

    if file_sha(ENGINE) != ENGINE_SHA:
        raise ValueError('deploy14 engine SHA mismatch')
    if args.wait_import_pid:
        identity = process_identity(args.wait_import_pid)
        if identity is None:
            raise ValueError('requested import PID is not live; inspect its terminal result first')
        save(status, dict(stage='wait_import', state='running', pid=os.getpid(),
                          import_pid=args.wait_import_pid, import_start=identity))
        while process_identity(args.wait_import_pid) == identity:
            time.sleep(10)
    # Import only new SHA; previously verified archives never replay.
    run('import', [sys.executable, str(ROOT / 'scripts/import_llm_run.py'), str(args.input),
                   '--content-addressed', '--dest', str(out / 'import'), '--executable', str(ENGINE),
                   '--report', str(out / 'import_report.md'), '--scan-out', str(out / 'scan_jobs.json')], (0, 1))
    rows = json.loads((out / 'import/import_progress.json').read_text())['archives']
    plan = json.loads((out / 'scan_jobs.json').read_text())
    completed = []
    failures = []
    for entry in plan['archives']:
        task = entry['task_id']
        result = out / 'scan/results' / f'{task}_s{entry["seed"]}.json'
        try:
            # --skip-complete：续跑包里已完成的存档不重放；register 自身幂等，仍会补齐漏登记。
            run(f'adaptive_{task}', [sys.executable, str(ROOT / 'scripts/takeover_scan.py'), 'adaptive',
                 str(out / 'scan_jobs.json'), '--out', str(out / 'scan'), '--task-id', task,
                 '--executable', str(ENGINE), '--workers', '2', '--skip-complete'])
            run(f'register_{task}', [sys.executable, str(ROOT / 'scripts/takeover_scan.py'), 'register',
                 str(result), '--out', str(out / 'registration' / task)])
        except SystemExit as error:
            # 引擎被 earlyoom 等外部压力打断时不要整条链停摆：记账后继续，下次重跑从这里续。
            failures.append(dict(task=task, returncode=getattr(error, 'code', 1)))
            save(out / 'failures.json', dict(schema='task105_failures_v1', failures=failures))
            print(f'{task} 失败（{getattr(error, "code", 1)}），继续下一份', flush=True)
            continue
        completed.append(task)
        save(out / 'completed_scans.json', dict(completed=completed, total=len(plan['archives'])))
    run('report', [sys.executable, str(ROOT / 'scripts/task105_results.py')])
    save(status, dict(stage='done', state='complete', imported=len(rows),
                      scanned=len(completed), deployed=False))


if __name__ == '__main__':
    main()
