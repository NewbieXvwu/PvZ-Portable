#!/usr/bin/env python3
"""Run the task148 paired candidate measurement, two simulator workers at most."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/home/newbiexvwu/pvz-logs/task148')


def main():
    resource.setrlimit(resource.RLIMIT_AS, (8000000 * 1024, 8000000 * 1024))
    provenance = json.loads((OUT / 'provenance.json').read_text())
    executable = Path(provenance['executable'])
    provenance['executable_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PVZ_NO_POPUP='1')
    for task in json.loads((OUT / 'manifest.json').read_text())['tasks']:
        print(task['task_id'], flush=True)
        subprocess.run([
            sys.executable, str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(OUT / 'student.pt'),
            '--value', str(OUT / 'value.pt'),
            '--static-value', str(OUT / 'static_value.pt'),
            '--executable', str(executable),
            '--resource-dir', provenance['resource_dir'],
            '--manifest', str(OUT / 'manifest.json'), '--task-id', task['task_id'],
            '--seeds', '8', '--workers', '2', '--arms', 'fast,fast_search',
            '--dagger-beta', '1', '--verify-samples', '9',
            '--student-candidates', '2', '--out', str(OUT / 'runs' / task['task_id'])
        ], check=True, env=env, cwd=ROOT)


if __name__ == '__main__':
    main()
