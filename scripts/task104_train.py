#!/usr/bin/env python3
"""按冻结配方串行训练 task104，并让正在训练的其他任务先运行。"""
import os
from pathlib import Path
import subprocess
import sys
import time

import dagger_experiment_runner as guard
from run_identity_validation import descendants, trainers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task104'
guard.OUT = OUT / 'logs/guard'
guard.OUT.mkdir(parents=True, exist_ok=True)
guard.descendants = descendants
guard.ENV = {**os.environ, 'PYTHONPATH': f'{ROOT}/python:{ROOT}/scripts',
             'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
             'MALLOC_ARENA_MAX': '2', 'CUDA_VISIBLE_DEVICES': '0'}

for run in ('a_s1', 'a_s2', 'b_s1', 'b_s2'):
    if (OUT / run / 'model.pt').exists():
        raise SystemExit(f'{run}: 检查点已存在，请先核实已完成的训练。')
    ready_since = None
    while True:
        busy = trainers()
        available = guard.available()
        if not busy and available >= 16 * 1024 * 1024:
            if ready_since is None:
                ready_since = time.monotonic()
            if time.monotonic() - ready_since >= 30:
                gpu = subprocess.check_output(
                    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
                if not gpu.strip():
                    break
        else:
            ready_since = None
        print(dict(run=run, foreign_trainers=busy, available_kib=available), flush=True)
        time.sleep(60)
    command = [sys.executable, str(ROOT / 'scripts/train_rollout_policy.py'),
               '--recipe', str(OUT / 'recipes' / f'{run}.json')]
    result = guard.run(command, run, cwd=ROOT,
                       pause_when=lambda owned: bool(set(trainers()) - set(owned)))
    if result['returncode']:
        raise SystemExit(result['returncode'])
