#!/usr/bin/env python3
"""task177: rerun the frozen task129 54-deck panel with one teacher arm.

The panel is the v5 deck manifest frozen by task129 (same decks, same seeds, 8 games each).
Each task runs search_play.py with one worker; at most ``--workers`` tasks at a time, as
task129 did. search_play resumes finished seeds from the arm's games.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = '/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'


def run_task(task, args):
    out = args.out / task['task_id']
    out.mkdir(parents=True, exist_ok=True)
    results = out / 'games.jsonl'
    if results.exists() and len(results.read_text().splitlines()) >= len(task['seeds']):
        rows = [json.loads(line) for line in results.read_text().splitlines()]
        return task['task_id'], sum(row['won'] for row in rows), len(rows), 0.0
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--manifest', str(args.manifest), '--task-id', task['task_id'],
               '--arms', 'fast_search', '--dagger-beta', '1',
               '--verify-samples', '9', '--student-candidates', '2',
               '--horizon', str(args.horizon), '--samples', str(args.samples),
               '--workers', '1', '--out', str(out),
               '--rollout-policy', str(args.rollout_policy),
               '--value', str(args.value), '--static-value', str(args.static_value),
               '--executable', str(args.executable), '--resource-dir', RESOURCES]
    started = time.monotonic()
    with (out / 'run.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''},
                       check=True)
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    return task['task_id'], sum(row['won'] for row in rows), len(rows), time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--rollout-policy', type=Path, required=True)
    parser.add_argument('--value', type=Path, required=True)
    parser.add_argument('--static-value', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=2500)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    tasks = json.loads(args.manifest.read_text())['tasks']
    won = games = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for task_id, w, n, seconds in executor.map(lambda task: run_task(task, args), tasks):
            won += w
            games += n
            print(f'{task_id}: {w}/{n} ({seconds:.0f}s) total {won}/{games}', flush=True)
    print(f'PANEL TOTAL {won}/{games} ({won / games:.3%}) in {time.monotonic() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
