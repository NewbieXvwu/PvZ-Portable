#!/usr/bin/env python3
"""task70: paired ABBA throughput between the production and candidate binaries.

16 student-arm games (day L7 m125 and pool L24 m125, 8 seeds each) run serially, one
game per engine process, with the engine order alternating ABBA per game so machine
drift cannot favour either side.  Pass criterion: median paired wall-time ratio
(new/old) <= 1.03.

Usage:
  python3 scripts/task70_throughput_abba.py --old <deploy3> --new <deploy4> \
      [--out artifacts/task70/throughput_abba.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from curriculum_queue import command_for  # noqa: E402

MULT_PANEL = ROOT / 'artifacts/stage4/multiplier_panel.json'
CELLS = (('day', 'day_l7_train_d7_m125', MULT_PANEL),
         ('pool', 'pool_l24_train_p9k_m125', MULT_PANEL))
SEEDS_PER_CELL = 8


def run_one(command: list[str]) -> dict:
    env = dict(__import__('os').environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed: {command}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}')
    return proc.stdout


def read_record(out_dir: Path) -> dict:
    lines = (out_dir / 'games.jsonl').read_text().splitlines()
    assert len(lines) == 1, out_dir
    return json.loads(lines[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task70/throughput_abba.json')
    args = p.parse_args()

    common = SimpleNamespace(
        python=Path(sys.executable),
        rollout_policy=MAIN / 'artifacts/student/dw4i_s1/model.pt',
        value=MAIN / 'artifacts/value_net/v4/model.pt',
        static_value=MAIN / 'artifacts/hard/value_branch1/model.pt',
        resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'),
        workers=1, executable=None)
    out = args.out.parent / 'throughput_runs'
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for cell, task_id, manifest in CELLS:
        document = json.loads(manifest.read_text())
        task = next(t for t in document['tasks'] if t['task_id'] == task_id)
        for seed in task['seeds'][:SEEDS_PER_CELL]:
            jobs.append((cell, task_id, manifest, seed))
    games = []
    for index, (cell, task_id, manifest, seed) in enumerate(jobs):
        order = ('old', 'new') if index % 2 == 0 else ('new', 'old')   # ABBA
        row = {'cell': cell, 'seed': seed, 'order': ''.join(order)}
        for tag in order:
            executable = args.old if tag == 'old' else args.new
            common.executable = executable
            label = f't{index:02d}_{cell}_{tag}'
            out_dir = out / label
            command = command_for(common, manifest, task_id, 0, 1, label,
                                  frames=out / f'frames_{tag}', out=out / tag, beta=0.0)
            # one game: command_for slices seeds from the task; use --seed-offset to land on seed
            offset = task_seeds_index(manifest, task_id, seed)
            command[command.index('--seed-offset') + 1] = str(offset)
            command[command.index('--seeds') + 1] = '1'
            started = time.time()
            run_one(command)
            record = read_record(out_dir / f'{label}_1' / 'games.jsonl')
            row[f'{tag}_seconds'] = record['seconds']
            row[f'{tag}_decisions'] = record['decisions']
            assert record['seed'] == seed, (record['seed'], seed)
        row['ratio'] = row['new_seconds'] / row['old_seconds']
        games.append(row)
        print(json.dumps(row), flush=True)

    ratios = [row['ratio'] for row in games]
    report = {'old': str(args.old), 'new': str(args.new), 'games': games,
              'median_ratio': statistics.median(ratios),
              'min_ratio': min(ratios), 'max_ratio': max(ratios),
              'pass': statistics.median(ratios) <= 1.03,
              'decision_mismatch': [row for row in games if row['old_decisions'] != row['new_decisions']]}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('median_ratio', 'min_ratio', 'max_ratio', 'pass')}), flush=True)


def task_seeds_index(manifest: Path, task_id: str, seed: int) -> int:
    document = json.loads(manifest.read_text())
    task = next(t for t in document['tasks'] if t['task_id'] == task_id)
    return task['seeds'].index(seed)


if __name__ == '__main__':
    main()
