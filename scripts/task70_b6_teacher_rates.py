#!/usr/bin/env python3
"""task70 B6.4: teacher cold start on the new _hv2 demo start cells (teacher_rates bases).

Per cell: 8 teacher games (dagger-beta 1) from ladder index 0 -- the latest demo decision,
where the teacher is already stable -- on the deploy5 engine with the production models
(dw4i_s1 + leaf v5 + desktop branch1).  Commands come from curriculum_queue.command_for,
the production shape for demo start cells (demo archive + decision + RESEED future seeds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from curriculum_queue import command_for  # noqa: E402

MANIFEST = ROOT / 'configs/human_demo_tasks.json'
GAMES = 8
CELLS = ['demo_fog_31_snow_m200_hv2', 'demo_fog_32_snow_m200_hv2',
         'demo_fog_33_cactus_snow_m200_hv2', 'demo_fog_33_cattail_snow_m200_hv2']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task70/b6_teacher_rates')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    models = {'rollout_policy': MAIN / 'artifacts/student/dw4i_s1/model.pt',
              'value': MAIN / 'artifacts/value_net/v5/model.pt',
              'static_value': MAIN / 'artifacts/hard/value_branch1/model.pt'}
    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    common = SimpleNamespace(python=Path(sys.executable),
                             resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'),
                             workers=args.workers, executable=args.executable, **models)
    document = json.loads(MANIFEST.read_text())
    report = {'executable': str(args.executable), 'executable_sha256': sha(args.executable),
              'models': {name: dict(path=str(path), sha256=sha(path)) for name, path in models.items()},
              'cells': {}}
    for task_id in CELLS:
        task = next(t for t in document['tasks'] if t['task_id'] == task_id)
        start = task['demo_start']
        decision = start['decisions'][start.get('index', 0)]
        seeds = list(range(int(start['future_seed_start']),
                           int(start['future_seed_start']) + int(start.get('future_seed_count', 256))))[:GAMES]
        label = task_id + '_search'
        out_dir = args.out / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        command = command_for(common, MANIFEST, task_id, 0, GAMES, label,
                              frames=out_dir / 'frames', out=out_dir / 'search', beta=1.0,
                              demo_archive=str(MAIN / start['archive']),
                              demo_decision=decision, future_seeds=seeds)
        env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.run(command, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f'{task_id}: cold start failed, see {out_dir}')
        rows = [json.loads(line)
                for line in (out_dir / 'search' / f'{label}_{GAMES}' / 'games.jsonl').read_text().splitlines()]
        assert len(rows) == GAMES, (task_id, len(rows))
        report['cells'][task_id] = {
            'decision': decision, 'games': GAMES,
            'wins': sum(r['won'] for r in rows),
            'loss_seeds': sorted(r['future_seed'] for r in rows if not r['won'])
                          if 'future_seed' in rows[0] else sorted(r['seed'] for r in rows if not r['won']),
        }
        print(task_id, report['cells'][task_id], flush=True)
    (args.out / 'teacher_rates.json').write_text(json.dumps(report, indent=1) + '\n')
    print('report:', args.out / 'teacher_rates.json')


if __name__ == '__main__':
    main()
