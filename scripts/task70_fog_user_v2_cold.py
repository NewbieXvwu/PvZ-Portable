#!/usr/bin/env python3
"""task70: preregistered teacher cold start for fog_user_v2_l31_snow (human_v2 observation).

32 teacher games (dagger-beta 1) on the evaluation sibling's seeds (49_100_000..31, disjoint
from every training block), played by the candidate deploy4 binary -- the fog human_v2
observation does not exist in deploy3.  Commands come from curriculum_queue.command_for, the
production command shape.  The measured rate replaces the task68 25/32 prior in
teacher_rates.json for the new family; the teacher searches with full knowledge, so this is
also the empirical check that the observation change does not disturb the teacher.

Usage:
  python3 scripts/task70_fog_user_v2_cold.py --executable <deploy4> \
      [--out artifacts/task70/cold] [--workers 4]
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

MANIFEST = ROOT / 'artifacts/task70/fog_user_v2_panel.json'
TASK_ID = 'fog_user_v2_l31_snow_eval'
GAMES = 32


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task70/cold')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--manifest', type=Path, default=MANIFEST,
                   help='override the panel (control runs: same seeds, other fog_observation)')
    p.add_argument('--task-id', default=TASK_ID)
    args = p.parse_args()

    models = {'rollout_policy': MAIN / 'artifacts/student/dw4i_s1/model.pt',
              'value': MAIN / 'artifacts/value_net/v4/model.pt',
              'static_value': MAIN / 'artifacts/hard/value_branch1/model.pt'}
    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    common = SimpleNamespace(python=Path(sys.executable), resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'),
                             workers=args.workers, executable=args.executable, **models)
    label = f'{args.task_id}_search'
    out_dir = args.out / 'search'
    args.out.mkdir(parents=True, exist_ok=True)
    command = command_for(common, args.manifest, args.task_id, 0, GAMES, label,
                          frames=args.out / 'frames', out=out_dir, beta=1.0)
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    proc = subprocess.run(command, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f'cold start failed: see {out_dir}')
    results_path = out_dir / f'{label}_{GAMES}' / 'games.jsonl'
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(rows) == GAMES, len(rows)
    document = json.loads(args.manifest.read_text())
    task = next(t for t in document['tasks'] if t['task_id'] == args.task_id)
    assert {r['seed'] for r in rows} == set(task['seeds'])
    report = {
        'task_id': args.task_id, 'manifest': str(args.manifest), 'games': GAMES,
        'wins': sum(r['won'] for r in rows),
        'rate': round(sum(r['won'] for r in rows) / GAMES, 4),
        'seeds': task['seeds'],
        'executable': str(args.executable), 'executable_sha256': sha(args.executable),
        'models': {name: dict(path=str(path), sha256=sha(path)) for name, path in models.items()},
        'mean_final_wave': round(sum(r['final_wave'] for r in rows) / GAMES, 2),
        'mean_decisions': round(sum(r['decisions'] for r in rows) / GAMES, 1),
    }
    (args.out / 'cold_start.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('games', 'wins', 'rate', 'mean_final_wave')}), flush=True)


if __name__ == '__main__':
    main()
