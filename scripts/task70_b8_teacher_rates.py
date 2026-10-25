#!/usr/bin/env python3
"""task70 B8.4: teacher cold start on ALL demo start cells (teacher_rates bases).

Same method as B6.4, generalized to the 13 registered demo start cells of a8d00858
(6 fog _hv2 + 7 roof): 8 teacher games (dagger-beta 1) per cell from ladder index 0
on the deploy5b engine with the production models (dw4i_s1 + leaf v5 + desktop
branch1).  Commands come from curriculum_queue.command_for, the production shape for
demo start cells (demo archive + decision + RESEED future seeds).  Cells whose
games.jsonl already has the full game count are skipped, so an interrupted run resumes.
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


def demo_archive(relative: str) -> Path:
    """Demo archives were recorded on the Mac and fetched into the local trees; try
    the main checkout, this tree, then the cq working tree."""
    for base in (MAIN, ROOT, MAIN / '.claude/worktrees/cq'):
        candidate = base / relative
        if (candidate / 'meta.json').is_file():
            return candidate
    raise FileNotFoundError(f'demo archive not found under MAIN/ROOT/cq: {relative}')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task70/b8_teacher_rates')
    p.add_argument('--workers', type=int, default=2)
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
    cells = [t for t in document['tasks'] if t.get('demo_start')]
    report_path = args.out / 'teacher_rates.json'
    report = ({'executable': str(args.executable), 'executable_sha256': sha(args.executable),
               'models': {name: dict(path=str(path), sha256=sha(path)) for name, path in models.items()},
               'cells': {}})
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report['executable'], report['executable_sha256'] = str(args.executable), sha(args.executable)
    args.out.mkdir(parents=True, exist_ok=True)
    for task in cells:
        task_id = task['task_id']
        start = task['demo_start']
        decision = start['decisions'][start.get('index', 0)]
        seeds = list(range(int(start['future_seed_start']),
                           int(start['future_seed_start']) + int(start.get('future_seed_count', 256))))[:GAMES]
        done = args.out / task_id / 'search' / f'{task_id}_search_{GAMES}' / 'games.jsonl'
        if done.is_file() and sum(1 for _ in done.open()) == GAMES:
            print(f'skip {task_id}: {GAMES} games already on file', flush=True)
            rows = [json.loads(line) for line in done.read_text().splitlines()]
            report['cells'][task_id] = {
                'decision': decision, 'games': GAMES,
                'wins': sum(r['won'] for r in rows),
                'loss_seeds': sorted(r.get('future_seed', r['seed']) for r in rows if not r['won'])}
            continue
        label = task_id + '_search'
        out_dir = args.out / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        command = command_for(common, MANIFEST, task_id, 0, GAMES, label,
                              frames=out_dir / 'frames', out=out_dir / 'search', beta=1.0,
                              demo_archive=str(demo_archive(start['archive'])),
                              demo_decision=decision, future_seeds=seeds)
        env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.run(command, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f'{task_id}: cold start failed, see {out_dir}')
        rows = [json.loads(line)
                for line in done.read_text().splitlines()]
        assert len(rows) == GAMES, (task_id, len(rows))
        report['cells'][task_id] = {
            'decision': decision, 'games': GAMES,
            'wins': sum(r['won'] for r in rows),
            'loss_seeds': sorted(r.get('future_seed', r['seed']) for r in rows if not r['won'])}
        print(task_id, report['cells'][task_id], flush=True)
        report_path.write_text(json.dumps(report, indent=1) + '\n')
    report_path.write_text(json.dumps(report, indent=1) + '\n')
    print('report:', report_path)


if __name__ == '__main__':
    main()
