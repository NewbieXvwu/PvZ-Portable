"""Prepare, resume and summarize the preregistered task45 cold start (at most four games).

Commands derive from curriculum_queue.command_for; outputs and logs remain under task45.
Completed cells are validated against the exact seed/arm set before reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from curriculum_queue import command_for


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(manifest, out):
    args = SimpleNamespace(python=Path(sys.executable),
        rollout_policy=MAIN / 'artifacts/student/dw4i_s1/model.pt',
        static_value=MAIN / 'artifacts/hard/value_branch1/model.pt',
        value=MAIN / 'artifacts/value_net/v4/model.pt', executable=ROOT / 'build/pvz-portable',
        resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'), workers=4,
        frames=out / 'unused_frames', out=out)
    tasks = {t['task_id']: t for t in json.loads(manifest.read_text())['tasks']}
    plan = dict(manifest=str(manifest), manifest_sha256=sha(manifest),
                models={k: dict(path=str(getattr(args, k)), sha256=sha(getattr(args, k)))
                        for k in ('rollout_policy', 'static_value', 'value', 'executable')}, cells=[])
    for level in range(31, 35):
        for suffix in ('f8', 'f8x', 'f8p'):
            task_id = f'fog_l{level}_eval_{suffix}'
            for arm in ('fast_search', 'fast'):
                label = f'{task_id}_{arm}'
                command = command_for(args, manifest, task_id, 0, 16, label)
                index = command.index('--frames')
                del command[index:index + 2]
                command[command.index('--dagger-beta') + 1] = '1' if arm == 'fast_search' else '0'
                command += ['--arms', arm]
                plan['cells'].append(dict(task_id=task_id, level=level, suffix=suffix, arm=arm,
                    seeds=tasks[task_id]['seeds'][:16], command=command,
                    out=command[command.index('--out') + 1]))
    return plan


def completed(cell):
    path = Path(cell['out']) / 'games.jsonl'
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = [(r['seed'], r['arm']) for r in rows]
    expected = {(s, cell['arm']) for s in cell['seeds']}
    if len(pairs) == len(expected) and set(pairs) == expected:
        if any('fog_diagnostics' not in r for r in rows):
            raise ValueError(f'incomplete diagnostics: {path}')
        return True
    if rows:
        raise ValueError(f'partial cell requires explicit recovery, refusing to overwrite: {path}')
    return False


def summarize(plan):
    summary = []
    for cell in plan['cells']:
        if not completed(cell):
            continue
        rows = [json.loads(line) for line in (Path(cell['out']) / 'games.jsonl').read_text().splitlines()]
        plants = Counter()
        positions = Counter()
        decks = {}
        agreement = Counter()
        for r in rows:
            agreement.update(r['fog_diagnostics']['teacher_agreement'])
            for p in r['fog_diagnostics']['opening_plant_positions']:
                plants[p['type']] += p['count']
                positions[p['type'], p['row'], p['col']] += p['count']
            key = ','.join(map(str, sorted(r['deck'])))
            deck = decks.setdefault(key, dict(games=0, won=0))
            deck['games'] += 1
            deck['won'] += r['won']
        summary.append(dict(task_id=cell['task_id'], arm=cell['arm'], games=len(rows),
             won=sum(r['won'] for r in rows), teacher_agreement=dict(agreement), opening_plant_counts=dict(plants),
             opening_plant_positions=[dict(type=k, row=r, col=c, count=n)
                                     for (k, r, c), n in sorted(positions.items())],
             deck_measurements=decks,
             losses=[dict(seed=r['seed'], final_wave=r['final_wave'],
                          **(r['fog_diagnostics']['loss'] or {'unattributed': True}))
                     for r in rows if not r['won']]))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'summary'])
    p.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/task45/fog_panel.json')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task45/cold')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / 'plan.json'
    if args.mode == 'prepare':
        if plan_path.exists():
            raise ValueError(f'plan already frozen: {plan_path}')
        plan = prepare(args.manifest, args.out)
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
        print(json.dumps(dict(cells=len(plan['cells']), games=384,
                              manifest_sha256=plan['manifest_sha256'], models=plan['models']), indent=2))
        return
    plan = json.loads(plan_path.read_text())
    assert sha(plan['manifest']) == plan['manifest_sha256'], 'manifest changed after preregistration'
    for model in plan['models'].values():
        assert sha(model['path']) == model['sha256'], f'input changed: {model["path"]}'
    if args.mode == 'run':
        for cell in plan['cells']:
            if completed(cell):
                continue
            print('START ' + cell['task_id'] + ' ' + cell['arm'], flush=True)
            folder = Path(cell['out'])
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'run.log').open('a') as stream:
                subprocess.run(cell['command'], stdout=stream, stderr=subprocess.STDOUT, check=True,
                               env={**__import__('os').environ, 'PVZ_NO_POPUP': '1'})
            assert completed(cell), f'cell did not finish: {cell["out"]}'
            (args.out / 'summary.json').write_text(json.dumps(summarize(plan), indent=2) + '\n')
            print('DONE ' + cell['task_id'] + ' ' + cell['arm'], flush=True)
    result = summarize(plan)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([dict(task_id=c['task_id'], arm=c['arm'], games=c['games'], won=c['won'])
                      for c in result], indent=2))


if __name__ == '__main__':
    main()
