"""Task45 C: preregistered difficulty ladder and matched collection-student comparison."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
OUT = ROOT / 'artifacts/task45/phase-c'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from curriculum_queue import command_for
from pvz_fog import task45_engine_pids
import pvz_constants

LEVELS = (31, 32, 33, 34)
RATES = ((.25, 'm025'), (.35, 'm035'), (.5, 'm050'))
STUDENTS = dict(dw4i=MAIN / 'artifacts/student/dw4i_s1/model.pt',
                dw5i=OUT / 'models/dw5i_s1.pt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(manifest, task_id, student, out, beta=1, seed_offset=0):
    args = SimpleNamespace(python=Path(sys.executable), rollout_policy=STUDENTS[student],
        static_value=MAIN / 'artifacts/hard/value_branch1/model.pt',
        value=MAIN / 'artifacts/value_net/v4/model.pt', executable=ROOT / 'build/pvz-portable',
        resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'), workers=4,
        frames=OUT / 'unused_frames', out=Path(out).parent)
    label = Path(out).name
    cmd = command_for(args, manifest, task_id, seed_offset, 16, label)
    at = cmd.index('--frames')
    del cmd[at:at + 2]
    cmd[cmd.index('--out') + 1] = str(out)
    cmd[cmd.index('--dagger-beta') + 1] = str(beta)
    return cmd + ['--arms', 'fast_search']


def inputs():
    return {name: dict(path=str(path), sha256=sha(path)) for name, path in {
        **STUDENTS, 'value': MAIN / 'artifacts/value_net/v4/model.pt',
        'static_value': MAIN / 'artifacts/hard/value_branch1/model.pt',
        'executable': ROOT / 'build/pvz-portable'}.items()}


def prepare_ladder():
    plan_path = OUT / 'ladder/plan.json'
    if plan_path.exists():
        raise ValueError('ladder already frozen; use run to resume')
    source = load(ROOT / 'artifacts/task45/fog_panel.json')
    tasks, cells = [], []
    manifest = OUT / 'ladder/manifest.json'
    old = load(ROOT / 'artifacts/task45/deck-diagnostic/plan.json')
    old_tasks = {t['task_id']: t for t in load(old['manifest'])['tasks']}
    frozen_inputs = inputs()
    for level in LEVELS:
        for suffix, mode in [('f8', 'human'), ('f8x', 'xray')]:
            base = next(t for t in source['tasks'] if t['task_id'] == f'fog_l{level}_eval_{suffix}')
            for rate, rate_suffix in RATES:
                task = copy.deepcopy(base)
                task.update(task_id=f'{base["task_id"]}_{rate_suffix}', zombie_count_multiplier=rate)
                tasks.append(task)
                out = OUT / 'ladder/runs' / task['task_id']
                cell = dict(task_id=task['task_id'], level=level, observation=mode, multiplier=rate,
                            student='dw4i', beta=1, seeds=task['seeds'][:16], out=str(out),
                            command=command(manifest, task['task_id'], 'dw4i', out))
                # Reuse exact matched 0.5-human evidence rather than count reruns as fresh games.
                if mode == 'human' and rate == .5:
                    previous = next(c for c in old['cells'] if c['task_id'] == task['task_id'])
                    assert old_tasks[task['task_id']] == task
                    assert old['models']['executable']['sha256'] == frozen_inputs['executable']['sha256']
                    assert old['models']['rollout_policy']['sha256'] == frozen_inputs['dw4i']['sha256']
                    cell['reuse'] = str(Path(previous['out']) / 'games.jsonl')
                    cell['reuse_sha256'] = sha(cell['reuse'])
                cells.append(cell)
    save(manifest, dict(schema_version=1, purpose='task45 C difficulty ladder', tasks=tasks))
    save(plan_path, dict(manifest=str(manifest), manifest_sha256=sha(manifest), inputs=frozen_inputs,
                         cells=cells, planned_games=384, reused_games=64))
    print(f'froze {len(cells)} ladder cells: 384 teacher games, including 64 prior matched games')


def rows_for(cell):
    path = Path(cell['out']) / 'games.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    identities = [(r['seed'], r['arm']) for r in rows]
    expected = {(s, 'fast_search') for s in cell['seeds']}
    if len(identities) != len(set(identities)) or not set(identities) <= expected:
        raise ValueError(f'wrong or duplicate seed/arm: {path}')
    if any('fog_diagnostics' not in r or (not r['won'] and r['fog_diagnostics']['loss'] is None) for r in rows):
        raise ValueError(f'missing terminal diagnostic evidence: {path}')
    return rows


def summarize(plan):
    result = []
    for cell in plan['cells']:
        rows = rows_for(cell)
        waves, zombies, joint, lanes = Counter(), Counter(), Counter(), Counter()
        names = {d['enum']: d['type'] for d in pvz_constants.zombie_defs()}
        for row in rows:
            loss = row['fog_diagnostics']['loss']
            if not row['won']:
                waves[str(loss['wave'])] += 1
                zombies[names[loss['type']]] += 1
                joint[f'wave{loss["wave"]}:{names[loss["type"]]}'] += 1
                lanes[str(loss['row'])] += 1
        result.append({k: cell[k] for k in ('task_id', 'level', 'observation', 'multiplier', 'student', 'beta')} |
            dict(games=len(rows), complete=len(rows) == 16, won=sum(r['won'] for r in rows),
                 loss_waves=dict(waves), loss_types=dict(zombies), loss_joint=dict(joint), loss_lanes=dict(lanes)))
    return result


def run(stage):
    plan = load(OUT / stage / 'plan.json')
    assert sha(plan['manifest']) == plan['manifest_sha256']
    for info in plan['inputs'].values():
        assert sha(info['path']) == info['sha256'], f'input changed: {info["path"]}'
    lock_path = OUT / 'stage.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for cell in plan['cells']:
            if len(rows_for(cell)) == 16:
                continue
            if task45_engine_pids():
                raise RuntimeError('live task45 games: cannot launch another four-process stage')
            folder = Path(cell['out'])
            folder.mkdir(parents=True, exist_ok=True)
            if cell.get('reuse'):
                assert sha(cell['reuse']) == cell['reuse_sha256']
                (folder / 'games.jsonl').write_bytes(Path(cell['reuse']).read_bytes())
                assert len(rows_for(cell)) == 16
                print('REUSE ' + cell['task_id'], flush=True)
            else:
                print(f'START {cell["task_id"]} {cell["student"]} beta={cell["beta"]}', flush=True)
                with (folder / 'run.log').open('a') as log:
                    subprocess.run(cell['command'], stdout=log, stderr=subprocess.STDOUT, check=True,
                                   env={**os.environ, 'PVZ_NO_POPUP': '1'})
                assert len(rows_for(cell)) == 16
                print('DONE ' + cell['task_id'], flush=True)
            save(OUT / stage / 'summary.json', summarize(plan))
        save(OUT / stage / 'summary.json', summarize(plan))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare-ladder', 'run', 'summary'])
    p.add_argument('--stage', default='ladder', choices=['ladder', 'comparison'])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare-ladder':
        prepare_ladder()
    elif args.mode == 'run':
        run(args.stage)
    else:
        save(OUT / args.stage / 'summary.json', summarize(load(OUT / args.stage / 'plan.json')))


if __name__ == '__main__':
    main()
