"""task79: frozen cold-start commands and per-tick opacity measurements."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PvZEnv, TaskSpec, fog_diagnostics
from pvz_frame_store import frame_path, read_frames
from pvz_tasks import profile_for_deck
from search_agreement_stats import wilson
from task66_fog_opacity import histogram

OUT = ROOT / 'artifacts/task79'
MANIFEST = ROOT / 'configs/human_demo_tasks.json'
EXECUTABLE = ROOT / 'build/pvz-portable'
RESOURCE = Path('/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')
MODELS = {
    'rollout-policy': Path('/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/student/dw4i_s1/model.pt'),
    'value': Path('/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/hard/value_v5/model.pt'),
    'static-value': Path('/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt'),
}


def tasks():
    return [task for task in json.loads(MANIFEST.read_text())['tasks']
            if task['task_id'].startswith('fog79_')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return dict(executable=dict(path=str(EXECUTABLE), sha256=sha(EXECUTABLE)),
                models={name: dict(path=str(path), sha256=sha(path))
                        for name, path in MODELS.items()},
                tasks=tasks(), samples=3, horizon=2500, dagger_beta=1,
                verify_samples=9, student_candidates=2, observation='human_v2', threshold=925)


def collect(task, trace=False):
    folder = OUT / ('traces' if trace else 'cold') / task['task_id']
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--manifest', str(MANIFEST), '--task-id', task['task_id'],
               '--executable', str(EXECUTABLE), '--resource-dir', str(RESOURCE),
               '--arms', 'fast_search', '--dagger-beta', '1', '--verify-samples', '9',
               '--student-candidates', '2', '--workers', '1' if trace else '4',
               '--seeds', '1' if trace else '32', '--out', str(folder)]
    for name, path in MODELS.items():
        command.extend(['--' + name, str(path)])
    if trace:
        command.extend(['--frames', str(OUT / 'frames'), '--frames-label', task['task_id']])
    with (folder / 'run.log').open('a') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1'})
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    print(task['task_id'], sum(row['won'] for row in rows), '/', len(rows), flush=True)


def measure(task):
    seed = task['seeds'][0]
    archive = OUT / 'frames' / f'seed_{seed}' / (task['task_id'] + '_fast_search')
    frames = read_frames(frame_path(archive))
    game = next(json.loads(line) for line in
                (OUT / 'traces' / task['task_id'] / 'games.jsonl').read_text().splitlines()
                if json.loads(line)['seed'] == seed)
    counts = Counter()
    columns = {col: Counter() for col in range(9)}
    opacities = Counter()
    mismatches = 0
    task_spec = TaskSpec(level=task['level'], seed=seed, profile=profile_for_deck(task['deck']),
                         fog_observation='human_v2')
    with PvZEnv(RESOURCE, executable=EXECUTABLE, headless=True) as env:
        observation, _ = env.reset(deck=task['deck'], task=task_spec)
        for frame in frames[1:]:
            action = frame['action']
            start_tick = observation['tick']
            if action['type'] != 'wait':
                observation, _, _, _, info = env.step(action)
                assert info['ok'] and not frame.get('action_rejected'), (task['task_id'], frame['tick'])
            while observation['tick'] < frame['tick']:
                observation, _, _, _, _ = env.step(dict(type='wait', ticks=1))
                raw = env.privileged_state()
                details = fog_diagnostics(raw)
                public = {zombie['id'] for zombie in observation['zombies']}
                for zombie in raw['zombies']:
                    if not zombie['on_board']:
                        continue
                    detail = details[zombie['id']]
                    col = max(0, min(8, int((zombie['x'] - 40) // 80)))
                    hidden = zombie['id'] not in public
                    values = dict(zombie_ticks=1, human_hidden=int(detail['cell_hidden']),
                                  human_v2_hidden=int(hidden),
                                  newly_visible=int(detail['cell_hidden'] and not hidden),
                                  newly_hidden=int(not detail['cell_hidden'] and hidden))
                    counts.update(values)
                    columns[col].update(values)
                    opacities[detail['opacity']] += 1
                    mismatches += detail['opacity'] != 925 and hidden != (detail['opacity'] >= 925)
                counts['ticks'] += 1
            if frame['terminal'] and not observation['terminal']:
                observation, _, _, _, _ = env.step(dict(
                    type='wait', ticks=action['ticks'] - (observation['tick'] - start_tick)))
            assert observation['tick'] == frame['tick'], (task['task_id'], frame['tick'])
        assert observation['terminal'] and observation['wave'] == game['final_wave'], (
            task['task_id'], observation['tick'], observation['terminal'], observation['wave'],
            observation['result'], game['final_tick'], game['final_wave'], game['won'])
        assert (observation['result'] == 1) == game['won']
        final_hash = env.state_hash()
        assert final_hash == game['state_hash'], (task['task_id'], final_hash, game['state_hash'])
    result = dict(task_id=task['task_id'], seed=seed, level=task['level'], **counts,
                  columns={str(col): dict(values) for col, values in columns.items()},
                  opacity_histogram=histogram(list(opacities.elements())),
                  public_visibility_mismatches=mismatches, final_state_hash=final_hash)
    path = OUT / 'opacity' / f"{task['level']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


OWN_OUTPUTS = {'configs/human_demo_tasks.json',
               'MEMORY/docs/fog-l36-39-results.json',
               'MEMORY/docs/fog-l36-39-controls.json'}


def seed_audit():
    """Every integer in every JSON of every local worktree, intersected with the new seeds.

    Own outputs of this task are listed separately: they contain the new seeds by
    construction, so they cannot count as evidence of a collision.
    """
    listing = subprocess.run(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    roots = [line.split('worktree ', 1)[1] for line in listing.splitlines()
             if line.startswith('worktree ')]
    new = {seed for task in tasks() for seed in task['seeds']}
    files = values = 0
    hits = Counter()
    for root in roots:
        for folder, subfolders, names in os.walk(root):
            subfolders[:] = [name for name in subfolders
                             if name not in ('.git', '.claude', 'node_modules',
                                             'artifacts', 'logs')]
            for name in names:
                if not name.endswith('.json'):
                    continue
                path = Path(folder) / name
                try:
                    data = json.loads(path.read_text())
                except (ValueError, OSError, UnicodeDecodeError):
                    continue
                files += 1
                stack = [data]
                while stack:
                    item = stack.pop()
                    if isinstance(item, dict):
                        stack.extend(item.values())
                    elif isinstance(item, list):
                        stack.extend(item)
                    elif isinstance(item, int) and not isinstance(item, bool):
                        values += 1
                        if item in new:
                            hits[str(path.relative_to(root))] += 1
    conflicts = {path: count for path, count in hits.items() if path not in OWN_OUTPUTS}
    # Listed without counts: this report is one of the own outputs, so counting its
    # own occurrences would make the file differ on every regeneration.
    result = dict(worktrees=roots, json_files_checked=files, integers_checked=values,
                  new_seeds=len(new), own_outputs=sorted(p for p in hits if p in OWN_OUTPUTS),
                  conflicts=conflicts)
    (OUT / 'seed-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'worktrees'}, ensure_ascii=False))
    return result


def report():
    snapshot = json.loads((OUT / 'task74-snapshot.json').read_text())
    cells = []
    for source, definitions in [('task74', snapshot['manifest']['tasks']), ('task79', tasks())]:
        for task in definitions:
            if task['level'] not in range(36, 40):
                continue
            rows = snapshot['games'][task['task_id']] if source == 'task74' else [
                json.loads(line) for line in
                (OUT / 'cold' / task['task_id'] / 'games.jsonl').read_text().splitlines()]
            assert len(rows) == len(task['seeds']) == len({row['seed'] for row in rows})
            assert {row['seed'] for row in rows} == set(task['seeds'])
            assert all(row['terminal'] and row['arm'] == 'fast_search' for row in rows)
            losses = []
            for row in rows:
                if row['won']:
                    continue
                breach = row.get('fog_diagnostics', {}).get('loss')
                losses.append(dict(seed=row['seed'], wave=row['final_wave'],
                                   wave_count=row['wave_count'], tick=row['final_tick'],
                                   breach=breach))
            wins = sum(row['won'] for row in rows)
            cells.append(dict(source=source, task_id=task['task_id'], level=task['level'],
                              deck=task['deck'], observation=task.get('fog_observation', 'human'),
                              wins=wins, games=len(rows), wilson95=wilson(wins, len(rows)),
                              loss_waves=dict(Counter(row['wave'] for row in losses)),
                              breach_types=dict(Counter(str(row['breach']['type'])
                                  if row['breach'] else 'unrecorded' for row in losses)),
                              losses=losses))
    r11_cells = []
    for path in sorted((OUT / 'cold_r11').glob('*/games.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        losses = [dict(seed=row['seed'], wave=row['final_wave'],
                       wave_count=row['wave_count'], tick=row['final_tick'],
                       breach=row.get('fog_diagnostics', {}).get('loss'))
                  for row in rows if not row['won']]
        wins = sum(row['won'] for row in rows)
        r11_cells.append(dict(task_id=path.parent.name, wins=wins, games=len(rows),
                              wilson95=wilson(wins, len(rows)),
                              seeds=sorted(row['seed'] for row in rows),
                              loss_waves=dict(Counter(row['wave'] for row in losses)),
                              breach_types=dict(Counter(str(row['breach']['type'])
                                  if row['breach'] else 'unrecorded' for row in losses)),
                              losses=losses))
    result = dict(task74_provenance=snapshot['provenance'],
                  task79_provenance=json.loads((OUT / 'cold-provenance.json').read_text()),
                  opacity=json.loads((OUT / 'opacity.json').read_text()),
                  seed_audit=seed_audit(), cells=cells, r11_cells=r11_cells)
    (ROOT / 'MEMORY/docs/fog-l36-39-results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    for cell in cells + r11_cells:
        print(cell['task_id'], cell['wins'], '/', cell['games'], cell['wilson95'])


def main():
    mode = sys.argv[1]
    os.environ['PVZ_NO_POPUP'] = '1'
    selected = list({task['level']: task for task in reversed(tasks())}.values())
    selected.sort(key=lambda task: task['level'])
    if mode in ('traces', 'cold'):
        identity = provenance()
        plan_path = OUT / f'{mode}-provenance.json'
        if plan_path.exists():
            assert json.loads(plan_path.read_text()) == identity, 'frozen inputs changed'
        else:
            plan_path.write_text(json.dumps(identity, indent=2) + '\n')
        if mode == 'traces':
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda task: collect(task, trace=True), selected))
        else:
            for task in tasks():
                collect(task)
    elif mode == 'opacity':
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(measure, selected))
        (OUT / 'opacity.json').write_text(json.dumps(results, indent=2) + '\n')
    elif mode == 'report':
        report()
    elif mode == 'seed-audit':
        seed_audit()
    else:
        raise ValueError('choose traces, opacity, cold, report or seed-audit')


if __name__ == '__main__':
    main()
