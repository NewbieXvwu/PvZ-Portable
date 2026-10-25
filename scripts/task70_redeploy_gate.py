#!/usr/bin/env python3
"""task70: engine-swap gate between the production binary and the redeploy candidate.

Runs the six production curriculum cells (day L7, night L14, pool L24, conveyor L10,
roof L41, fog fog_user_l31_snow) x {teacher beta=1, student beta=0} x 8 seeds against both
engines with commands built by curriculum_queue.command_for (the production command shape),
then requires every logical outcome field and every recorded action to match exactly.

state_hash digests are reported but allowed to differ: task63 changed how FIELD_SHAKE
particles are seeded, and the SaveGame image stores particle bytes.  The digest itself is
not part of the pass criteria; scripts/task70_particle_probe.py narrows sampled differences
to the particle chunk.

Usage:
  python3 scripts/task70_redeploy_gate.py --old <deploy3/pvz-portable> --new <deploy4/pvz-portable> \
      --out artifacts/task70/gate [--workers 4] [--seeds 8]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from curriculum_queue import command_for  # noqa: E402
from pvz_frame_store import read_frames  # noqa: E402

MULT_PANEL = ROOT / 'artifacts/stage4/multiplier_panel.json'
# (cell, task_id, manifest).  Multiplier cells come from the production panel; the roof and
# fog cells run at multiplier 1.0 from their source panels, as in production.
CELLS = (
    ('day', 'day_l7_train_d7_m125', MULT_PANEL),
    ('night', 'night_l14_train_n10_m125', MULT_PANEL),
    ('pool', 'pool_l24_train_p9k_m125', MULT_PANEL),
    ('conveyor', 'day_l10_orig_train_m125', MULT_PANEL),
    ('roof', 'roof_l41_train_r7', ROOT / 'artifacts/task46/roof_panel.json'),
    ('fog', 'fog_user_l31_snow', ROOT / 'artifacts/task68/fog_user_panel.json'),
)
ARMS = (('search', 1.0), ('fast', 0.0))

# Logical per-game fields that must be identical across engines (state_hash excluded on
# purpose; seconds is wall time).
COMPARE_FIELDS = ('won', 'final_wave', 'final_tick', 'decisions', 'planted_slots',
                  'planted_types', 'mowers_used', 'terminal', 'wave_count', 'searched',
                  'overrides', 'final_zombies', 'shovel_events', 'replant_events')


def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with log.open('a') as out:
        started = time.time()
        proc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f'command failed ({proc.returncode}): {command}; see {log}')
    print(f'  done in {time.time() - started:.0f}s -> {log}', flush=True)


def frames_for(frames_dir: Path, label: str, seed: int) -> list[dict]:
    # search_play names every archive <label>_fast_search regardless of dagger-beta.
    path = frames_dir / f'seed_{seed}' / f'{label}_fast_search' / 'frames.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    return read_frames(path)


def action_stream(frames: list[dict]) -> list:
    return [(f.get('tick'), f.get('action'), bool(f.get('action_rejected'))) for f in frames]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True, help='production (deploy3) binary')
    p.add_argument('--new', type=Path, required=True, help='candidate (deploy4) binary')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task70/gate')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--cells', default='', help='comma-separated subset of cells (smoke runs)')
    p.add_argument('--skip-run', action='store_true', help='compare existing outputs only')
    args = p.parse_args()
    cells = CELLS if not args.cells else tuple(
        c for c in CELLS if c[0] in set(args.cells.split(',')))

    models = {
        'rollout_policy': MAIN / 'artifacts/student/dw4i_s1/model.pt',
        'value': MAIN / 'artifacts/value_net/v4/model.pt',
        'static_value': MAIN / 'artifacts/hard/value_branch1/model.pt',
    }
    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f'{name}: {path}')
    common = SimpleNamespace(
        python=Path(sys.executable), resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'),
        workers=args.workers, **models)

    tags = ('old', 'new')
    executables = {'old': args.old, 'new': args.new}
    for tag in tags:
        if not executables[tag].exists():
            raise FileNotFoundError(f'{tag}: {executables[tag]}')
    if not args.skip_run:
        for tag in tags:
            out_dir = args.out / tag
            frames_dir = args.out / f'frames_{tag}'
            common.executable = executables[tag]
            print(f'== running {tag} ({executables[tag]})', flush=True)
            for cell, task_id, manifest in cells:
                for arm, beta in ARMS:
                    label = f'{cell}_{arm}_{tag}'
                    # task70 B6: a set whose games.jsonl is already complete (one row per seed)
                    # is kept -- worker count does not change the games, so an interrupted run
                    # resumes instead of redoing finished sets.
                    done = out_dir / f'{label}_{args.seeds}' / 'games.jsonl'
                    if done.is_file() and sum(1 for _ in done.open()) == args.seeds:
                        print(f'   skip {label}: {args.seeds} games already on file', flush=True)
                        continue
                    command = command_for(common, manifest, task_id, 0, args.seeds, label,
                                          frames=frames_dir, out=out_dir, beta=beta)
                    run_command(command, args.out / f'{tag}.log')
            print(f'== {tag} complete', flush=True)

    # ---- comparison ----
    evidence = {'old_executable': str(args.old), 'new_executable': str(args.new), 'seeds': args.seeds,
                'cells': []}
    all_ok = True
    state_hash_compared = state_hash_differ = 0
    # Per-command out dirs are named <cell>_<arm>_<tag>_<games>; the game seeds come from
    # the manifest task (command_for uses seed-offset 0).
    rows = {tag: {} for tag in tags}
    for tag in tags:
        for results in sorted((args.out / tag).glob('*/games.jsonl')):
            label = results.parent.name.rsplit('_', 1)[0]
            cell, arm, _tag = label.rsplit('_', 2)
            for line in results.read_text().splitlines():
                r = json.loads(line)
                key = r['seed'] if r.get('future_seed') is None else r['future_seed']
                rows[tag][(cell, arm, key)] = r
    task_seeds = {}
    for _cell, task_id, manifest in cells:
        document = json.loads(manifest.read_text())
        task = next(t for t in document['tasks'] if t['task_id'] == task_id)
        task_seeds[task_id] = task['seeds'][:args.seeds]
    for cell, task_id, _manifest in cells:
        cell_report = {'cell': cell, 'task_id': task_id, 'games': 0, 'mismatches': [],
                       'state_hash_same': 0, 'state_hash_differ': 0}
        for arm, _beta in ARMS:
            for seed in task_seeds[task_id]:
                old = rows['old'].get((cell, arm, seed))
                new = rows['new'].get((cell, arm, seed))
                if old is None or new is None:
                    cell_report['mismatches'].append({'seed': seed, 'arm': arm, 'error': 'missing game record'})
                    all_ok = False
                    continue
                cell_report['games'] += 1
                for field in COMPARE_FIELDS:
                    if old.get(field) != new.get(field):
                        cell_report['mismatches'].append(
                            {'seed': seed, 'arm': arm, 'field': field,
                             'old': old.get(field), 'new': new.get(field)})
                        all_ok = False
                if old.get('state_hash') and new.get('state_hash'):
                    state_hash_compared += 1
                    if old['state_hash'] == new['state_hash']:
                        cell_report['state_hash_same'] += 1
                    else:
                        cell_report['state_hash_differ'] += 1
                        state_hash_differ += 1
                # action sequences (frames archives are keyed by the actual game seed)
                old_frames = frames_for(args.out / 'frames_old', f'{cell}_{arm}_old', old['seed'])
                new_frames = frames_for(args.out / 'frames_new', f'{cell}_{arm}_new', new['seed'])
                old_actions, new_actions = action_stream(old_frames), action_stream(new_frames)
                if old_actions != new_actions:
                    first = next(i for i, (a, b) in enumerate(zip(old_actions, new_actions)) if a != b)
                    cell_report['mismatches'].append(
                        {'seed': seed, 'arm': arm, 'field': 'actions', 'first_index': first,
                         'old': old_actions[first], 'new': new_actions[first],
                         'len_old': len(old_actions), 'len_new': len(new_actions)})
                    all_ok = False
        evidence['cells'].append(cell_report)
    evidence['state_hash'] = {'compared': state_hash_compared, 'differ': state_hash_differ}
    evidence['pass'] = all_ok
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'gate_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=1) + '\n')
    for cell_report in evidence['cells']:
        print(f"{cell_report['cell']}: {cell_report['games']} games, "
              f"{len(cell_report['mismatches'])} mismatches, "
              f"state_hash same/differ {cell_report['state_hash_same']}/{cell_report['state_hash_differ']}")
    print('GATE', 'PASS' if all_ok else 'FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
