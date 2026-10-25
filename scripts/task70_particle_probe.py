#!/usr/bin/env python3
"""task70: narrow engine-swap state_hash differences to the particle bytes.

For sampled student-arm gate games this replays the recorded action stream against a
series of engine runs and hashes the state after every decision.  The hash itself comes
from BRANCH_SNAPSHOT_FAST W:0 -- the one hashing path both the production (deploy3) and
candidate (deploy4) binaries implement, with identical semantics (restore parent, wait 0
ticks, hash the SaveGame image + RNG + counters).

Per game it reports:
  - every-decision observations must stay byte-identical between engines (the gate
    already proved this from the frame archives; the probe re-checks at hash time);
  - deploy4 vs deploy3 hash differences per tick (allowed; task63 changed FIELD_SHAKE
    seeding from heap addresses to DataArray ids, and particle positions are part of
    the SaveGame image);
  - deploy4 vs deploy4 (two runs): must be identical everywhere (determinism restored);
  - deploy3 vs deploy3 (two runs): expected to differ wherever a FIELD_SHAKE particle
    has evolved -- the pre-task63 nondeterminism, which is exactly why deploy4 may
    disagree with the recorded archives' hashes.

Usage:
  python3 scripts/task70_particle_probe.py --gate artifacts/task70/gate \
      --old <deploy3> --new <deploy4> [--cells day,fog] [--games-per-cell 1] \
      [--out artifacts/task70/particle_probe.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_event_env import EventWaitEnv  # noqa: E402
from pvz_env import PlayerProfileContext, TaskSpec  # noqa: E402
from pvz_frame_store import read_frames  # noqa: E402

RESOURCE_DIR = Path('/home/newbiexvwu/.cache/pvz-research-resources')
MULT_PANEL = ROOT / 'artifacts/stage4/multiplier_panel.json'
CELLS = (('day', 'day_l7_train_d7_m125', MULT_PANEL),
         ('night', 'night_l14_train_n10_m125', MULT_PANEL),
         ('pool', 'pool_l24_train_p9k_m125', MULT_PANEL),
         ('conveyor', 'day_l10_orig_train_m125', MULT_PANEL),
         ('roof', 'roof_l41_train_r7', ROOT / 'artifacts/task46/roof_panel.json'),
         ('fog', 'fog_user_l31_snow', ROOT / 'artifacts/task68/fog_user_panel.json'))


def profile_from(meta_task):
    profile = meta_task.get('profile')
    if not profile:
        return PlayerProfileContext()
    if isinstance(profile, dict):
        return PlayerProfileContext(**profile)
    return PlayerProfileContext()


def task_spec_from(task_definition, seed):
    """Same construction as search_play's _play for the manifest task."""
    from pvz_tasks import profile_for_deck
    from render_episode import training_task_kwargs
    kw = training_task_kwargs(task_definition, seed)
    return TaskSpec(level=kw['level'], seed=kw['seed'], playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck']), kw.get('first_aid_owned', False)),
                    **{k: v for k, v in kw['task_extra'].items() if v is not None})


def state_hash_via_branch(env) -> str:
    """BRANCH_SNAPSHOT_FAST W:0 digest of the current state (works on both engines)."""
    parent = env.snapshot()
    try:
        branches = env.branch_snapshot(parent, [{'type': 'wait', 'ticks': 0}])
        record = branches[0]
        digest = record.get('state_hash')
        if record.get('snapshot_id'):
            env.release_snapshot(record['snapshot_id'])
        return str(digest) if digest is not None else ''
    finally:
        env.release_snapshot(parent)


def replay_and_hash(executable, frames, task_definition, seed, resource_dir=RESOURCE_DIR):
    """Replay one recorded action stream; return (reset hash, obs digests, state hashes)."""
    observations, hashes = [], []
    with EventWaitEnv(resource_dir, executable=executable, headless=True) as env:
        obs = env.reset(deck=task_definition['deck'], task=task_spec_from(task_definition, seed))[0]
        reset_hash = state_hash_via_branch(env)
        previous_tick = obs['tick']
        for frame in frames:
            tick = frame.get('tick')
            if tick is None:
                continue
            if tick > previous_tick:
                obs, _r, _done, _trunc, _info = env.step({'type': 'wait', 'ticks': tick - previous_tick})
                previous_tick = tick
            action = frame.get('action')
            if action is not None:
                obs, _r, _done, _trunc, _info = env.step(action)
            observations.append(hashlib.sha256(
                json.dumps(obs, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
            hashes.append(state_hash_via_branch(env))
    return reset_hash, observations, hashes


def compare(left, right):
    diffs = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    return {'ticks': min(len(left), len(right)), 'differ': len(diffs),
            'first_differing_index': diffs[0] if diffs else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate', type=Path, default=ROOT / 'artifacts/task70/gate')
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task70/particle_probe.json')
    p.add_argument('--cells', default='day,night,pool,conveyor,roof,fog')
    p.add_argument('--games-per-cell', type=int, default=1)
    args = p.parse_args()

    report = {'games': []}
    panel = {t['task_id']: t for t in json.loads(MULT_PANEL.read_text())['tasks']}
    source_panels = {}
    for cell, task_id, panel_path in CELLS:
        document = json.loads(Path(panel_path).read_text())
        source_panels[cell] = (next(t for t in document['tasks'] if t['task_id'] == task_id)
                               if task_id in {t['task_id'] for t in document['tasks']}
                               else panel[task_id])
    for cell in args.cells.split(','):
        # sample the student arm; its action stream is policy-determined and replayable
        task_definition = source_panels[cell]
        samples = sorted((args.gate / 'frames_old').glob('seed_*'))
        done = 0
        for seed_dir in samples:
            label = f'{cell}_fast_old'
            archive = seed_dir / f'{label}_fast_search'
            if not (archive / 'frames.npz').exists():
                continue
            seed = int(seed_dir.name[len('seed_'):])
            frames = read_frames(archive / 'frames.npz')
            reset_o1, obs_o1, hash_o1 = replay_and_hash(args.old, frames, task_definition, seed)
            reset_n1, obs_n1, hash_n1 = replay_and_hash(args.new, frames, task_definition, seed)
            reset_n2, obs_n2, hash_n2 = replay_and_hash(args.new, frames, task_definition, seed)
            reset_o2, obs_o2, hash_o2 = replay_and_hash(args.old, frames, task_definition, seed)
            game = {
                'cell': cell, 'seed': seed, 'label': label,
                'reset_state_hash': {'old_run1': reset_o1, 'old_run2': reset_o2,
                                     'new_run1': reset_n1, 'new_run2': reset_n2},
                'old_vs_new_observations': compare(obs_o1, obs_n1),
                'old_vs_new_state_hash': compare(hash_o1, hash_n1),
                'new_run2_observations': compare(obs_n1, obs_n2),
                'new_run2_state_hash': compare(hash_n1, hash_n2),
                'old_run2_observations': compare(obs_o1, obs_o2),
                'old_run2_state_hash': compare(hash_o1, hash_o2),
            }
            report['games'].append(game)
            print(json.dumps(game), flush=True)
            done += 1
            if done >= args.games_per_cell:
                break

    totals = {
        'old_vs_new_hash_ticks': sum(g['old_vs_new_state_hash']['differ'] for g in report['games']),
        'old_vs_new_hash_compared': sum(g['old_vs_new_state_hash']['ticks'] for g in report['games']),
        'new_self_hash_diff': sum(g['new_run2_state_hash']['differ'] for g in report['games']),
        'old_self_hash_diff': sum(g['old_run2_state_hash']['differ'] for g in report['games']),
        'observation_diffs_anywhere': sum(
            g[k]['differ'] for g in report['games']
            for k in ('old_vs_new_observations', 'new_run2_observations', 'old_run2_observations')),
    }
    report['totals'] = totals
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
