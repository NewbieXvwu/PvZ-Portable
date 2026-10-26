"""Label one frozen qc archive at one horizon, using task43's audited teacher channel.

Usage: absolute script path ARCHIVE_INDEX HORIZON. Each process owns one game;
the caller may start at most two (the amended task cap). No production defaults change.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from task78_data import MAIN, sha
from task78_state import state_digest
from task78_replay import equal, open_archive, physical
from task78_student_context import initial_state, context_identity
from value_teacher_feasibility import sample_root, stats_new, write_json
import search_play as teacher


def full_leaf_rollout(env, obs, first, until_tick, root=None, seed=None):
    """Diagnostic equivalent of compact ROLLOUT_VALUE, retaining task43's full leaf.

    Production now omits leaf features/income on the scored path. The collector
    needs the full ROLLOUT channel and scores the identical board with LEAF_VALUES.
    """
    history = teacher._W.get('student_history') if teacher._W['fast'].obs != 'v1' else None
    leaf = env.rollout(first, until_tick, sim_skip=teacher._W.get('rollout_sim_skip', 0),
                       snapshot_id=root, seed=seed, student_history=history)
    response = dict(env._last_response)
    metrics = teacher._W.setdefault('rollout_metrics', {})
    for key in ('decisions', 'simulation_seconds', 'network_seconds'):
        metrics[key] = metrics.get(key, 0) + response[key]
    leaf['value'] = env.leaf_values(income=leaf['sun_income_rate'])[0]
    return leaf


def paths(manifest):
    """Derive the replay record and label directory that belong to one frozen manifest."""
    name = Path(manifest).name
    replay = Path(manifest).with_name(name.replace('sources', 'replay'))
    if Path(manifest).parent!=ROOT/'artifacts/task78':
        labels=Path(manifest).parent/'labels'
    else:
        labels = ROOT/('artifacts/task78/labels_old' if name.startswith('old') else 'artifacts/task78/labels')
    return replay, labels


def main():
    index, horizon = map(int, sys.argv[1:3])
    source = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT/'artifacts/task78/sources.json'
    manifest = json.loads(source.read_text())
    if not 0 <= index < len(manifest['archives']) or horizon not in (2500, 4000, 6000):
        raise ValueError(f'requires archive index 0..{len(manifest["archives"])-1} and horizon 2500/4000/6000')
    row = manifest['archives'][index]
    replay_path, label_root = paths(source)
    replay = json.loads(replay_path.read_text())
    if not replay['passed'] or replay['source_manifest_sha256'] != sha(source):
        raise ValueError('all frozen archives must pass exact root replay before labelling')
    executable = ROOT/'build/pvz-portable'
    if replay['executable_sha256'] != sha(executable):
        raise ValueError('engine differs from replay verification')
    output = label_root/f'archive_{index:02d}/h{horizon}'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = dict(pid=os.getpid(), started=time.time(), source_manifest_sha256=sha(source),
                   source=row, horizon=horizon, executable_sha256=sha(executable),
                   teacher_source_sha256=sha(ROOT/'scripts/search_play.py'),
                   label_channel_sha256=sha(ROOT/'scripts/value_teacher_feasibility.py'))
        models = dict(value=MAIN/'artifacts/value_net/v5/model.pt',
                      rollout_policy=MAIN/'artifacts/student/dw4i_s1/model.pt',
                      static_value=MAIN/'artifacts/hard/value_branch1/model.pt')
        run['models'] = {k:dict(path=str(p), sha256=sha(p)) for k,p in models.items()}
        if row.get('demo_archive'):
            run['student_context'] = context_identity(row)
        identity = {k:v for k,v in run.items() if k not in ('pid', 'started')}
        if (output/'identity.json').exists():
            if json.loads((output/'identity.json').read_text()) != identity:
                raise ValueError('previous job identity changed')
        else:
            write_json(output/'identity.json', identity)
        write_json(output/'run.json', run)
        keep, bins = teacher._export_values(str(models['value']), str(models['static_value']))
        teacher._init_worker(bins[0], str(models['rollout_policy']), 61, bins[1])
        task_id = 'task78_archived_source'
        write_json(output/'archive_task.json', dict(tasks=[dict(task_id=task_id,
            level=row['level'], deck=row['deck'], seeds=[row['world']['seed']],
            zombie_count_multiplier=row['multiplier'])]))
        args = teacher.parser().parse_args(['--out', str(output), '--value', str(models['value']),
            '--rollout-policy', str(models['rollout_policy']), '--static-value', str(models['static_value']),
            '--manifest', str(output/'archive_task.json'), '--task-id', task_id,
            '--verify-samples', '9', '--student-candidates', '2', '--horizon', str(horizon),
            '--executable', str(executable), '--resource-dir', '/home/newbiexvwu/.cache/pvz-research-resources'])
        env = None
        try:
            from pvz_frame_store import read_frames
            from pvz_value_net import observation_context
            from render_episode import _frame
            if sha(row['frames_path']) != row['frames_sha256']:
                raise ValueError('archive frames changed')
            frames = read_frames(Path(row['frames_path']))
            teacher._W['demo_actions'] = []
            if row['demo_archive']:
                from pvz_frame_store import frame_path
                prefix = read_frames(frame_path(Path(row['demo_archive'])))
                teacher._W['demo_actions'] = [(before['tick'], after['action'])
                    for before, after in zip(prefix, prefix[1:])
                    if (after.get('action') or {}).get('type') == 'plant']
            roots = sorted({r['frame'] for r in row['roots']})
            results = []
            for position, target in enumerate(roots):
                # One fresh engine per root. A search at an earlier root leaves hidden state
                # that SNAPSHOT does not restore, so a later root in the same process can
                # differ from the frozen replay (roof L42/L43, 2026-10-26). Replaying the
                # prefix in a new process reproduces the verified root bit for bit.
                env, obs, _ = open_archive(row, executable)
                try:
                    teacher._W['env'] = env
                    env.load_rollout_net(teacher._W['rollout_file'])
                    env.load_value_net(bins[0])
                    env.load_value_net(bins[1], static=True)
                    teacher._W['context'] = observation_context(obs)
                    state = initial_state(row, teacher._W['fast'].obs)
                    for i in range(target+1):
                        if not equal(physical(_frame(obs)), physical(frames[i])):
                            raise ValueError(f'prefix changed at input frame {i}')
                        teacher._fast_action(obs, state)
                        teacher._W['student_history'] = state.get('history')
                        if i == target:
                            # A fresh search at the frozen source root. Common world/noise
                            # schedules are identical across the three horizons.
                            decision_seed = int(manifest.get('seed_base', 78200000)) + index*100 + position
                            formal = teacher._rollout_policy
                            try:
                                teacher._rollout_policy = full_leaf_rollout
                                _, summary = sample_root(env, obs, deepcopy(state), args, decision_seed,
                                                          stats_new(), output/f'root_{i}')
                            finally:
                                teacher._rollout_policy = formal
                            # sample_root leaves the engine exactly at the root, so this is
                            # the same instant as root_state_sha256, minus the never-written
                            # GridItem field that a cold replay cannot reproduce.
                            root_path = output/f'root_{i}'/'root.json'
                            record = json.loads(root_path.read_text())
                            record['root_state_canonical_sha256'] = state_digest(env.privileged_state())
                            write_json(root_path, record)
                            results.append(dict(frame=i, **summary))
                            print(json.dumps(results[-1]), flush=True)
                            break
                        if frames[i+1].get('targets'):
                            root = env.snapshot_fast(); env.restore(root); env.release_snapshot(root)
                        action = frames[i+1]['action']; state['action'] = action
                        obs, *_, info = env.step(action)
                        if not info.get('ok'):
                            if not frames[i+1].get('action_rejected'):
                                raise ValueError('prefix action unexpectedly rejected')
                            obs, *_ = env.step(dict(type='wait', ticks=60))
                finally:
                    env.close()
                    env = None
                    teacher._W['env'] = None
            write_json(output/'complete.json', dict(identity=identity, roots=results,
                finished=time.time(), wall_seconds=time.time()-run['started']))
        finally:
            teacher._close_worker_env()
            if env is not None:
                env.close()
            keep.cleanup()


if __name__ == '__main__':
    main()
