"""Verify frozen qc roots against the unmodified task78 engine, one game at a time."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from task78_data import sha


def physical(frame):
    annotations = {'action', 'label', 'targets', 'verification', 'action_rejected', 'eaten'}
    return {k: v for k, v in frame.items() if k not in annotations}


def equal(actual, expected):
    # Archives serialize tuples as lists, so compare canonical JSON values.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def open_archive(row, executable):
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from human_demo import replay
    env = EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources',
                       executable=str(executable), headless=True)
    try:
        meta = json.loads((Path(row['archive'])/'meta.json').read_text())
        if row['demo_archive']:
            obs, demo_meta, _ = replay(env, Path(row['demo_archive']), decision=row['demo_decision'])
            if row['world']['future_seed'] is None:
                raise ValueError('demo continuation missing future seed')
            env.reseed(row['world']['future_seed'])
            # Reseeding changes hidden future waves; public current state is unchanged.
            verification = demo_meta['hash_verification']
        else:
            task = TaskSpec(level=row['level'], seed=row['world']['seed'], playthrough=2,
                            profile=profile_for_deck(tuple(row['deck']),
                                meta['task'].get('first_aid_owned', False)),
                            zombie_count_multiplier=row['multiplier'],
                            fog_observation=meta['task'].get('fog_observation'))
            obs, _ = env.reset(deck=tuple(row['deck']), task=task)
            verification = 'cold reset from archived seed/deck/multiplier/fog mode'
        return env, obs, verification
    except BaseException:
        env.close()
        raise


def verify(row, executable, observe_input=None):
    from pvz_frame_store import read_frames
    from render_episode import _frame
    archive = Path(row['archive'])
    if sha(row['frames_path']) != row['frames_sha256'] or sha(archive/'meta.json') != row['meta_sha256']:
        raise ValueError('source identity changed')
    frames = read_frames(Path(row['frames_path']))
    stop = max(r['frame'] for r in row['roots'])
    roots = []
    env, obs, verification = open_archive(row, executable)
    try:
        for i in range(stop+1):
            actual, expected = physical(_frame(obs)), physical(frames[i])
            if not equal(actual, expected):
                fields = [k for k in actual.keys() | expected.keys()
                          if not equal(actual.get(k), expected.get(k))]
                raise ValueError(f'frame {i} tick {obs["tick"]}: physical mismatch {fields}')
            if observe_input is not None:
                observe_input(env, obs, i)
            if i in {r['frame'] for r in row['roots']}:
                roots.append(dict(frame=i, tick=obs['tick'], engine_sha256=env._debug_state_sha256()))
            if i == stop:
                break
            if frames[i+1].get('targets'):
                root = env.snapshot_fast()
                env.restore(root)
                env.release_snapshot(root)
            obs, *_, info = env.step(frames[i+1]['action'])
            if (not info.get('ok')) != bool(frames[i+1].get('action_rejected')):
                raise ValueError(f'frame {i+1}: action acceptance differs')
            if not info.get('ok'):
                obs, *_ = env.step(dict(type='wait', ticks=60))
        return dict(archive=row['archive'], group=row['group'], status='verified',
                    frames=stop+1, roots=roots, start_verification=verification)
    finally:
        env.close()


def main():
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'artifacts/task78/sources.json'
    output = manifest.with_name(manifest.name.replace('sources', 'replay'))
    rows = json.loads(manifest.read_text())['archives']
    executable = ROOT/'build/pvz-portable'
    results = []
    started = time.time()
    executable_sha = sha(executable)
    with mp.get_context('fork').Pool(4) as pool:
        for result in pool.imap_unordered(verify_job, [(row, executable) for row in rows]):
            if sha(executable) != executable_sha:
                raise ValueError('engine changed during root verification')
            results.append(result)
            print(json.dumps(result), flush=True)
            target = output
            target.write_text(json.dumps(dict(source_manifest_sha256=sha(manifest),
                executable=str(executable), executable_sha256=executable_sha, workers=4,
                complete=len(results)==len(rows),
                passed=len(results)==len(rows) and all(r['status']=='verified' for r in results),
                wall_seconds=time.time()-started, results=results), indent=1)+'\n')


def verify_job(job):
    row, executable = job
    try:
        return verify(row, executable)
    except Exception as exc:
        return dict(archive=row['archive'], group=row['group'], status='failed', reason=str(exc))


if __name__ == '__main__':
    main()
