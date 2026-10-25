"""Real-engine check: retained full leaves have production compact scores and state."""
from __future__ import annotations

from copy import deepcopy
import json

from task78_data import MAIN, ROOT, sha
from task78_replay import verify
from task78_collect import full_leaf_rollout
from value_teacher_feasibility import stats_new, write_json
import search_play as teacher


def main():
    manifest = ROOT/'artifacts/task78/sources.json'
    source = json.loads(manifest.read_text())
    replay = json.loads((ROOT/'artifacts/task78/replay.json').read_text())
    if not replay['complete']:
        raise ValueError('wait for the four replay workers before opening another game')
    selected = {}
    for row in source['archives']:
        selected.setdefault(row['group'], row)
    keep, bins = teacher._export_values(str(MAIN/'artifacts/value_net/v5/model.pt'),
                                       str(MAIN/'artifacts/hard/value_branch1/model.pt'))
    teacher._init_worker(bins[0], str(MAIN/'artifacts/student/dw4i_s1/model.pt'), 61, bins[1])
    results = []
    try:
        for group, row in selected.items():
            state = {}
            def check(env, obs, i):
                from pvz_value_net import observation_context
                teacher._W['context'] = observation_context(obs)
                if i == 0:
                    env.load_rollout_net(teacher._W['rollout_file'])
                    env.load_value_net(bins[0])
                    env.load_value_net(bins[1], static=True)
                proposal = teacher._fast_action(obs, state)
                teacher._W['student_history'] = deepcopy(state.get('history'))
                if i == row['roots'][0]['frame']:
                    root = env.snapshot()
                    try:
                        for horizon in (2500, 4000, 6000):
                            for action in (proposal, teacher.NOOP):
                                scored = teacher._rollout_policy(env, obs, action, obs['tick']+horizon,
                                                                 root, 78299999)
                                state_hash = env._debug_state_sha256()
                                full = full_leaf_rollout(env, obs, action, obs['tick']+horizon,
                                                         root, 78299999)
                                error = abs(full['value']-scored['value'])
                                same = state_hash == env._debug_state_sha256()
                                if not same or error > 1e-6:
                                    raise ValueError(f'{group} h{horizon}: state={same}, score error={error}')
                                results.append(dict(group=group, frame=i, horizon=horizon, action=action,
                                                    state_identical=same, value_error=error))
                    finally:
                        env.restore(root)
                        env.release_snapshot(root)
                # Reconstruct the student's history along the archived executed prefix.
                from pvz_frame_store import read_frames
                # Frames are loaded outside the callback; no archive or cache writes.
                if i+1 < len(frames):
                    state['action'] = frames[i+1]['action']
            from pathlib import Path
            from pvz_frame_store import read_frames
            frames = read_frames(Path(row['frames_path']))
            verify(row, ROOT/'build/pvz-portable', check)
            print(group+' passed', flush=True)
        write_json(ROOT/'artifacts/task78/leaf_channel_check.json',
                   dict(passed=True, source_manifest_sha256=sha(manifest),
                        executable_sha256=sha(ROOT/'build/pvz-portable'), cases=results))
    finally:
        keep.cleanup()


if __name__ == '__main__':
    main()
