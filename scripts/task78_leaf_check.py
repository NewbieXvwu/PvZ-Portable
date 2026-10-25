"""Real-engine check: the retained full leaf describes the same state as the scored leaf.

Production scores continuations inside C++ (ROLLOUT_VALUE) and returns only a value, no
frame, so the collector substitutes a full ROLLOUT at the same seed/tick and scores the
identical board itself. This checks the substitution is state-preserving and that the
retained frame encodes into finite encoder-v2 rows.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from task78_data import MAIN, ROOT, sha
from task78_replay import verify
from task78_collect import full_leaf_rollout
from value_teacher_feasibility import write_json
import search_play as teacher


def main():
    manifest = ROOT/'artifacts/task78/sources.json'
    source = json.loads(manifest.read_text())
    replay = json.loads((ROOT/'artifacts/task78/replay.json').read_text())
    if not replay['complete'] or replay['source_manifest_sha256'] != sha(manifest):
        raise ValueError('frozen archives must finish exact root replay first')
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
            frames = None

            def check(env, obs, i):
                from pvz_value_net import encode_frame, observation_context
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
                                scored = teacher._rollout_policy(env, obs, action,
                                                                 obs['tick']+horizon, root, 78299999)
                                scored_hash = env._debug_state_sha256()
                                full = full_leaf_rollout(env, obs, action, obs['tick']+horizon,
                                                         root, 78299999)
                                full_hash = env._debug_state_sha256()
                                if full_hash != scored_hash:
                                    raise ValueError(f'{group} h{horizon}: trajectory differs from production')
                                from render_episode import _frame
                                board, scalars = encode_frame(_frame(full), 2)
                                finite = bool(board.isfinite().all() and scalars.isfinite().all())
                                if not finite or tuple(board.shape) != (92, 6, 10):
                                    raise ValueError(f'{group} h{horizon}: encoded row invalid {board.shape} {finite}')
                                value = full['value']
                                if not (0.0 <= value <= 1.0):
                                    raise ValueError(f'{group} h{horizon}: leaf value out of range {value}')
                                results.append(dict(group=group, frame=i, horizon=horizon,
                                                    action=action, state_identical=True,
                                                    scored_value=scored['value'], leaf_value=value,
                                                    encoded_finite=True))
                    finally:
                        env.restore(root)
                        env.release_snapshot(root)
                if i+1 < len(frames):
                    state['action'] = frames[i+1]['action']

            from pvz_frame_store import read_frames
            frames = read_frames(Path(row['frames_path']))
            verify(row, ROOT/'build/pvz-portable', check)
            print(group+' passed', flush=True)
        write_json(ROOT/'artifacts/task78/leaf_channel_check.json',
                   dict(passed=True, source_manifest_sha256=sha(manifest),
                        executable_sha256=sha(ROOT/'build/pvz-portable'),
                        baseline_sha256=sha(MAIN/'artifacts/value_net/v5/model.pt'),
                        cases=results))
        print('leaf channel check passed: %d cases' % len(results), flush=True)
    finally:
        keep.cleanup()


if __name__ == '__main__':
    main()
