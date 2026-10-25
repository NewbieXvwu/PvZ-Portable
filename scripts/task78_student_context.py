"""Restore and audit the public student history at archived task78 roots.

This follows search_play._play's demonstration initialization. It never starts an
engine or reads outcomes; a physical replay alone cannot establish history identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]

from pvz_frame_store import frame_path, read_frames
from pvz_student_observation import StudentHistory
from task78_data import sha


def context_identity(row):
    """Seal the demonstration input and initializer used by newly collected jobs."""
    if not row.get('demo_archive'):
        return None
    return dict(mode='search_play_demo_prefix_v1',
                initializer_sha256=sha(Path(__file__)),
                frames_sha256=sha(frame_path(Path(row['demo_archive']))),
                decision=row['demo_decision'])


def initial_state(row, obs):
    """The state immediately before the first archived continuation input."""
    if not row.get('demo_archive'):
        return {}
    frames = read_frames(frame_path(Path(row['demo_archive'])))
    decision = row['demo_decision']
    if type(decision) is not int or not 1 <= decision <= len(frames):
        raise ValueError('invalid demonstration prefix length')
    prefix = frames[:decision]
    state = dict(frame=prefix[-2] if len(prefix) > 1 else None,
                 action=prefix[-1].get('action'))
    if obs != 'v1':
        state['history'] = StudentHistory(obs)
        for frame in prefix[:-1]:
            state['history'].extra(frame)
    return state


def expected_histories(row, obs):
    """History after the student's feature update at each selected input root."""
    from value_teacher_feasibility import history_record
    if obs == 'v2fogstack4':
        raise ValueError('legacy root records omit the temporal stack; cannot certify it')
    frames = read_frames(Path(row['frames_path']))
    if sha(row['frames_path']) != row['frames_sha256']:
        raise ValueError('source continuation frames changed')
    roots = {r['frame'] for r in row['roots']}
    if not roots or min(roots) < 0 or max(roots) >= len(frames):
        raise ValueError('invalid selected root indices')
    state = initial_state(row, obs)
    if obs != 'v1' and 'history' not in state:
        state['history'] = StudentHistory(obs)
    result = {}
    for i, frame in enumerate(frames[:max(roots)+1]):
        if state.get('history') is not None:
            state['history'].extra(frame)
        if i in roots:
            # On all but the initial input, previous frame/action are reconstructed
            # from the same recorded continuation. Root history is serialized after
            # _fast_action, so its previous_frame is already the current input.
            result[i] = history_record(state)['history']
    return result


def validate_root_history(row, frame, root, expected=None):
    """Reject labels whose student observation differs despite equal board state."""
    saved = root['history']['history']
    obs = saved['obs'] if saved is not None else 'v1'
    expected = expected_histories(row, obs) if expected is None else expected
    if saved != expected[frame]:
        raise ValueError(f'student history differs from source prefix at frame {frame}: '
                         f'saved={saved}, expected={expected[frame]}')
    # The old collector lacked the preceding frame/action only at input zero.
    # Later inputs have both from the archived continuation. Do not silently
    # certify an initial demonstration root from its post-action record.
    if row.get('demo_archive') and frame == 0:
        raise ValueError('initial demo root needs an explicit pre-action context receipt')


def audit(manifest, labels):
    source = json.loads(manifest.read_text())
    checks = []
    for index, row in enumerate(source['archives']):
        if not row.get('demo_archive'):
            continue
        per_obs = {}
        prefix_path = frame_path(Path(row['demo_archive']))
        for horizon in (2500, 4000, 6000):
            job = labels/f'archive_{index:02d}/h{horizon}'
            for selected in row['roots']:
                frame = selected['frame']
                path = job/f'root_{frame}/root.json'
                if not path.is_file():
                    checks.append(dict(index=index, horizon=horizon, frame=frame,
                                       group=row['group'], passed=False, reason='missing root'))
                    continue
                root = json.loads(path.read_text())
                saved = root['history']['history']
                obs = saved['obs'] if saved is not None else 'v1'
                if obs not in per_obs:
                    per_obs[obs] = expected_histories(row, obs)
                reason = None
                try:
                    validate_root_history(row, frame, root, per_obs[obs])
                except ValueError as error:
                    reason = str(error)
                checks.append(dict(index=index, horizon=int(job.name[1:]), frame=frame,
                    group=row['group'], passed=reason is None, reason=reason,
                    saved=saved, expected=per_obs[obs][frame], root_sha256=sha(path),
                    demo_frames_sha256=sha(prefix_path), demo_decision=row['demo_decision']))
    return dict(source_manifest_sha256=sha(manifest), engine_processes_started=0,
                checked_roots=len(checks), failed_roots=sum(not c['passed'] for c in checks),
                passed=bool(checks) and all(c['passed'] for c in checks), checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'artifacts/task78/sources.json')
    parser.add_argument('--labels', type=Path, default=ROOT/'artifacts/task78/labels')
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/task78/student_context_audit.json')
    args = parser.parse_args()
    result = audit(args.manifest, args.labels)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
