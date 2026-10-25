"""Re-score task72b's 94 frozen fog states with two value nets on identical candidates.

task72b recorded task/seed/decision/tick, the actual candidate set, v5's per-horizon values
and the terminal truth for wait and static_best. The rows carry no board state, so this
script replays the demo archive to the same decision, checks the tick identity, and reruns
the teacher search with each arm. Only the leaf value net changes; candidates and the common
future schedule are the search's own, so the arms are paired on identical states.

Workers own one game each; the caller starts at most four.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from task78_data import MAIN, sha
from value_teacher_feasibility import stats_new, write_json

MAIN_ROOT = Path('/home/newbiexvwu/PvZ-search')
ARMS = {}


def key(action):
    return 'wait' if action.get('type') == 'wait' else f"{action['row']},{action['col']}"


def search_args(horizon):
    from argparse import Namespace
    return Namespace(horizon=horizon, min_gap=300, per_card=1, samples=3, halving=True,
                     verify_samples=9, student_candidates=2, max_actions=1000000,
                     rollout_sim_skip=61)


def run_group(job):
    """One (task, decision) state: replay once, then score every row's futures with each arm."""
    task, decision, rows, horizon, arms = job
    import search_play as teacher
    from human_demo import replay
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import read_frames
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    try:
        demos = {t['task_id']: t.get('demo_start') for t in
                 json.loads((ROOT/'configs/human_demo_tasks.json').read_text())['tasks']}
        archive = MAIN_ROOT/demos[task]['archive']
        executable = ROOT/'build/pvz-portable'
        keep, bins = teacher._export_values(str(MAIN/'artifacts/value_net/v5/model.pt'),
                                           str(MAIN/'artifacts/hard/value_branch1/model.pt'))
        teacher._init_worker(bins[0], str(MAIN/'artifacts/student/dw4i_s1/model.pt'), 61, bins[1])
        teacher._W['env'] = env = EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources',
                                               executable=str(executable), headless=True)
        try:
            obs, meta, _ = replay(env, archive, decision=decision)
            frames = read_frames(archive)
            history = StudentHistory(teacher._W['fast'].obs)
            for frame in frames[:decision]:
                history.extra(frame)
            env.load_rollout_net(teacher._W['rollout_file'])
            out = []
            for row in rows:
                if int(obs['tick']) != int(row['tick']):
                    raise ValueError(f"{task} d{decision}: tick {obs['tick']} != {row['tick']}")
                teacher._W['student_history'] = history
                teacher._W['demo_actions'] = []
                teacher._W['context'] = observation_context(obs)
                values = {}
                for name, path in arms.items():
                    keep_arm, arm_bins = teacher._export_values(
                        path, str(MAIN/'artifacts/hard/value_branch1/model.pt'))
                    try:
                        env.load_value_net(arm_bins[0])
                        env.load_value_net(arm_bins[1], static=True)
                        teacher._W['context'] = observation_context(obs)
                        teacher._W['student_history'] = history
                        proposed = teacher._fast_action(obs, {})
                        stats = stats_new()
                        teacher._search(env, obs, proposed, search_args(horizon),
                                        int(row['seed']), stats)
                        values[name] = {key(a): v for a, v in stats['last_targets']}
                    finally:
                        keep_arm.cleanup()
                truth = row.get('truth') or {}
                static_cell = ','.join(str(v) for v in row['static_best'])
                pair = None
                if 'wait' in truth and static_cell in truth:
                    gap = truth[static_cell]['win_rate'] - truth['wait']['win_rate']
                    if abs(gap) >= 0.25:
                        pair = dict(true_better=static_cell if gap > 0 else 'wait', gap=gap)
                out.append(dict(task=task, decision=decision, tick=row['tick'],
                                wave=row['wave'], seed=row['seed'], horizon=horizon,
                                values=values, truth_pair=pair,
                                truth={k: v.get('win_rate') for k, v in truth.items()}))
            return dict(status='ok', decision=decision, rows=out)
        finally:
            teacher._close_worker_env()
            env.close()
            keep.cleanup()
    except Exception as exc:
        return dict(status='failed', decision=decision, reason=f'{type(exc).__name__}: {exc}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=2500)
    parser.add_argument('--arms', default='v5=/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt,'
                                         f'v6={ROOT}/artifacts/task78/v6/model.pt')
    parser.add_argument('--out', default=str(ROOT/'artifacts/task78/truth'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    arms = dict(spec.split('=', 1) for spec in args.arms.split(','))
    for path in arms.values():
        if not Path(path).is_file():
            raise ValueError(f'arm checkpoint missing: {path}')
    rows = [json.loads(line) for line in
            (ROOT/'MEMORY/docs/task72b-probe-rows.jsonl').read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task'], row['decision'])].append(row)
    jobs = [(task, decision, group, args.horizon, arms)
            for (task, decision), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    with mp.get_context('fork').Pool(args.workers) as pool:
        for result in pool.imap_unordered(run_group, jobs):
            results.append(result)
            print(json.dumps(dict(decision=result['decision'], status=result['status'],
                                  rows=len(result.get('rows', [])),
                                  reason=result.get('reason'))), flush=True)
            write_json(out/f"h{args.horizon}.json",
                       dict(horizon=args.horizon,
                            arms={k: dict(path=v, sha256=sha(v)) for k, v in arms.items()},
                            executable_sha256=sha(ROOT/'build/pvz-portable'),
                            groups=len(jobs), done=len(results),
                            wall_seconds=time.time()-started,
                            complete=len(results) == len(jobs),
                            passed=all(r['status'] == 'ok' for r in results),
                            results=results))
    failed = [r for r in results if r['status'] != 'ok']
    print(json.dumps(dict(groups=len(jobs), failed=len(failed), wall=time.time()-started)))


if __name__ == '__main__':
    main()
