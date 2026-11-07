"""task251：只读生产引擎，用现有 SEARCH/ROLLOUT 记录升级链诊断，可按组续跑。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/task251'
WORK = DATA / 'work'
WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  VECLIB_MAXIMUM_THREADS='1', TMPDIR=str(WORK))
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import search_play

DECK29 = [1, 16, 35, 10, 42, 39, 44, 30, 21, 17]
DECK22 = [1, 16, 35, 10, 42, 36, 17, 4, 19, 21]
WATCH = {10, 35, 39, 42, 44}
original_search = search_play._search
probe_counts = Counter()


def diagnostic_search(env, obs, policy, args, decision_seed, stats):
    """记录根及原始响应，少量重用 C++ 续走取得每张关键卡的后续种植计数。"""
    seed, decision = divmod(decision_seed, 10000)
    directory = Path(args.out)
    raw_search = env.search
    response = None
    probes = []

    def record_search(root, action, **kwargs):
        nonlocal response
        response = raw_search(root, action, **kwargs)
        # 每卡首次候选，以及波次>=10后的首次候选，各抽一个共享未来。
        # 这批探针只取证，不向 SEARCH 回写动作或分数。
        if response['searched']:
            deck = json.loads(args.task_json)['deck']
            for packet, row, col, value, upgrade, base in response.get('targets', []):
                if packet < 0 or upgrade or deck[packet] not in WATCH:
                    continue
                plant = deck[packet]
                phase = 0 if probe_counts[(seed, plant, 0)] == 0 else 1
                if probe_counts[(seed, plant, phase)] or (phase == 1 and obs['wave'] < 10):
                    continue
                probe_counts[(seed, plant, phase)] += 1
                first = dict(type='plant', packet=packet, row=row, col=col)
                leaf = env.rollout(first, obs['tick'] + args.horizon,
                    snapshot_id=root, seed=(decision_seed * 1000) % 2**32,
                    sim_skip=args.rollout_sim_skip, student_history=kwargs['student_history'])
                details = env._last_response
                leaf_value = (float(leaf['result'] == 1) if leaf['terminal']
                              else env.leaf_values()[0])
                probes.append(dict(plant=plant, phase=phase, first=first,
                    future=(decision_seed * 1000) % 2**32, search_mean=value,
                    leaf_value=leaf_value, tick=leaf['tick'], wave=leaf['wave'],
                    terminal=leaf['terminal'], result=leaf['result'],
                    decisions=details['decisions'], planted_types=details['planted_types'],
                    plants=leaf['plants'], sun=leaf.get('sun')))
        return response

    env.search = record_search
    try:
        chosen = original_search(env, obs, policy, args, decision_seed, stats)
    finally:
        env.search = raw_search
    legal = (obs.get('legal_actions') or {}).get('plants') or []
    record = dict(seed=seed, decision=decision, tick=obs['tick'], wave=obs['wave'],
        sun=obs.get('sun'), packets=obs['packets'], plants=obs['plants'],
        zombies=obs['zombies'], defenses=obs.get('defenses'),
        legal_counts=dict(Counter(a['packet'] for a in legal)),
        policy=policy, chosen=chosen, response=response, probes=probes)
    with (directory / f'decisions_{seed}.jsonl').open('w' if decision == 0 else 'a') as stream:
        stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', default='l29_m025,l29_m050,l22_m050,l29_student')
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--student', type=Path,
        default=Path.home() / 'pvz-mac-collect/models/student_d65f4c4f6477.pt')
    options = parser.parse_args()
    models = Path.home() / 'pvz-mac-collect/models'
    engine = Path.home() / 'pvz-mac-collect/build2/pvz-portable'
    inputs = {str(p): dict(bytes=p.stat().st_size,
              sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in [engine, options.student, models/'value.pt', models/'static.pt', models/'gptbc.pt']}
    (DATA / 'inputs.json').write_text(json.dumps(inputs, indent=1) + '\n')
    search_play._search = diagnostic_search
    for group in options.groups.split(','):
        level = 22 if group.startswith('l22') else 29
        multiplier = .25 if group.endswith('m025') else .5
        seeds = list(range(243002000, 243002004)) if level == 22 else list(range(251029000, 251029008))
        task = dict(task_id=group, level=level, deck=DECK22 if level == 22 else DECK29,
                    seeds=seeds, pairs=len(seeds), zombie_count_multiplier=multiplier)
        directory = DATA / group
        directory.mkdir(exist_ok=True)
        sys.argv = [__file__, '--rollout-policy', str(options.student),
            '--rollout-policy-gpt', str(models/'gptbc.pt'), '--value', str(models/'value.pt'),
            '--static-value', str(models/'static.pt'), '--manifest', str(DATA/'unused.json'),
            '--task-json', json.dumps(task), '--task-id', group, '--out', str(directory),
            '--workers', str(options.workers), '--seeds', str(len(seeds)),
            '--executable', str(engine), '--resource-dir', str(Path.home()/'.cache/pvz-research-resources'),
            '--arms', 'fast' if group.endswith('student') else 'fast_search',
            '--dagger-beta', '1', '--native-games', '--per-card', '1',
            '--student-candidates', '2', '--samples', '3', '--verify-samples', '9',
            '--halving', '--halving-first-samples', '1', '--horizon', '2500',
            '--min-gap', '300', '--rollout-sim-skip', '61',
            '--frames', str(DATA/'frames'/group), '--frames-label', group]
        print('RUN', group, flush=True)
        search_play.main()


if __name__ == '__main__':
    main()
