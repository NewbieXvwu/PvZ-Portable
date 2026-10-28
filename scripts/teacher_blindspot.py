"""task120：取消绝对增益保护区，沿用原搜索与独立未来复核。"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault('PVZ_NO_POPUP', '1')

import search_play as sp


def play(job):
    sp.MARGIN = 0.0
    result = sp._play(job)
    result.update(task_id=job['args'].task_id, variant='B')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    jobs, baseline, configs = [], [], {}
    for task_id in ('task118_l33_cactus', 'task118_l34_cactus', 'task118_l39_control'):
        source = options.source / 'runs' / task_id
        config = json.loads((source / 'config.json').read_text())
        flags = []
        for name in ('rollout_policy', 'value', 'static_value', 'manifest', 'task_id',
                     'resource_dir', 'executable', 'per_card', 'student_candidates',
                     'samples', 'horizon', 'verify_samples', 'min_gap', 'rollout_sim_skip'):
            flags.extend(['--' + name.replace('_', '-'), str(config[name])])
        args = sp.parser().parse_args(flags + ['--out', str(options.out / task_id)])
        args.halving = str(config['halving']).lower() == 'true'
        games = [json.loads(line) for line in (source / 'games.jsonl').read_text().splitlines()]
        baseline.extend(dict(row, task_id=task_id, variant='A') for row in games)
        jobs.extend(dict(args=args, seed=row['seed'], arm='fast_search') for row in games)
        configs[task_id] = dict(config, margin=0.0, frames=None, dagger_beta=1.0,
                               out=str(options.out), workers=options.workers,
                               source_config=str(source / 'config.json'))
    (options.out / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
    (options.out / 'config.json').write_text(json.dumps(configs, indent=2) + '\n')
    keep, values = sp._export_values(config['value'], config['static_value'])
    try:
        sp._load_nets(values[0], config['rollout_policy'], values[1])
        sp._freeze_startup()
        started = time.monotonic()
        with (options.out / 'games.jsonl').open('w') as stream, mp.get_context('fork').Pool(
                options.workers, initializer=sp._init_worker,
                initargs=(values[0], config['rollout_policy'], args.rollout_sim_skip, values[1])) as pool:
            for completed, result in enumerate(pool.imap_unordered(play, jobs), 1):
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(f"{completed}/{len(jobs)} {result['task_id']} {result['seed']} "
                      f"won={result['won']} {time.monotonic() - started:.0f}s", flush=True)
            pool.close()
            pool.join()
    finally:
        keep.cleanup()
        if sp._W.get('rollout_tmp'):
            sp._W['rollout_tmp'].cleanup()


if __name__ == '__main__':
    main()
