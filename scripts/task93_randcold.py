"""Finish the preregistered 32-deck endpoint after the expanded dose pipeline exits.

Keeps the main experiment lock throughout evaluation. Existing complete runs are
verified and reused; only the two expanded models may launch new games (workers=2).
"""
import argparse
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import select
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import task93_validation as queue
from task93_analyze import comparison, named_runs


def verify_config(path, template, model=None):
    actual = json.loads(path.read_text())
    if {k: v for k, v in actual.items() if k not in ('rollout_policy', 'out')} != {
            k: v for k, v in template.items() if k not in ('rollout_policy', 'out')}:
        raise ValueError(f'supplementary evaluation settings differ: {path}')
    if model is not None and actual['rollout_policy'] != str(model):
        raise ValueError(f'supplementary evaluation model differs: {path}')


def summarize(paths, task, template):
    seeds = task['seeds']
    decks = {s: queue.deck_for_seed(task, s) for s in seeds}
    for path in paths.values():
        verify_config(path.with_name('config.json'), template)
    runs, sources = named_runs([f'{name}={path}' for name, path in paths.items()], seeds, decks, task)
    for name, source in sources.items():
        model = Path(json.loads(paths[name].with_name('config.json').read_text())['rollout_policy'])
        source.update(model=str(model), model_sha256=queue.sha(model))
    return dict(task_id=task['task_id'], total=len(seeds),
        wins={name: sum(row['won'] for row in run.values()) for name, run in runs.items()},
        sources=sources,
        paired={f'{a}_minus_{b}': comparison(runs[a], runs[b], seeds) for a, b in combinations(runs, 2)},
        interpretation='Descriptive, 32 card sets with one environment seed each; not an admission criterion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/task93/expand')
    parser.add_argument('--wait-for-pid', type=int)
    args = parser.parse_args()
    if args.wait_for_pid:
        handle = os.pidfd_open(args.wait_for_pid)
        try:
            print(f'waiting for dose controller {args.wait_for_pid}', flush=True)
            while not select.select([handle], [], [], 10)[0]:
                pass
        finally:
            os.close(handle)
    with (args.root/'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads((args.root/'validation_status.json').read_text())['phase'] != 'complete':
            raise ValueError('dose pipeline did not complete; no supplementary games launched')
        queue.DATA, queue.LOCK_FD = args.root, lock.fileno()
        for seed in (1, 2):
            if not queue.complete_model(seed):
                raise ValueError(f'model {seed} has no valid completion proof')
        folder = args.root/'randcold'
        folder.mkdir(exist_ok=True)
        template = json.loads((folder/'seed_1_0173/config.json').read_text())
        task = next(t for t in json.loads(Path(template['manifest']).read_text())['tasks']
                    if t['task_id'] == template['task_id'])
        if len(task['seeds']) != 32 or len({tuple(sorted(queue.deck_for_seed(task, s))) for s in task['seeds']}) != 32:
            raise ValueError('expected 32 distinct preregistered card sets')
        queue.DATA = folder
        for seed in (1, 2):
            model = args.root/f'seed_{seed}/student/model.pt'
            output = folder/f'seed_{seed}_0875'
            games = output/'games.jsonl'
            if games.exists():
                verify_config(games.with_name('config.json'), template, model)
            if queue.complete_evaluation(games, task, model):
                continue
            queue.run([sys.executable, str(ROOT/'scripts/search_play.py'), '--rollout-policy', str(model),
                '--value', template['value'], '--manifest', template['manifest'], '--task-id', task['task_id'],
                '--executable', template['executable'], '--resource-dir', template['resource_dir'],
                '--seeds', '32', '--workers', '2', '--arms', 'fast', '--dagger-beta', '0', '--out', str(output)],
                f'eval_seed_{seed}_0875')
            verify_config(output/'config.json', template, model)
            if not queue.complete_evaluation(games, task, model):
                raise ValueError('supplementary evaluation incomplete')
        paths = {name: folder/name/'games.jsonl' for name in
                 ('r11', 'seed_1_0173', 'seed_2_0173', 'seed_1_0875', 'seed_2_0875')}
        report = summarize(paths, task, template)
        report['complete'] = True
        (folder/'comparison_all_doses.json').write_text(json.dumps(report, indent=2)+'\n')
        queue.status('complete')


if __name__ == '__main__':
    main()
