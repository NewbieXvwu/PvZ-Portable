"""任务127：冻结第22轮标签子集，训练并比较四个种子的EMA。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pvz_recipe import materialize_recipe, repository_root

VARIANTS = dict(base=0.0, ema=0.999)
TASKS = ('day_l7_eval_d7', 'day_l7_eval_d7p_holdout1')


def prepare(options):
    subset = json.loads((options.subset / 'subset.json').read_text())
    source = json.loads((options.subset / 'source_recipe.json').read_text())
    (options.out / 'subset.json').write_text(json.dumps(subset, indent=2) + '\n')
    for seed in range(1, 5):
        for variant, decay in VARIANTS.items():
            name = f'{variant}_s{seed}'
            materialize_recipe(options.subset / 'source_recipe.json', options.out / 'recipes' / f'{name}.json',
                data=options.subset, labels=subset['labels'],
                label_weight={key: value for key, value in source['label_weight'].items() if key in subset['labels']},
                out=options.out / 'models' / name, device='cpu', threads=3, precompute_workers=1,
                cache_dir=options.out / 'cache', read_cache_dir=None, seed=seed, ema_decay=decay)
    panel = json.loads((ROOT / 'configs/adventure_scoreboard.json').read_text())
    original = next(task for task in panel['tasks'] if task['level'] == 7)
    original = dict(original, task_id=TASKS[0], seeds=list(range(103062048, 103062080)))
    identity = json.loads((repository_root() / 'artifacts/stage4/day_panel_identity.json').read_text())
    heldout = next(task for task in identity['tasks'] if task['task_id'] == TASKS[1])
    heldout = dict(heldout, seeds=list(range(103063072, 103063104)))
    (options.out / 'eval_manifest.json').write_text(json.dumps(dict(schema_version=1,
        tasks=[original, heldout]), indent=2) + '\n')


def one(options):
    import torch
    import train_rollout_policy as trainer
    torch.set_num_interop_threads(1)
    subset = json.loads((options.subset / 'subset.json').read_text())
    paths = [options.subset / 'segments' / entry['name'] for entry in subset['segments']]

    def load(data, **kwargs):
        return trainer.read_segments(paths, data, row_observer=kwargs.get('row_observer'),
                                     source_weight=kwargs.get('source_weight'))

    trainer.load = load
    sys.argv = ['train_rollout_policy.py', '--recipe',
                str(options.out / 'recipes' / f'{options.variant}_s{options.seed}.json')]
    trainer.main()


def train(options):
    for seed in range(1, 5):
        for variant in VARIANTS:
            name = f'{variant}_s{seed}'
            if (options.out / 'models' / name / 'model.pt').is_file():
                continue
            with (options.out / f'{name}.log').open('w') as stream:
                subprocess.run([sys.executable, __file__, '--phase', 'one', '--subset',
                    str(options.subset), '--out', str(options.out), '--seed', str(seed), '--variant', variant],
                    stdout=stream, stderr=subprocess.STDOUT, check=True)
            print(f'trained {name}', flush=True)


def evaluate(options):
    for seed in range(1, 5):
        for variant in VARIANTS:
            name = f'{variant}_s{seed}'
            for task in TASKS:
                directory = options.out / 'eval' / name / task
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
                    '--rollout-policy', str(options.out / 'models' / name / 'model.pt'),
                    '--manifest', str(options.out / 'eval_manifest.json'), '--task-id', task,
                    '--seeds', '32', '--workers', '1', '--arms', 'fast', '--out', str(directory)]
                for field in ('value', 'static_value', 'executable', 'resource_dir'):
                    command.extend(['--' + field.replace('_', '-'), str(getattr(options, field))])
                with (directory / 'run.log').open('w') as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
                print(f'evaluated {name} {task}', flush=True)
    report(options)


def report(options):
    result = dict(subset=json.loads((options.out / 'subset.json').read_text()), groups={})
    for task in TASKS:
        groups = {}
        for variant in VARIANTS:
            wins = []
            for seed in range(1, 5):
                path = options.out / 'eval' / f'{variant}_s{seed}' / task / 'games.jsonl'
                games = [json.loads(line) for line in path.read_text().splitlines()]
                if len(games) != 32 or len({row['seed'] for row in games}) != 32 or not all(row['terminal'] for row in games):
                    raise ValueError(f'incomplete evaluation: {path}')
                wins.append(sum(game['won'] for game in games))
            groups[variant] = dict(wins=wins, mean=statistics.mean(wins),
                range=max(wins) - min(wins), stdev=statistics.stdev(wins))
        result['groups'][task] = groups
    heldout = result['groups'][TASKS[1]]
    result['criterion'] = dict(half_stdev=heldout['ema']['stdev'] <= heldout['base']['stdev'] / 2,
                              mean_preserved=heldout['ema']['mean'] >= heldout['base']['mean'] - 2)
    result['passed'] = all(result['criterion'].values())
    result['variance_reproduced'] = heldout['base']['range'] > 2
    (options.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'subset'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'train', 'one', 'evaluate', 'report'), required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=range(1, 5))
    parser.add_argument('--variant', choices=VARIANTS)
    for field in ('value', 'static_value', 'executable', 'resource_dir'):
        parser.add_argument('--' + field.replace('_', '-'), type=Path)
    options = parser.parse_args()
    options.subset = options.subset.resolve()
    options.out = options.out.resolve()
    options.out.mkdir(parents=True, exist_ok=True)
    if options.phase == 'evaluate' and any(getattr(options, field) is None
            for field in ('value', 'static_value', 'executable', 'resource_dir')):
        parser.error('evaluate requires value, static-value, executable and resource-dir')
    globals()[options.phase](options)


if __name__ == '__main__':
    main()
