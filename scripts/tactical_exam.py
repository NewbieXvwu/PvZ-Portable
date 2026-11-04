#!/usr/bin/env python3
"""Fixed, paired tactical monitor. Games, replay and scoring belong to search_play.

The bank is deliberately separate from curriculum manifests: exam outcomes never
become training archives, gate scores or promotion conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_promotion import summarize


def training_split(training):
    # Compare registered source paths without touching desktop-only /home mounts on Mac.
    archives = {os.path.abspath(t['drill_start']['archive'])
                for t in training.get('tasks', []) if t.get('drill_start')}
    blocks = [(t['drill_start']['future_seed_start'],
               t['drill_start']['future_seed_start'] + t['drill_start']['future_seed_count'])
              for t in training.get('tasks', []) if t.get('drill_start')]
    return archives, blocks


def check_split(bank, training, *, split=None):
    """Manifest-only split check; never replay a training archive here."""
    archives, blocks = training_split(training) if split is None else split
    seen = set()
    for item in bank['items']:
        drill = item['task']['drill_start']
        if os.path.abspath(drill['archive']) in archives:
            raise ValueError(f"exam archive is used by training: {item['id']}")
        if not drill.get('targets') or not item['seeds']:
            raise ValueError(f"unscorable exam item: {item['id']}")
        for seed in item['seeds']:
            key = (item['id'], seed)
            if key in seen or any(low <= seed < high for low, high in blocks):
                raise ValueError(f"duplicate/training exam seed: {key}")
            seen.add(key)


def paired_report(rows, items, models):
    expected = {(item['id'], seed): item['skill'] for item in items for seed in item['seeds']}
    outcomes = {name: {} for name in models}
    errors = []
    for row in rows:
        if row['status'] != 'ok':
            errors.append({k: row[k] for k in ('item', 'model', 'seed', 'error')})
            continue
        key = (row['item'], row['seed'])
        side = outcomes[row['model']]
        if key not in expected or key in side or type(row['won']) is not bool:
            raise ValueError(f'invalid/duplicate exam outcome: {key}')
        side[key] = row['won']

    def comparison(left, right):
        pairs = [dict(candidate=outcomes[left][key], incumbent=outcomes[right][key],
                      skill=skill) for key, skill in expected.items()
                 if key in outcomes[left] and key in outcomes[right]]

        def block(group, count):
            result = dict(expected_pairs=count, missing_pairs=count - len(group),
                          **(summarize(group) if group else dict(pairs=0)))
            if group:
                result.update(candidate_losses=len(group) - result['candidate_wins'],
                              incumbent_losses=len(group) - result['incumbent_wins'])
            return result

        return dict(direction=f'{left}-{right}', overall=block(pairs, len(expected)),
                    skills={skill: block([p for p in pairs if p['skill'] == skill], count)
                            for skill, count in sorted(Counter(expected.values()).items())})

    comparisons = {f'{side}-incumbent': comparison(side, 'incumbent')
                   for side in models if side != 'incumbent'}
    if 'A' in models and 'B' in models:
        comparisons['A-B'] = comparison('A', 'B')
    complete = all(len(outcomes[side]) == len(expected) for side in models)
    return dict(status='ok' if complete else 'partial', veto=False,
                items=len(items), seeds_per_model=len(expected), models=models,
                errors=errors, comparisons=comparisons)


_CONFIG = {}


def _init(config):
    import search_play
    from pvz_rollout_policy import export_student, load_student
    # Each worker owns one simulator and only reads immutable model files.
    search_play._W.clear()
    first = next(iter(config['models'].values()))
    search_play._init_worker(config['value_bins']['value'], first, 61,
                             config['value_bins']['static'])
    nets = {}
    for index, (side, path) in enumerate(config['models'].items()):
        net = load_student(path)
        binary = str(Path(config['scratch']) / f'{os.getpid()}-{index}.bin')
        export_student(net, binary)
        nets[side] = (net, binary)
    if config.get('teacher_prior'):
        net = load_student(config['teacher_prior'])
        binary = str(Path(config['scratch']) / f'{os.getpid()}-gpt.bin')
        export_student(net, binary)
        nets['teacher_gpt'] = (net, binary)
    search_play._W['drill_prefix_cache'] = {}
    _CONFIG.update(config, nets=nets)


def _run(job):
    import search_play
    side, item, seed = job
    try:
        task = copy.deepcopy(item['task'])
        drill = task['drill_start']
        if item.get('source_root') in _CONFIG['roots']:
            drill['archive'] = str(Path(_CONFIG['roots'][item['source_root']]) / item['archive'])
        terrain = item['skill'].split('|')[0]
        prior_path = search_play._terrain_policy(_CONFIG['models'][side],
            _CONFIG.get('teacher_prior') if side == 'teacher' else None, task)
        prior = ('teacher_gpt' if side == 'teacher' and prior_path == _CONFIG.get('teacher_prior')
                 else side)
        net, binary = _CONFIG['nets'][prior]
        value = search_play._leaf_value(_CONFIG['value_bins']['value'],
                                        _CONFIG['value_bins']['roof'], task)
        search_play._W.update(fast=net, rollout_file=binary,
                              value_files=(value, _CONFIG['value_bins']['static']))
        env = search_play._W.get('env')
        if env is not None:
            env.load_rollout_net(binary)
            env._teacher_value_loaded = False
        args = search_play.parser().parse_args([
            '--rollout-policy', _CONFIG['models'][side], '--value', _CONFIG['value'],
            '--manifest', str(ROOT / 'configs/human_demo_tasks.json'),
            '--task-id', task['task_id'], '--task-json', json.dumps(task),
            '--demo-archive', drill['archive'],
            '--demo-decision', str(drill['decisions'][drill['index']]),
            '--executable', _CONFIG['executable'], '--resource-dir', _CONFIG['resource_dir'],
            '--out', _CONFIG['out'], '--dagger-beta', '1'])
        if side == 'teacher':
            from drill_run import TEACHER
            for key, value in TEACHER.items():
                setattr(args, key, value)
            args.save_candidates = int(terrain == 'fog')
        result = search_play._play(dict(args=args, seed=task['seeds'][0],
                                       future_seed=seed,
                                       arm='fast_search' if side == 'teacher' else 'fast'))
        return dict(status='ok', item=item['id'], model=side, seed=seed,
                    won=result['won'], seconds=result['seconds'], drill=result['drill'])
    except Exception as error:
        # search_play closes its simulator on a game exception. Other items can proceed.
        return dict(status='error', item=item['id'], model=side, seed=seed,
                    error=f'{type(error).__name__}: {error}')


def _run_item(item):
    return [_run((side, item, seed)) for seed in item['seeds'] for side in _CONFIG['models']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--models', required=True, help='JSON model paths, including incumbent')
    parser.add_argument('--teacher-prior', type=Path, help='production GPT prior on fog/roof during calibration')
    parser.add_argument('--training-manifest', type=Path, nargs='+',
                        default=[ROOT / 'configs/task235_drills.json'])
    parser.add_argument('--archive-root', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--value', type=Path, required=True)
    parser.add_argument('--value-roof', type=Path)
    parser.add_argument('--static-value', type=Path)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('--workers must be 1..4')
    models = json.loads(args.models)
    if 'incumbent' not in models:
        parser.error('--models must include incumbent')
    bank = json.loads(args.bank.read_text())
    if bank.get('status') == 'insufficient_calibration':
        parser.error('exam calibration has fewer than 100 or more than 200 selected items')
    for manifest in args.training_manifest:
        check_split(bank, json.loads(manifest.read_text()))
    items = bank['items'][:args.limit] if args.limit else bank['items']
    if not items:
        parser.error('empty exam')
    roots = dict(entry.split('=', 1) for entry in args.archive_root)
    args.out.mkdir(parents=True, exist_ok=True)
    # Keep all exports and Python/engine scratch under this run, including on Mac.
    import tempfile
    os.environ['TMPDIR'] = str(args.out.resolve())
    tempfile.tempdir = str(args.out.resolve())
    import search_play
    from drill_run import export_values
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='exam-', dir=args.out) as scratch:
        keep = argparse.Namespace(name=scratch)
        bins = export_values(args.value, args.value_roof or args.value,
                             args.static_value or args.value, keep)
        config = dict(models=models, roots=roots, value_bins=bins, scratch=scratch,
                      teacher_prior=str(args.teacher_prior) if args.teacher_prior else None,
                      value=str(args.value), executable=str(args.executable),
                      resource_dir=str(args.resource_dir), out=str(args.out))
        # All models for one item are adjacent and use exactly the same future seeds.
        game_count = sum(len(item['seeds']) for item in items) * len(models)
        rows = []
        with (args.out / 'results.jsonl').open('w') as stream:
            with mp.get_context('fork').Pool(args.workers, initializer=_init, initargs=(config,)) as pool:
                for batch in pool.imap_unordered(_run_item, items):
                    rows.extend(batch)
                    for row in batch:
                        stream.write(json.dumps(row) + '\n')
                        if row['status'] != 'ok':
                            print(json.dumps(row), flush=True)
                    stream.flush()
                    print(f'exam {len(rows)}/{game_count} ({time.monotonic()-started:.1f}s)', flush=True)
        result = paired_report(rows, items, models)
        result.update(bank=str(args.bank), seconds=round(time.monotonic()-started, 3),
                      workers=args.workers, games=game_count,
                      teacher_prior=str(args.teacher_prior) if args.teacher_prior else None)
        (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
