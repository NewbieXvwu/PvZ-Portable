"""Bounded value candidate training and held-out comparison for the curriculum loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]


def held_out(seed):
    return hashlib.sha256(str(seed).encode()).digest()[0] % 5 == 0


def terrain(level):
    from pvz_constants import background_for_level
    return background_for_level(level)


def catalog(request):
    from render_episode import training_task_kwargs
    groups = {group.get('frame_label', group['label'] + '_fast_search'): group
              for group in request['groups']}
    records = []
    for path in sorted(Path(request['data']).glob('*/*/meta.json')):
        group = groups.get(path.parent.name)
        if group is None:
            continue
        meta = json.loads(path.read_text())
        base_seed = meta['task']['seed']
        seed = meta['task'].get('future_seed', base_seed)
        if meta.get('demo_archive') and 'future_seed' not in meta['task']:
            seed = int(path.parent.parent.name.rsplit('_f', 1)[1])
        definition = group['definition']
        kwargs = training_task_kwargs(definition, base_seed)
        if meta.get('demo_archive'):
            kwargs.update(demo_archive=meta['demo_archive'], demo_decision=meta['demo_decision'])
        deck = meta['task'].get('deck', kwargs['deck'])
        records.append(dict(path=str(path), seed=seed, base_seed=base_seed, zone=terrain(kwargs['level']),
                            deck=','.join(map(str, sorted(deck))), group=group['base'],
                            kwargs=kwargs, won=meta['outcome']['won']))
    return records


def shares(records):
    total = len(records)
    return {field: {name: dict(games=count, share=count / max(1, total))
                    for name, count in sorted(Counter(record[field] for record in records).items())}
            for field in ('zone', 'deck', 'group')}


def balanced(records, limit):
    from itertools import zip_longest
    buckets = defaultdict(list)
    for record in records:
        buckets[record['zone']].append(record)
    rng = random.Random(155)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = [record for row in zip_longest(*(buckets[name] for name in sorted(buckets)))
               for record in row if record is not None]
    return ordered[:limit]


def significantly_worse(differences, tolerance):
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / max(1, len(differences) - 1)
    return mean > tolerance or mean > 1.96 * math.sqrt(variance / len(differences))


def decide(calibration, positions, zones):
    from value_ranking_check import report
    reasons, metrics = [], {}
    for zone in ['overall', *sorted(zones)]:
        games = [row for row in calibration if zone == 'overall' or row['zone'] == zone]
        roots = [row for row in positions if zone == 'overall' or row['zone'] == zone]
        ranking = report(roots, 0.25, 0.02)
        independent = {root['seed'] for root in roots}
        if (len({game['seed'] for game in games}) < 8 or len(independent) < 4
                or ranking.get('candidate', {}).get('pairs', 0) < 16):
            reasons.append(f'{zone}: insufficient held-out games or informative pairs')
            continue
        brier = {name: sum(row[name] for row in games) / len(games) for name in ('incumbent', 'candidate')}
        metrics[zone] = dict(games=len(games), brier=brier, ranking=ranking)
        if zone == 'overall':
            if brier['candidate'] >= brier['incumbent']:
                reasons.append('overall: Brier did not improve')
            if ranking['candidate']['pair_accuracy'] <= ranking['incumbent']['pair_accuracy']:
                reasons.append('overall: pair accuracy did not improve')
        else:
            per_seed = defaultdict(list)
            for game in games:
                per_seed[game['seed']].append(game['candidate'] - game['incumbent'])
            if significantly_worse([sum(values) / len(values) for values in per_seed.values()], 0.02):
                reasons.append(f'{zone}: Brier worsened')
            differences = []
            for seed in independent:
                scores = report([root for root in roots if root['seed'] == seed], 0.25, 0.02)
                if scores['candidate']['pairs']:
                    differences.append(scores['incumbent']['pair_accuracy'] - scores['candidate']['pair_accuracy'])
            if differences and significantly_worse(differences, 0.03):
                reasons.append(f'{zone}: pair accuracy worsened')
    return dict(approved=not reasons, reasons=reasons, metrics=metrics)


def calibration_scores(records, models):
    import torch
    from pvz_frame_store import frame_path, read_frames
    from pvz_value_net import context_vector
    from train_value_net import wave_starts
    from value_ranking_check import _score
    rows = []
    for record in records:
        meta = json.loads(Path(record['path']).read_text())
        frames = read_frames(frame_path(Path(record['path']).parent))
        starts = wave_starts(frames)
        if not starts:
            continue
        selected = [frames[index] for index in starts.values()]
        context = meta['context']
        context = torch.from_numpy(context_vector(context['multiplier'], context['roster']))
        row = dict(zone=record['zone'], seed=record['seed'])
        for name, model in models.items():
            values = _score(model, selected, context)
            row[name] = sum((value - record['won']) ** 2 for value in values) / len(values)
        rows.append(row)
    return rows


def ranking_positions(records, request, candidate):
    import value_ranking_check as ranking
    args = SimpleNamespace(rollout_policy=request['student'],
        value_map=dict(incumbent=request['incumbent'], candidate=str(candidate)),
        resource_dir=request['resource_dir'], executable=request['executable'],
        cells=1, random_cells=0, samples=4, horizon=2500, sim_skip=29,
        finish_sim_skip=None, dump=None, dump_shards=None, min_wave=3, min_sun=0,
        student_waits=False, position_rate=0.03, max_positions=2)
    ranking._init(args)
    positions = []
    try:
        for record in balanced(records, 40):
            roots, _ = ranking._game((record['seed'], record['kwargs']))
            for root in roots:
                root['zone'] = record['zone']
            positions.extend(roots)
    finally:
        ranking._W['env'].close()
        ranking._W['tmp'].cleanup()
        ranking._W.clear()
    return positions


def train_candidate(records, request, checkpoint, out):
    import numpy as np
    import torch
    import train_value_net as training
    from pvz_value_net import load_value_net
    initial = load_value_net(request['incumbent'])
    started = time.monotonic()
    training.CONTEXT, training.ENCODER_VERSION = initial.context, initial.encoder_version
    games, tensors = training.load_games(Path(request['data']), meta_paths=[Path(row['path']) for row in records],
                                         frames_per_game=32)
    zones = {row['path']: row['zone'] for row in records}
    ordered = sorted(records, key=lambda row: row['path'])
    tensors['seed_of_game'] = [row['seed'] for row in ordered]
    counts = Counter(row['zone'] for row in records)
    factors = torch.tensor([len(records) / (len(counts) * counts[zones[row['path']]]) for row in ordered])
    tensors['weights'] *= factors[tensors['game_of'].long()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(155)
    model = training.train_model(tensors, set(range(len(games))), device, 8, 155, initial=initial)
    checkpoint.update(state_dict=model.cpu().state_dict(),
                      trained_on=sorted(set(checkpoint['trained_on']) | {row['seed'] for row in records}),
                      source='curriculum-value-loop', width=model.conv[0].out_channels)
    temporary = out / 'model.building'
    torch.save(checkpoint, temporary)
    temporary.replace(out / 'model.pt')
    return dict(device=device, seconds=time.monotonic() - started, games=len(games),
                rows=len(tensors['labels']), epochs_max=8, frames_per_game_max=32)


def run(request, out):
    import torch
    from pvz_value_net import load_value_net
    torch.set_num_threads(1)
    checkpoint = torch.load(request['incumbent'], map_location='cpu', weights_only=False)
    seen = set(checkpoint['trained_on'])
    records = catalog(request)
    train = balanced([row for row in records if not held_out(row['seed'])], 2048)
    test = balanced([row for row in records if held_out(row['seed'])
                     and row['seed'] not in seen and row['base_seed'] not in seen], 160)
    result = dict(approved=False, incumbent=request['incumbent'], candidate=str(out / 'model.pt'),
                  round=request['round'], composition=shares(records), train_composition=shares(train),
                  test_composition=shares(test), split='stable seed hash, 20% held out; ancestry excluded')
    (out / 'data.json').write_text(json.dumps(dict(result, terrain_weighting='equal total game weight per terrain'), indent=2) + '\n')
    zones = {row['zone'] for row in records}
    counts = Counter(row['zone'] for row in test)
    if len({row['seed'] for row in train}) < 8 or any(counts[zone] < 8 for zone in zones) or not zones:
        return dict(result, reasons=['insufficient unseen held-out coverage or training seeds'])
    result['training'] = train_candidate(train, request, checkpoint, out)
    models = dict(incumbent=load_value_net(request['incumbent']), candidate=load_value_net(out / 'model.pt'))
    calibration = calibration_scores(test, models)
    positions = ranking_positions(test, request, out / 'model.pt')
    result.update(decide(calibration, positions, zones))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, default=2700)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def expired(_signal, _frame):
        raise TimeoutError('value candidate exceeded the GPU idle-window budget')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.budget_seconds)
    try:
        result = run(request, args.out)
    except Exception as error:
        result = dict(approved=False, round=request['round'], incumbent=request['incumbent'],
                      reasons=[f'{type(error).__name__}: {error}'])
    finally:
        signal.alarm(0)
    result['seconds'] = time.monotonic() - started
    temporary = args.out / 'decision.building'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.out / 'decision.json')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
