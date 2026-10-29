"""Small matched-capacity student comparison, holding out complete deck multisets.

Three arms (CNN 64, CNN 81, entity attention 64) trained on the same packed rows for three
random seeds each; reports the mean and range per metric, and separates decks seen in training
from decks never seen.  The 2026-10-29 review found the first entity run lost on unseen decks
because the deck leaked into the plant-or-wait context; the entity arm here is the corrected
encoder (board-only pooling, group embedding, four heads, learned relative tables, deck dropout).
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from pack_training_rows import build_from_archives
from pvz_rollout_policy import RolloutNet

ARMS = (('cnn64', 64, 'grid'), ('cnn81', 81, 'grid'), ('entity64', 64, 'entity_attn'))
STATS = ('action_acc', 'type_acc', 'plant_action_acc', 'packet_acc', 'cell_acc',
         'auc', 'wait_false_plant', 'wait_logit', 'plant_logit')


def select_archives(data):
    groups = defaultdict(list)
    for seed in sorted(os.scandir(data), key=lambda entry: entry.name):
        if not seed.is_dir() or not seed.name.startswith('seed_'):
            continue
        for label in sorted(os.scandir(seed.path), key=lambda entry: entry.name):
            if not re.fullmatch(r'd(?:7|8|9|10)p[1-4]m_fast_search', label.name):
                continue
            directory = Path(label.path)
            task = json.loads((directory / 'meta.json').read_text())['task']
            groups[(task['level'], tuple(sorted(task['deck'])))].append(directory / 'frames.npz')
    rng = random.Random(153)
    chosen, seen = [], set()
    for level in (7, 8, 9, 10):
        candidates = sorted(key for key, paths in groups.items() if key[0] == level and len(paths) >= 5)
        rng.shuffle(candidates)
        for _, deck in candidates[:6]:
            seen.add(deck)
            paths = sorted(groups[(level, deck)])
            rng.shuffle(paths)
            for index, path in enumerate(paths[:5]):
                chosen.append((path, level, deck, 'train' if index < 4 else 'seen'))
    for level in (7, 8, 9, 10):
        candidates = sorted(key for key in groups if key[0] == level and key[1] not in seen)
        rng.shuffle(candidates)
        for _, deck in candidates[:8]:
            chosen.append((sorted(groups[(level, deck)])[0], level, deck, 'unseen'))
    return chosen


def auc(scores, labels):
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores))
    positive = labels.sum()
    negative = len(labels) - positive
    if not positive or not negative:
        return float('nan')
    return float((ranks[labels == 1].sum() - positive * (positive - 1) / 2) / (positive * negative))


def metrics_for(logits, packet_logits, cell_logits, plant, packet, cell, weight, levels, splits):
    predicted = np.stack((logits > 0, packet_logits, cell_logits), axis=1)
    target_plant = plant.astype(bool)
    type_correct = predicted[:, 0].astype(bool) == target_plant
    packet_correct = predicted[:, 1] == packet
    cell_correct = predicted[:, 2] == cell
    action_correct = type_correct & (~target_plant | (packet_correct & cell_correct))
    result = {}
    for split in ('seen', 'unseen'):
        for level in (None, 7, 8, 9, 10):
            selected = (splits == split) & ((levels == level) if level else True)
            planted = selected & target_plant
            wait = selected & ~target_plant
            w = weight[selected]
            result[f'{split}/L{level}' if level else split] = {
                'rows': int(selected.sum()), 'plants': int(planted.sum()),
                'action_acc': float((action_correct[selected] * w).sum() / w.sum()),
                'type_acc': float((type_correct[selected] * w).sum() / w.sum()),
                'plant_action_acc': float(action_correct[planted].mean()),
                'packet_acc': float(packet_correct[planted].mean()),
                'cell_acc': float(cell_correct[planted].mean()),
                'auc': auc(logits[selected], plant[selected].astype(int)),
                'wait_false_plant': float((predicted[wait, 0] * weight[wait]).sum() / weight[wait].sum()),
                'wait_logit': float(logits[wait].mean()),
                'plant_logit': float(logits[planted].mean())}
    return result


def summarize(per_seed):
    result = {}
    for key in per_seed[0]:
        entry = {'rows': per_seed[0][key]['rows'], 'plants': per_seed[0][key]['plants']}
        for metric in STATS:
            values = [seed[key][metric] for seed in per_seed]
            entry[metric] = {'mean': float(np.mean(values)), 'min': float(min(values)),
                             'max': float(max(values))}
        result[key] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seeds', default='1,2,3')
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(',')]
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = select_archives(args.data)
    assignment = {path: (level, split) for path, level, _, split in chosen}
    levels, splits = [], []

    def observe(path, rows):
        level, split = assignment[path]
        levels.extend([level] * len(rows))
        splits.extend([split] * len(rows))

    started = time.perf_counter()
    loaded, _, _, _ = build_from_archives(
        args.data, {path.parent.name for path, *_ in chosen}, wait_keep=.1,
        obs='v2timer', packet_head='identity', arch='entity_attn',
        cache_dir=args.out / 'cache', archive_paths=[path for path, *_ in chosen], row_observer=observe)
    boards, scalars, plant, packet, cell, weight, _, _, _, packet_info = loaded
    levels, splits = np.asarray(levels), np.asarray(splits)
    plant_np, packet_np, cell_np, weight_np = (plant.numpy(), packet.numpy(), cell.numpy(), weight.numpy())
    train = torch.from_numpy(np.flatnonzero(splits == 'train'))
    held = torch.from_numpy(np.flatnonzero(splits != 'train'))
    wait_weight = weight[train][plant[train] == 0].sum()
    positive_weight = (wait_weight / plant[train].sum()).sqrt().to(args.device)
    manifest = [{'path': str(path), 'level': level, 'deck': deck, 'split': split}
                for path, level, deck, split in chosen]
    report = {'archives': manifest, 'rows': len(plant), 'pack_seconds': time.perf_counter() - started,
              'device': args.device, 'epochs': args.epochs, 'seeds': seeds, 'batch_size': 128,
              'loss': 'weighted BCE + packet CE + cell CE; hard teacher labels; wait_keep=0.1', 'arms': {}}
    print(json.dumps({key: value for key, value in report.items() if key != 'archives'}), flush=True)
    common = dict(pool='meanmax', obs='v2timer', row_pool=True, packet_head='identity')
    for name, width, architecture in ARMS:
        per_seed, seconds = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            net = RolloutNet(width, arch=architecture, **common).to(args.device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
            steps = args.epochs * math.ceil(len(train) / 128)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

            def forward(indices):
                scalar_batch = scalars.batch(indices, args.device)
                if architecture == 'grid':
                    scalar_batch = scalar_batch[:, :net.scalar[0].in_features]
                return net(boards.batch(indices, args.device), scalar_batch,
                           packet[indices].clamp_min(0).long().to(args.device),
                           packet_info=packet_info[indices].to(args.device))

            started = time.perf_counter()
            for epoch in range(args.epochs):
                net.train()
                generator = torch.Generator().manual_seed(seed * 1000 + epoch)
                order = train[torch.randperm(len(train), generator=generator)]
                for indices in order.split(128):
                    output = forward(indices)
                    target = plant[indices].float().to(args.device)
                    weights = weight[indices].to(args.device)
                    loss = (functional.binary_cross_entropy_with_logits(
                        output['plant'], target, pos_weight=positive_weight, reduction='none') * weights).sum() / weights.sum()
                    planted = target > .5
                    if planted.any():
                        packet_loss = functional.cross_entropy(output['packet'][planted], packet[indices].long().to(args.device)[planted], reduction='none')
                        cell_loss = functional.cross_entropy(output['cell'][planted], cell[indices].long().to(args.device)[planted], reduction='none')
                        loss += ((packet_loss + cell_loss) * weights[planted]).sum() / weights[planted].sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    schedule.step()
                print(json.dumps({'arm': name, 'seed': seed, 'epoch': epoch + 1,
                                  'seconds': round(time.perf_counter() - started, 1)}), flush=True)
            net.eval()
            logits, packet_logits, cell_logits = [], [], []
            with torch.inference_mode():
                for indices in held.split(128):
                    output = forward(indices)
                    logits.append(output['plant'].cpu())
                    packet_logits.append(output['packet'].argmax(1).cpu())
                    cell_logits.append(output['cell'].argmax(1).cpu())
            index = held.numpy()
            metrics = metrics_for(torch.cat(logits).numpy(), torch.cat(packet_logits).numpy(),
                                  torch.cat(cell_logits).numpy(), plant_np[index],
                                  packet_np[index], cell_np[index], weight_np[index],
                                  levels[index], splits[index])
            per_seed.append(metrics)
            seconds.append(time.perf_counter() - started)
            torch.save({'state_dict': net.cpu().state_dict(), 'arch': architecture, 'width': width, **common},
                       args.out / f'{name}_s{seed}.pt')
            report['arms'][name] = {'parameters': sum(parameter.numel() for parameter in net.parameters()),
                                    'seeds_run': [s for s in seeds[:len(per_seed)]],
                                    'seconds': seconds, 'per_seed': per_seed,
                                    'metrics': summarize(per_seed)}
            (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({name: metrics['unseen']}), flush=True)
        print(json.dumps({name: report['arms'][name]['metrics']['unseen']}), flush=True)


if __name__ == '__main__':
    main()
