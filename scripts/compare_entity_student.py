"""Small matched-capacity student comparison, holding out complete deck multisets."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
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
    train = torch.from_numpy(np.flatnonzero(splits == 'train'))
    wait_weight = weight[train][plant[train] == 0].sum()
    positive_weight = (wait_weight / plant[train].sum()).sqrt().to(args.device)
    manifest = [{'path': str(path), 'level': level, 'deck': deck, 'split': split}
                for path, level, deck, split in chosen]
    report = {'archives': manifest, 'rows': len(plant), 'pack_seconds': time.perf_counter() - started,
              'device': args.device, 'epochs': args.epochs, 'seed': 1, 'batch_size': 128,
              'loss': 'weighted BCE + packet CE + cell CE; hard teacher labels; wait_keep=0.1', 'arms': {}}
    print(json.dumps({key: value for key, value in report.items() if key != 'archives'}), flush=True)
    common = dict(pool='meanmax', obs='v2timer', row_pool=True, packet_head='identity')
    for name, width, architecture in (('cnn64', 64, 'grid'), ('cnn81', 81, 'grid'),
                                      ('entity64', 64, 'entity_attn')):
        torch.manual_seed(1)
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
            generator = torch.Generator().manual_seed(153 + epoch)
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
            print(json.dumps({'arm': name, 'epoch': epoch + 1, 'seconds': time.perf_counter() - started}), flush=True)
        net.eval()
        predictions = []
        indices = torch.from_numpy(np.flatnonzero(splits != 'train'))
        with torch.inference_mode():
            for chunk in indices.split(128):
                output = forward(chunk)
                predictions.append(torch.stack((output['plant'] > 0, output['packet'].argmax(1),
                                                output['cell'].argmax(1)), dim=1).cpu())
        predicted = torch.cat(predictions)
        target_plant = plant[indices].bool()
        type_correct = predicted[:, 0].bool() == target_plant
        packet_correct = predicted[:, 1] == packet[indices]
        cell_correct = predicted[:, 2] == cell[indices]
        action_correct = type_correct & (~target_plant | (packet_correct & cell_correct))
        metrics = {}
        for split in ('seen', 'unseen'):
            for level in (None, 7, 8, 9, 10):
                selected = (splits[indices] == split) & ((levels[indices] == level) if level else True)
                mask = torch.from_numpy(selected)
                planted = mask & target_plant
                weights = weight[indices][mask]
                metrics[f'{split}/L{level}' if level else split] = {
                    'rows': int(mask.sum()), 'plants': int(planted.sum()),
                    'action_acc': float((action_correct[mask] * weights).sum() / weights.sum()),
                    'type_acc': float((type_correct[mask] * weights).sum() / weights.sum()),
                    'plant_action_acc': float(action_correct[planted].float().mean()),
                    'packet_acc': float(packet_correct[planted].float().mean()),
                    'cell_acc': float(cell_correct[planted].float().mean())}
        report['arms'][name] = {'parameters': sum(parameter.numel() for parameter in net.parameters()),
                                'seconds': time.perf_counter() - started, 'metrics': metrics}
        torch.save({'state_dict': net.cpu().state_dict(), 'arch': architecture, 'width': width, **common}, args.out / f'{name}.pt')
        (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({name: report['arms'][name]}), flush=True)


if __name__ == '__main__':
    main()
