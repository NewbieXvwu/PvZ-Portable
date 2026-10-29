"""Interruptible CPU heldout-row metrics on a frozen pack, by deck exposure and terrain.

Counts use actual row denominators (plant-only heads never count waits). Deck identity
is the sorted public packet kinds, retaining duplicate cards and ignoring slot order.
Empty seen/unseen strata are reported explicitly, never fabricated from seed holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import signal
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
import torch
import torch.nn.functional as F
from pvz_constants import background_for_level
from pvz_recipe import load_recipe
from pvz_rollout_policy import load_student
from train_rollout_policy import read_pack
from gpu_window_runner import write_json


def deck_keys(kinds):
    ordered = np.ascontiguousarray(np.sort(np.asarray(kinds, dtype=np.int8), axis=1))
    return ordered.view(np.dtype((np.void, ordered.shape[1]))).reshape(-1)


@lru_cache(maxsize=50)
def terrain_for_level(level):
    return {'BACKGROUND_1_DAY': 'day', 'BACKGROUND_2_NIGHT': 'night', 'BACKGROUND_3_POOL': 'pool',
            'BACKGROUND_4_FOG': 'fog', 'BACKGROUND_5_ROOF': 'roof', 'BACKGROUND_6_BOSS': 'roof'}[background_for_level(level)]


def add_metrics(target, pred_type, planted, packet_ok, cell_ok, losses):
    target['rows'] += len(planted)
    target['plants'] += int(planted.sum())
    target['type_correct'] += int((pred_type == planted).sum())
    target['plant_recalled'] += int((pred_type & planted).sum())
    target['packet_correct'] += int(packet_ok.sum())
    target['cell_correct'] += int(cell_ok.sum())
    target['joint_correct'] += int(((pred_type == planted) & (~planted | (packet_ok & cell_ok))).sum())
    target['nll_sum'] += float(losses.sum())


def empty_metrics():
    return dict(rows=0, plants=0, type_correct=0, plant_recalled=0, packet_correct=0,
                cell_correct=0, joint_correct=0, nll_sum=0.0)


def report_metrics(counts):
    result = dict(counts)
    for key, numerator, denominator in (
            ('type_acc', 'type_correct', 'rows'), ('plant_recall', 'plant_recalled', 'plants'),
            ('packet_acc', 'packet_correct', 'plants'), ('cell_acc', 'cell_correct', 'plants'),
            ('joint_acc', 'joint_correct', 'rows'), ('nll', 'nll_sum', 'rows')):
        result[key] = counts[numerator] / counts[denominator] if counts[denominator] else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--merge-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--eval-data', type=Path, help='external heldout corpus, separate from training data')
    parser.add_argument('--manifest', type=Path, help='task157 fixed train/seen/unseen/terrain_sample split')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--tokens', type=Path, help='task162 sparse decision tokens aligned with --pack')
    args = parser.parse_args()
    recipe = load_recipe(args.recipe, 'training')
    torch.set_num_threads(1)
    interrupted = False
    ready = False
    def stop(_signum, _frame):
        nonlocal interrupted
        if not ready:
            raise SystemExit(75)
        interrupted = True
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    # Metadata is read once per source, never game-frame decompression/replay.
    spans, position = [], 0
    eval_data = args.eval_data or Path(recipe['data'])
    fixed_archives = json.loads(args.manifest.read_text())['archives'] if args.manifest else []
    split_by_path = {str(Path(a['path']).relative_to('frames')): a['split'] for a in fixed_archives} if args.manifest else None
    level_by_path = {str(Path(a['path']).relative_to('frames')): a['level'] for a in fixed_archives}
    terrains = ['day', 'night', 'pool', 'fog', 'roof', 'unknown']
    def observe(path, rows):
        nonlocal position
        metadata = path.parent / 'meta.json'
        level = (level_by_path[str(path.relative_to(eval_data))] if args.manifest else
                 json.loads(metadata.read_text()).get('context', {}).get('level') if metadata.exists() else None)
        terrain = terrain_for_level(level) if level is not None else 'unknown'
        split = split_by_path[str(path.relative_to(eval_data))] if split_by_path is not None else None
        spans.append((position, position + len(rows), terrains.index(terrain), split))
        position += len(rows)
    data = read_pack(args.pack, eval_data, observe, merge_dir=args.merge_dir)
    boards, scalars, plant, packets, cells, _weight, seeds = data[:7]
    info = data[9 + bool(recipe['aux_weight'])] if recipe['packet_head'] == 'identity' else None
    tokens = data[-1] if recipe['obs'] == 'v3' else None
    if args.tokens:
        if recipe['obs'] != 'v3':
            parser.error('--tokens requires obs=v3')
        from pvz_research_data import decision_tokens
        tokens = decision_tokens(args.tokens, len(seeds), recipe.get('token_input', 'compact'))
    if info is None:
        raise ValueError('deck-stratified evaluation requires an identity packet pack')
    rng = random.Random(recipe['validation']['seed'])
    unique_seeds = sorted(set(seeds)); rng.shuffle(unique_seeds)
    val_seeds = set(unique_seeds[:int(len(unique_seeds) * recipe['validation']['fraction'])])
    is_val = np.fromiter((seed in val_seeds for seed in seeds), bool, len(seeds))
    if args.manifest:
        is_val[:] = True  # Pack contains only the external heldout archives.
    validation = np.flatnonzero(is_val)
    terrain_of = np.empty(len(seeds), dtype=np.uint8)
    split_of = np.full(len(seeds), '', dtype='U14')
    for start, stop_at, terrain, split in spans:
        terrain_of[start:stop_at] = terrain
        if args.manifest:
            if split == 'train':
                raise ValueError('external evaluation pack contains training archives')
            split_of[start:stop_at] = split
    state_path = args.out / 'eval_progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        seen_cursor=0, next_row=0, seen_decks=[], groups={name: empty_metrics() for name in
            ['all', 'seen', 'unseen', 'terrain_sample'] + terrains + [f'{deck}/{terrain}' for deck in ('seen', 'unseen') for terrain in terrains]})
    seen = {bytes.fromhex(value) for value in state['seen_decks']}
    ready = True
    def save():
        state['seen_decks'] = sorted(key.hex() for key in seen)
        write_json(state_path, state)
    for start in range(state['seen_cursor'], 0 if args.manifest else len(seeds), 65536):
        stop_at = min(len(seeds), start + 65536)
        kinds = np.asarray(info.columns[0])[start:stop_at][~is_val[start:stop_at]]
        seen.update(key.tobytes() for key in np.unique(deck_keys(kinds)))
        state['seen_cursor'] = stop_at
        if interrupted:
            save(); raise SystemExit(75)
    save()
    net = load_student(args.out / 'model.pt')  # CPU only, including future recipe architectures.
    with torch.inference_mode():
        for start in range(state['next_row'], len(validation), args.batch_size):
            if interrupted:
                save(); raise SystemExit(75)
            idx = validation[start:start + args.batch_size]
            planted = plant[idx].bool()
            pk, cell = packets[idx].long(), cells[idx].long()
            packet_info = info[idx]
            extra = {'tokens': tokens.batch(idx, 'cpu')} if tokens is not None else {}
            out = net(boards.batch(idx, 'cpu'), scalars.batch(idx, 'cpu'), pk.clamp(min=0), packet_info=packet_info, **extra)
            pred_type = out['plant'] > 0
            packet_ok = (out['packet'].argmax(1) == pk) & planted
            cell_ok = (out['cell'].argmax(1) == cell) & planted
            nll = F.binary_cross_entropy_with_logits(out['plant'], planted.float(), reduction='none')
            nll[planted] += F.cross_entropy(out['packet'][planted], pk[planted], reduction='none')
            nll[planted] += F.cross_entropy(out['cell'][planted], cell[planted], reduction='none')
            exposed = (split_of[idx]=='seen' if args.manifest else
                       np.array([key.tobytes() in seen for key in deck_keys(packet_info[:, :, 0].numpy())]))
            unseen = split_of[idx]=='unseen' if args.manifest else ~exposed
            masks = dict(all=np.ones(len(idx), dtype=bool), seen=exposed, unseen=unseen,
                         terrain_sample=split_of[idx]=='terrain_sample')
            for terrain, name in enumerate(terrains):
                masks[name] = terrain_of[idx] == terrain
                masks['seen/' + name] = masks[name] & exposed
                masks['unseen/' + name] = masks[name] & unseen
            for name, mask in masks.items():
                add_metrics(state['groups'][name], pred_type[mask], planted[mask], packet_ok[mask], cell_ok[mask], nll[mask])
            state['next_row'] = start + len(idx)
            if start % (args.batch_size * 64) == 0:
                save()
    save()
    write_json(args.out / 'heldout.json', dict(model=str(args.out / 'model.pt'), split=str(args.manifest) if args.manifest else recipe['validation'],
        deck_identity=('task157 manifest split; terrain samples are separate' if args.manifest else
                       'sorted public packet kinds including duplicates; seen in training rows'),
        groups={name: report_metrics(counts) for name, counts in state['groups'].items()}))


if __name__ == '__main__':
    main()
