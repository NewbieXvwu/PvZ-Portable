"""Interruptible heldout-row metrics on a frozen pack, by deck exposure and terrain.

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


def evaluation_order(validation, preview_rows):
    """Uniform common prefix, followed by every remaining heldout row unchanged."""
    if not preview_rows or preview_rows >= len(validation):
        return validation, len(validation)
    from pvz_research_data import keep_training_rows
    prefix = keep_training_rows(validation, preview_rows / len(validation))
    return np.r_[validation[prefix], validation[~prefix]], int(prefix.sum())


def peer_comparisons(groups, peers):
    """Matched-prefix differences in existing output, without writing into control runs."""
    focus = ['all', 'unseen', 'day', 'night', 'pool', 'fog', 'roof']
    focus += ['unseen/' + zone for zone in ('day', 'night', 'pool', 'fog', 'roof')]
    current = {name: report_metrics(groups[name]) for name in focus}
    comparisons = {}
    for path, peer in peers.items():
        result = {}
        for name in focus:
            control = report_metrics(peer['groups'][name])
            result[name] = dict(rows=current[name]['rows'], control=control,
                delta={key: current[name][key] - control[key]
                       if current[name][key] is not None and control[key] is not None else None
                       for key in ('type_acc', 'plant_recall', 'packet_acc', 'cell_acc', 'joint_acc', 'nll')})
        comparisons[path] = result
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--merge-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--deck-holdout', type=Path)
    parser.add_argument('--tokens', type=Path)
    parser.add_argument('--training-budget', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--peer-out', type=Path, action='append', default=[], help='compare these trained runs in the same input/token pass')
    parser.add_argument('--preview-rows', type=int, default=0, help='publish a uniform heldout prefix, then continue the complete heldout set')
    parser.add_argument('--preview-only', action='store_true', help='stop after the existing common heldout prefix')
    parser.add_argument('--peers-inline', action='store_true', help='keep peer comparisons in this run heldout.json; never replace control outputs')
    parser.add_argument('--cached-preview-only', action='store_true', help='Mac handoff: evaluate the cached prefix without original frame archives; report partial')
    args = parser.parse_args()
    recipe = load_recipe(args.recipe, 'training')
    row_fraction = json.loads(args.training_budget.read_text())['train_row_fraction'] if args.training_budget else 1.0
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
    terrains = ['day', 'night', 'pool', 'fog', 'roof', 'unknown']
    data_dir = Path(recipe['data'])
    frozen = args.deck_holdout.parent / 'raw' if args.deck_holdout else None
    excluded = set(json.loads((args.tokens/'ready.json').read_text())['excluded_sources']) if args.cached_preview_only else None
    def observe(path, rows):
        nonlocal position
        metadata = path.parent / 'meta.json'
        level = json.loads(metadata.read_text()).get('context', {}).get('level') if metadata.exists() else None
        terrain = terrain_for_level(level) if level is not None else 'unknown'
        # Archives whose frames are gone keep their pack rows, but neither side can score them.
        keep = (str(path.relative_to(data_dir)) not in excluded if excluded is not None else
                (frozen / path.relative_to(data_dir)).exists() if frozen else path.exists())
        spans.append((position, position + len(rows), terrains.index(terrain), keep))
        position += len(rows)
    data = read_pack(args.pack, data_dir, observe, merge_dir=args.merge_dir)
    boards, scalars, plant, packets, cells, _weight, seeds = data[:7]
    info = data[9 + bool(recipe['aux_weight'])] if recipe['packet_head'] == 'identity' else None
    if info is None:
        raise ValueError('deck-stratified evaluation requires an identity packet pack')
    rng = random.Random(recipe['validation']['seed'])
    unique_seeds = sorted(set(seeds)); rng.shuffle(unique_seeds)
    val_seeds = set(unique_seeds[:int(len(unique_seeds) * recipe['validation']['fraction'])])
    is_val = np.fromiter((seed in val_seeds for seed in seeds), bool, len(seeds))
    if args.deck_holdout:
        from pvz_research_data import extra_validation
        is_val |= extra_validation(args.deck_holdout, len(seeds))
    token_store = None
    if args.tokens:
        from pvz_research_data import SparseTokens
        token_store = SparseTokens(args.tokens)
    terrain_of = np.empty(len(seeds), dtype=np.uint8)
    scored = np.ones(len(seeds), dtype=bool)
    for start, stop_at, terrain, available in spans:
        terrain_of[start:stop_at] = terrain
        scored[start:stop_at] = available
    validation = np.flatnonzero(is_val & scored)
    validation, preview_count = evaluation_order(validation, args.preview_rows)
    state_path = args.out / 'eval_progress.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        seen_cursor=0, next_row=0, seen_decks=[], groups={name: empty_metrics() for name in
            ['all', 'seen', 'unseen'] + terrains + [f'{deck}/{terrain}' for deck in ('seen', 'unseen') for terrain in terrains]})
    seen = {bytes.fromhex(value) for value in state['seen_decks']}
    peers = state.setdefault('peers', {})
    for out in args.peer_out:
        peers.setdefault(str(out), dict(groups={name: empty_metrics() for name in state['groups']}))
    ready = True
    def save():
        state['seen_decks'] = sorted(key.hex() for key in seen)
        write_json(state_path, state)
    for start in range(state['seen_cursor'], len(seeds), 65536):
        stop_at = min(len(seeds), start + 65536)
        from pvz_research_data import keep_training_rows
        kinds = np.asarray(info.columns[0])[start:stop_at][~is_val[start:stop_at] & keep_training_rows(np.arange(start, stop_at), row_fraction)]
        seen.update(key.tobytes() for key in np.unique(deck_keys(kinds)))
        state['seen_cursor'] = stop_at
        if interrupted:
            save(); raise SystemExit(75)
    save()
    net = load_student(args.out / 'model.pt').to(args.device)
    models = [(args.out, net, state['groups'])] + [
        (out, load_student(out / 'model.pt').to(args.device), peers[str(out)]['groups']) for out in args.peer_out]
    router = getattr(getattr(net, 'trunk', [None])[-1], 'route_counts', None)
    if router is not None and 'routing' in state:
        router.copy_(torch.tensor(state['routing']))
    def publish(complete):
        for folder, _model, groups in models:
            if args.peers_inline and folder != args.out:
                continue
            write_json(folder / 'heldout.json', dict(model=str(folder / 'model.pt'), split=recipe['validation'],
                deck_identity='sorted public packet kinds including duplicates; seen in training rows',
                groups={name: report_metrics(counts) for name, counts in groups.items()},
                routing_by_terrain=state.get('routing') if folder == args.out else None,
                evaluation_complete=complete, evaluated_rows=state['next_row'], heldout_rows=len(validation),
                preview_rows=preview_count, device=args.device,
                **(dict(comparisons=peer_comparisons(groups, peers)) if args.peers_inline else {})))
        print(json.dumps(dict(stage='complete' if complete else 'preview', rows=state['next_row'],
                              heldout_rows=len(validation), models=[str(folder) for folder, _, _ in models])), flush=True)
    with torch.inference_mode():
        limit = preview_count if args.cached_preview_only or args.preview_only else len(validation)
        while state['next_row'] < limit:
            start = state['next_row']
            if interrupted:
                save(); raise SystemExit(75)
            end = min(start + args.batch_size, limit, preview_count if start < preview_count else limit)
            idx = validation[start:end]
            planted = plant[idx].bool()
            pk, cell = packets[idx].long(), cells[idx].long()
            packet_info = info[idx]
            tokens = token_store.batch(idx, args.device) if token_store is not None else None
            if args.cached_preview_only:
                terrain_of[idx] = tokens['terrain'].cpu().numpy()
            exposed = np.array([key.tobytes() in seen for key in deck_keys(packet_info[:, :, 0].numpy())])
            masks = dict(all=np.ones(len(idx), dtype=bool), seen=exposed, unseen=~exposed)
            for terrain, name in enumerate(terrains):
                masks[name] = terrain_of[idx] == terrain
                masks['seen/' + name] = masks[name] & exposed
                masks['unseen/' + name] = masks[name] & ~exposed
            # Decode/collate public tokens and read board rows once for all compared models.
            b, s = boards.batch(idx, args.device), scalars.batch(idx, args.device)
            public_info, labelled_packet = packet_info.to(args.device), pk.clamp(min=0).to(args.device)
            for _folder, model, groups in models:
                extra = dict(tokens=tokens) if getattr(model, 'requires_tokens', False) else {}
                output = model(b, s, labelled_packet, packet_info=public_info, **extra)
                out = {key: value.cpu() for key, value in output.items()}
                pred_type = out['plant'] > 0
                packet_ok = (out['packet'].argmax(1) == pk) & planted
                cell_ok = (out['cell'].argmax(1) == cell) & planted
                nll = F.binary_cross_entropy_with_logits(out['plant'], planted.float(), reduction='none')
                nll[planted] += F.cross_entropy(out['packet'][planted], pk[planted], reduction='none')
                nll[planted] += F.cross_entropy(out['cell'][planted], cell[planted], reduction='none')
                for name, mask in masks.items():
                    add_metrics(groups[name], pred_type[mask], planted[mask], packet_ok[mask], cell_ok[mask], nll[mask])
            state['next_row'] = start + len(idx)
            if router is not None:
                state['routing'] = router.tolist()
            if (start // args.batch_size) % 64 == 0:
                save()
            if args.preview_rows and not state.get('preview_published') and state['next_row'] >= preview_count:
                state['preview_published'] = True
                save(); publish(False)
    save()
    publish(not (args.cached_preview_only or args.preview_only))


if __name__ == '__main__':
    main()
