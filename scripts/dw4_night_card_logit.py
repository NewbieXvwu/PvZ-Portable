"""task23 step 6.2: does an unseen card in hand change the student's sun-shroom choice?

Read-only, no games. On n9 night archives (deck [0,1,2,3,4,5,7,8,9], slot 9 empty) compare the
packet-head and plant-head outputs under four hands:

  base              the archived hand
  +11 unaffordable  slot 9 = grave buster (11), active, affordable=0
  +11 affordable    slot 9 = grave buster (11), active, affordable=1
  +10 affordable    slot 9 = fume-shroom (10), active, affordable=1 (a second unseen card)

For every archived decision (plant and wait) this records the sun-shroom packet probability, its rank,
whether it is the packet argmax, the plant-head probability, and the joint action probability
(sigmoid(plant) * softmax over packets). The same numbers are aggregated over all decisions and over
the subset where the student actually planted. This separates "any unfamiliar card perturbs the input"
from "11 specifically", and shows which card gains the mass.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
import torch
from pvz_frame_store import archive_seed, cache_episode, cache_packets, cache_identity_scene, read_frames
from pvz_rollout_policy import load_student

SUN_SHROOM = 9
GRAVE_BUSTER = 11
FUME_SHROOM = 10
VARIANTS = {
    'base': None,
    'add11_unaffordable': (GRAVE_BUSTER, 0),
    'add11_affordable': (GRAVE_BUSTER, 1),
    'add10_affordable': (FUME_SHROOM, 1),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hand(base: np.ndarray, change) -> np.ndarray:
    result = base.copy()
    if change is not None:
        kind, affordable = change
        free = np.flatnonzero(result[:, 0] < 0)
        assert len(free) == 1, ('expected exactly one empty slot', result[:, 0])
        result[free[0]] = (kind, 1.0, 1.0, float(affordable))
    return result


def frame_path(archive: Path) -> Path:
    return archive/'frames.npz' if (archive/'frames.npz').exists() else archive/'frames.jsonl'


def select_archives(root: Path, first_seed: int, last_seed: int, levels: set[int]):
    chosen = []
    for folder in sorted(root.iterdir()):
        if not folder.name.startswith('seed_'):
            continue
        seed = archive_seed(folder.name)
        if not first_seed <= seed <= last_seed:
            continue
        for archive in sorted(folder.iterdir()):
            if not frame_path(archive).exists():
                continue
            level = json.loads((archive/'meta.json').read_text())['context']['level']
            if level in levels:
                chosen.append((archive, level))
    return chosen


def summarize(values):
    probability = np.array([v['probability'] for v in values])
    top = np.array([v['top1'] for v in values])
    rank = np.array([v['rank'] for v in values])
    joint = np.array([v['joint'] for v in values])
    plant = np.array([v['plant'] for v in values])
    return dict(states=int(len(values)),
                mean_plant_head_probability=float(plant.mean()),
                plant_head_fire_share=float((plant > 0.5).mean()),
                mean_sun_shroom_packet_probability=float(probability.mean()),
                median_sun_shroom_packet_probability=float(np.median(probability)),
                sun_shroom_packet_top1_share=float(top.mean()),
                mean_sun_shroom_rank=float(rank.mean()),
                mean_joint_sun_shroom_action_probability=float(joint.mean()))


def run(args):
    torch.set_num_threads(1)
    net = load_student(args.model).eval()
    assert net.obs == 'v2' and net.packet_head == 'identity' and not net.context
    cache = args.cache
    records = {name: {'all': [], 'plant': []} for name in VARIANTS}
    gains = {name: Counter() for name in VARIANTS}
    levels = Counter()
    used = []
    wanted = {level: 0 for level in args.levels}
    for archive, level in select_archives(args.frames_root, args.first_seed, args.last_seed, set(args.levels)):
        if wanted[level] >= args.per_level:
            continue
        frames = read_frames(frame_path(archive))
        encoded = cache_episode(frame_path(archive), cache, 'v2')
        with np.load(encoded, allow_pickle=False) as f:
            rows = f['rows']; boards = f['boards'].astype(np.float32); scalars = f['scalars'].copy()
        with np.load(cache_packets(frame_path(archive), cache), allow_pickle=False) as f:
            assert np.array_equal(rows, f['rows']); packets = f['packets']
        with np.load(cache_identity_scene(frame_path(archive), cache), allow_pickle=False) as f:
            assert np.array_equal(rows, f['rows']); scalars[:, 4] = f['wave_scalar']
        planting = np.array([(frames[int(row)].get('action') or {}).get('type') == 'plant' for row in rows])
        if not planting.any():
            continue
        index = np.flatnonzero(planting)[:args.per_level - wanted[level]]
        # Keep every decision of the sampled games, not only the planting rows: the plant head can
        # flip a wait into a plant, which is the mechanism the live A/B is meant to test.
        levels[level] += int(planting.sum()); wanted[level] += len(index)
        used.append(str(archive))
        for start in range(0, len(rows), 256):
            stop = start + 256
            board = torch.from_numpy(boards[start:stop]); scalar = torch.from_numpy(scalars[start:stop])
            base = packets[start:stop]
            slot = int(np.flatnonzero(base[0][:, 0] == SUN_SHROOM)[0])
            plant_mask = planting[start:stop]
            for name, change in VARIANTS.items():
                info = torch.from_numpy(np.stack([hand(row, change) for row in base]))
                with torch.inference_mode():
                    out = net(board, scalar, None, packet_info=info)
                    probability = out['packet'].softmax(1)
                    chosen = probability.argmax(1)
                    plant = torch.sigmoid(out['plant'])
                values = dict(probability=probability[:, slot].tolist(),
                              top1=(chosen == slot).tolist(),
                              rank=(probability > probability[:, slot:slot+1]).sum(1).tolist(),
                              joint=(plant * probability[:, slot]).tolist(),
                              plant=plant.tolist())
                for i in range(len(board)):
                    item = {field: values[field][i] for field in values}
                    records[name]['all'].append(item)
                    if plant_mask[i]:
                        records[name]['plant'].append(item)
                kinds = np.stack([hand(row, change)[:, 0] for row in base]).astype(int).tolist()
                for row, choice in enumerate(chosen.tolist()):
                    gains[name][int(kinds[row][choice])] += 1
        print(json.dumps({'archives': len(used), 'states': sum(levels.values()), 'levels': dict(levels)}), flush=True)
        if all(value >= args.per_level for value in wanted.values()):
            break
    report = dict(purpose='task23 step 6.2 unseen-card hand perturbation on n9 night decisions',
                  model=str(args.model), model_sha256=digest(args.model),
                  archives=len(used), planting_decisions=dict(levels),
                  variants={name: {key: summarize(values) for key, values in parts.items()}
                            for name, parts in records.items()},
                  packet_argmax_counts={name: {str(kind): count for kind, count in sorted(counter.items())}
                                        for name, counter in gains.items()},
                  archive_list=used,
                  note='rank is 0-based among the ten packet slots (empty slots masked to -1e9); '
                       'joint = sigmoid(plant) * softmax(packet) for the sun-shroom slot; '
                       'argmax counts use the card type actually in the chosen slot')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: {key: summarize(values) for key, values in parts.items()}
                      for name, parts in records.items()}, indent=2))
    print(json.dumps(report['packet_argmax_counts'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--frames-root', type=Path, default=Path('/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2'))
    p.add_argument('--first-seed', type=int, default=1900000)
    p.add_argument('--last-seed', type=int, default=1929439)
    p.add_argument('--cache', type=Path, default=ROOT/'artifacts/task23/cache')
    p.add_argument('--levels', type=int, action='append', default=None,
                   help='night levels to sample (default 11,12,13: the n9 archives that exist)')
    p.add_argument('--per-level', type=int, default=100,
                   help='planting decisions sampled per level; all decisions of those games are scored')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not args.levels:
        args.levels = [11, 12, 13]
    for name in ('model', 'frames_root', 'cache', 'out'):
        if not getattr(args, name).is_absolute():
            p.error('all paths must be absolute: ' + name)
    run(args)


if __name__ == '__main__':
    main()
