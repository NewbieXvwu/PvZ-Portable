#!/usr/bin/env python3
"""task229: uniform weight soup over same-initialization curriculum candidates.

The production A recipe retrains from one fixed initialization every round
(recipes/dw4i2_v2timer_clean_a.json), so r101..r105 A differ only by the data
seen in 8 epochs.  This script averages their weights (Wortsman et al. 2022,
uniform soup) and screens the result on holdout training rows.

  python3 scripts/task229_soup.py build  --models-dir DIR --out DIR
  python3 scripts/task229_soup.py screen --pack PACK --data FRAMES --models a.pt,b.pt --out screen.json

Soups keep every metadata key of the newest member checkpoint; only state_dict
changes, so python/pvz_rollout_policy.load_student reads them unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_rollout_policy import load_student  # noqa: E402
from train_rollout_policy import BatchSource, load, load_batch  # noqa: E402


def _checkpoint(path: Path):
    document = torch.load(path, map_location='cpu', weights_only=False)
    state = document['state_dict']
    return document, state


def average(members: list[Path], out: Path, base: Path | None = None) -> dict:
    """Uniform average of state_dict tensors; float tensors averaged, others must match exactly."""
    loaded = [_checkpoint(path) for path in members]
    keys = set(loaded[0][1])
    for path, (_, state) in zip(members[1:], loaded[1:]):
        if set(state) != keys:
            raise SystemExit(f'{path}: state_dict keys differ from {members[0]}')
    averaged = {}
    for key in keys:
        tensors = [state[key] for _, state in loaded]
        first = tensors[0]
        if any(t.shape != first.shape for t in tensors[1:]):
            raise SystemExit(f'{key}: shape mismatch {[list(t.shape) for t in tensors]}')
        if first.is_floating_point():
            averaged[key] = torch.stack(tensors).mean(0).to(first.dtype)
        else:
            if any(not torch.equal(t, first) for t in tensors[1:]):
                raise SystemExit(f'{key}: non-float tensor differs across members')
            averaged[key] = first.clone()
    document = dict((base and _checkpoint(base)[0]) or loaded[-1][0])
    document['state_dict'] = averaged
    document['soup'] = dict(method='uniform', members=[str(p) for p in members],
                            base=str(base) if base else str(members[-1]))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(document, out)
    return dict(path=str(out), members=[str(p) for p in members], size=out.stat().st_size,
                tensors=len(averaged), params=int(sum(t.numel() for t in averaged.values())))


def screen(pack: Path, data: Path, models: list[Path], batch: int = 1024) -> dict:
    loaded = load(data, labels=set(), pack_file=pack)
    boards, scalars, plant, packet, cell, weight, _seed, _values, cands = loaded[:9]
    packet_info = loaded[9] if len(loaded) > 9 else None
    source = BatchSource(boards, scalars, plant, packet, cell, weight, packet_info, cands, None)
    positive = float(plant[weight > 0].sum())
    waits = float(weight[plant < 0.5].sum())
    pos_weight = torch.tensor([waits / max(1.0, positive)]).sqrt()
    rows = len(plant)
    report = dict(pack=str(pack), rows=rows, pos_weight=float(pos_weight), models={})
    for model in models:
        net = load_student(str(model)).eval()
        totals = dict(weight=0.0, plant=0.0, packet=0.0, cell=0.0,
                      type_correct=0.0, packet_correct=0.0, cell_correct=0.0,
                      joint_correct=0.0, planted=0.0, joint=0.0)
        with torch.no_grad():
            for start in range(0, rows, batch):
                idx = list(range(start, min(start + batch, rows)))
                b, s, y, pk, c, w, info, _cd, _aux = load_batch(source, source.prepare(idx), 'cpu')
                planted = y > 0.5
                out = net(b, s, pk.clamp(min=0), **({'packet_info': info} if info is not None else {}))
                plant_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out['plant'], y, pos_weight=pos_weight, reduction='none')
                pred_plant = out['plant'] > 0
                packet_pred = out['packet'].argmax(1)
                cell_pred = out['cell'].argmax(1)
                joint = (pred_plant == planted) & (~planted | ((packet_pred == pk) & (cell_pred == c)))
                totals['weight'] += float(w.sum())
                totals['plant'] += float((plant_loss * w).sum())
                totals['type_correct'] += float(((pred_plant == planted).float() * w).sum())
                totals['joint_correct'] += float((joint.float() * w).sum())
                totals['joint'] += float(w.sum())
                if planted.any():
                    wp = w[planted]
                    totals['packet'] += float((torch.nn.functional.cross_entropy(
                        out['packet'][planted], pk[planted], reduction='none') * wp).sum())
                    totals['cell'] += float((torch.nn.functional.cross_entropy(
                        out['cell'][planted], c[planted], reduction='none') * wp).sum())
                    totals['packet_correct'] += float((packet_pred[planted] == pk[planted]).float().sum() * 1.0)
                    totals['cell_correct'] += float((cell_pred[planted] == c[planted]).float().sum() * 1.0)
                    totals['planted'] += float(planted.sum())
        wsum = max(totals['weight'], 1e-30)
        planted = max(totals['planted'], 1.0)
        report['models'][str(model)] = dict(
            weighted_loss=(totals['plant'] + totals['packet'] + totals['cell']) / wsum,
            plant_bce=totals['plant'] / wsum,
            packet_ce=totals['packet'] / wsum,
            cell_ce=totals['cell'] / wsum,
            type_acc=totals['type_correct'] / wsum,
            packet_acc=totals['packet_correct'] / planted,
            cell_acc=totals['cell_correct'] / planted,
            joint_acc=totals['joint_correct'] / wsum,
            params=sum(p.numel() for p in net.parameters()))
    ranked = sorted(report['models'], key=lambda key: report['models'][key]['weighted_loss'])
    report['ranking'] = [dict(model=key, **report['models'][key]) for key in ranked]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--models-dir', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    build.add_argument('--base', type=Path, help='checkpoint whose metadata the soup keeps (default: newest member)')
    screen_cmd = sub.add_parser('screen')
    screen_cmd.add_argument('--pack', type=Path, required=True)
    screen_cmd.add_argument('--data', type=Path, required=True)
    screen_cmd.add_argument('--models', required=True)
    screen_cmd.add_argument('--out', type=Path, required=True)
    screen_cmd.add_argument('--batch', type=int, default=1024)
    args = parser.parse_args()
    if args.command == 'build':
        folder = args.models_dir
        groups = {
            'soup_a5': [folder / f'r{round}_s1.pt' for round in range(101, 106)],
            'soup_a4': [folder / f'r{round}_s1.pt' for round in range(102, 106)],
            'soup_ab': [folder / f'r{round}_s{seed}.pt' for round in range(101, 106) for seed in (1, 2)],
        }
        manifest = {name: average([path for path in members if path.exists()], args.out / f'{name}.pt',
                                  base=folder / 'r105_s1.pt')
                    for name, members in groups.items()}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'soup_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        models = [Path(item) for item in args.models.split(',') if item]
        report = screen(args.pack, args.data, models, args.batch)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
