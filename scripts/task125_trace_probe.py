"""Summarize existing gate games and the small task125 trace sample, without replay."""
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_frame_store import read_frames
from pvz_rollout_policy import identity_scalars, load_student, packet_features
from pvz_student_observation import StudentHistory
from train_rollout_policy import _episode_context

BASE = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')
GATE = BASE / 'round_000021/evaluations/40830e597d79325dc7fd'
OUT = ROOT / 'artifacts/task125'


def games(path):
    return {j['seed']: j for j in map(json.loads, path.read_text().splitlines())}


@torch.inference_mode()
def score_snapshot(path, index, nets):
    frames = read_frames(path)
    result = dict(source=str(path.relative_to(ROOT)), input_frame=index,
                  tick=frames[index]['tick'], wave=frames[index]['wave'], sun=frames[index]['sun'],
                  packets=frames[index]['packets'], models={})
    for name, net in nets.items():
        history = StudentHistory(net.obs)
        ctx = _episode_context(path) if net.context else None
        for i, frame in enumerate(frames[:index + 1]):
            board, scalars = history.features(frame, frames[i - 1] if i else None, ctx)
        info = packet_features(frame)[None]
        scalars = identity_scalars(frame, scalars)
        h, g = net.encode(board[None], scalars[None], sparse_input=True, packet_info=info)
        logits = net.packet_logits(g, info)[0]
        perm = torch.tensor([6, 5, 4, 3, 2, 1, 0, 7, 8, 9])
        reordered = info[:, perm]
        _, gp = net.encode(board[None], scalars[None], sparse_input=True, packet_info=reordered)
        perm_logits = net.packet_logits(gp, reordered)[0]
        sunflower_slot = int(torch.nonzero(info[0, :, 0] == 1)[0, 0])
        cell = net.cell_logits(h, g, torch.tensor([sunflower_slot]), info)[0]
        result['models'][name] = dict(
            plant_logit=float(net.plant(g)[0, 0]),
            packet_logits={str(int(info[0, i, 0])): float(logits[i])
                           for i in range(10) if info[0, i, 0] >= 0},
            sunflower_r0c0=float(cell[0]), sunflower_r4c0=float(cell[36]),
            packet_permutation_max_error=float((perm_logits - logits[perm]).abs().max()),
            aggregate_permutation_max_error=float((gp - g).abs().max()))
    return result


def main():
    torch.set_num_threads(1)
    receipts, trace_receipts, snapshots = {}, [], []
    for task in ('day_l7_eval_d7p_holdout1', 'day_l9_eval_d7p_holdout2', 'day_l7_eval_d7'):
        refs = {s: games(GATE / s / task / 'games.jsonl') for s in ('candidate', 'incumbent')}
        harm = {seed for seed, j in refs['candidate'].items() if not j['won'] and refs['incumbent'][seed]['won']}
        gain = {seed for seed, j in refs['candidate'].items() if j['won'] and not refs['incumbent'][seed]['won']}
        receipt = dict(harm=len(harm), gain=len(gain), sides={})
        for side, items in refs.items():
            counter = collections.Counter()
            for seed in harm:
                counter.update(items[seed]['planted_types'])
            receipt['sides'][side] = dict(wins=sum(j['won'] for j in items.values()),
                                         harm_pair_planted_types=dict(counter),
                                         loss_waves=dict(collections.Counter(j['final_wave'] for j in items.values() if not j['won'])))
            trace_path = OUT / 'traces' / side / task
            if not (trace_path / 'games.jsonl').exists():
                continue
            for seed, j in games(trace_path / 'games.jsonl').items():
                fields = ('won', 'final_wave', 'final_tick', 'decisions', 'state_hash', 'planted_types')
                trace_receipts.append(dict(task=task, side=side, seed=seed,
                                           equal={k: j[k] == items[seed][k] for k in fields},
                                           outcome={k: j[k] for k in ('won', 'final_wave', 'final_tick', 'decisions')},
                                           frames_sha256=hashlib.sha256((trace_path / 'frames' / f'seed_{seed}/play_fast/frames.npz').read_bytes()).hexdigest()))
        receipts[task] = receipt
    first_differences = []
    for task, seeds in [('day_l7_eval_d7p_holdout1', (45488, 45489)),
                        ('day_l9_eval_d7p_holdout2', (45694,))]:
        for seed in seeds:
            paths = {s: OUT / 'traces' / s / task / 'frames' / f'seed_{seed}/play_fast/frames.npz'
                     for s in ('candidate', 'incumbent')}
            frames = {s: read_frames(p) for s, p in paths.items()}
            for i, (a, b) in enumerate(zip(frames['candidate'][1:], frames['incumbent'][1:]), 1):
                if a['action'] != b['action']:
                    first_differences.append(dict(task=task, seed=seed, decision=i,
                        tick=frames['candidate'][i-1]['tick'], wave=frames['candidate'][i-1]['wave'],
                        sun=frames['candidate'][i-1]['sun'], candidate=a['action'], incumbent=b['action'],
                        same_prior_frame=frames['candidate'][i-1] == frames['incumbent'][i-1]))
                    if task.endswith('holdout1'):
                        snapshots.append((paths['candidate'], i-1))
                    break
            if seed == 45488:
                for i, f in enumerate(frames['incumbent'][1:], 1):
                    if f['action'].get('packet') == 5 and f['action']['type'] == 'plant':
                        snapshots.append((paths['incumbent'], i-1))
                        break
    nets = {name: load_student(BASE / f'round_{r:06d}/student/model.pt')
            for name, r in (('r11', 11), ('r21_A', 21))}
    scored = [score_snapshot(p, i, nets) for p, i in snapshots]
    cohort = set(range(45488, 45520))
    comparison = {}
    for name, path in [('r11', GATE / 'incumbent/day_l7_eval_d7p_holdout1/games.jsonl'),
                       ('r21_A_seed1', GATE / 'candidate/day_l7_eval_d7p_holdout1/games.jsonl'),
                       *[(label, OUT / 'additional' / label / 'games.jsonl')
                         for label in ('round20_seed1', 'round20_seed2', 'round21_recipeB_seed2')]]:
        js = games(path)
        comparison[name] = dict(wins=sum(js[s]['won'] for s in cohort), games=len(cohort),
                                outcomes={str(s): js[s]['won'] for s in sorted(cohort)},
                                games_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    receipt = dict(tasks=receipts, trace_reproduction=trace_receipts,
                   first_action_differences=first_differences, shared_state_scores=scored,
                   same_first32_seeds=sorted(cohort), first32_models=comparison,
                   additional_native_games=96, recorded_trace_games=6,
                   warning='32-seed diagnostic sample; not a replacement promotion gate or causal retraining ablation')
    (OUT / 'heldout_probe.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(first_differences=first_differences, scores=scored,
                          first32={k:v['wins'] for k,v in comparison.items()}), indent=2))


if __name__ == '__main__':
    main()
