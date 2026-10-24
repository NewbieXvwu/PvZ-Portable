"""Task58 bounded, restartable, same-data training and held-out prediction."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Replace this process before heavy imports when it is only scheduling work.
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'run-all':
    os.execv(sys.executable,[sys.executable,str(Path(__file__).with_name('memory_probe_queue.py')),
                            *sys.argv[2:]])

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_memory_probe import CONFIGS, PREDICTION_DIR, configure_probe_inference, make_probe
from memory_probe_data import OUT, digest, write_json


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def load_episodes(arm, splits):
    corpus = json.loads((OUT / 'corpus.json').read_text())
    result = []
    for row in corpus['episodes']:
        if row['split'] not in splits:
            continue
        path = OUT / 'episodes' / f"{row['id']}.npz"
        assert digest(path) == row['pack_sha256']
        with np.load(path, allow_pickle=False) as data:
            result.append(dict(meta=row, board=torch.from_numpy(data[f'{arm}_board']),
                scalar=torch.from_numpy(data[f'{arm}_scalar']), target=torch.from_numpy(data['target']),
                dt=torch.from_numpy(data['dt'].astype(np.float32)), fog=torch.from_numpy(data['fog'])))
    return result


def batches(episodes, device, batch_size, shuffle):
    order = list(range(len(episodes)))
    if shuffle:
        random.shuffle(order)
    for base in range(0, len(order), batch_size):
        group = [episodes[i] for i in order[base:base+batch_size]]
        length, batch = max(len(e['board']) for e in group), len(group)
        packed = {}
        for name in ('board', 'scalar', 'target', 'dt', 'fog'):
            example = group[0][name]
            values = torch.zeros((length, batch, *example.shape[1:]), dtype=example.dtype)
            for j, e in enumerate(group):
                values[:len(e[name]), j] = e[name]
            packed[name] = values.to(device)
        valid = torch.zeros((length, batch, 1, 1), dtype=torch.bool, device=device)
        for j, e in enumerate(group):
            valid[:len(e['board']), j] = True
        packed['valid'] = valid
        yield packed


def objective(logits, target, valid):
    # Unweighted reconstruction on every valid lawn cell, with a conditional type head.
    target = target.float()
    mask = valid.expand_as(target[:, :, 0])
    occupancy = F.binary_cross_entropy_with_logits(logits[:, :, 0], target[:, :, 0], reduction='none')
    occ_loss = occupancy[mask].mean()
    occupied = mask & target[:, :, 0].bool()
    types = F.binary_cross_entropy_with_logits(logits[:, :, 1:], target[:, :, 1:], reduction='none')
    type_loss = types.permute(0, 1, 3, 4, 2)[occupied].mean() if occupied.any() else types.sum() * 0
    return occ_loss + .25 * type_loss


def validate(model, episodes, device):
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for batch in batches(episodes, device, 8, False):
            state = None
            for base in range(0, len(batch['board']), 32):
                sl = slice(base, base+32)
                logits, state = model(batch['board'][sl], batch['scalar'][sl], batch['dt'][sl], state)
                loss = F.binary_cross_entropy_with_logits(logits[:, :, 0],
                    batch['target'][sl, :, 0].float(), reduction='none')
                mask = batch['fog'][sl] & batch['valid'][sl]
                total += float(loss[mask].sum())
                count += int(mask.sum())
    if not count:
        raise RuntimeError('no validation hidden cells')
    return total / count


def train_one(kind, arm, seed, epochs):
    torch.set_num_threads(1)
    available = mem_available()
    # Imports and other workloads can consume memory between the parent's check and
    # child startup. Remain before CUDA initialization until the child's gate passes.
    while available < 10 * 1024 ** 3:
        print(json.dumps(dict(wait='child MemAvailable', bytes=available,
                              gpu_training_started=False)), flush=True)
        time.sleep(30)
        available = mem_available()
    gate = dict(mem_available_bytes=available, required_bytes=10 * 1024 ** 3,
                time_limit_seconds=3600, start_unix=time.time(), kind=kind, arm=arm, seed=seed)
    folder = OUT / 'models' / f'{kind}-{arm}-s{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'resource_gate.json', gate)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    episodes = load_episodes(arm, ('train', 'validation'))
    train = [e for e in episodes if e['meta']['split'] == 'train']
    validation = [e for e in episodes if e['meta']['split'] == 'validation']
    model = make_probe(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
    best, best_epoch, state_dict = float('inf'), 0, None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        total, steps = 0., 0
        for batch in batches(train, device, 8, True):
            state = None
            for base in range(0, len(batch['board']), 32):
                sl = slice(base, base+32)
                if not batch['valid'][sl].any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits, state = model(batch['board'][sl], batch['scalar'][sl], batch['dt'][sl], state)
                loss = objective(logits, batch['target'][sl], batch['valid'][sl])
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if state is not None:
                    state = state.detach()
                total += float(loss.detach())
                steps += 1
        val = validate(model, validation, device)
        record = dict(epoch=epoch, train_loss=total/steps, validation_hidden_logloss=val,
                      elapsed_seconds=time.time()-gate['start_unix'])
        print(json.dumps(record), flush=True)
        history.append(record)
        if val < best:
            best, best_epoch = val, epoch
            state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
    torch.save(dict(state_dict=state_dict, kind=kind, arm=arm, seed=seed,
                    width=model.width), folder / 'model.pt')
    summary = dict(**gate, elapsed_seconds=time.time()-gate['start_unix'], epochs=epochs,
        best_epoch=best_epoch, validation_hidden_logloss=best,
        parameters=sum(p.numel() for p in model.parameters()), width=model.width,
        train_episodes=len(train), validation_episodes=len(validation),
        corpus_sha256=digest(OUT/'corpus.json'), checkpoint_sha256=digest(folder/'model.pt'),
        history=history, torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    write_json(folder / 'result.json', summary)


def run_all(epochs):
    os.execv(sys.executable,[sys.executable,str(ROOT/'scripts/memory_probe_queue.py'),
                            '--epochs',str(epochs)])


def predict():
    torch.set_num_threads(1)
    configure_probe_inference()
    device = torch.device('cuda')
    # No optimization: held-out inference uses the validation-selected checkpoint only.
    for arm in ('human', 'xray'):
        episodes = load_episodes(arm, ('test',))
        for seed in (1, 2, 3):
            for kind in CONFIGS:
                folder = OUT / 'models' / f'{kind}-{arm}-s{seed}'
                result = json.loads((folder / 'result.json').read_text())
                assert result['corpus_sha256'] == digest(OUT/'corpus.json')
                assert result['checkpoint_sha256'] == digest(folder/'model.pt')
                model = make_probe(kind).to(device)
                ckpt = torch.load(folder / 'model.pt', map_location=device, weights_only=True)
                model.load_state_dict(ckpt['state_dict'])
                model.eval()
                for episode in episodes:
                    target = folder / PREDICTION_DIR / f"{episode['meta']['id']}.npz"
                    if target.exists():
                        with np.load(target, allow_pickle=False) as existing:
                            assert str(existing['checkpoint_sha256']) == result['checkpoint_sha256']
                            assert str(existing['corpus_sha256']) == result['corpus_sha256']
                            assert str(existing['episode_pack_sha256']) == episode['meta']['pack_sha256']
                            assert str(existing['precision']) == 'ieee_fp32'
                        continue
                    state, outputs = None, []
                    with torch.inference_mode():
                        for base in range(0, len(episode['board']), 32):
                            sl = slice(base, base+32)
                            logits, state = model(episode['board'][sl, None].to(device),
                                episode['scalar'][sl, None].to(device),
                                episode['dt'][sl, None].to(device), state)
                            probabilities = logits.sigmoid()[:, 0]
                            probabilities[:, 1:] *= probabilities[:, :1]
                            outputs.append(probabilities.cpu().numpy())
                    target.parent.mkdir(exist_ok=True)
                    np.savez_compressed(target, probabilities=np.concatenate(outputs),
                        checkpoint_sha256=result['checkpoint_sha256'],
                        corpus_sha256=result['corpus_sha256'],
                        episode_pack_sha256=episode['meta']['pack_sha256'], precision='ieee_fp32')
                print('PREDICTED ' + folder.name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['train-one', 'run-all', 'predict'])
    p.add_argument('--kind', choices=CONFIGS)
    p.add_argument('--arm', choices=['human', 'xray'])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--epochs', type=int, default=20)
    args = p.parse_args()
    if args.stage == 'train-one':
        train_one(args.kind, args.arm, args.seed, args.epochs)
    elif args.stage == 'run-all':
        run_all(args.epochs)
    else:
        predict()
