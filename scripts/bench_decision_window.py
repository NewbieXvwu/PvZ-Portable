"""Measure real task162 optimizer throughput in a production-admitted GPU window.

Uses stratified actual r42 rows and corresponding public frame tokens, not synthetic
tensors. No full token sidecar is needed to time these bounded training samples.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import signal
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
import numpy as np
import torch
import torch.nn.functional as F
from gpu_window_runner import write_json
from train_rollout_policy import read_pack
from pvz_frame_store import read_frames
from pvz_student_tokens import frame_tokens, collate_tokens
from pvz_rollout_policy import create_student
from pvz_recipe import load_recipe
from eval_window_policy import terrain_for_level
from decision_window_budget import windows_48h, choose_budget, markdown


def samples(sources):
    db = sqlite3.connect(sources/'sources.sqlite')
    records = list(db.execute('SELECT source,n FROM sources ORDER BY source'))
    offsets = {}; at = 0
    for source, n in records:
        offsets[source] = at; at += n
    random.Random(162).shuffle(records)
    selected, tokens, strata = [], [], {}
    for source, n in records:
        meta = json.loads((sources/'raw'/Path(source).parent/'meta.json').read_text())
        level = meta.get('context', {}).get('level')
        if level is None:
            continue
        terrain = terrain_for_level(level)
        if strata.get(terrain, 0) >= 6:
            continue
        raw_rows = np.frombuffer(db.execute('SELECT rows FROM sources WHERE source=?', (source,)).fetchone()[0], np.int32)
        frames = read_frames(sources/'raw'/source)
        for local in np.unique(np.linspace(0, n-1, min(8, n), dtype=int)):
            selected.append(offsets[source]+local)
            tokens.append(frame_tokens(frames[int(raw_rows[local])-1]))
        strata[terrain] = strata.get(terrain, 0)+1
        if len(strata) == 5 and min(strata.values()) >= 6:
            break
    return np.asarray(selected), tokens, strata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--measure-seconds', type=float, default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    interrupted, ready = False, False
    def stop(*_):
        nonlocal interrupted
        if not ready:
            raise SystemExit(75)
        interrupted = True
    signal.signal(signal.SIGUSR1, stop); signal.signal(signal.SIGTERM, stop)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU probe requires CUDA; no CPU proxy rate is accepted')
    torch.cuda.set_per_process_memory_fraction(.22)
    corpus = json.loads((args.state/'corpus/corpus.json').read_text())
    study = json.loads((args.state/'decision-study.json').read_text())
    manifest = json.loads((ROOT/'configs/decision_r42.json').read_text())
    sources = args.state/'decision-data/split'
    loaded = read_pack(Path(corpus['pack']), Path(load_recipe(Path(study['base_recipe']))['data']),
                       merge_dir=Path(corpus['merge_dir']))
    boards, scalars, plants, packets, cells, weight, seeds, values, candidates, info = loaded
    selected, public_tokens, strata = samples(sources)
    unique = sorted(set(seeds)); random.Random(0).shuffle(unique)
    validation = set(unique[:int(len(unique)*.1)])
    is_val = np.fromiter((s in validation for s in seeds), dtype=bool, count=len(seeds))
    is_val |= np.load(sources/'unseen.npy', mmap_mode='r')
    training_rows = int((~is_val).sum())
    progress = args.out/'progress.json'
    state = json.loads(progress.read_text()) if progress.exists() else dict(arms={})
    ready = True
    for arm in manifest['arms']:
        name = arm['name']
        if name in state['arms']:
            continue
        recipe_path = next(args.state.joinpath('decision-recipes').glob(f'r42_{name}_s1.json'))
        recipe = load_recipe(recipe_path)
        micro = arm['micro_batch_size']
        while True:
            net = create_student(arm['arch'], width=arm['width'], obs='v2timer', pool=recipe['pool'],
                row_pool=recipe['row_pool'], packet_head='identity').cuda().train()
            optimizer = torch.optim.AdamW(net.parameters(), **{k:v for k,v in recipe['optimizer'].items() if k!='name'})
            elapsed, rows, steps = 0., 0, 0
            torch.cuda.reset_peak_memory_stats()
            try:
                while elapsed < args.measure_seconds or steps < 3:
                    local = np.random.default_rng(162+steps).choice(len(selected), size=recipe['batch_size'])
                    ids = selected[local]
                    ww, pp = weight[ids], plants[ids].bool()
                    denom = ww.sum().clamp_min(1e-30).cuda()
                    plant_denom = ww[pp].sum().clamp_min(1e-30).cuda()
                    cd = candidates[ids].float()
                    same = (cd[:,:,0] == packets[ids,None]) & ~torch.isnan(cd[:,:,2]) & pp[:,None]
                    cell_denom = ww[same.sum(1)>=2].sum().clamp_min(1e-30).cuda()
                    torch.cuda.synchronize(); began = time.perf_counter()
                    optimizer.zero_grad()
                    for begin in range(0, len(ids), micro):
                        if interrupted:
                            raise SystemExit(75)
                        idx, loc = ids[begin:begin+micro], local[begin:begin+micro]
                        y, pk, cell, w = plants[idx].cuda().float(), packets[idx].cuda().long(), cells[idx].cuda().long(), weight[idx].cuda()
                        kwargs = dict(packet_info=info[idx].cuda())
                        if arm.get('tokens'):
                            kwargs['tokens'] = collate_tokens([public_tokens[i] for i in loc], 'cuda')
                        out = net(boards.batch(idx, 'cuda'), scalars.batch(idx, 'cuda'), pk.clamp_min(0), **kwargs)
                        loss = (F.binary_cross_entropy_with_logits(out['plant'], y, reduction='none')*w).sum()/denom
                        planted = y > .5
                        if planted.any():
                            loss = loss + ((F.cross_entropy(out['packet'][planted], pk[planted], reduction='none')+
                                F.cross_entropy(out['cell'][planted], cell[planted], reduction='none'))*w[planted]).sum()/plant_denom
                        cand = candidates[idx].cuda().float()
                        mask = (cand[:,:,0] == pk[:,None]) & ~torch.isnan(cand[:,:,2]) & planted[:,None]
                        known = mask.sum(1)>=2
                        if known.any():
                            logits = out['cell'][known].gather(1, cand[known,:,1].nan_to_num(0).clamp_min(0).long())
                            student = logits.masked_fill(~mask[known], -1e9).log_softmax(1)
                            teacher = (cand[known,:,2].nan_to_num(0)/recipe['soft_temperature']).masked_fill(~mask[known], -1e9).softmax(1)
                            kl = (teacher*(teacher.clamp_min(1e-12).log()-student)).masked_fill(~mask[known], 0).sum(1)
                            loss = loss + recipe['cell_soft_weight']*(kl*w[known]).sum()/cell_denom
                        if getattr(net, 'routing_loss', None) is not None:
                            loss = loss + .01*net.routing_loss*len(idx)/len(ids)
                        loss.backward()
                    optimizer.step(); torch.cuda.synchronize()
                    seconds = time.perf_counter()-began
                    if steps >= 2:
                        elapsed += seconds; rows += len(ids)
                    steps += 1
                state['arms'][name] = dict(rows_per_second=rows/elapsed, rows=rows, seconds=elapsed,
                    micro_batch_size=micro, batch_size=recipe['batch_size'], peak_cuda_bytes=torch.cuda.max_memory_reserved())
                write_json(progress, state)
                print(json.dumps(dict(arm=name, **state['arms'][name])), flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                if micro == 1:
                    raise
                micro = max(1, micro//2)
                print(f'{name}: reducing probe micro batch to {micro}', flush=True)
            finally:
                out = loss = kwargs = logits = student = teacher = kl = cand = None
                del net, optimizer
                gc.collect(); torch.cuda.empty_cache()
    windows = windows_48h(Path.home()/'pvz-logs/curriculum_state.pipeline')
    budget = choose_budget({name: record['rows_per_second'] for name, record in state['arms'].items()}, training_rows,
                           windows['eligible_fraction'])
    host_free = min(shutil.disk_usage(path).free for path in ('/mnt/c','/mnt/d') if Path(path).exists())
    cache_cap = min(2*(1<<30), max(0, host_free-13*(1<<30)))
    if cache_cap < (1<<27):
        raise RuntimeError(f'not enough host disk headroom for the bounded token cache: {host_free} bytes free')
    sizes = [len(sample['ids'])*11 + np.count_nonzero(sample['features'])*5 + 41 for sample in public_tokens]
    bytes_per_row = float(np.percentile(sizes, 95))*1.5
    disk_fraction = min(1.0, .95*cache_cap/(training_rows*bytes_per_row))
    if disk_fraction < budget['train_row_fraction']:
        ratio = disk_fraction/budget['train_row_fraction']
        budget['train_row_fraction'] = disk_fraction
        for key in ('planned_seed1_days','planned_seed1_gpu_hours','planned_all_gpu_hours','planned_all_days'):
            budget[key] *= ratio
        for entry in budget['per_job'].values():
            entry['planned_gpu_hours'] *= ratio
    budget.update(sidecar_max_bytes=cache_cap, estimated_sidecar_bytes=training_rows*budget['train_row_fraction']*bytes_per_row,
        sparse_bytes_per_row_with_margin=bytes_per_row, host_free_bytes=host_free,
        disk_rule='at most 2 GiB token cache, leave 13 GiB host headroom at planning and stop construction below 12 GiB; CPU heldout tokens stream from original archives')
    report = dict(revision=args.revision, measured=time.time(), windows=windows, budget=budget, strata=strata,
                  rates=state['arms'], sample_rows=len(selected))
    write_json(args.out/'report.json', report)
    (args.out/'report.md').write_text(markdown(report))
    write_json(args.out/'ready.json', dict(budget, revision=args.revision,
        micro_batches={name: record['micro_batch_size'] for name, record in state['arms'].items()}))


if __name__ == '__main__':
    main()
