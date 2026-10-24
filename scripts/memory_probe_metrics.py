"""Task58 held-out metrics, paired episode uncertainty, and CPU streaming costs."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_memory_probe import CONFIGS, make_probe
from memory_probe_data import OUT, digest, write_json


def auc(y, p):
    y = np.asarray(y, dtype=bool).ravel()
    p = np.asarray(p).ravel()
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)) + 1, len(p)]
    starts = np.r_[0, ends[:-1]]
    ranks = (starts + 1 + ends) / 2.
    rank_sum = np.dot(np.add.reduceat(sorted_y.astype(np.int64), starts), ranks)
    return float((rank_sum - positives * (positives+1)/2) / (positives*negatives))


def stats(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if not len(y):
        return dict(cells=0, positive=0, auc=None, logloss=None)
    ranking_auc = auc(y, p)
    clipped = np.clip(p, 1e-7, 1-1e-7)
    loss = -np.where(y, np.log(clipped), np.log1p(-clipped))
    return dict(cells=len(y), positive=int(y.sum()), auc=ranking_auc, logloss=float(loss.mean()))


def masks(data, events):
    fog, age, ticks = data['fog'], data['age'], data['tick']
    result = dict(hidden=fog, age_0_120=fog & (age >= 0) & (age <= 120),
                  age_121_600=fog & (age > 120) & (age <= 600),
                  age_over_600=fog & (age > 600), never_visible=fog & (age < 0))
    before, after = np.zeros_like(fog), np.zeros_like(fog)
    for event in events:
        tick = event['tick']
        post_rows = np.flatnonzero((ticks >= tick) & (ticks <= tick + 600))
        if not len(post_rows):
            continue
        common_cells = fog[post_rows[-1]]
        before |= ((ticks >= tick-600) & (ticks < tick))[:, None, None] & common_cells
        after |= ((ticks >= tick) & (ticks <= tick+600))[:, None, None] & common_cells & fog
    result.update(return_before=before, return_after=after)
    return result


def measure_run(kind, arm, seed, episodes):
    folder = OUT / 'models' / f'{kind}-{arm}-s{seed}'
    result = json.loads((folder / 'result.json').read_text())
    assert result['corpus_sha256'] == digest(OUT/'corpus.json')
    assert result['checkpoint_sha256'] == digest(folder/'model.pt')
    buckets, episode_metrics = {}, []
    type_buckets = [[] for _ in range(34)]
    for row in episodes:
        path = OUT / 'episodes' / f"{row['id']}.npz"
        assert digest(path) == row['pack_sha256']
        with np.load(path, allow_pickle=False) as data, np.load(
                folder / 'test' / f"{row['id']}.npz", allow_pickle=False) as pred:
            y, probability = data['target'], pred['probabilities']
            assert probability.shape == y.shape
            assert np.isfinite(probability).all()
            episode = dict(id=row['id'], level=row['level'], seed=row['seed'], returns=len(row['returns']))
            for name, mask in masks(data, row['returns']).items():
                values = (y[:, 0][mask], probability[:, 0][mask])
                buckets.setdefault(name, []).append(values)
                episode[name] = stats(*values)
            for zombie_type in range(34):
                type_buckets[zombie_type].append((y[:, zombie_type+1][data['fog']],
                                                  probability[:, zombie_type+1][data['fog']]))
            episode_metrics.append(episode)
    combined = {name: stats(np.concatenate([a for a, b in pairs]),
                            np.concatenate([b for a, b in pairs])) for name, pairs in buckets.items()}
    types = {}
    for kind_id, pairs in enumerate(type_buckets):
        y, p = np.concatenate([a for a,b in pairs]), np.concatenate([b for a,b in pairs])
        metric = stats(y, p)
        # Empty types have no AUC; keep their count explicit without a misleading macro average.
        types[str(kind_id)] = metric
    macro = [v['auc'] for v in types.values() if v['auc'] is not None]
    return dict(kind=kind, arm=arm, seed=seed, parameters=result['parameters'],
        best_epoch=result['best_epoch'], validation_logloss=result['validation_hidden_logloss'],
        elapsed_seconds=result['elapsed_seconds'], metrics=combined, types=types,
        type_macro_auc=float(np.mean(macro)) if macro else None,
        episodes=episode_metrics)


def paired_comparisons(runs, bootstrap=2000):
    lookup = {(r['kind'],r['arm'],r['seed']):r for r in runs}
    outputs = {}
    rng = np.random.default_rng(58)
    for bucket in ('hidden', 'age_0_120', 'age_121_600', 'age_over_600', 'return_after'):
        outputs[bucket] = {}
        for kind in CONFIGS[1:]:
            seed_results = []
            episode_losses = []
            for seed in (1,2,3):
                base, memory, ceiling = [lookup[k, a, seed] for k,a in
                    [('current','human'),(kind,'human'),('current','xray')]]
                bl, ml, xl = [r['metrics'][bucket]['logloss'] for r in (base,memory,ceiling)]
                ba, ma, xa = [r['metrics'][bucket]['auc'] for r in (base,memory,ceiling)]
                seed_results.append(dict(seed=seed, logloss_improvement=bl-ml if bl is not None else None,
                    logloss_gap_recovered=(bl-ml)/(bl-xl) if bl is not None and bl>xl else None,
                    auc_gap_recovered=(ma-ba)/(xa-ba) if None not in (ba,ma,xa) and xa>ba else None))
                rows = []
                for b,m,x in zip(base['episodes'],memory['episodes'],ceiling['episodes']):
                    assert b['id'] == m['id'] == x['id']
                    n = b[bucket]['cells']
                    assert n == m[bucket]['cells'] == x[bucket]['cells']
                    rows.append([n, *[e[bucket]['logloss'] or 0 for e in (b,m,x)]])
                episode_losses.append(rows)
            values = np.mean(np.asarray(episode_losses), axis=0)
            n = values[:,0]
            selected = rng.integers(0, len(n), size=(bootstrap,len(n)))
            counts = n[selected].sum(1)
            sums = (values[:,1:] * n[:,None])[selected].sum(1)
            valid = counts > 0
            means = sums[valid] / counts[valid,None]
            delta = means[:,0] - means[:,1]
            gaps = means[:,0] - means[:,2]
            recovered = delta[gaps>0] / gaps[gaps>0]
            outputs[bucket][kind] = dict(seeds=seed_results,
                episode_bootstrap_logloss_improvement_95ci=np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
                episode_bootstrap_recovered_95ci=np.quantile(recovered,[.025,.975]).tolist() if len(recovered) else None,
                episodes_with_cells=int(np.sum(n>0)),
                bootstrap_valid=len(delta), bootstrap_samples=bootstrap,
                note='paired episode bootstrap; average initialization losses before resampling; frame-count weighted')
    return outputs


def benchmark():
    torch.set_num_threads(1)
    torch.manual_seed(58)
    results = {}
    for kind in CONFIGS:
        model = make_probe(kind).eval()
        b, s, dt = torch.zeros(1,1,102,6,10), torch.zeros(1,1,157), torch.full((1,1),60.)
        # Nonzero synthetic encoded values avoid denormal benchmarks and do not affect training.
        b.normal_(); s.normal_()
        state, timings = None, []
        with torch.inference_mode():
            for step in range(1100):
                start = time.perf_counter_ns()
                _, state = model(b,s,dt,state)
                elapsed = (time.perf_counter_ns()-start)/1000
                if step >= 100:
                    timings.append(elapsed)
        results[kind] = dict(width=model.width, parameters=sum(p.numel() for p in model.parameters()),
            p50_us=float(np.median(timings)),p95_us=float(np.quantile(timings,.95)),
            state_bytes=state.numel()*state.element_size() if state is not None else 0)
    cpu = next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')), platform.processor())
    return dict(cpu=cpu,torch_version=torch.__version__,threads=1,batch=1,warmup=100,steps=1000,
                scope='encoded tensors through output logits; includes streaming state update; excludes v2fog encoding',
                configurations=results)


def report():
    corpus = json.loads((OUT / 'corpus.json').read_text())
    episodes = [e for e in corpus['episodes'] if e['split']=='test']
    runs=[]
    for seed in (1,2,3):
        for kind in CONFIGS:
            for arm in ('human','xray'):
                r = measure_run(kind,arm,seed,episodes)
                runs.append(r)
                print(json.dumps({k:r[k] for k in ('kind','arm','seed')} | r['metrics']['hidden']),flush=True)
    result = dict(corpus_sha256=digest(OUT/'corpus.json'), test_episodes=len(episodes),
        test_returns=sum(len(e['returns']) for e in episodes), runs=runs,
        comparisons=paired_comparisons(runs), cpu=benchmark())
    # Per-episode arrays stay with the ignored evidence; git carries conclusions only.
    write_json(OUT / 'evaluation.json', result)
    result['full_evaluation_sha256'] = digest(OUT / 'evaluation.json')
    result['runs'] = [{k: v for k, v in run.items() if k != 'episodes'} for run in runs]
    write_json(ROOT / 'MEMORY/docs/memory-probe-results.json',result)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['report','benchmark'])
    args=p.parse_args()
    if args.stage=='benchmark':
        write_json(OUT/'cpu-benchmark.json',benchmark())
    else:
        report()
