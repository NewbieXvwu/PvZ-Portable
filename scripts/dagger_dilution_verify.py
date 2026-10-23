"""Audit the frozen task15 execution without changing models, frames or inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(plan, folder):
    freeze = read(folder / 'evaluation_freeze.json')
    for mapping in (plan['source_files'], freeze['source_files']):
        for name, digest in mapping.items():
            assert sha(ROOT / name) == digest, name
    contract = plan['eval_contract']
    assert sha(folder / 'frozen_eval_manifest.json') == contract['manifest_sha256']
    assert sha(contract['executable']) == contract['executable_sha256']
    return {'training_files': len(plan['source_files']),
            'evaluation_files': len(freeze['source_files']),
            'evaluation_freeze_sha256': sha(folder / 'evaluation_freeze.json'),
            'manifest_sha256': contract['manifest_sha256'],
            'executable_sha256': contract['executable_sha256']}


def verify_model(job, plan, plan_sha):
    import torch
    out = Path(job['out'])
    record = read(out / 'experiment_complete.json')
    assert record['name'] == job['name'] and record['seed'] == job['seed']
    assert record['cohort'] == job['cohort'] and record['run_plan_sha256'] == plan_sha
    if 'recipe' in job:
        expected_command = [str(ROOT/'scripts/train_rollout_policy.py'), '--recipe', job['recipe']]
    else:
        # Historical result validation only: these arguments are never launched.
        expected_command = [str(ROOT / 'scripts/train_rollout_policy.py'), *plan['common_args'],
                            '--labels', ','.join(plan['cohorts'][job['cohort']]['labels']),
                            '--seed', str(job['seed']), '--out', str(out), *job['extra_args']]
    assert record['command'] == expected_command
    digest = sha(out / 'model.pt')
    assert digest == record['model_sha256']
    model = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    expected = {'width': 64, 'pool': 'meanmax', 'packet_balance': .5, 'context': False,
                'obs': 'v2', 'soft_weight': 0., 'spread_weight': 8., 'spread_cap': 10.,
                'lr_schedule': 'cosine', 'ema': 0., 'cell_soft_weight': 3., 'label_weight': {}}
    assert all(model[k] == v for k, v in expected.items()), job['name']
    assert model['labels'].split(',') == plan['cohorts'][job['cohort']]['labels']
    assert all(torch.isfinite(v).all() for v in model['state_dict'].values())
    metrics = read(out / 'metrics.json')
    assert model['history'] == metrics['history']
    history = model['history']
    steps = (42456 if job['arm'] == 'S'
             else plan['cohorts'][job['cohort']]['optimizer_steps_12epochs'])
    assert len(history) == (10 if job['arm'] == 'S' else 12)
    assert [v['epoch'] for v in history] == list(range(1, len(history) + 1))
    assert all(math.isfinite(v) for row in history for v in row.values())
    if job['arm'] == 'S':
        assert model['optimizer_steps'] == model['max_steps'] == steps
    else:
        assert 'optimizer_steps' not in model and 'max_steps' not in model
    if job['arm'] in 'MTIR':
        spec = plan['row_weight_specs'][job['arm']]
        meta = model['row_intervention']
        assert meta['name'] == job['arm'] and meta['multipliers_sha256'] == spec['multipliers_sha256']
        assert meta['row_identity_sha256'] == spec['row_identity_sha256']
        assert meta['sha256'] == sha(meta['path'])
    else:
        assert 'row_intervention' not in model
    assert metrics['data_memory']['rows'] == plan['cohorts'][job['cohort']]['rows']
    return {'name': job['name'], 'seed': job['seed'], 'cohort': job['cohort'],
            'model_sha256': digest, 'epochs': len(history), 'optimizer_steps': steps,
            'steps_evidence': 'checkpoint counter' if job['arm'] == 'S' else 'full epochs and frozen loop',
            'last_val_loss': history[-1]['val_loss'], 'wall_s': record['wall_s'],
            'data_loads_so_far': record['data_loads_so_far']}


def verify_inventory(plan, folder):
    path = folder / 'audit/inventory.json'
    assert sha(path) == plan['audit_inventory_sha256']
    entries = read(path)
    total = 0
    for i, entry in enumerate(entries):
        source = Path(entry['source'])
        stat = source.stat()
        assert stat.st_size == entry['source_bytes'] and stat.st_mtime_ns == entry['source_mtime_ns'], str(source)
        assert sha(source) == entry['source_sha256'], str(source)
        assert sha(source.parent / 'meta.json') == entry['meta_sha256'], str(source)
        total += stat.st_size
        if (i + 1) % 4096 == 0:
            print(json.dumps({'verified_source_archives': i + 1}), flush=True)
    assert len(entries) == 33919
    return {'source_archives': len(entries), 'bytes_read': total,
            'inventory_sha256': plan['audit_inventory_sha256'],
            'all_archive_and_metadata_sha256_unchanged': True,
            'all_sizes_and_mtimes_unchanged': True}


def verify_weight_files(plan, folder):
    import numpy as np
    reports = {}
    for name, spec in plan['row_weight_specs'].items():
        assert read(folder / 'interventions' / (name + '.json')) == spec
        values = np.load(folder / 'interventions' / spec['multipliers'], mmap_mode='r', allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (spec['rows'],)
        digest = hashlib.sha256()
        for start in range(0, len(values), 1_000_000):
            digest.update(values[start:start + 1_000_000].tobytes())
        assert digest.hexdigest() == spec['multipliers_sha256']
        reports[name] = {'rows': len(values), 'multipliers_sha256': digest.hexdigest()}
    return reports


def verify_inference(stats, plan, folder):
    """Rebuild raw paired clusters; check Student tails with incomplete beta."""
    import numpy as np
    import mpmath as mp
    sys.path.insert(0, str(ROOT / 'scripts'))
    from render_episode import load_training_task
    contract = plan['eval_contract']
    seeds = load_training_task(folder / 'frozen_eval_manifest.json', contract['task_ids'][0])['seeds'][:1024]
    assert len(set(seeds)) == 1024
    wins = {a: np.empty((2, 6, 1024), dtype=float) for a in 'BFMTIRS'}
    for job in plan['jobs']:
        for i, task in enumerate(contract['task_ids']):
            rows = read_games(folder / 'eval' / job['name'] / task / 'games.jsonl')
            assert set(rows) == set(seeds)
            wins[job['arm']][job['seed'] - 1, i] = [int(rows[s]['won']) for s in seeds]
    for arm, values in wins.items():
        result = stats['arms'][arm]
        assert result['wins_by_training_seed_and_tier'] == values.sum(axis=2).astype(int).tolist()
        assert np.allclose(result['mean_rate_by_training_seed'], values.mean(axis=(1, 2)), rtol=0, atol=1e-15)
        assert np.isclose(result['macro_win_rate'], values.mean(), rtol=0, atol=1e-15)
    pvalues = {}
    for name in stats['primary_family']:
        a, b = name.split('-')
        delta = wins[a] - wins[b]
        clusters = delta.mean(axis=(0, 1))
        result = stats['contrasts'][name]
        assert np.isclose(result['difference'], clusters.mean(), rtol=0, atol=1e-15)
        assert np.allclose(result['per_training_seed_difference'], delta.mean(axis=(1, 2)), rtol=0, atol=1e-15)
        assert result['df'] == 1023
        if clusters.std(ddof=1) == 0:
            pvalue = 0. if clusters.mean() else 1.
        else:
            t = float(abs(clusters.mean()) / (clusters.std(ddof=1) / math.sqrt(1024)))
            with mp.workdps(40):
                x = mp.mpf(1023) / (1023 + mp.mpf(t)**2)
                pvalue = float(mp.betainc(mp.mpf(1023)/2, mp.mpf('.5'), 0, x, regularized=True))
        assert abs(pvalue - result['t_p']) < 1e-9, (name, pvalue, result['t_p'])
        pvalues[name] = pvalue
    maximum = 0.
    for i, name in enumerate(sorted(pvalues, key=pvalues.get)):
        maximum = max(maximum, min(1., (7 - i) * pvalues[name]))
        assert abs(maximum - stats['contrasts'][name]['holm_p']) < 7e-9
    return {'raw_games_recounted': 86016, 'clusters': 1024,
            'independent_incomplete_beta_pvalues_match': True, 'independent_holm7_match': True}


def read_games(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1024
    result = {r['seed']: r for r in rows}
    assert len(result) == 1024 and all(r['arm'] == 'fast' and r['searched'] == 0 for r in rows)
    return result


def resource_summary(folder):
    stages = {'training': folder / 'training.result.json',
              'evaluation_all': folder / 'evaluation_all.result.json'}
    stages.update({str(p.relative_to(folder)): p for p in (folder / 'eval').glob('eval_*.result.json')})
    results = {k: read(p) for k, p in stages.items()}
    assert len(results) == 86 and all(v['returncode'] == 0 for v in results.values())
    pauses = {}
    for path in [folder / 'training.resources.jsonl', folder / 'evaluation_all.resources.jsonl']:
        paused_samples = 0
        previous = -1.
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row['elapsed_s'] >= previous
            previous = row['elapsed_s']
            paused_samples += bool(row['paused'])
        pauses[path.stem] = paused_samples
    return {'stages': results, 'peak_rss_kib': max(v['peak_rss_kib'] for v in results.values()),
            'peak_swap_kib': max(v['peak_swap_kib'] for v in results.values()),
            'minimum_available_kib': min(v['minimum_available_kib'] for v in results.values()),
            'paused_samples': pauses, 'memory_rule': 'pause own subtrees below2GiB, resume>=3GiB'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    folder = args.plan.parent
    plan = read(args.plan)
    plan_sha = sha(args.plan)
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'plan_sha256': plan_sha}
    report['sources'] = verify_sources(plan, folder)
    complete = read(folder / 'training_complete.json')
    assert len(complete) == 14 and {x['name'] for x in complete} == {j['name'] for j in plan['jobs']}
    report['models'] = [verify_model(j, plan, plan_sha) for j in plan['jobs']]
    assert max(v['data_loads_so_far'] for v in report['models']) == 2
    report['weight_files'] = verify_weight_files(plan, folder)
    stats = read(args.results)
    assert stats['models'] == 14 and stats['total_games'] == 86016
    assert stats['primary_family'] == ['B-F', 'M-F', 'T-F', 'I-R', 'I-F', 'R-F', 'S-F']
    assert stats['environment_seed_clusters'] == 1024 and stats['training_seeds'] == [1, 2]
    assert len(stats['game_files_sha256']) == 84
    for path, digest in stats['game_files_sha256'].items():
        assert sha(path) == digest, path
    assert stats['model_sha256'] == {m['name']: m['model_sha256'] for m in report['models']}
    report['evaluation'] = {'models': 14, 'tiers': 6, 'games_per_tier': 1024, 'total_games': 86016,
                            'results_sha256': sha(args.results), 'game_files_verified': 84}
    report['inference_crosscheck'] = verify_inference(stats, plan, folder)
    assert read(folder / 'eval/evaluation_complete.json') == {'models': 14, 'tiers': 6, 'games': 86016}
    report['resources'] = resource_summary(folder)
    report['readonly_inputs'] = verify_inventory(plan, folder)
    tests = read(folder / 'targeted_validation.json')
    assert tests['unique_tests_passed'] == 13 and tests['remaining_skips'] == 0
    report['targeted_tests'] = tests
    # Verify identity again after the full input scan.
    assert verify_sources(plan, folder) == report['sources']
    report['complete'] = True
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'execution_verified': True, 'models': 14, 'games': 86016}), flush=True)


if __name__ == '__main__':
    main()
