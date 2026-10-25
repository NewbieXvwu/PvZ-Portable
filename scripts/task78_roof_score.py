"""Audit and score the independent roof panel without starting game processes.

Each root/action/horizon is paired across three common futures by action JSON,
never by candidate ordinal. Full coverage is 24 worlds * 3 futures * 3 horizons.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]

from task78_data import sha
from task78_export import audit_labels
from task78_roof_truth import OUT, frozen
from value_teacher_feasibility import write_json

HORIZONS = (2500, 4000, 6000)
FUTURES = 3
MIN_GAP = .25
EPS = 1e-9


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'))


def paired_candidates(futures):
    """Validate the same actual actions and combine all three terminal futures."""
    if len(futures) != FUTURES:
        raise ValueError('exactly three common futures required')
    maps = []
    for records in futures:
        mapping = {action_key(record['action']): record for record in records}
        if len(mapping) != len(records):
            raise ValueError('duplicate action in a common future')
        maps.append(mapping)
    if any(set(mapping) != set(maps[0]) for mapping in maps[1:]):
        raise ValueError('actual candidate actions differ across common futures')
    result = []
    for key in sorted(maps[0]):
        rows = [mapping[key] for mapping in maps]
        if len({row['common_future'] for row in rows}) != FUTURES:
            raise ValueError('common futures are duplicated')
        if any(row['outcome'].get('terminal') is not True for row in rows):
            raise ValueError('nonterminal truth continuation')
        for row in rows:
            leaf=row['leaf_frame']
            if leaf['terminal'] and (leaf.get('result') not in (1,2) or
                    (leaf['result']==1) != row['outcome']['won']):
                raise ValueError('terminal leaf and continuation truth differ')
        result.append(dict(action=json.loads(key),
            truth=sum(float(row['outcome']['won']) for row in rows)/FUTURES,
            values={name:sum(row['values'][name] for row in rows)/FUTURES for name in ('v5','v6')},
            terminal_leaves=sum(row['leaf_frame']['terminal'] for row in rows)))
    return result


def consistency(roots, include_terminal=False):
    """All selected worlds stay in the audit, including roots with no eligible pairs."""
    worlds = defaultdict(lambda: dict(pairs=0, v5=0., v6=0.))
    totals = dict(roots=len(roots), candidates=0, terminal_leaves=0,
                  terminal_excluded_pairs=0, truth_ties=0, small_truth_gaps=0,
                  pairs=0, v5=0., v6=0., prediction_ties=dict(v5=0,v6=0))
    for root in roots:
        cluster = worlds[root['world']]
        candidates = root['candidates']
        totals['candidates'] += len(candidates)
        totals['terminal_leaves'] += sum(c['terminal_leaves'] for c in candidates)
        for left, right in combinations(candidates, 2):
            if not include_terminal and (left['terminal_leaves'] or right['terminal_leaves']):
                totals['terminal_excluded_pairs'] += 1
                continue
            gap = left['truth'] - right['truth']
            if abs(gap) < EPS:
                totals['truth_ties'] += 1
                continue
            if abs(gap) < MIN_GAP:
                totals['small_truth_gaps'] += 1
                continue
            totals['pairs'] += 1
            cluster['pairs'] += 1
            for name in ('v5','v6'):
                delta = left['values'][name] - right['values'][name]
                correct = .5 if abs(delta)<EPS else float((delta>0)==(gap>0))
                totals[name] += correct
                cluster[name] += correct
                totals['prediction_ties'][name] += int(abs(delta)<EPS)
    totals['worlds'] = dict(worlds)
    totals['rates'] = {name:totals[name]/totals['pairs'] if totals['pairs'] else None
                       for name in ('v5','v6')}
    return totals


def improvement_interval(stats, draws=10000, seed=780):
    """Paired world bootstrap, v6 minus v5; undefined denominators stay visible."""
    import numpy as np
    if not stats['pairs']:
        return dict(status='insufficient_evidence', lower=None, upper=None,
                    point=None, resamples=draws, undefined_resamples=draws)
    worlds = list(stats['worlds'].values())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        selected = [worlds[i] for i in rng.integers(len(worlds),size=len(worlds))]
        denominator = sum(w['pairs'] for w in selected)
        if denominator:
            samples.append(sum(w['v6']-w['v5'] for w in selected)/denominator)
    undefined = draws-len(samples)
    # A zero-denominator resample supplies no rate, not a fabricated zero.
    # Preserve the defined distribution as a diagnostic; a complete interval is
    # required before this implementation can claim the registered gate passed.
    return dict(status='evaluated' if not undefined else 'insufficient_evidence',
                point=stats['rates']['v6']-stats['rates']['v5'],
                lower=float(np.quantile(samples,.025)) if not undefined else None,
                upper=float(np.quantile(samples,.975)) if not undefined else None,
                resamples=draws, undefined_resamples=undefined)


def score_leaf(record, context, models):
    """Both models read one immutable full-precision leaf; terminals bypass both."""
    import torch
    from pvz_value_net import encode_frame
    frame = record['leaf_frame']
    if frame['terminal']:
        if frame.get('result') not in (1,2):
            raise ValueError('terminal leaf has no valid result')
        return {name:float(frame['result']==1) for name in models}
    values = {}
    with torch.inference_mode():
        for name, model in models.items():
            board, scalars = encode_frame(frame, model.encoder_version)
            if model.context:
                scalars = torch.cat([scalars,torch.tensor(context,dtype=torch.float32)])
            value = torch.sigmoid(model(board[None],scalars[None])).item()
            if not 0<=value<=1:
                raise ValueError('nonfinite or out-of-range leaf prediction')
            values[name] = value
    return values


def coverage(root=OUT):
    """Verify the sealed source cohort and list every required collection job."""
    if root != OUT:
        raise ValueError('only the preregistered independent roof panel is supported')
    plan = frozen()
    manifests = []
    expected_worlds = {json.dumps(dict(seed=w['seed'],future_seed=None),sort_keys=True)
                       for w in plan['worlds']}
    if len(expected_worlds)!=24 or plan['common_futures']!=FUTURES:
        raise ValueError('roof world/future coverage changed')
    jobs, missing = [], []
    for k in range(FUTURES):
        path = root/f'k{k}/sources.json'
        source = json.loads(path.read_text())
        if (source.get('training_eligible') is not False or
                source['common_future_index']!=k or source['seed_base']!=plan['seed_bases'][k] or
                source['plan_sha256']!=sha(root/'plan.json') or len(source['archives'])!=24 or
                {json.dumps(a['world'],sort_keys=True) for a in source['archives']}!=expected_worlds or
                any(len(a['roots'])!=2 for a in source['archives'])):
            raise ValueError('independent roof source identity/coverage changed')
        if manifests and source['archives']!=manifests[0][1]['archives']:
            raise ValueError('roots differ across common futures')
        manifests.append((path,source))
        for index in range(24):
            for horizon in HORIZONS:
                job = dict(k=k,index=index,horizon=horizon,manifest=str(path))
                jobs.append(job)
                if not (path.parent/f'labels/archive_{index:02d}/h{horizon}/complete.json').is_file():
                    missing.append(job)
    return manifests,jobs,missing


def evaluate(candidate, target):
    manifests, jobs, missing = coverage()
    receipt = dict(evaluation_scope='independent_roof_terminal_panel',
                   plan_sha256=sha(OUT/'plan.json'), candidate_sha256=sha(candidate),
                   horizons=list(HORIZONS),primary_horizon=2500, common_futures=FUTURES,
                   expected_jobs=len(jobs),missing_jobs=missing,complete=False,
                   training_eligible=False,engine_processes_started=0)
    write_json(OUT/'collection_jobs.json',dict(jobs=jobs))
    if missing:
        receipt['status']='incomplete'
        write_json(target,receipt)
        return receipt
    import torch
    torch.set_num_threads(1)
    from pvz_value_net import load_value_net
    baseline = Path(frozen()['inputs']['value']['path'])
    models = dict(v5=load_value_net(baseline),v6=load_value_net(candidate))
    training_worlds = {json.dumps(row['world'],sort_keys=True) for row in
                       json.loads((ROOT/'artifacts/task78/sources.json').read_text())['archives']}
    panel_worlds = {json.dumps(row['world'],sort_keys=True) for row in manifests[0][1]['archives']}
    if training_worlds & panel_worlds:
        raise ValueError('roof evaluation worlds overlap training sources')
    for name,path in (('v5',baseline),('v6',candidate)):
        checkpoint = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if set(checkpoint.get('trained_on',[])) & {row['world']['seed'] for row in manifests[0][1]['archives']}:
            raise ValueError(f'{name} has seen roof evaluation seeds')
        if name=='v6':
            receipt['candidate_preregistered_corpus']=checkpoint.get('preregistered_corpus') is True
    by_root = defaultdict(dict)
    provenance = {}
    for manifest,source in manifests:
        k = source['common_future_index']
        replay_path=manifest.with_name('replay.json')
        replay=json.loads(replay_path.read_text())
        if (not replay.get('passed') or replay['source_manifest_sha256']!=sha(manifest) or
                replay['executable_sha256']!=frozen()['inputs']['executable']['sha256']):
            raise ValueError('roof source replay is incomplete or belongs to another manifest')
        provenance[str(manifest)] = sha(manifest)
        provenance[str(replay_path)] = sha(replay_path)
        for index,row in enumerate(source['archives']):
            world=json.dumps(row['world'],sort_keys=True)
            root_order=sorted(r['frame'] for r in row['roots'])
            for horizon in HORIZONS:
                records,hashes=audit_labels(index,horizon,row,manifest,replay)
                provenance.update(hashes)
                for frame,root,record in records:
                    seed=source['seed_base']+index*100+root_order.index(frame)
                    if (root['decision_seed']!=seed or root['common_future']!=(seed*1000)%2**32):
                        raise ValueError('roof common-future schedule differs from frozen manifest')
                    entry=by_root[(world,frame,horizon)].setdefault(k,dict(
                        signature=[root['root_state_sha256'],root['history'],root['context']],rows=[]))
                    record['values']=score_leaf(record,root['context'],models)
                    entry['rows'].append(record)
    horizons = {}
    root_rows = []
    root_signatures = {}
    for (world,frame,horizon),futures in sorted(by_root.items()):
        if set(futures)!=set(range(FUTURES)) or any(
                future['signature']!=futures[0]['signature'] for future in futures.values()):
            raise ValueError('roof root/history/context differs across futures')
        signature=root_signatures.setdefault((world,frame),futures[0]['signature'])
        if signature!=futures[0]['signature']:
            raise ValueError('roof root/history/context differs across horizons')
        root_rows.append(dict(world=world,frame=frame,horizon=horizon,
            candidates=paired_candidates([futures[k]['rows'] for k in range(FUTURES)])))
    if len(root_rows)!=48*len(HORIZONS):
        raise ValueError('roof root/horizon coverage is incomplete')
    for horizon in HORIZONS:
        selected=[row for row in root_rows if row['horizon']==horizon]
        primary=consistency(selected)
        interval=improvement_interval(primary)
        horizons[horizon]=dict(nonterminal=primary,improvement=interval,
            all_candidates_diagnostic=consistency(selected,include_terminal=True),
            passed=(interval['point']>0 and interval['lower']>=0)
                   if interval['status']=='evaluated' else None)
    receipt.update(complete=True,status='evaluated',by_horizon=horizons,roots=48,
                   primary_gate_passed=horizons[2500]['passed'],
                   provenance=provenance,scored_roots=root_rows)
    write_json(target,receipt)
    return receipt


def main():
    import os
    os.environ.setdefault('OMP_NUM_THREADS','1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'artifacts/task78/v6/model.pt')
    parser.add_argument('--out',type=Path,default=OUT/'scored.json')
    args=parser.parse_args()
    result=evaluate(args.candidate,args.out)
    print(json.dumps({k:v for k,v in result.items() if k not in
                      ('missing_jobs','provenance','scored_roots','by_horizon')},indent=1))
    print('missing_jobs',len(result.get('missing_jobs',[])))


if __name__=='__main__':
    main()
