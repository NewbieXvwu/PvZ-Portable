"""Strict raw-evidence audit of task43 sparse v5 retraining and all three acceptance gates.

Pass the absolute sparse plan path after training/calibration/all 208 games finish.
No games, uploads, or default changes are performed here.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys

from value_teacher_feasibility import digest, write_json
from value_teacher_sparse import load_plan
from value_leaf_dispersion import describe
from value_widen_ab import paired

CALIBRATION_COHORTS={'day_d7':256,'day_d7p':256,'day_l7_control':256,
                     'night_n9_n10':256,'night_original_conveyor':32}


def game_rows(path,expected):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seeds=[r['seed'] for r in rows]
    if len(seeds)!=len(set(seeds)) or set(seeds)!=set(expected):
        raise ValueError('games do not exactly cover frozen seeds')
    for row in rows:
        if type(row['seed']) is not int or type(row['won']) is not bool:
            raise ValueError('game seed/win types are invalid')
        if type(row['decisions']) is not int or not 0<row['decisions']<3000:
            raise ValueError('game completion is ambiguous at the action cap')
    return {r['seed']:r for r in rows}


def probe_rows(paths,task,expected,counts):
    rows=[];seen=set();by_seed={s:0 for s in expected}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line);seed=row['seed'];key=(seed,row['decision'])
            if row['task']!=task or type(seed) is not int or seed not in by_seed:
                raise ValueError('unexpected probe task/seed')
            if key in seen:raise ValueError('duplicate probe decision')
            seen.add(key);by_seed[seed]+=1
            if type(row['decision']) is not int or not 0<=row['decision']<3000:
                raise ValueError('invalid probe decision')
            for name in ('std','top1_top2','mean'):
                if not math.isfinite(row[name]) or not 0<=row[name]<=1:
                    raise ValueError('invalid probe value')
            if not 2<=row['scored']<=row['candidates'] or type(row['overridden']) is not bool:
                raise ValueError('invalid candidate/probe metadata')
            rows.append(row)
    if by_seed!=counts:raise ValueError('missing/excess probe decisions for planned games')
    if not rows:raise ValueError('no usable leaf probes')
    return rows


def paired_gate(result):
    return result['delta_wins']>=-2 and not (
        result['delta_wins']<0 and result['mcnemar_exact_two_sided']<.05)


def check_config(directory,plan,task,model,manifest,games):
    c=json.loads((directory/'config.json').read_text())
    expected=dict(rollout_policy=plan['teacher']['rollout_policy'],static_value=plan['teacher']['static_value'],
        value=str(model),manifest=str(manifest),task_id=task,executable=plan['teacher']['executable'])
    for key,value in expected.items():
        if c[key]!=value:raise ValueError(f'wrong frozen config: {directory} {key}')
    numeric=dict(seeds=games,seed_offset=0,workers=4,per_card=1,student_candidates=2,
        samples=3,horizon=2500,verify_samples=9,min_gap=300,rollout_sim_skip=61,dagger_beta=1)
    for key,value in numeric.items():
        if float(c[key])!=value:raise ValueError(f'wrong teacher parameter: {key}')
    if str(c['halving']).lower()!='true':raise ValueError('halving differs from frozen teacher')
    if c['arms']!='fast_search':raise ValueError('evaluation did not run the search teacher')


def audit(plan):
    out=Path(plan['out']);evaluation=out/'evaluation';model=Path(plan['training']['out'])/'model.pt'
    checkpoint_sha=digest(model);baseline=Path(plan['teacher']['value'])
    if digest(Path(plan['evaluation']['manifest']))!=plan['evaluation']['manifest_sha256']:
        raise ValueError('frozen evaluation tasks changed')
    hashes={str(model):checkpoint_sha};probe_result={};all_game_count=0
    for arm,checkpoint in (('v3',baseline),('v4',model)):
        probe_result[arm]={}
        for task in ('pool_l24_eval_p9k','pool_l26_eval_p9k'):
            directory=evaluation/'dispersion'/arm/task;expected=plan['evaluation']['dispersion_seeds']
            games=game_rows(directory/'games.jsonl',expected)
            check_config(directory,plan,task,checkpoint,Path(plan['evaluation']['manifest']),4)
            counts={s:min(80,games[s]['searched']) for s in expected}
            parts=sorted(directory.glob('leaves.*.jsonl'))
            rows=probe_rows(parts,task,expected,counts)
            probe_result[arm][task]=describe(rows);all_game_count+=len(games)
            for path in [directory/'games.jsonl',directory/'config.json',*parts]:hashes[str(path)]=digest(path)
    pairs={}
    for label,task in [('l21','pool_l21_eval_p9k'),('day','day_l7_z15_eval'),('night','night_l14_eval_n10')]:
        directory=evaluation/f'ab_{label}';frozen=json.loads((directory/'plan.json').read_text())
        manifest=Path(frozen['manifest']);tasks=json.loads(manifest.read_text())['tasks']
        if len(tasks)!=1 or tasks[0]['task_id']!=task or tasks[0]['seeds']!=plan['evaluation']['seeds']:
            raise ValueError('paired task/frozen seed list differs')
        if frozen['v3']!=str(baseline) or frozen['v4']!=str(model):raise ValueError('paired model paths differ')
        original=next(t for t in json.loads(Path(plan['evaluation']['manifest']).read_text())['tasks'] if t['task_id']==task)
        if tasks[0]!=original:raise ValueError('paired deck/task semantics differ')
        data=[]
        for arm,checkpoint in (('v3',baseline),('v4',model)):
            d=directory/arm/task;games=game_rows(d/'games.jsonl',plan['evaluation']['seeds'])
            check_config(d,plan,task,checkpoint,manifest,32)
            data.append({s:r['won'] for s,r in games.items()});all_game_count+=len(games)
            for p in (d/'games.jsonl',d/'config.json'):hashes[str(p)]=digest(p)
        result=paired(*data);saved=json.loads((directory/'paired.json').read_text())
        if saved.get('complete') is not True:raise ValueError('saved paired report is incomplete')
        for key,value in result.items():
            if saved['results'][task][key]!=value:raise ValueError('saved/raw paired statistics differ')
        pairs[task]=result
        for p in (manifest,directory/'plan.json',directory/'paired.json'):hashes[str(p)]=digest(p)
    if all_game_count!=208:raise ValueError('all three acceptance gates require exactly 208 games')
    # Model provenance and validation split are checked independently of the game reports.
    import torch
    ck=torch.load(model,map_location='cpu',weights_only=False)
    if not all(torch.isfinite(v).all().item() for v in ck['state_dict'].values()):raise ValueError('nonfinite model')
    if ck['init']!=str(baseline) or ck['decision_data']!=str(out/'data') or ck['decision_fraction']!=.5:
        raise ValueError('wrong warm start or decision data recipe')
    if ck['decision_seeds']!=64 or not set(plan['seeds'])<=set(ck['trained_on']):
        raise ValueError('checkpoint does not include the complete frozen corpus')
    exported=json.loads((out/'export.json').read_text())
    if exported['status']!='complete' or {r['seed'] for r in exported['seeds']}!=set(plan['seeds']) or len(exported['seeds'])!=64:
        raise ValueError('incomplete exported corpus')
    if {p.name for p in (out/'data').glob('*.pt')}!={f'seed_{s}.pt' for s in plan['seeds']}:
        raise ValueError('unexpected or missing training shard')
    for row in exported['seeds']:
        p=out/'data'/f"seed_{row['seed']}.pt"
        if digest(p)!=row['sha256']:raise ValueError('training shard changed after export')
        shard=torch.load(p,map_location='cpu',weights_only=False,mmap=True);n=row['rows']
        if shard['boards'].shape!=(n,92,6,10) or shard['scalars_v2'].shape!=(n,154):
            raise ValueError('invalid encoder-v2 shard dimensions')
        if not all(torch.isfinite(shard[k]).all().item() for k in ('boards','scalars_v2','won')):
            raise ValueError('nonfinite training shard')
        if not (shard['seed']==row['seed']).all().item() or not (shard['kind']==0).all().item():
            raise ValueError('mixed seed or row type in teacher leaf shard')
        if not ((shard['won']==0)|(shard['won']==1)).all().item() or shard['won'].sum().item()!=row['wins']:
            raise ValueError('teacher labels differ from export receipt')
        hashes[str(p)]=row['sha256']
    total_rows=sum(r['rows'] for r in exported['seeds'])
    if ck['decision_fit_rows']+ck['decision_validation_rows']!=total_rows:
        raise ValueError('checkpoint trained on an incomplete decision corpus')
    pair_accuracy_path=Path(plan['training']['out'])/'pair_accuracy.json'
    accuracy=json.loads(pair_accuracy_path.read_text())
    if accuracy['v5']['checkpoint_sha256']!=checkpoint_sha or accuracy['init']['checkpoint_sha256']!=digest(baseline):
        raise ValueError('held-out pair report scored another checkpoint')
    if accuracy['v5']['cohorts']['pool_decision']['seeds']!=len(plan['held_out_seeds']):
        raise ValueError('held-out decision seed count differs')
    if accuracy['v5']['cohorts']['pool_decision']['rows']!=ck['decision_validation_rows']:
        raise ValueError('held-out report omitted validation rows')
    a6=Path(plan['evaluation']['calibration_manifest'])
    if digest(a6)!=plan['evaluation']['calibration_manifest_sha256']:raise ValueError('A6 changed')
    audit_seeds={g['seed'] for g in json.loads(a6.read_text())}
    if set(ck['trained_on'])&audit_seeds:raise ValueError('external A6 audit leakage')
    calibration_path=evaluation/'calibration'/'metrics.json';calibration=json.loads(calibration_path.read_text())
    canonical=__import__('hashlib').sha256(json.dumps(json.loads(a6.read_text()),sort_keys=True).encode()).hexdigest()
    if calibration['manifest_sha256']!=canonical:raise ValueError('calibration uses a different A6 manifest')
    for name,sha in [('v4',digest(baseline)),('v5',checkpoint_sha)]:
        if calibration['models'][name]['sha256']!=sha:raise ValueError('calibration scored another checkpoint')
    calibration_delta={}
    for group,count in CALIBRATION_COHORTS.items():
        row=calibration['cohorts'][group];old=row['models']['v4']['calibration'];new=row['models']['v5']['calibration']
        if row['games']!=count or old['n']!=new['n'] or old['n']<=0:
            raise ValueError('missing or mismatched external calibration samples')
        if any(not math.isfinite(d[k]) or not 0<=d[k]<=1 for d in (old,new) for k in ('brier','ece')):
            raise ValueError('invalid external calibration score')
        calibration_delta[group]=dict(games=row['games'],wins=row['wins'],brier_old=old['brier'],
            brier_new=new['brier'],brier_delta=new['brier']-old['brier'],ece_old=old['ece'],
            ece_new=new['ece'],ece_delta=new['ece']-old['ece'])
    for p in (pair_accuracy_path,calibration_path,a6):hashes[str(p)]=digest(p)
    gates=dict(a=all(v['mean_std']>=.02 and v['median_gap']>.02 for v in probe_result['v4'].values()),
        b=paired_gate(pairs['pool_l21_eval_p9k']),
        c=all(paired_gate(pairs[t]) for t in ('day_l7_z15_eval','night_l14_eval_n10')))
    return dict(status='complete',training_complete=True,acceptance_complete=True,games=all_game_count,
        adoption_gates=gates,all_three_pass=all(gates.values()),hf_upload_required=all(gates.values()),
        hf_uploaded=False,default_changed=False,model_sha256=dict(v4=digest(baseline),v5=checkpoint_sha),
        dispersion=probe_result,paired=pairs,held_out_pair_accuracy=accuracy,calibration=calibration_delta,
        calibration_brier_no_regression=all(r['brier_delta']<=0 for r in calibration_delta.values()),
        artifact_sha256=hashes,
        caveats=['Pairwise candidates share roots/seeds; pair counts are not independent games.',
                 'Nonsignificant paired results do not establish equivalence beyond the stated net-win gate.',
                 'External calibration reflects archived policies; pool all-loss Brier is not a ranking gate.'])


if __name__=='__main__':
    p=load_plan(sys.argv[1]);target=Path(p['out'])/'evaluation'/'acceptance.json'
    target.unlink(missing_ok=True)
    result=audit(p);write_json(target,result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=1))
