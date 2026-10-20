"""Frozen seven-contrast inference, jointly clustered by held-out environment seed."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from teacher_audit_stats import paired

CONTRASTS=('B-F','M-F','T-F','I-R','I-F','R-F','S-F')


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);m=len(ordered);last=0.;result={}
    for rank,name in enumerate(ordered):
        last=max(last,min(1.,pvalues[name]*(m-rank)));result[name]=last
    return result


def summarize(wins,mowers=None):
    """Each arm: train_seed x tier x environment_seed. Train seeds are fixed replicates."""
    shape=next(iter(wins.values())).shape
    assert shape[0]==2 and shape[1]==6 and shape[2]>=32
    assert set(wins)==set('BFMTIRS') and all(v.shape==shape for v in wins.values())
    assert all(np.isin(v,[0,1]).all() for v in wins.values())
    envs=shape[2];picks=np.random.default_rng(20261020).integers(0,envs,size=(10000,envs))
    arms={}
    for name,values in wins.items():
        each=values.sum(axis=2)
        vec=values.mean(axis=(0,1))
        arms[name]={'wins_by_training_seed_and_tier':each.astype(int).tolist(),
                    'mean_rate_by_training_seed':values.mean(axis=(1,2)).tolist(),
                    'macro_win_rate':float(vec.mean()),
                    'macro_wins_per1024_equivalent':float(vec.mean()*envs),
                    'env_seed_cluster_ci95':np.quantile(vec[picks].mean(axis=1),[.025,.975]).tolist(),
                    'absolute_training_seed_gap':float(abs(values[0].mean()-values[1].mean()))}
        if mowers is not None:arms[name]['mean_mowers_used_by_training_seed_and_tier']=mowers[name].mean(axis=2).tolist()
    comparisons={};pvalues={}
    for name in CONTRASTS:
        left,right=name.split('-');delta=wins[left]-wins[right]
        vec=delta.mean(axis=(0,1));test=paired(vec)
        boot=vec[picks].mean(axis=1);each=delta.mean(axis=(1,2))
        tiers=delta.mean(axis=0)
        result={'difference':float(vec.mean()),'per_training_seed_difference':each.tolist(),
                'both_training_seeds_positive':bool((each>0).all()),
                'both_training_seeds_negative':bool((each<0).all()),
                't_ci95':test['ci95'],'t_p':test['p_adjusted'],'df':envs-1,
                'env_seed_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
                'tier_differences':tiers.mean(axis=1).tolist(),
                'tier_env_seed_cluster_ci95':[np.quantile(tiers[i][picks].mean(axis=1),[.025,.975]).tolist() for i in range(6)]}
        if mowers is not None:result['mowers_used_difference']=float((mowers[left]-mowers[right]).mean())
        comparisons[name]=result;pvalues[name]=test['p_adjusted']
    adjusted=holm(pvalues)
    for name,result in comparisons.items():
        result['holm_p']=adjusted[name]
        result['supports_positive_effect']=(result['both_training_seeds_positive']
            and result['env_seed_cluster_ci95'][0]>0 and adjusted[name]<.05)
        result['supports_negative_effect']=(result['both_training_seeds_negative']
            and result['env_seed_cluster_ci95'][1]<0 and adjusted[name]<.05)
    return {'arms':arms,'contrasts':comparisons,'training_seeds':[1,2],'tiers':['z15','z16','z17','z18','z19','z20'],
            'environment_seed_clusters':envs,'bootstrap_replicates':10000,'bootstrap_seed':20261020,
            'primary_family':list(CONTRASTS),'inference_scope':'conditional on these two trained initializations; all six tiers and both models resampled jointly by env seed; no population-level training-seed inference',
            'recommendation_rule':'supported I-F/M-F/T-F/S-F practical improvement with both seed directions shown; no untested combination or best-seed claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--eval',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());contract=plan['eval_contract']
    from render_episode import load_training_task
    manifest=args.plan.parent/'frozen_eval_manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==contract['manifest_sha256']
    seed_lists=[load_training_task(manifest,t)['seeds'][:contract['seeds_each']] for t in contract['task_ids']]
    assert all(s==seed_lists[0] for s in seed_lists) and len(seed_lists[0])==1024
    seeds=seed_lists[0];shape=(2,6,len(seeds));wins={a:np.empty(shape,float) for a in 'BFMTIRS'}
    mowers={a:np.empty(shape,float) for a in wins};files={};model_hashes={}
    completed=json.loads((args.plan.parent/'training_complete.json').read_text())
    expected_models={v['name']:v['model_sha256'] for v in completed}
    for job in plan['jobs']:
        model=Path(job['out'])/'model.pt';digest=hashlib.sha256(model.read_bytes()).hexdigest()
        assert digest==expected_models[job['name']];model_hashes[job['name']]=digest
        for tier,task in enumerate(contract['task_ids']):
            dest=args.eval/job['name']/task;path=dest/'games.jsonl'
            rows=[json.loads(v) for v in path.read_text().splitlines()]
            assert len(rows)==len(seeds) and {r['seed'] for r in rows}==set(seeds)
            by_seed={r['seed']:r for r in rows};assert len(by_seed)==len(rows)
            assert all(r['arm']=='fast' and r['searched']==0 for r in rows)
            config=json.loads((dest/'config.json').read_text())
            identity=json.loads((dest/'evaluation_identity.json').read_text())
            assert identity['model_sha256']==digest and identity['manifest_sha256']==contract['manifest_sha256']
            assert identity['executable_sha256']==contract['executable_sha256']
            assert config['rollout_policy']==str(model) and config['task_id']==task
            assert config['executable']==contract['executable'] and config['workers']==str(contract['workers'])
            ix=(job['seed']-1,tier)
            wins[job['arm']][ix]=[int(by_seed[s]['won']) for s in seeds]
            mowers[job['arm']][ix]=[by_seed[s]['mowers_used'] for s in seeds]
            summary=json.loads((dest/'summary.json').read_text())['fast']
            assert summary['won']==sum(int(r['won']) for r in rows)
            files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=summarize(wins,mowers)
    report.update(models=14,total_games=14*6*1024,game_files_sha256=files,model_sha256=model_hashes)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:{k:v for k,v in c.items() if k in ['difference','per_training_seed_difference','env_seed_cluster_ci95','holm_p','supports_positive_effect']} for n,c in report['contrasts'].items()},indent=2))


if __name__=='__main__':main()
