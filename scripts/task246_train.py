"""Task246 mixed episode/decision training through task239's existing trainer."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from task239_train import TERRAINS, load_rows, metrics, run_arm
from task243_train import pack, predict
from train_value_branches import pair_accuracy
from pvz_value_net import load_value_net
from value_teacher_feasibility import write_json
from task244_experiment import tempered_model


def rank(prob, data, mask):
    d={"seed":torch.from_numpy(data['seed']),"position":torch.from_numpy(data['op']),
       "candidate":torch.from_numpy(data['candidate']),"won":torch.from_numpy(data['truth'])}
    score=prob*data['live_fraction']+data['terminal_win_fraction']
    return pair_accuracy(torch.from_numpy(score),d,torch.from_numpy(np.flatnonzero(mask)),.25)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16)
    args=p.parse_args();base=args.data.resolve();started=time.time()
    torch.set_num_threads(1)
    features=base/'features'
    selected={j['game'] for j in json.loads((base/'plan.json').read_text())['jobs']}
    for path in (base/'branches').glob('game_*/rows.npz'):
        meta=json.loads((path.parent/'complete.json').read_text())
        if meta['game'] not in selected:continue
        target=features/f"{meta['terrain']}-branch-{meta['game']:04d}.npz"
        if not target.exists():target.symlink_to(path)
    pack(features,base/'packed')
    data=load_rows(base/'packed')
    split=json.loads((base/'split.json').read_text())
    train,val,test=({tuple(x) for x in split[k]} for k in ('train','validation','test'))
    terrain=np.concatenate([[t]*len(data[t]['y']) for t in TERRAINS])
    columns={k:np.concatenate([data[t][k] for t in TERRAINS]) for k in
        ('seed','game','source','candidate','op','future','level','y','truth','live_fraction','terminal_win_fraction')}
    test_mask=np.array([(t,int(s)) in test for t,s in zip(terrain,columns['seed'])])
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    predictions={};training={}
    for name,init in [('finetune',base/'models/production.pt'),('scratch',None)]:
        state=load_value_net(init).state_dict() if init else None
        report,checkpoint,prob,_=run_arm(data,2,64,246,args.epochs,device,train,val,lr=1e-4,
            dropout=0,weight_decay=1e-4,init_state=state,batch_size=512,decision_fraction=.5)
        checkpoint['state_dict']={k:v.cpu() for k,v in checkpoint['state_dict'].items()}
        checkpoint.update(task='task246',trained_on=sorted({s for _,s in train|val}),
            validation_seeds=sorted({s for _,s in val}),decision_fraction=.5,init=str(init) if init else None)
        torch.save(checkpoint,base/'models'/f'{name}.pt')
        predictions[name]=prob
        training[name]=report
        del checkpoint,state
        gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    for name in ('production','task244'):
        paths={t:base/'models'/('roof.pt' if name=='production' and t=='roof' else
                               'production.pt' if name=='production' else 'v5-t3.pt') for t in TERRAINS}
        predictions[name]=predict(data,paths,device)
    for name in ('finetune','scratch'):
        path=tempered_model(base/'models'/f'{name}.pt',base/'models'/f'{name}-t3.pt',3)
        predictions[f'{name}-t3']=predict(data,{t:path for t in TERRAINS},device)
    report=dict(task='task246',device=str(device),training=training,episode={},ranking={},seconds=time.time()-started)
    for scope in ('all',*TERRAINS):
        scope_mask=np.ones(len(terrain),bool) if scope=='all' else terrain==scope
        episode=test_mask&scope_mask&(columns['source']==0)
        branch=test_mask&scope_mask&(columns['source']==1)
        report['episode'][scope]={n:metrics(v[episode],columns['y'][episode]) for n,v in predictions.items()}
        report['ranking'][scope]={n:rank(v,columns,branch) for n,v in predictions.items()}
    rng=np.random.default_rng(246)
    report['ranking_bootstrap']={}
    for scope in ('all','pool'):
        mask=test_mask&(columns['source']==1)&((terrain==scope) if scope!='all' else True)
        games=np.unique(columns['game'][mask])
        values={name:[rank(prob,columns,mask&(columns['game']==game)) for game in games]
                for name,prob in predictions.items()}
        draws=rng.integers(len(games),size=(2000,len(games)))
        intervals={}
        for name in ('finetune','scratch','finetune-t3','scratch-t3'):
            numerator=np.array([v['pair_accuracy']*v['pairs'] for v in values[name]])
            control=np.array([v['pair_accuracy']*v['pairs'] for v in values['task244']])
            pairs=np.array([v['pairs'] for v in values[name]])
            delta=(numerator[draws].sum(1)-control[draws].sum(1))/np.maximum(1,pairs[draws].sum(1))
            intervals[name]=dict(low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),games=len(games))
        report['ranking_bootstrap'][scope]=intervals
    report['eligible_for_panel']={name:bool(
        report['ranking']['all'][name]['pair_accuracy']-report['ranking']['all']['task244']['pair_accuracy']>=.05 and
        report['ranking']['pool'][name]['pair_accuracy']>report['ranking']['pool']['task244']['pair_accuracy'] and
        report['ranking_bootstrap']['all'][name]['low']>0) for name in ('finetune-t3','scratch-t3')}
    write_json(base/'offline.json',report)
    np.savez_compressed(base/'predictions.npz',**predictions,**columns,terrain=terrain,test=test_mask)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
