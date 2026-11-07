"""Task246 mixed episode/decision training through task239's existing trainer."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
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


def prepare_new_labels(base, source):
    """Read completed task254 games into an isolated, branch-only training directory."""
    if base == source or source in base.parents or base in source.parents:
        raise ValueError('--data and --new-labels must be separate directories')
    if any((base/name).exists() for name in ('features', 'packed', 'branches', 'models')):
        raise ValueError('--new-labels requires a fresh --data directory')
    jobs = [j for j in json.loads((source/'new-plan.json').read_text())['jobs']
            if (source/'branches'/f"game_{j['game']:04d}"/'complete.json').exists()]
    if not jobs:
        raise ValueError('no completed games in --new-labels')
    for name in ('features', 'branches', 'models'):
        (base/name).mkdir(parents=True, exist_ok=True)
    split = {key: [] for key in ('train', 'validation', 'test')}
    for job in jobs:
        directory = f"game_{job['game']:04d}"
        (base/'branches'/directory).symlink_to(source/'branches'/directory)
        split[job['partition']].append([job['terrain'], job['probe_seed']])
    for name in ('production.pt', 'roof.pt', 'finetune-t3.pt'):
        shutil.copy2(source/'models'/name, base/'models'/name)
    write_json(base/'plan.json', dict(jobs=jobs))
    write_json(base/'split.json', split)


def rank(prob, data, mask):
    d={"seed":torch.from_numpy(data['game']),"position":torch.from_numpy(data['op']),
       "candidate":torch.from_numpy(data['candidate']),"won":torch.from_numpy(data['truth'])}
    return pair_accuracy(torch.from_numpy(prob),d,torch.from_numpy(np.flatnonzero(mask)),.25)


def candidate_rows(base, jobs, columns, predictions):
    """Aggregate all common futures, including candidates with only terminal leaves."""
    indices=defaultdict(list)
    for i in np.flatnonzero(columns['source']==1):
        indices[int(columns['game'][i]),int(columns['op'][i]),int(columns['candidate'][i])].append(i)
    records=[]
    scores={name:[] for name in predictions}
    for job in jobs:
        directory=base/'branches'/f"game_{job['game']:04d}"
        selection=json.loads((directory/'selection.json').read_text())
        for point in selection['decisions']:
            root=directory/f'root_{point}'
            labels=json.loads((root/'labels.json').read_text())
            grouped=defaultdict(list)
            for row in labels:grouped[row['candidate']].append(row)
            for candidate,rows in grouped.items():
                ix=indices[job['game'],point,candidate]
                terminal_wins=sum(r['won'] and r['leaf_terminal'] for r in rows)
                if len(ix)!=sum(not r['leaf_terminal'] for r in rows):
                    raise ValueError(f'{root}: missing candidate {candidate} features')
                records.append(dict(game=job['game'],op=point,candidate=candidate,terrain=job['terrain'],
                    level=job['level'] if 'level' in job else job['task']['level'],
                    seed=selection.get('probe_seed',json.loads((directory/'complete.json').read_text())['seed']),
                    truth=sum(r['won'] for r in rows)/len(rows),samples=len(rows),
                    action=rows[0]['action']))
                for name,prob in predictions.items():
                    scores[name].append((float(prob[ix].sum())+terminal_wins)/len(rows))
    data={key:np.asarray([r[key] for r in records]) for key in ('game','op','candidate','terrain','level','seed','truth')}
    return data,{name:np.asarray(value,dtype=np.float32) for name,value in scores.items()},records


def l23_report(base, paths, device):
    jobs=json.loads((base/'l23-plan.json').read_text())['jobs']
    # A diagnostic game whose teacher continuation hit the action cap has no rows; skip it.
    jobs=[job for job in jobs if (base/'branches'/f"game_{job['game']:04d}"/'rows.npz').exists()]
    shards=[]
    for job in jobs:
        with np.load(base/'branches'/f"game_{job['game']:04d}"/'rows.npz') as stored:
            shards.append({key:stored[key] for key in stored.files})
    data={key:np.concatenate([shard[key] for shard in shards]) for key in shards[0]}
    prob={name:predict({'pool':data},{'pool':path},device) for name,path in paths.items()}
    _,scores,records=candidate_rows(base,jobs,data,prob)
    for i,record in enumerate(records):
        record['prediction']={name:float(value[i]) for name,value in scores.items()}
    decisions=[]
    for job in jobs:
        rows=[r for r in records if r['game']==job['game']]
        bottom=next((r for r in rows if (r['action'].get('row'),r['action'].get('col'))==(4,4)),None)
        top=next((r for r in rows if (r['action'].get('row'),r['action'].get('col'))==(0,0)),None)
        decisions.append(dict(seed=job['probe_seed'],game=job['game'],candidates=rows,
            r4c4_minus_r0c0=(dict(truth=bottom['truth']-top['truth'],
                prediction={name:bottom['prediction'][name]-top['prediction'][name] for name in paths})
                if bottom and top else None)))
    return decisions


def balanced_report(base, data, jobs, columns, train, val, test, predictions, training, device):
    """Task254 calibration and held-out ranking, using the same training/aggregation path."""
    calibration = {}
    calibrated = []
    raw_logits = predict(data, {t:base/'models/finetune.pt' for t in TERRAINS}, device, logits=True)
    offset = 0
    for name in TERRAINS:
        if name not in data:
            continue
        rows = data[name]
        logits = raw_logits[offset:offset+len(rows['y'])]
        mask = np.array([(name, int(seed)) in val for seed in rows['seed']])
        if not mask.any():
            raise ValueError(f'{name}: no validation games for temperature fitting')
        losses = {}
        for temperature in (.25, .5, .75, 1., 1.5, 2., 3., 4., 6., 8.):
            loss = np.logaddexp(0, logits/temperature) - rows['y']*logits/temperature
            losses[temperature] = float(np.mean([
                loss[mask & (rows['source'] == source)].mean() for source in (0, 1)
                if np.any(mask & (rows['source'] == source))]))
        temperature = min(losses, key=losses.get)
        tempered_model(base/'models/finetune.pt', base/f'models/balanced-{name}.pt', temperature)
        calibrated.append(np.exp(-np.logaddexp(0, -logits/temperature)))
        calibration[name] = dict(temperature=temperature, validation_rows=int(mask.sum()),
                                 validation_loss=losses[temperature], grid=losses)
        offset += len(rows['y'])
    predictions['balanced'] = np.concatenate(calibrated)
    ranking, scores, _ = candidate_rows(base, jobs, columns, predictions)
    heldout = np.array([(t, int(s)) in test for t, s in zip(ranking['terrain'], ranking['seed'])])
    results = {}
    for scope in ('all', *TERRAINS, *sorted(set(map(int, ranking['level'])))):
        mask = heldout & (True if scope == 'all' else
                         ranking['level'] == scope if isinstance(scope, int) else ranking['terrain'] == scope)
        results[str(scope)] = {name: rank(prob, ranking, mask) for name, prob in scores.items()}
    origins = {origin: {scope: {name: rank(prob, ranking, heldout & (ranking['terrain'] == scope) &
               ((ranking['game'] < 10000) if origin == 'original' else (ranking['game'] >= 10000)))
               for name, prob in scores.items()} for scope in ('pool', 'fog')}
               for origin in ('original', 'new')}
    # Freeze this admission rule before running a new-seed whole-game panel.
    eligible = all(results[t]['balanced']['pairs'] > 0 and
                   results[t]['balanced']['pair_accuracy'] >= results[t]['production']['pair_accuracy'] and
                   results[t]['balanced']['pair_accuracy'] >= results[t]['finetune-t3']['pair_accuracy'] + .05
                   for t in ('pool', 'fog'))
    eligible = eligible and all(results[str(level)]['balanced']['pairs'] > 0 and
        results[str(level)]['balanced']['pair_accuracy'] >= results[str(level)]['production']['pair_accuracy'] and
        results[str(level)]['balanced']['pair_accuracy'] > results[str(level)]['finetune-t3']['pair_accuracy']
        for level in (21, 22, 24, 31, 36, 37))
    report = dict(task='task254', training=training, calibration=calibration, ranking=results,
                  ranking_by_origin=origins, eligible_for_panel=bool(eligible),
                  gate='pool/fog >= production and >= finetune-t3 +0.05; L21/22/24/31/36/37 '
                       'have held-out pairs, >= production and > finetune-t3',
                  split_games=dict(train=len(train), validation=len(val), test=len(test)))
    test_mask = np.array([(t, int(s)) in test for t in data for s in data[t]['seed']])
    report['heldout'] = {name: metrics(prob[test_mask], columns['y'][test_mask])
                         for name, prob in predictions.items()} if test_mask.any() else {}
    write_json(base/'offline.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--task',choices=('task246','task254'),default='task246')
    p.add_argument('--new-labels',type=Path,
        help='read completed task254 labels; train from finetune-t3 without original task246 data')
    p.add_argument('--device',choices=('cpu','mps'),help='default: MPS when available')
    args=p.parse_args();base=args.data.resolve();started=time.time()
    torch.set_num_threads(1)
    if args.new_labels:
        if args.task != 'task254':p.error('--new-labels requires --task task254')
        prepare_new_labels(base, args.new_labels.resolve())
    features=base/'features'
    jobs=json.loads((base/'plan.json').read_text())['jobs']
    selected={j['game'] for j in jobs}
    for path in (base/'branches').glob('game_*/rows.npz'):
        meta=json.loads((path.parent/'complete.json').read_text())
        if meta['game'] not in selected:continue
        target=features/f"{meta['terrain']}-branch-{meta['game']:04d}.npz"
        if not target.exists():target.symlink_to(path)
    pack(features,base/'packed')
    data=load_rows(base/'packed')
    split=json.loads((base/'split.json').read_text())
    train,val,test=({tuple(x) for x in split[k]} for k in ('train','validation','test'))
    terrain=np.concatenate([[t]*len(data[t]['y']) for t in TERRAINS if t in data])
    columns={k:np.concatenate([data[t][k] for t in TERRAINS if t in data]) for k in
        ('seed','game','source','candidate','op','future','level','y','truth','live_fraction','terminal_win_fraction')}
    test_mask=np.array([(t,int(s)) in test for t,s in zip(terrain,columns['seed'])])
    device=torch.device(args.device or ('mps' if torch.backends.mps.is_available() else 'cpu'))
    predictions={};training={}
    arms = [('finetune',base/'models'/('finetune-t3.pt' if args.new_labels else 'production.pt'))]
    decision_fraction = 1. if args.new_labels else .5
    if args.task == 'task246': arms.append(('scratch',None))
    for name,init in arms:
        state=load_value_net(init).state_dict() if init else None
        report,checkpoint,prob,_=run_arm(data,2,64,246,args.epochs,device,train,val,lr=1e-4,
            dropout=0,weight_decay=1e-4,init_state=state,batch_size=512,decision_fraction=decision_fraction)
        checkpoint['state_dict']={k:v.cpu() for k,v in checkpoint['state_dict'].items()}
        checkpoint.update(task=args.task,trained_on=sorted({s for _,s in train|val}),
            validation_seeds=sorted({s for _,s in val}),decision_fraction=decision_fraction,init=str(init) if init else None)
        torch.save(checkpoint,base/'models'/f'{name}.pt')
        predictions[name]=prob
        training[name]=report
        del checkpoint,state
        gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
    if args.task == 'task254':
        for name in ('production', 'finetune-t3'):
            paths={t:base/'models'/('roof.pt' if name=='production' and t=='roof' else f'{name}.pt') for t in TERRAINS}
            predictions[name]=predict(data,paths,device)
        balanced_report(base,data,jobs,columns,train,val,test,predictions,training,device)
        if args.new_labels:
            # Preserve the existing terrain routing: roof keeps the frozen production leaf.
            frozen = {t: 'finetune-t3.pt' if t == 'roof' else 'production.pt'
                      for t in TERRAINS if t not in data}
            for terrain, model in frozen.items():
                shutil.copy2(base/'models'/model, base/f'models/balanced-{terrain}.pt')
            report = json.loads((base/'offline.json').read_text())
            report.update(route='finetune-t3 + new branches only', frozen_terrains=frozen,
                          source=str(args.new_labels.resolve()), device=str(device), seconds=time.time()-started)
            write_json(base/'offline.json', report)
        return
    for name in ('production','task244'):
        paths={t:base/'models'/('roof.pt' if name=='production' and t=='roof' else
                               'production.pt' if name=='production' else 'v5-t3.pt') for t in TERRAINS}
        predictions[name]=predict(data,paths,device)
    for name in ('finetune','scratch'):
        path=tempered_model(base/'models'/f'{name}.pt',base/'models'/f'{name}-t3.pt',3)
        predictions[f'{name}-t3']=predict(data,{t:path for t in TERRAINS},device)
    ranking,rank_predictions,_=candidate_rows(base,jobs,columns,predictions)
    rank_test=np.array([(t,int(s)) in test for t,s in zip(ranking['terrain'],ranking['seed'])])
    report=dict(task='task246',device=str(device),training=training,episode={},ranking={},seconds=time.time()-started)
    for scope in ('all',*TERRAINS):
        scope_mask=np.ones(len(terrain),bool) if scope=='all' else terrain==scope
        episode=test_mask&scope_mask&(columns['source']==0)
        branch=rank_test&((ranking['terrain']==scope) if scope!='all' else True)
        report['episode'][scope]={n:metrics(v[episode],columns['y'][episode]) for n,v in predictions.items()}
        report['ranking'][scope]={n:rank(v,ranking,branch) for n,v in rank_predictions.items()}
    rng=np.random.default_rng(246)
    report['ranking_bootstrap']={}
    for scope in ('all','pool'):
        mask=rank_test&((ranking['terrain']==scope) if scope!='all' else True)
        games=np.unique(ranking['game'][mask])
        values={name:[rank(prob,ranking,mask&(ranking['game']==game)) for game in games]
                for name,prob in rank_predictions.items()}
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
    report['l23']=l23_report(base,{name:base/'models'/filename for name,filename in
        [('production','production.pt'),('task244','v5-t3.pt'),('finetune','finetune.pt'),
         ('scratch','scratch.pt'),('finetune-t3','finetune-t3.pt'),('scratch-t3','scratch-t3.pt')]},device)
    write_json(base/'offline.json',report)
    np.savez_compressed(base/'predictions.npz',**predictions,**columns,terrain=terrain,test=test_mask)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
