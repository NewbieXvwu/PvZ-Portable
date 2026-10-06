from __future__ import annotations
import json, math, pickle, random, sqlite3, sys, time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path.home()/'PvZAgent-gru-bc-level7-v1'
DATA=OUT/'scripted_level7_bc_samples_v2.sqlite'
CHECKPOINT=Path.home()/'PvZAgent-bc-handoff/best_model_pipeline.pt'
TEACHER=ROOT/'scripts/bc_pipeline/scripted_baseline_d1.py'
RESOURCE=str(Path.home()/'.cache/pvz-research-resources')
TRAIN_FIRST=1_400_000
SEEN_COUNT=256
UNSEEN_FIRST=30_000
UNSEEN_COUNT=256
BATCH=128
MAX_EPOCHS=50
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
import torch
import pvz_agent_model as api
from pvz_agent_model import GameplayModelV1, replay_log_probs, select_action


def batches(payloads, batch_size):
    for start in range(0,len(payloads),batch_size):
        yield [pickle.loads(payload) for payload in payloads[start:start+batch_size]]


def eval_loss(model, payloads):
    model.eval(); total=0.0; count=0
    with torch.no_grad():
        for rows in batches(payloads,BATCH):
            outputs,_=model.forward_sequences([[row] for row in rows],[None]*len(rows))
            logp,_=replay_log_probs(model,outputs,rows)
            total-=float(logp.float().sum())
            count+=len(rows)
    return total/count


def exact_match(pred, target):
    if pred.get('type')!=target.get('type'): return False
    kind=target['type']
    if kind=='plant':
        return all(pred.get(k)==target.get(k) for k in ('packet','row','col'))
    if kind=='shovel': return all(pred.get(k)==target.get(k) for k in ('row','col'))
    return all(pred.get(k)==target.get(k) for k in ('ticks','until'))


def accuracy_metrics(model, payloads):
    model.eval(); counts=Counter(); correct=Counter(); plant_packet_n=plant_packet_ok=0
    with torch.no_grad():
        for rows in batches(payloads,BATCH):
            outputs,_=model.forward_sequences([[row] for row in rows],[None]*len(rows))
            for row,out in zip(rows,outputs):
                target=row['action']; kind=target['type']
                pred,_,_=select_action(model,out,row['legal'],deterministic=True)
                counts[kind]+=1
                correct[kind]+=int(exact_match(pred,target))
                if kind=='plant':
                    valid=set(row['legal']['packets'])
                    packet_ids=out['packet_ids']
                    logits=out['packet_logits'].detach().cpu().tolist()
                    candidates=[(logits[i],packet) for i,packet in enumerate(packet_ids) if packet in valid]
                    predicted_packet=max(candidates)[1]
                    plant_packet_n+=1
                    plant_packet_ok+=int(predicted_packet==target['packet'])
    return {'exact_action_by_teacher_type':{
                kind:{'correct':correct[kind],'total':counts[kind],
                      'accuracy':correct[kind]/counts[kind] if counts[kind] else None}
                for kind in ('plant','wait','shovel')},
            'plant_packet_conditional_accuracy':{
                'correct':plant_packet_ok,'total':plant_packet_n,
                'accuracy':plant_packet_ok/plant_packet_n if plant_packet_n else None}}


def run_eval(model, teacher, seeds, sampled):
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    deck=(0,1,2,3,4,5,7)
    env=EventWaitEnv(RESOURCE,headless=True)
    wins=truncated=decisions=0
    action_counts=Counter(); packet_counts=Counter(); plant_type_counts=Counter()
    action_limit=4000
    for episode_i,seed in enumerate(seeds,1):
        torch.manual_seed(int(seed))
        task=TaskSpec(level=7,seed=int(seed),playthrough=2,profile=teacher.profile_for_deck(deck))
        obs,_=env.reset(deck=deck,task=task)
        prev_action=None; prev_wait=None; elapsed=0; events={}; n=0
        while not obs['terminal'] and n<action_limit:
            with torch.no_grad():
                output=model.step(obs,None,prev_action,elapsed,events,prev_wait)
                action,_,_=select_action(model,output,obs,deterministic=not sampled)
            if action['type']=='plant':
                packet_counts[str(action['packet'])]+=1
                plant_type_counts[str(int(obs['packets'][action['packet']]['type']))]+=1
            action_counts[action['type']]+=1
            before=obs
            obs,_,done,_,info=env.step(action)
            if not info.get('ok'): raise RuntimeError(f'student action rejected seed={seed}: {action}')
            prev_action=action; prev_wait=info.get('wait_result')
            elapsed=int(info.get('ticks_advanced',0)); events=info.get('events') or {}
            n+=1; decisions+=1
            if done: break
        is_terminal=bool(obs['terminal'])
        truncated+=int(not is_terminal)
        wins+=int(int(obs.get('result',0))==1)
        if episode_i%32==0:
            print(f'eval sampled={sampled} episodes={episode_i}/{len(seeds)} wins={wins} decisions={decisions}',flush=True)
    env.close()
    return {'episodes':len(seeds),'wins':wins,'win_rate':wins/len(seeds),
            'truncated':truncated,'decisions':decisions,
            'action_counts':dict(action_counts),'plant_packet_counts':dict(packet_counts),
            'plant_type_counts':dict(plant_type_counts)}


def main():
    started=time.time()
    ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    config=ckpt['config']
    if config!={'layers':6,'width':256,'heads':8,'ff_width':1024,
                'gru_layers':2,'gru_width':256,'critic_width':256,'critic_layers':1,
                'input_flags':7,'wait_mode':'events','wait_mask':'progress_v1'}:
        raise RuntimeError(f'expected mainline configuration differs: {config}')
    if not DATA.exists(): raise FileNotFoundError(DATA)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.manual_seed(281_005)
    random.seed(281_005)
    model=GameplayModelV1(config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    conn=sqlite3.connect(DATA)
    train_payloads=[]; val_payloads=[]
    for split,payload in conn.execute('SELECT split,payload FROM samples ORDER BY id'):
        (val_payloads if split else train_payloads).append(payload)
    conn.close()
    if len(train_payloads)+len(val_payloads)<1_000_000 or not val_payloads:
        raise RuntimeError(f'BC dataset too small: train={len(train_payloads)} val={len(val_payloads)}')
    print(f'loaded train={len(train_payloads)} validation={len(val_payloads)} serialized rows',flush=True)
    curve=[]; best=float('inf'); stale=0
    best_path=OUT/'best_model.pt'
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); random.shuffle(train_payloads)
        loss_sum=0.0; seen=0
        for rows in batches(train_payloads,BATCH):
            outputs,_=model.forward_sequences([[row] for row in rows],[None]*len(rows))
            logp,_=replay_log_probs(model,outputs,rows)
            loss=-logp.float().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            loss_sum+=float(loss.detach().cpu())*len(rows); seen+=len(rows)
        train_loss=loss_sum/seen
        val_loss=eval_loss(model,val_payloads)
        curve.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss})
        improved=val_loss < best-1e-7
        if improved:
            best=val_loss; stale=0
            torch.save({'config':config,'model_state_dict':model.state_dict(),'epoch':epoch,
                        'validation_loss':val_loss},best_path)
        else: stale+=1
        print(f'epoch={epoch} train_loss={train_loss:.6f} validation_loss={val_loss:.6f} '
              f'best={best:.6f} stale={stale}',flush=True)
        if stale>=2: break
    best_ckpt=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict']); model.eval()
    model.to('cpu'); torch.set_num_threads(8)
    accuracy=accuracy_metrics(model,val_payloads)
    manifest=json.loads((OUT/'collection_manifest.json').read_text())
    report={'schema_version':1,'device':device,'initialization':'random, exact mainline architecture',
            'configuration':config,'dataset':manifest['dataset'],'training':{
                'epochs_run':len(curve),'best_epoch':best_ckpt['epoch'],'best_validation_loss':best,
                'loss_curve':curve,'batch_size':BATCH,'optimizer':'Adam','learning_rate':3e-4,
                'early_stop':'two consecutive epochs without validation-loss decrease or 50 epochs',
                'sequence_mode':'independent one-transition sequences; recurrent hidden reset per decision'},
            'validation_metrics':accuracy,'model_path':str(best_path),
            'training_seconds':round(time.time()-started,1)}
    (OUT/'training_results.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'phase':'training_done','epochs':len(curve),'best_epoch':best_ckpt['epoch'],
                      'best_validation_loss':best,'validation_metrics':accuracy,
                      'seconds':report['training_seconds']},ensure_ascii=False),flush=True)
    sampled_seen=run_eval(model,importlib_teacher(),range(TRAIN_FIRST,TRAIN_FIRST+SEEN_COUNT),True)
    sampled_unseen=run_eval(model,importlib_teacher(),range(UNSEEN_FIRST,UNSEEN_FIRST+UNSEEN_COUNT),True)
    greedy_unseen=run_eval(model,importlib_teacher(),range(UNSEEN_FIRST,UNSEEN_FIRST+UNSEEN_COUNT),False)
    eval_report={'sampled_seen':sampled_seen,'sampled_unseen':sampled_unseen,
                 'greedy_unseen':greedy_unseen,'greedy_caveat':'structurally invalid per handoff: factored action argmax favors wait'}
    (OUT/'evaluation_results.json').write_text(json.dumps(eval_report,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'phase':'evaluation_done','results':eval_report},ensure_ascii=False),flush=True)


def importlib_teacher():
    import importlib.util
    spec=importlib.util.spec_from_file_location('teacher_d1_eval',TEACHER)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


if __name__=='__main__': main()
