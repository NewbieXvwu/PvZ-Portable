"""Task56 evidence audit. All output stays in this worktree; source episodes are read-only."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/newbiexvwu/PvZ-search')
OUT=ROOT/'artifacts/task56'
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n')


def inventory():
    from pvz_frame_store import frame_path,read_frames
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    def freeze(name,path):
        path=Path(path);data=path.read_bytes();dest=OUT/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        sources[name]=dict(path=str(path),snapshot=str(dest),sha256=hashlib.sha256(data).hexdigest())
        return json.loads(data)
    state=freeze('qc-state.json','/home/newbiexvwu/pvz-logs/curriculum_state.json')
    log_path=Path('/home/newbiexvwu/pvz-logs/curriculum.log');log_bytes=log_path.read_bytes()
    (OUT/'sources/qc.log').write_bytes(log_bytes)
    sources['qc.log']=dict(path=str(log_path),snapshot=str(OUT/'sources/qc.log'),sha256=hashlib.sha256(log_bytes).hexdigest())
    command=freeze('A-command.json',MAIN/'.claude/worktrees/dw5/artifacts/task36/train_A.command.json')['command']
    labels=command[command.index('--labels')+1].split(',')
    a_sources=freeze('A-source-rows.json',MAIN/'.claude/worktrees/dw5/artifacts/task36/dw5i_A_s1/source_rows.json')
    ledger=freeze('ledger.json',MAIN/'.claude/worktrees/dw5/artifacts/task36/data_ledger.json')
    inv=freeze('dw4-inventory.json',MAIN/'.claude/worktrees/dw4/artifacts/task23/dataset_inventory.json')
    cells={k:v for k,v in state['cells'].items() if v['level']==20}
    groups=[g for entry in state['log'] for g in entry['groups'] if 'l20_' in g.get('task_id','') or 'n20' in g.get('label','')]
    rows=[]
    frames_root=MAIN/'artifacts/student/frames_v2'
    paths=sorted(frames_root.glob('*/n20o1_fast_search/meta.json'))
    for path in paths:
        meta=read(path);archive=frame_path(path.parent);frames=read_frames(archive)
        actual=Counter();target=Counter();wait_card=0;bad=0
        for before,after in zip(frames,frames[1:]):
            action=after.get('action') or {};label=after.get('label') or action
            if action.get('type')=='plant':actual[before['packets'][action['packet']][1]]+=1
            if label.get('type')=='plant':
                slot=label['packet'];kind=before['packets'][slot][1];target[kind]+=1
                bad+=kind<0 or not before['packets'][slot][2]
            if action.get('type')=='wait' and any(p[1]>=0 and p[2] for p in before['packets']):wait_card+=1
        rows.append(dict(archive=str(archive),sha256=sha(archive),seed=meta['task']['seed'],
                         decisions=len(frames)-1,won=meta['outcome']['won'],
                         actual_plantings=sum(actual.values()),actual_types=dict(actual),
                         labelled_plantings=sum(target.values()),labelled_types=dict(target),
                         waits_with_ready_card=wait_card,label_invalid_packet=bad,
                         model_training_inclusion=dict(dw4=any(x['archive']==str(path.parent.relative_to(frames_root)) for x in inv),
                                                       A=path.parent.name in labels)))
    old=MAIN/'.claude/worktrees/l20diag/artifacts/task26'
    comparisons={}
    for name in ('student_16','teacher_8'):
        path=old/name/'games.jsonl';gs=[json.loads(line) for line in path.read_text().splitlines()]
        comparisons[name]=dict(path=str(path),sha256=sha(path),games=len(gs),wins=sum(g['won'] for g in gs),
                               planted_slots=[g.get('planted_slots',g.get('planted',{})) for g in gs],
                               searched=sum(g['searched'] for g in gs))
    answer=dict(status='incomplete',captured_at=datetime.now().astimezone().isoformat(),sources=sources,
                qc=dict(cells=cells,collected_groups=groups,offsets={k:v for k,v in state['offsets'].items() if 'l20_' in k},
                        matching_log_lines=log_bytes.decode().count('night_l20_train_n10_orig')),
                actual_training=dict(dw4_archives=len(inv),dw4_L20_archives=sum('/n20' in x['archive'] for x in inv),
                                     A_rows=a_sources['rows'],A_labels=len(labels),A_L20_labels=[x for x in labels if x.startswith('n20')],
                                     A_source_L20_labels={k:v for k,v in a_sources['sources'].items() if k.startswith('n20')},
                                     ledger_L20_archives={arm:sum(x['levels'].get('20',0) for x in ledger[arm]['labels']) for arm in ('dw4i','dw5i')},
                                     A_L14_source={k:v for k,v in a_sources['sources'].items() if k.startswith('n14t')}),
                n20o1=rows,old_comparisons=comparisons,
                scope='n20o1 is bootstrap beta0 DAgger, not ongoing qc sampling; encoding/packing/native parity still to check')
    write(OUT/'inventory.json',answer)
    print(json.dumps(dict(games=len(rows),executed_plantings=sum(r['actual_plantings'] for r in rows),
                          labelled_plantings=sum(r['labelled_plantings'] for r in rows),
                          waits_with_ready_card=sum(r['waits_with_ready_card'] for r in rows),
                          qc_new_groups=len(groups),training=answer['actual_training']),ensure_ascii=False))


def packing():
    import numpy as np
    import torch
    from pack_training_rows import pack
    from pvz_frame_store import frame_path,read_frames
    from pvz_training_pack import pack_options,archive_fingerprint
    from pvz_rollout_policy import packet_features,identity_scalars,load_student
    from pvz_student_observation import StudentHistory
    from train_rollout_policy import load
    torch.set_num_threads(1)
    original=read(OUT/'inventory.json')
    archives=[Path(x['archive']) for x in original['n20o1']]
    controls=sorted((MAIN/'artifacts/student/frames_v2').glob('*/n14t1_fast_search/meta.json'))[:4]
    archives += [frame_path(p.parent) for p in controls]
    assert len(archives)==36
    view=OUT/'pack_view';labels=set()
    for path in archives:
        target=view/path.parent.parent.name/path.parent.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink():assert target.resolve()==path.parent.resolve()
        else:target.symlink_to(path.parent.resolve(),target_is_directory=True)
        labels.add(path.parent.name)
    opts=pack_options(labels,.1,'v2',True,0,'identity',False,sources=archive_fingerprint(view,labels))
    index=pack(view,opts,OUT/'cache',1)
    keys=[]
    def observer(path,rows):keys.extend((path,int(row)) for row in rows)
    loaded=load(view,labels,.1,obs='v2',cells=True,packet_head='identity',row_observer=observer)
    boards,scalars,plant,packet,cell,weight,seeds,values,candidates,packets=loaded
    assert len(keys)==len(plant)
    source_index={}
    for j,(path,i) in enumerate(keys):source_index.setdefault(path,[]).append((j,i))
    label_stats={};positive_types={};rows_checked=0
    for path,entries in source_index.items():
        fs=read_frames(path);history=StudentHistory('v2');positions={i:j for j,i in entries}
        label=path.parent.name;stat=label_stats.setdefault(label,Counter())
        raw_positive=sum((f.get('label') or f.get('action') or {}).get('type')=='plant' and not f.get('action_rejected') for f in fs[1:])
        stat['raw_positive']+=raw_positive
        for i in range(1,len(fs)):
            b,s=history.features(fs[i-1],fs[i-2] if i>=2 else None)
            if i not in positions:continue
            j=positions[i];idx=torch.tensor([j]);a=fs[i].get('label') or fs[i].get('action')
            s=identity_scalars(fs[i-1],s)
            np.testing.assert_array_equal(boards.batch(idx,'cpu')[0].numpy(),b.numpy().astype(np.float16).astype(np.float32))
            np.testing.assert_array_equal(scalars.batch(idx,'cpu')[0].numpy(),s.numpy())
            np.testing.assert_array_equal(packets[idx][0].numpy(),packet_features(fs[i-1]).numpy())
            is_plant=a['type']=='plant';assert bool(plant[j])==is_plant
            assert int(packet[j])==(a['packet'] if is_plant else -1)
            assert int(cell[j])==(a['row']*9+a['col'] if is_plant else -1)
            expected_weight=1 if is_plant or fs[i].get('targets') else 10
            assert float(weight[j])==expected_weight
            if is_plant:
                positive_types[j]=fs[i-1]['packets'][a['packet']][1]
                assert int(packets.columns[0][j,a['packet']])==positive_types[j]>=0
            stat['packed_rows']+=1;stat['packed_positive']+=is_plant
            stat['weight_sum']+=float(weight[j]);stat['positive_weight_sum']+=float(weight[j]) if is_plant else 0
            rows_checked+=1
    assert label_stats['n20o1_fast_search']['packed_positive']==575
    assert all(s['raw_positive']==s['packed_positive'] for s in label_stats.values())
    def distribution(items):
        return dict(n=len(items),min=float(min(items)),median=float(np.median(items)),max=float(max(items)))
    model_stats={}
    for name,path in [('dw4i',MAIN/'artifacts/student/dw4i_s1/model.pt'),('A',MAIN/'.claude/worktrees/dw5/artifacts/task36/dw5i_A_s1/model.pt')]:
        net=load_student(path);assert net.obs=='v2' and net.packet_head=='identity' and not net.context
        details=[];losses=[]
        with torch.inference_mode():
            for start in range(0,len(plant),64):
                idx=torch.arange(start,min(start+64,len(plant)));info=packets[idx]
                b=boards.batch(idx,'cpu');s=scalars.batch(idx,'cpu');pk=packet[idx].long().clamp(min=0)
                outputs=net(b,s,pk,packet_info=info)
                mask=plant[idx].bool()
                if mask.any():
                    loss=torch.nn.functional.cross_entropy(outputs['packet'][mask],packet[idx][mask].long())+torch.nn.functional.cross_entropy(outputs['cell'][mask],cell[idx][mask].long())
                    assert torch.isfinite(loss);losses.append(float(loss))
                for k,j in enumerate(idx.tolist()):
                    if not plant[j]:continue
                    p=int(packet[j]);c=int(cell[j]);pl=float(outputs['plant'][k]);pk_logits=outputs['packet'][k];cl_logits=outputs['cell'][k]
                    ppk=float(torch.softmax(pk_logits,0)[p]);pcl=float(torch.softmax(cl_logits,0)[c]);pplant=float(torch.sigmoid(outputs['plant'][k]))
                    details.append(dict(row=j,archive=str(keys[j][0]),source_row=keys[j][1],label=keys[j][0].parent.name,
                                        kind=positive_types[j],plant_logit=pl,plant_probability=pplant,
                                        teacher_packet_rank=int((pk_logits>pk_logits[p]).sum())+1,
                                        teacher_packet_probability=ppk,teacher_cell_rank=int((cl_logits>cl_logits[c]).sum())+1,
                                        teacher_cell_probability=pcl,joint_training_head_probability=pplant*ppk*pcl))
        summary={}
        for label in labels:
            rs=[x for x in details if x['label']==label]
            summary[label]=dict(teacher_plant_rows=len(rs),student_gate_calls_plant=sum(x['plant_logit']>0 for x in rs),
                               plant_logit=distribution([x['plant_logit'] for x in rs]),
                               plant_probability=distribution([x['plant_probability'] for x in rs]),
                               teacher_packet_rank=distribution([x['teacher_packet_rank'] for x in rs]),
                               teacher_cell_rank=distribution([x['teacher_cell_rank'] for x in rs]),
                               teacher_packet_probability=distribution([x['teacher_packet_probability'] for x in rs]),
                               joint_probability=distribution([x['joint_training_head_probability'] for x in rs]),
                               per_kind={str(kind):dict(rows=sum(x['kind']==kind for x in rs),calls_plant=sum(x['kind']==kind and x['plant_logit']>0 for x in rs)) for kind in sorted({x['kind'] for x in rs})})
        write(OUT/f'logits-{name}.json',details)
        model_stats[name]=dict(path=str(path),sha256=sha(path),labels=summary,batches=len(losses),all_supervised_losses_finite=True)
    result=dict(status='complete-for-selected-corpus',index=str(index),index_sha256=sha(index),pack_version=2,
                archives=len(archives),rows=rows_checked,labels={k:dict(v) for k,v in label_stats.items()},
                original_model_L20_share=0,prospective_model_L20_share='explicitly added to this audit view; no optimisation or model change',
                checks='all retained inputs, packet identity/slot, cell and base row weight exactly match archive; all positive labels retained',
                probability_scope='training-head softmax: packets include nonempty inactive slots; cells all54. Legal-mask inference separately checked live; not calibrated action probabilities.',
                models=model_stats)
    write(OUT/'packing.json',result)
    print(json.dumps(result,ensure_ascii=False))


def native():
    """Replay the fresh eight diagnostics in one engine; compare heads on64 preset states."""
    import numpy as np
    import torch
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from pvz_frame_store import frame_path,read_frames
    from pvz_rollout_policy import load_student,export_student,identity_scalars,packet_features,act,ranked_plants
    from pvz_student_observation import StudentHistory
    from render_episode import _frame,load_training_task,training_task_kwargs
    torch.set_num_threads(1)
    assert (OUT/'matched4/summary.json').is_file(),'finish the bounded game runner before native audit'
    paths=sorted((OUT/'diagnostic_frames').glob('*/*/meta.json'))
    assert len(paths)==8
    nets={name:load_student(path) for name,path in [('dw4i',MAIN/'artifacts/student/dw4i_s1/model.pt'),('A',MAIN/'.claude/worktrees/dw5/artifacts/task36/dw5i_A_s1/model.pt')]}
    for name,net in nets.items():export_student(net,OUT/f'{name}.bin')
    binary=MAIN/'.claude/worktrees/deploy3/build/pvz-portable'
    task_def=load_training_task(MAIN/'artifacts/stage4/night_panel.json','night_l20_eval_n10_orig')
    rows=[];games=[];max_error=0.;frames_checked=0
    with EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources',executable=binary,headless=True) as env:
        for meta_path in paths:
            meta=read(meta_path);path=frame_path(meta_path.parent);fs=read_frames(path)
            teacher=meta['task']['arm']=='fast_search'
            candidates=[i for i in range(1,len(fs)) if ((fs[i]['action']['type']=='plant') if teacher else any(p[1]>=0 and p[2] for p in fs[i-1]['packets']))]
            chosen={candidates[int(k)] for k in np.linspace(0,len(candidates)-1,8).round().astype(int)}
            assert len(chosen)==8
            kw=training_task_kwargs(task_def,meta['task']['seed'])
            task=TaskSpec(level=20,seed=kw['seed'],playthrough=2,profile=profile_for_deck(tuple(kw['deck'])),**kw['task_extra'])
            live,_=env.reset(deck=tuple(kw['deck']),task=task)
            history=StudentHistory('v2');plant_types=Counter();wait_card=wait_legal=0;free_positions=empty_positions=duplicate_positions=0
            for i in range(1,len(fs)):
                # _frame uses tuples in memory; the lossless JSON archive uses lists.
                frame=json.loads(json.dumps(_frame(live)))
                # Every public archived frame must match before using its action as a replay tape.
                for key,v in frame.items():
                    if v!=fs[i-1].get(key):
                        write(OUT/'replay-mismatch.json',dict(path=str(path),decision=i,key=key,expected=fs[i-1].get(key),observed=v,tick=frame['tick']))
                        raise AssertionError((str(path),i,key))
                frame['action']=fs[i-1].get('action')
                legal=live['legal_actions']['plants'];info=packet_features(frame)[None]
                action=fs[i]['action'];has_card=any(p[1]>=0 and p[2] for p in frame['packets'])
                if action['type']=='plant':plant_types[frame['packets'][action['packet']][1]]+=1
                elif action['type']=='wait':wait_card+=has_card;wait_legal+=bool(legal)
                active=[p[1] for p in frame['packets'] if p[1]>=0]
                empty_positions+=len(active)<10;duplicate_positions+=len(set(active))<len(active)
                free_positions+=any(frame['packets'][a['packet']][-1]>frame['sun'] for a in legal)
                for a in legal:
                    p=a['packet'];assert info[0,p,0]>=0 and info[0,p,1]==1 and info[0,p,3]==1
                if i in chosen:
                    for name,net in nets.items():
                        env.load_rollout_net(OUT/f'{name}.bin')
                        native_out=env.rollout_predict(fs[i-2]['tick'] if i>=2 else None,
                                                      bool(frame.get('action') and frame['action']['type']=='plant'),student_history=history)
                        b,s=history.features(frame,fs[i-2] if i>=2 else None);s=identity_scalars(frame,s)
                        with torch.inference_mode():
                            h,g=net.encode(b[None],s[None],packet_info=info)
                            pairs=[(net.plant(g).flatten(),torch.tensor([native_out['plant']])),(net.packet_logits(g,info)[0],torch.tensor(native_out['packet']))]
                            if native_out['cell_packet']>=0:pairs.append((net.cell_logits(h,g,torch.tensor([native_out['cell_packet']]),info)[0],torch.tensor(native_out['cell'])))
                            for py,cpp in pairs:
                                error=float((py-cpp).abs().max());max_error=max(max_error,error)
                                torch.testing.assert_close(py,cpp,rtol=1e-5,atol=1e-5)
                                assert int(py.argmax())==int(cpp.argmax())
                            python_action=act(net,frame,fs[i-2] if i>=2 else None,legal,history=history)
                            assert python_action==native_out['action'],(path,i,name,python_action,native_out['action'])
                            ranked=ranked_plants(net,frame,fs[i-2] if i>=2 else None,legal,history=history)
                            target=None
                            if action['type']=='plant':
                                match=lambda a:all(a[k]==action[k] for k in ('packet','row','col'))
                                index=next(j for j,(_,a) in enumerate(ranked) if match(a))
                                target=dict(teacher_action=action,kind=frame['packets'][action['packet']][1],legal_plant_rank=index+1,
                                            legal_joint_preference=ranked[index][0])
                            rows.append(dict(archive=str(path),seed=meta['task']['seed'],arm=meta['task']['arm'],decision=i,tick=frame['tick'],model=name,
                                             plant_logit=native_out['plant'],plant_probability=float(torch.sigmoid(net.plant(g))[0,0]),
                                             python_action=python_action,native_action=native_out['action'],target=target))
                else:history.extra(frame)
                live,*_=env.step(action)
                frames_checked+=1
            terminal=json.loads(json.dumps(_frame(live)))
            for key,v in terminal.items():assert v==fs[-1].get(key),(str(path),'terminal',key)
            games.append(dict(seed=meta['task']['seed'],arm=meta['task']['arm'],archive=str(path),sha256=sha(path),
                              won=meta['outcome']['won'],decisions=len(fs)-1,plantings=sum(plant_types.values()),planted_types=dict(plant_types),
                              waits_with_ready_card=wait_card,waits_with_legal_plant=wait_legal,free_card_states=free_positions,
                              empty_slot_states=empty_positions,duplicate_states=duplicate_positions))
    assert len(rows)==128
    answer=dict(status='complete',executable=str(binary),executable_sha256=sha(binary),games=games,
                frames_exact=frames_checked+len(paths),selected_root_states=64,model_root_pairs=128,
                logits_max_abs_error=max_error,all_head_argmax_and_greedy_actions_equal=True,
                full_legal_packet_affordability_matches_engine=True,
                probability_scope='rank among legal plant actions and P(plant)*P(packet|legal packets)*P(cell|legal cells); greedy decision still thresholds plant at0; preferences are not calibrated probabilities',rows=rows)
    write(OUT/'native.json',answer)
    print(json.dumps({k:v for k,v in answer.items() if k!='rows'},ensure_ascii=False))


def continuations():
    """Exact2500-tick continuation and encoded-input parity from four fixed teacher roots."""
    from copy import deepcopy
    import numpy as np
    import torch
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from pvz_frame_store import frame_path,read_frames
    from pvz_rollout_policy import load_student,export_student,packet_features,identity_scalars
    from pvz_student_observation import StudentHistory
    from render_episode import _frame,load_training_task,training_task_kwargs
    import search_play
    torch.set_num_threads(1)
    canonical=lambda x:json.loads(json.dumps(x))
    paths=sorted((OUT/'diagnostic_frames').glob('*/t56diag_fast_search/meta.json'));assert len(paths)==4
    task_def=load_training_task(MAIN/'artifacts/stage4/night_panel.json','night_l20_eval_n10_orig')
    nets={name:load_student(path) for name,path in [('dw4i',MAIN/'artifacts/student/dw4i_s1/model.pt'),('A',MAIN/'.claude/worktrees/dw5/artifacts/task36/dw5i_A_s1/model.pt')]}
    rows=[];total_bits=0
    with EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources',executable=MAIN/'.claude/worktrees/deploy3/build/pvz-portable',headless=True) as env:
        for meta_path in paths:
            meta=read(meta_path);fs=read_frames(frame_path(meta_path.parent));seed=meta['task']['seed']
            point=next(i for i in range(1,len(fs)) if fs[i-1]['tick']>=8000 and fs[i]['action']['type']=='plant')
            kw=training_task_kwargs(task_def,seed)
            live,_=env.reset(deck=tuple(kw['deck']),task=TaskSpec(level=20,seed=seed,playthrough=2,profile=profile_for_deck(tuple(kw['deck'])),**kw['task_extra']))
            history=StudentHistory('v2')
            for i in range(1,point):
                frame=_frame(live);history.extra(frame);live,*_=env.step(fs[i]['action'])
            history.extra(_frame(live));assert canonical(_frame(live))=={k:v for k,v in fs[point-1].items() if k in _frame(live)}
            root=env.snapshot();root_obs=deepcopy(live)
            for name,net in nets.items():
                export_student(net,OUT/f'{name}.bin');env.load_rollout_net(OUT/f'{name}.bin')
                future_seed=560000+seed;trace=OUT/f'continuation-{seed}-{name}.jsonl'
                first=dict(type='wait',ticks=60,until='timeout');until=root_obs['tick']+2500
                actual=env.rollout(first,until,snapshot_id=root,seed=future_seed,sim_skip=61,student_history=deepcopy(history),trace_path=trace)
                env.observe() # Ensure trace file is closed before reading on this frozen binary.
                records=[json.loads(line) for line in trace.read_text().splitlines()]
                replay=deepcopy(history);captured={}
                hook=net.scalar[0].register_forward_pre_hook(lambda _module,args:captured.update(scalars=args[0].detach().clone()))
                with torch.inference_mode():
                    for record in records:
                        frame=_frame(record['observation']);frame['sun_income_rate']=record['income']
                        frame['action']={'type':'plant' if record['previous_plant'] else 'wait'}
                        previous={'tick':record['previous_tick']} if record['previous_tick']>=0 else None
                        b,s=replay.features(frame,previous);s=identity_scalars(frame,s)
                        net.encode(b[None],s[None],packet_info=packet_features(frame)[None])
                        for py,cpp in [(b.flatten().numpy(),record['board']),(captured['scalars'].flatten().numpy(),record['scalars'])]:
                            np.testing.assert_array_equal(py.view(np.uint32),np.asarray(cpp,np.float32).view(np.uint32));total_bits+=py.size
                hook.remove()
                search_play._W.clear();search_play._W.update(fast=net,context=None,student_history=deepcopy(history),lean_rollouts=True,batch_waits=False,rollout_sim_skip=61)
                python_states=[];original=search_play._fast_action
                def observed(obs,fast_state):
                    action=original(obs,fast_state);python_states.append(canonical(_frame(obs)));return action
                search_play._fast_action=observed
                try:expected=search_play._python_rollout_reference(env,root_obs,first,until,root,future_seed)
                finally:search_play._fast_action=original;search_play._W.clear()
                assert canonical(_frame(actual))==canonical(_frame(expected)),(seed,name,'leaf')
                # Native trace includes the supplied first WAIT; Python calls the policy only after it.
                assert records[0]['observation']['tick']==root_obs['tick']
                assert len(python_states)==len(records)-1,(seed,name,'decision count',len(python_states),len(records))
                for py,record in zip(python_states,records[1:]):
                    cpp=canonical(_frame(record['observation']));cpp['sun_income_rate']=record['income']
                    assert py==cpp,(seed,name,'trajectory state')
                rows.append(dict(seed=seed,model=name,root_decision=point,root_tick=root_obs['tick'],future_seed=future_seed,
                                 decisions=len(records),student_decisions=len(python_states),forced_first_actions=1,end_tick=actual['tick'],trace=str(trace),trace_sha256=sha(trace),all_float32_inputs_bit_exact=True,full_trajectory_and_leaf_exact=True))
            env.release_snapshot(root)
    write(OUT/'continuations.json',dict(status='complete',root_model_pairs=8,encoded_float32_compared=total_bits,rows=rows))
    print(json.dumps(dict(root_model_pairs=8,decisions=sum(r['decisions'] for r in rows),encoded_float32_compared=total_bits),ensure_ascii=False))


def verify():
    """Read-only final evidence audit; never opens a game or starts optimisation."""
    import numpy as np
    import torch
    from train_rollout_policy import read_pack
    d=read(ROOT/'MEMORY/docs/task56-l20-gap-evidence.json')
    for s in d['sources'].values():assert sha(s['snapshot'])==s['sha256']
    for path,digest in d.get('production_code_sha256',{}).items():assert sha(ROOT/path)==digest
    inventory=read(OUT/'inventory.json');assert sha(OUT/'inventory.json')==d['inventory_sha256']
    ledger=read(d['sources']['ledger.json']['snapshot'])
    assert all(sum(x['levels'].get('20',0) for x in ledger[model]['labels'])==0 for model in ('dw4i','dw5i'))
    command=read(d['sources']['A-command.json']['snapshot'])['command'];labels=command[command.index('--labels')+1].split(',')
    assert len(labels)==281 and 'n20o1_fast_search' not in labels
    for x in inventory['n20o1']:assert sha(x['archive'])==x['sha256']
    assert len(inventory['n20o1'])==32 and sum(x['labelled_plantings'] for x in inventory['n20o1'])==575
    assert not inventory['qc']['collected_groups'] and inventory['qc']['matching_log_lines']==0
    assert inventory['qc']['offsets']=={'night_l20_train_n10_orig':32}
    for key in ('packing','native','continuations','diagnostic_plan'):
        s=d[key];assert sha(s['source'])==s.get('source_sha256',s.get('sha256'))
    plan=read(d['diagnostic_plan']['source'])
    for path,h in plan['sha256'].items():assert sha(path)==h,path
    assert plan['workers']==2 and plan['seeds']==[30000,30001,30002,30003]
    native_result=read(d['native']['source']);assert len(native_result['games'])==8 and len(native_result['rows'])==128
    for g in native_result['games']:assert sha(g['archive'])==g['sha256']
    groups={arm:[g for g in native_result['games'] if g['arm']==arm] for arm in ('fast','fast_search')}
    assert sum(g['won'] for g in groups['fast'])==0 and sum(g['plantings'] for g in groups['fast'])==0
    assert sum(g['won'] for g in groups['fast_search'])==4 and sum(g['plantings'] for g in groups['fast_search'])==231
    assert all(x['python_action']==x['native_action'] for x in native_result['rows'])
    assert sum(g['decisions']+1 for g in native_result['games'])==4634
    continuation=read(d['continuations']['source']);assert continuation['root_model_pairs']==8
    assert sum(x['student_decisions'] for x in continuation['rows'])==328
    assert continuation['encoded_float32_compared']==2077152
    for x in continuation['rows']:
        assert x['all_float32_inputs_bit_exact'] and x['full_trajectory_and_leaf_exact']
        assert sha(x['trace'])==x['trace_sha256']
    keys=[]
    def observer(path,rows):keys.extend((path,int(row)) for row in rows)
    packed=read_pack(Path(d['packing']['index']),OUT/'pack_view',observer)
    plant,packet,cell,weight=packed[2:6];packets=packed[-1]
    selected=np.array([j for j,(p,i) in enumerate(keys) if p.parent.name=='n20o1_fast_search' and plant[j]],np.int64)
    assert len(plant)==4428 and len(selected)==575
    slots=packet[selected].long().numpy()
    assert (packets.columns[0][selected,slots]>=0).all()
    assert (packets.columns[1][selected,slots]==1).all() and (packets.columns[3][selected,slots]==1).all()
    assert torch.equal(weight[selected],torch.ones(575)) and (cell[selected]>=0).all() and (cell[selected]<54).all()
    if d['status']=='complete':
        assert not d['pending'] and all(x['passed'] for x in d['completion_audit']['requirements'].values())
        assert d['diagnosis']['new_candidate_trained'] is False
        report=ROOT/'MEMORY/docs/task56-l20-student-gap.md'
        assert sha(report)==d['completion_audit']['report_sha256']
        import re
        for link in re.findall(r'\]\(([^)]+)\)',report.read_text()):
            path=re.sub(r':\d+$','',link.strip('<>').split('#',1)[0])
            if path and not path.startswith(('https://','http://')):assert (report.parent/Path(path)).is_file(),path
    return dict(passed=True,original_L20_training_rows=0,retained_L20_plants=575,matched_games=8,
                teacher_wins=4,student_wins=0,replayed_frames=4634,prediction_pairs=128,
                continuation_pairs=8,student_continuation_decisions=328,float32_inputs=2077152,
                scope='diagnosis/data-coverage branch; no new candidate or promotion claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['inventory','packing','native','continuations','verify'])
    args=parser.parse_args()
    if args.phase=='inventory':inventory()
    elif args.phase=='packing':packing()
    elif args.phase=='native':native()
    elif args.phase=='continuations':continuations()
    elif args.phase=='verify':print(json.dumps(verify(),ensure_ascii=False))
