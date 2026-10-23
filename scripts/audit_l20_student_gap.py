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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['inventory','packing'])
    args=parser.parse_args()
    if args.phase=='inventory':inventory()
    elif args.phase=='packing':packing()
