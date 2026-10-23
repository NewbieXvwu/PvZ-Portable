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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['inventory'])
    args=parser.parse_args()
    if args.phase=='inventory':inventory()
