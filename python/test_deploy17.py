"""task165: small allocation/recipe/pack fixtures; never start a production service."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import curriculum_queue as q
import make_random_deck_cells as decks


def fixture(tmp_path):
    archive=tmp_path/'archive';archive.mkdir()
    (archive/'meta.json').write_text(json.dumps(dict(verified=True,task=dict(seed=42))))
    cells={};tasks={}
    for kind,n in [('opening',8),('demo',275),('rd',16),('mult',43)]:
        for i in range(n):
            name=f'{kind}_l9_{i:03d}'
            cells[name]=dict(level=9,mult_index=0,last_round=1,windows={},family='fixture')
            tasks[name]=dict(task_id=name,level=9,seeds=list(range(128)))
            if kind=='opening': cells[name]['opening_specialist']=True
            if kind=='rd': cells[name]['random_deck_cell']=True
            if kind=='demo':
                cells[name].update(archive=str(archive),task_id=name,seeds=list(range(128)),ladder=[100])
                tasks[name]['seeds']=[42]
    state=dict(round=47,cells=cells,offsets={},retired_cells={},log=[])
    return state,tasks,{k:tmp_path/'panel.json' for k in tasks}


def counts(state,groups):
    totals=dict(opening=0,demo=0,random=0,multiplier=0)
    for group in groups:
        cell=state['cells'].get(group['base'],state['retired_cells'].get(group['base'],{}))
        kind='opening' if cell.get('opening_specialist') else 'random' if cell.get('random_deck_cell') else 'demo' if cell.get('archive') else 'multiplier'
        totals[kind]+=group['games']
    return totals


def test_deploy17_allocation_and_partial_resume(tmp_path):
    original,tasks,manifests=fixture(tmp_path)
    def plan(state,**kw): return q.plan_round(state,tasks,{},[1.0],manifests,768,48,**kw)[0]
    legacy=plan(copy.deepcopy(original))
    assert counts(original,legacy)==dict(opening=192,demo=576,random=0,multiplier=0)
    state=copy.deepcopy(original)
    proposed=plan(state,demo_start_share=.25,random_deck_share=.25)
    assert counts(state,proposed)==dict(opening=192,demo=192,random=192,multiplier=192)
    complete=[proposed[0]]+[g for g in proposed if g['base'].startswith('rd_')][:4]+[g for g in proposed if g['base'].startswith('demo_')][:4]
    state['completed_groups']=complete
    for group in complete:
        state['offsets'][group['task_id']]=group['offset']+group['games']
        if not group['base'].startswith('opening'):state['cells'][group['base']]['last_round']=48
    demo=next(g['base'] for g in complete if g['base'].startswith('demo_'))
    state['retired_cells'][demo]=state['cells'].pop(demo)
    resumed=plan(state,demo_start_share=.25,random_deck_share=.25)
    assert counts(state,complete+resumed)==dict(opening=192,demo=192,random=192,multiplier=192)
    assert not {g['label'] for g in resumed}&{g['label'] for g in complete}


def test_deploy17_scarcity_never_borrows_demo_quota(tmp_path):
    state,tasks,manifests=fixture(tmp_path)
    state['cells']={k:v for k,v in state['cells'].items() if not k.startswith('rd_')}
    plan,_=q.plan_round(state,tasks,{},[1.0],manifests,768,48,demo_start_share=.25,random_deck_share=.25)
    totals=counts(state,plan)
    assert totals['demo']<=192 and totals['multiplier']>192 and sum(totals.values())<=768


def test_deploy17_recipe_and_frozen_pool(tmp_path):
    a=json.loads((ROOT/'recipes/dw4i2_v2timer.json').read_text())
    b=json.loads((ROOT/'recipes/dw4i2_v2timer_history10.json').read_text())
    assert {k for k in a if a[k]!=b[k]}=={'label_weight'}
    assert len(a['label_weight'])==33 and all(k.startswith('z') for k in a['label_weight'])
    assert set(a['label_weight'].values())=={1.5} and set(b['label_weight'].values())=={1.0}
    registry=json.loads(decks.REGISTRY.read_text())
    holdouts=[ROOT/'configs/deploy17_deck_holdouts.json']
    pool=decks.load_frozen_pool(ROOT/'configs/deploy17_random_deck_pool.json',registry,holdouts)
    training={tuple(sorted(t['deck'])) for t in pool+registry['tasks'] if t.get('split','train')=='train'}
    assert not training&decks.heldout_signatures(registry,holdouts)
    assert len(pool)>0 and any('task148' in t.get('also_sources',[]) for t in pool)
    duplicate=tmp_path/'bad.json';duplicate.write_text(json.dumps(dict(tasks=[pool[0],pool[0]])))
    with pytest.raises(ValueError):decks.load_frozen_pool(duplicate,registry,holdouts)


def test_deploy17_shared_pack_weights_match_physical_pack(tmp_path):
    import torch
    from test_incremental_pack import fixture as rows,options
    from pack_training_rows import pack
    import train_rollout_policy as trainer
    from pvz_training_rows import LabelWeightAdjustment
    data=tmp_path/'data';rows(data,['fresh','z_old'])
    a=pack(data,options(data,['fresh','z_old'],label_weight={'z_old':1.5}),tmp_path/'cache')
    adjust=LabelWeightAdjustment({'z_old':1.0},{'z_old':1.5})
    shared=trainer.read_pack(a,data,row_observer=adjust.observe)
    original=shared[5].clone();adjust.apply(shared[5])
    b=pack(data,options(data,['fresh','z_old'],label_weight={'z_old':1.0}),tmp_path/'cache')
    physical=trainer.read_pack(b,data)
    torch.testing.assert_close(shared[5],physical[5],rtol=0,atol=1e-6)
    assert not torch.equal(original,shared[5])
    torch.testing.assert_close(trainer.read_pack(a,data)[5],original,rtol=0,atol=0)
    assert torch.equal(shared[2],physical[2]) and torch.equal(shared[3],physical[3])


def test_deploy17_training_cli_uses_shared_pack_without_training(tmp_path,monkeypatch):
    import torch
    import train_rollout_policy as trainer
    from test_incremental_pack import fixture as rows,options
    from pack_training_rows import pack
    from pvz_recipe import write_recipe
    data=tmp_path/'data';rows(data,['fresh','z_old'])
    a=pack(data,options(data,['fresh','z_old'],label_weight={'z_old':1.5}),tmp_path/'cache')
    packed=trainer.read_pack(a,data)[5].clone()
    common=dict(schema_version=1,kind='training',data=str(data),labels=['fresh','z_old'],device='cpu',threads=1,
        wait_keep=1.0,obs='v1',packet_head='slot',cell_soft_weight=0,source_row_report=False,
        out=str(tmp_path/'student'))
    write_recipe(tmp_path/'a.json',dict(common,label_weight={'z_old':1.5}))
    write_recipe(tmp_path/'b.json',dict(common,label_weight={'z_old':1.0}))
    monkeypatch.setattr(sys,'argv',['train','--recipe',str(tmp_path/'b.json'),'--pack-recipe',str(tmp_path/'a.json')])
    monkeypatch.setattr(trainer,'Metrics',Mock())
    monkeypatch.setattr(trainer,'set_oom_score_adj',lambda *_:None)
    observed={};original=trainer.load
    def load(*args,**kwargs):
        observed['weights']=kwargs['label_weight']
        result=original(*args,**kwargs);observed['result']=result;return result
    monkeypatch.setattr(trainer,'load',load)
    class Loaded(Exception):pass
    class StopBeforeSplit:
        def __init__(self,*_):pass
        def shuffle(self,*_):raise Loaded
    monkeypatch.setattr(trainer.random,'Random',StopBeforeSplit)
    with pytest.raises(Loaded):trainer.main()
    assert observed['weights']=={'z_old':1.5}
    assert not torch.equal(observed['result'][5],packed)
    note=json.loads((tmp_path/'student/shared_pack_weights.json').read_text())
    assert note['factors']=={'z_old':2/3}


def test_deploy17_pipeline_accepts_label_only_alternate(tmp_path):
    import threading
    pipeline=q.TrainingPipeline.__new__(q.TrainingPipeline)
    pipeline.args=SimpleNamespace(recipe=ROOT/'recipes/dw4i2_v2timer.json',
        alternate_recipe=ROOT/'recipes/dw4i2_v2timer_history10.json',candidates_per_recipe=2,
        state=tmp_path/'state.json',log=tmp_path/'qc.log')
    pipeline.directory=tmp_path/'pipeline';pipeline.build_workers=1
    pipeline.state=dict(round=1,log=[],training_jobs=[])
    pipeline._plan_build=Mock();pipeline._snapshot_selection=Mock();pipeline._seeded=Mock()
    pipeline.executor=Mock();pipeline.futures=[]
    pipeline.enqueue()
    job=pipeline.state['training_jobs'][0]
    assert [r['recipe_family'] for r in job['candidate_recipes']]==['A','A','B','B']
    assert job['training_seeds']==[1,2,3,4]


@pytest.mark.parametrize('active',[False,True])
def test_deploy17_watchdog_launch_is_inert_until_active(tmp_path,active):
    import subprocess
    result=subprocess.run(['bash','-c','source "$1"; watchdog_extract_start "$2"','test',
        str(ROOT/'scripts/qc_watchdog.sh'),str(ROOT/'scripts/qc_deploy17.sh')],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    script=tmp_path/'start.sh';script.write_text(result.stdout.replace('$HOME',str(tmp_path)))
    logs=tmp_path/'pvz-logs';logs.mkdir()
    (logs/'qc_deploy16.mode').write_text('v2timer\n');(logs/'qc_deploy16.active').touch()
    if active:
        (logs/'qc_deploy17.mode').write_text('datamix\n');(logs/'qc_deploy17.active').touch()
    result=subprocess.run(['bash','-c','tmux() { printf "%s\\n" "$*"; }; source "$1"','test',str(script)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert 'artifacts/production/deploy16/pvz-portable' in result.stdout
    assert '--training-parallelism 1' in result.stdout and '--gate-version 5' in result.stdout
    assert result.stdout.count('--extra-manifest')==1
    if active:
        assert 'history10.json' in result.stdout and 'task165/training_panel.json' in result.stdout
        assert '--demo-start-share 0.25 --random-deck-share 0.25' in result.stdout
    else:
        assert 'rebalanced_p05_v2timer.json' in result.stdout and '--demo-start-share' not in result.stdout


def test_deploy17_boundary_barrier_is_scoped_and_resumes_busy_scheduler(tmp_path,monkeypatch):
    import deploy17_boundary as b
    import signal
    proc=tmp_path/'proc';worktree=tmp_path/'cq'
    def process(pid,args,children=''):
        folder=proc/str(pid);(folder/'task'/str(pid)).mkdir(parents=True)
        (folder/'task'/str(pid)/'children').write_text(children)
        (folder/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
        (folder/'status').write_text('State:\tT (stopped)\n')
        (folder/'stat').write_text(f'{pid} (python) '+' '.join(['T']+['0']*18+['123']+['0']*2))
    process(101,['python',str(worktree/'scripts/curriculum_queue.py')],'202')
    process(202,['python',str(worktree/'scripts/search_play.py')])
    realpath=Path
    monkeypatch.setattr(b,'Path',lambda value:proc if str(value)=='/proc' else realpath(value))
    signals=[];monkeypatch.setattr(b.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    assert b.freeze_if_idle(101,worktree) is None
    assert signals==[(101,signal.SIGSTOP),(101,signal.SIGCONT)]
    (proc/'202/cmdline').write_bytes(b'python\0trainer\0');signals.clear()
    record=tmp_path/'scheduler.json'
    paused=b.freeze_if_idle(101,worktree,record)
    assert record.is_file() and signals==[(101,signal.SIGSTOP)]
    assert paused['children']==[dict(pid=202,start='123')]
    (proc/'202/stat').write_text('202 (reused) '+' '.join(['T']+['0']*18+['999']))
    b.act(paused['children'][0],signal.SIGTERM)
    assert signals==[(101,signal.SIGSTOP)]
    # stop also catches children forked by a background builder after the saved boundary.
    process(303,['python','new-builder'])
    (proc/'202/task/202/children').write_text('303')
    monkeypatch.setattr(sys,'argv',['boundary','stop','--worktree',str(worktree),
        '--log',str(tmp_path/'log'),'--record',str(record)])
    b.main()
    assert (303,signal.SIGTERM) in signals and signals[-1]==(101,signal.SIGCONT)
    assert {pid for pid,_ in signals}<={101,202,303}


def test_deploy17_backup_restore_only_temporary_files(tmp_path):
    import subprocess
    text=(ROOT/'scripts/qc_deploy17.sh').read_text().split('# --- switch:',1)[0]
    script=tmp_path/'functions.sh';script.write_text(text.replace('$HOME',str(tmp_path)))
    result=subprocess.run(['bash','-c','''
source "$1"
O="$2/backup"; mkdir -p "$W/scripts" "$O"
FILES=(scripts/example.py recipes/new.json)
GENERATED=artifacts/task165/training_panel.json
printf 'original\n' > "$W/scripts/example.py"
backup_files
printf 'changed\n' > "$W/scripts/example.py"
mkdir -p "$W/recipes"; printf 'new\n' > "$W/recipes/new.json"
restore_files
[[ "$(cat "$W/scripts/example.py")" = original ]]
[[ ! -e "$W/recipes/new.json" ]]
[[ -f "$O/backed_up" ]]
''','test',str(script),str(tmp_path)],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr


def test_deploy17_boundary_waits_for_durable_group_record(tmp_path):
    import deploy17_boundary as b
    log=tmp_path/'curriculum.log';state=tmp_path/'curriculum_state.json'
    log.write_text('2026-10-29T12:00:00 round 48 c48task_o24 task mult=1 rc=0 20s\n')
    state.write_text(json.dumps(dict(completed_groups=[])))
    assert not b.committed_boundary(log,state)
    state.write_text(json.dumps(dict(completed_groups=[dict(label='c48task_o24')])))
    assert b.committed_boundary(log,state)
    log.write_text('2026-10-29T12:00:00 round 48 c48task_o24: group skipped: failed\n')
    assert b.committed_boundary(log,state)
    with log.open('a') as f:f.write('2026-10-29T12:00:00 round 48 c48task_o48: workers=2/16\n')
    assert not b.committed_boundary(log,state)
