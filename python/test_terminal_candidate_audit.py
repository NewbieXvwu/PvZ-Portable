"""Prevent silent coverage loss or reuse of incomplete expensive terminal labels."""
import gzip
import json
from pathlib import Path
import sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import terminal_candidate_audit as d


def test_full_legal_space_includes_shovels_and_wait_without_topk():
    plants=[dict(packet=p,row=r,col=c) for p in (1,4) for r in (0,4) for c in (0,8)]
    shovels=[dict(row=0,col=1),dict(row=4,col=8)]
    candidates=d.all_candidates(dict(legal_actions=dict(plants=plants,shovels=shovels)))
    assert len(candidates)==11
    assert candidates[0]==dict(type='wait',ticks=60,until='timeout')
    assert {(a['packet'],a['row'],a['col']) for a in candidates if a['type']=='plant'}=={(a['packet'],a['row'],a['col']) for a in plants}
    assert {(a['row'],a['col']) for a in candidates if a['type']=='shovel'}=={(a['row'],a['col']) for a in shovels}
    assert candidates==d.all_candidates(dict(legal_actions=dict(plants=plants,shovels=[[1,0],[8,4]])))
    with pytest.raises(ValueError):d.all_candidates(dict(legal_actions=dict(plants=[plants[0],plants[0]])))


def test_sampling_uses_time_without_outcomes_and_deduplicates():
    eligible=[dict(decision=i,tick=t) for i,t in enumerate((0,4900,5100,9900))]
    assert [x['tick'] for x in d.pick_roots(eligible,3)]==[0,4900,9900]
    assert [x['tick'] for x in d.pick_roots(eligible,1)]==[4900]
    assert d.pick_roots([],3)==[]
    assert len(d.pick_roots(eligible[:1],3))==1


def test_common_futures_stable_and_stage_separated(monkeypatch):
    monkeypatch.setattr(d,'PLAN',{'spec':{'namespace':'fixture'}})
    first=[d.future_seed('cell',15,k) for k in range(3)]
    assert first==[d.future_seed('cell',15,k) for k in range(3)]
    assert len(set(first))==3
    assert not set(first)&{d.future_seed('cell',15,k,'confirm') for k in range(3)}


def test_terminal_trace_identity_and_coverage_required(tmp_path):
    path=tmp_path/'trace.gz'
    with gzip.open(path,'wt') as stream:stream.write('{"decision":0}\n')
    identity={'candidate':1,'future':2}
    row=dict(identity=identity,outcome=dict(terminal=True,result=2,won=False,decisions=1),
        checks={x:True for x in ('root_engine','root_history','root_income','legal_candidate','terminal','no_rejected_actions')},trace_sha256=d.sha(path))
    assert d.validate_label(row,identity,path)==row
    with pytest.raises(ValueError):d.validate_label(row,dict(candidate=2),path)
    row['outcome']['terminal']=False
    with pytest.raises(ValueError):d.validate_label(row,identity,path)
    row['outcome']['terminal']=True;row['outcome']['decisions']=2
    with pytest.raises(ValueError,match='decision chain'):d.validate_label(row,identity,path)


def test_cpu_receipts_do_not_double_count_waited_children(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'OUT',tmp_path)
    folder=tmp_path/'runs'/'one';folder.mkdir(parents=True)
    (folder/'usage-1.json').write_text(json.dumps(dict(cpu_seconds=12)))
    (folder/'usage-controller.json').write_text(json.dumps(dict(cpu_seconds=2)))
    (folder/'complete.json').write_text(json.dumps(dict(controller_children_cpu_seconds=15)))
    assert d.spent_cpu()==15
    (folder/'complete.json').write_text(json.dumps(dict(cases=8)))
    assert d.spent_cpu()==14


def test_frozen_selection_resume_ignores_later_cpu_usage(tmp_path,monkeypatch):
    import roof_terminal_pipeline as pipeline
    monkeypatch.setattr(d,'OUT',tmp_path/'artifacts')
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(d,'check_plan',lambda:{'inputs':'unchanged'})
    trajectory=d.OUT/'episodes'/'cell'/'trajectory.json'
    d.save(trajectory,dict(prefix=[1,2,3]))
    selection=d.OUT/'selection-active.json'
    d.save(selection,dict(jobs=[],root_trajectory_sha256={'cell':d.sha(trajectory)}))
    summary=dict(root_selection_sha256=d.sha(selection),cpu_seconds_before_screen=100)
    d.save(tmp_path/'MEMORY/docs/task46-roof-terminal-sampling.json',summary)
    monkeypatch.setattr(d,'spent_cpu',lambda:9999)
    assert pipeline.select()==summary
    d.save(trajectory,dict(prefix=[1,2,4]))
    with pytest.raises(ValueError,match='trajectory changed'):pipeline.select()


def test_timing_amendment_keeps_both_stages_and_requires_completion(tmp_path,monkeypatch):
    import roof_terminal_pipeline as pipeline
    monkeypatch.setattr(d,'OUT',tmp_path)
    manifests=[]
    for stage,tick,cpu in [('pilot',5000,50),('pilot-end',10000,5)]:
        job=dict(cell=dict(id='cell',level=42),point=dict(decision=tick//60,tick=tick),stage=stage)
        name=stage+'-jobs.json';manifests.append(name);d.save(tmp_path/name,dict(jobs=[job]))
        folder=tmp_path/'episodes'/'cell'/f'root_{tick//60}'
        d.save(folder/'root.json',dict(root_frame=dict(packets=[[0,33]])))
        d.save(folder/f'{stage}_0_0.json',dict(identity=dict(action=dict(type='plant',packet=0)),cpu_seconds=cpu))
        d.save(folder/f'{stage}-complete.json',dict(labels=1))
    plan={'spec':{'timing_job_manifests':manifests}}
    times=pipeline.timing_samples(plan)
    assert times[(42,5000,'plant:33')]==[50]
    assert times[(42,10000,'plant:33')]==[5]
    (folder/'pilot-end-complete.json').unlink()
    with pytest.raises(ValueError,match='incomplete'):pipeline.timing_samples(plan)


def test_sparse_cpu_selection_preserves_named_and_reset_seed_coverage(tmp_path,monkeypatch):
    import roof_terminal_pipeline as pipeline
    monkeypatch.setattr(d,'OUT',tmp_path)
    a=dict(id='a',level=42,seed=1);b=dict(id='b',level=42,seed=2)
    archive=dict(id='archive',level=42,seed=1,archive=True)
    named=dict(cell=archive,point=dict(decision=99),samples=2,stage='screen')
    monkeypatch.setattr(pipeline,'selections',lambda *args:[named])
    # Cheap root wins despite opposite, irrelevant outcome metadata; outside-window root excluded.
    d.save(tmp_path/'episodes/b/trajectory.json',dict(eligible=[
        dict(decision=1,tick=60,candidate_count=30,won=True),
        dict(decision=2,tick=120,candidate_count=20,won=False),
        dict(decision=3,tick=11000,candidate_count=1,won=True)]))
    plan=dict(cells=[a,b,archive],spec=dict(window_end_tick=10000))
    selected=pipeline.budget_selections(plan,{(42,0,'wait'):[10]})
    assert selected[0]==named
    assert [(j['cell']['seed'],j['point']['decision']) for j in selected]==[(1,99),(2,2)]
