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
