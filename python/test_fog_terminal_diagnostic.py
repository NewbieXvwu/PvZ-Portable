"""Full legal coverage and reproducible terminal evidence are essential for appendix E."""
import gzip
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from fog_terminal_diagnostic import all_candidates,pick_roots,future_seed,validate_label,sha


def test_candidates_cover_all_legal_slots_and_cells_without_filtering():
    plants=[dict(packet=p,row=r,col=c) for p in (1,6,7) for r in (0,3) for c in (0,8)]
    actions=all_candidates({'legal_actions':{'plants':list(reversed(plants))}})
    assert actions[0]==dict(type='wait',ticks=60,until='timeout')
    assert {(a['packet'],a['row'],a['col']) for a in actions[1:]}=={(a['packet'],a['row'],a['col']) for a in plants}
    assert len(actions)==13
    with pytest.raises(ValueError,match='duplicate'):
        all_candidates({'legal_actions':{'plants':[plants[0],plants[0]]}})


def test_root_selection_is_temporal_and_deduplicates_sparse_trajectories():
    roots=[dict(decision=i,tick=t) for i,t in enumerate((0,1000,3900,4100,7900))]
    assert [r['tick'] for r in pick_roots(roots,3)]==[0,3900,7900]
    assert [r['tick'] for r in pick_roots(roots,2)]==[0,7900]
    assert len(pick_roots(roots[:1],3))==1
    assert pick_roots([],3)==[]


def test_common_futures_stable_and_confirmation_is_separate():
    screen=[future_seed('l31-human-39000000',40,k) for k in range(3)]
    test=[future_seed('l31-human-39000000',40,k,'confirm') for k in range(16)]
    assert len(set(screen))==3 and len(set(test))==16 and not set(screen)&set(test)
    assert screen==[future_seed('l31-human-39000000',40,k) for k in range(3)]


def test_incomplete_or_changed_terminal_evidence_cannot_be_reused(tmp_path):
    trace=tmp_path/'trace.gz'
    with gzip.open(trace,'wt') as f:f.write('{"decision":0}\n')
    identity={'future':41,'candidate':3}
    row=dict(identity=identity,outcome=dict(terminal=True,result=2,won=False,decisions=1),
       checks={k:True for k in ('root_engine','root_history','root_income','legal_candidate','terminal','no_rejected_actions')},trace_sha256=sha(trace))
    assert validate_label(row,identity,trace) is row
    with pytest.raises(ValueError):validate_label(row,{'future':42},trace)
    row['outcome']['terminal']=False
    with pytest.raises(ValueError):validate_label(row,identity,trace)
    row['outcome']['terminal']=True;row['checks']={}
    with pytest.raises(ValueError):validate_label(row,identity,trace)
    row['checks']={k:True for k in ('root_engine','root_history','root_income','legal_candidate','terminal','no_rejected_actions')}
    row['outcome']['decisions']=2
    with pytest.raises(ValueError,match='decision chain'):validate_label(row,identity,trace)
