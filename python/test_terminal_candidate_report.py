from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import terminal_candidate_report as report


def test_ranking_ties_are_half_credit_and_truth_ties_excluded():
    assert report.pair_counts([1,2],[0,0])==dict(pairs=1,correct=.5,value_ties=1,accuracy=.5)
    assert report.pair_counts([1,1],[0,1])['pairs']==0
    assert report.pair_counts([1,1.5],[0,1],gap=1)['pairs']==0


def fixture(counts,values,best):
    wait=dict(type='wait',ticks=60,until='timeout');plant=dict(type='plant',packet=1,row=2,col=3)
    group=dict(cell=dict(id='fixture'),point=dict(decision=0),student_position=0,
        candidates=[dict(index=0,action=wait,means=dict(won=0,wave=8,tick=18000)),
                    dict(index=1,action=plant,means=dict(won=0,wave=10,tick=22000))])
    trace=dict(static_actions=[plant],static_values=[.4],per_card=1,candidates=[wait,plant],
        best_index=best,student_index=0,before_verification_values=values,
        after_verification_values=values,halving_counts=counts,required_halving_samples=3,
        chosen=wait,margin=.02)
    return group,trace


def test_margin_and_halving_are_distinct_attribution_stages(monkeypatch):
    group,trace=fixture([3,3],[.5,.515],1)
    monkeypatch.setattr(report.diag,'load',lambda _:trace)
    got=report.attribution(group)['stages']['wave']
    assert got['observed_improvers']==got['initial_margin_blocked']==1
    group,trace=fixture([3,1],[.5,.4],0)
    got=report.attribution(group)['stages']['wave']
    assert got['halving_pruned']==1 and 'initial_margin_blocked' not in got


def test_static_omission_is_separate_from_student_candidate_recovery(monkeypatch):
    group,trace=fixture([3,3],[.5,.49],0)
    trace['static_actions']=[];trace['static_values']=[]
    monkeypatch.setattr(report.diag,'load',lambda _:trace)
    got=report.attribution(group)['stages']['wave']
    assert got['leaf_ranking_not_top']==1 and 'candidate_generation_omission' not in got
    trace['candidates']=trace['candidates'][:1]
    got=report.attribution(group)['stages']['wave']
    assert got['candidate_generation_omission']==1


def test_independent_repro_checks_trace_and_value_semantics(tmp_path):
    import gzip
    import pytest
    from roof_terminal_repro import compare
    original={k:{} for k in ('identity','outcome','v4_short_value','short_leaf','teacher_leaf','terminal_state_sha256','checks')}
    a=tmp_path/'a.gz';b=tmp_path/'b.gz'
    for path in (a,b):
        with gzip.open(path,'wb') as stream:stream.write(b'{"chosen":"wait"}\n')
    assert compare(original,dict(original),a,b)['teacher_action_chain_equal']
    changed=dict(original,v4_short_value=.3)
    with pytest.raises(ValueError,match='v4_short_value'):compare(original,changed,a,b)
    with gzip.open(b,'wb') as stream:stream.write(b'{"chosen":"plant"}\n')
    with pytest.raises(ValueError,match='action_chain'):compare(original,dict(original),a,b)
