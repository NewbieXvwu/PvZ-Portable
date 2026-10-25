"""Regressions for the task76 coverage and terminal-analysis audit findings."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import roof_candidates_v5_analysis as audit

WAIT=dict(type='wait',ticks=60,until='timeout')
A=dict(type='plant',packet=0,row=0,col=0)
B=dict(type='plant',packet=0,row=0,col=1)


def row(candidates,expanded,values,executed=A):
    return dict(decision=0,tick=0,wave=0,production_candidates=candidates,
        evaluated_candidates=expanded,static_scores=[dict(action=A,value=.9),dict(action=B,value=.2)],
        legal_actions=[A,B,WAIT],rounds=[[dict(action=a,value=v) for a,v in values]],executed=executed)


def test_recorded_extra_student_candidate_is_not_a_static_omission():
    r=row([A,B,WAIT],[A,B,WAIT],[(A,.1),(B,.8),(WAIT,.05)],B)
    result=audit.decision_measure(r)
    assert result['global_complete'] and result['global_best_omitted'] is False
    assert result['global_best_static_omitted'] is True
    assert result['chosen_outside_production'] is False


def test_unscored_legal_actions_are_unknown_not_zero_omissions():
    result=audit.decision_measure(row([A,WAIT],[A,WAIT],[(A,.1),(WAIT,.05)]))
    assert not result['global_complete'] and result['scored_legal_plants']==1
    assert result['global_best_omitted'] is None and result['global_regret'] is None


def test_tied_best_cell_inside_production_prevents_false_omission():
    result=audit.decision_measure(row([A,WAIT],[A,WAIT,B],[(B,.8),(A,.8),(WAIT,.1)]))
    assert result['global_complete'] and result['global_best_omitted'] is False
    assert result['global_regret']==0


def test_predicted_ties_get_symmetric_half_credit():
    assert audit.direction_score(0,1)==.5
    assert audit.direction_score(0,-1)==.5
    assert audit.direction_score(-.1,-1)==1
    assert audit.direction_score(.1,-1)==0


def test_common_worlds_are_root_and_stage_specific():
    common=audit.future_seeds(42,45000000,119,'terminal',8)
    assert len(common)==len(set(common))==8 and all(0<=v<2**32 for v in common)
    assert common==audit.future_seeds(42,45000000,119,'terminal',8)
    assert common!=audit.future_seeds(43,45000000,119,'terminal',8)
    assert common!=audit.future_seeds(42,45000000,119,'leaf',8)


def test_nondiscriminating_terminal_root_is_not_ranking_evidence():
    document=dict(actions=[dict(action=a,terminal=[dict(won=False)]*8,
        leaf={'value':{'2500':[v]*9}}) for a,v in [(A,.3),(B,.4),(WAIT,.1)]])
    result=audit.metrics(document)
    assert result['all_actions_zero_wins'] and not result['discriminating']
    assert result['pairs']==0 and result['accuracy'] is None


def test_argmax_compares_sets_without_adversarial_tie_breaking():
    document=dict(actions=[dict(action=a,terminal=[dict(won=(i<n)) for i in range(8)],
        leaf={'value':{'2500':[v]*9}}) for a,n,v in [(A,2,.3),(B,2,.5),(WAIT,0,.1)]])
    result=audit.metrics(document)
    assert result['discriminating'] and result['tie_aware_argmax_overlap']
    assert result['pairs']==2 and result['accuracy']==1


def test_memory_cap_rejects_excess_workers_before_launch(monkeypatch):
    import pytest
    monkeypatch.setattr(audit.base, 'native_game_processes', lambda: [])
    with pytest.raises(ValueError, match='one or two'):
        audit.base.require_game_capacity(3)
    audit.base.require_game_capacity(2)


def test_existing_games_reserve_capacity_for_qc(monkeypatch):
    import pytest
    monkeypatch.setattr(audit.base, 'native_game_processes',
                        lambda: [dict(pid=123, executable='/qc/pvz-portable')])
    with pytest.raises(RuntimeError, match='qc has priority'):
        audit.base.require_game_capacity(2)
    audit.base.require_game_capacity(1)


def test_full_global_budget_starts_no_game(monkeypatch):
    import pytest
    monkeypatch.setattr(audit.base, 'native_game_processes', lambda: [dict(pid=1), dict(pid=2)])
    with pytest.raises(RuntimeError, match='no task76 game started'):
        audit.base.require_game_capacity(1)
