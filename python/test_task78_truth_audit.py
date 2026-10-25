"""Keep the third labelled arm and distinguish insufficient signal from zero accuracy."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import task78_truth_audit as audit


@pytest.fixture
def row():
    truth={}
    for action,wins in [('wait',0),('0,1',1),('0,2',2)]:
        outcomes=[1.]*wins+[0.]*(8-wins)
        truth[action]=dict(wins=outcomes,finals=outcomes[:],win_rate=wins/8)
    return dict(task='probe',seed=101,decision=1,tick=100,cells=[[0,1],[0,2]],n_cells=2,
                static_best=[0,1],truth=truth,
                values={h:{'wait':.2,'0,1':.3,'0,2':.4} for h in audit.HORIZONS})


def test_third_labelled_candidate_can_supply_the_only_registered_pair(row):
    result=audit.analyse([row],expected_count=1)
    assert result['all_labelled_pairs']==3
    assert result['eligible_truth_pairs']==1
    assert {result['eligible_pairs'][0]['left'],result['eligible_pairs'][0]['right']}=={'wait','0,2'}
    assert result['static_vs_wait_gap_counts']=={'0.125':1}
    assert result['ranking_signal_status']=='leaf_scores_and_terminal_classification_required'
    assert result['leaf_scores_recomputed'] is False


@pytest.mark.parametrize('corruption',['rate','short_futures','nonbinary_final','missing_arm','unknown_arm','missing_horizon'])
def test_bad_summary_cannot_be_certified(row,corruption):
    if corruption=='rate':row['truth']['wait']['win_rate']=.5
    elif corruption=='short_futures':row['truth']['wait']['wins'].pop()
    elif corruption=='nonbinary_final':row['truth']['wait']['finals'][0]=.01
    elif corruption=='missing_arm':del row['truth']['0,1']
    elif corruption=='unknown_arm':row['truth']['5,8']=deepcopy(row['truth']['wait'])
    else:del row['values']['4000']
    with pytest.raises(ValueError):audit.analyse([row],expected_count=1)


def test_tick_distinguishes_same_future_and_continuation_ordinal(row):
    second=deepcopy(row);second['tick']+=60
    assert audit.analyse([row,second],expected_count=2)['states']==2
    with pytest.raises(ValueError,match='duplicate'):
        audit.analyse([row,row],expected_count=2)


def test_original_all_94_states_have_no_eligible_pair_even_with_all_third_arms():
    rows=[json.loads(line) for line in audit.REFERENCE.read_text().splitlines()]
    result=audit.analyse(rows)
    assert result['labelled_candidates_per_state']=={'2':16,'3':78}
    assert result['recorded_candidate_future_outcomes']==2128
    assert result['all_labelled_pairs']==250
    assert result['absolute_gap_counts']=={'0.0':236,'0.125':14}
    assert result['eligible_truth_pairs']==0
    assert result['nonterminal_eligible_pairs_upper_bound']==0
    assert result['ranking_signal_status']=='insufficient_evidence'
    assert result['value_consistency']['v6'] is None
    assert result['value_consistency']['bootstrap_interval'] is None
    assert result['horizon_stability_evaluated'] is False
