"""Three-future truth must be paired by action and keep terminal bypass visible."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
import task78_roof_score as roof


def test_root_signatures_pair_raw_only_and_masked_labels_across_futures():
    legacy=dict(root_state_sha256='raw-first',history={'inputs':[1,2]},context=[3])
    replayed=dict(engine_sha256='raw-first',engine_canonical_sha256='stable')
    current=dict(legacy,root_state_sha256='raw-second',root_state_canonical_sha256='stable')
    another_replay=dict(engine_sha256='raw-third',engine_canonical_sha256='stable')
    assert roof.root_signature(legacy,replayed)==roof.root_signature(current,another_replay)
    assert roof.root_signature(legacy,replayed)[0]=='stable'
    assert roof.root_signature(legacy,dict(engine_sha256='raw-first'))[0]=='raw-first'


@pytest.mark.parametrize('fault',['different_state','missing_mask','missing_replay_mask'])
def test_root_signatures_reject_unproven_state_identity(fault):
    root=dict(root_state_sha256='raw-first',root_state_canonical_sha256='stable',history={},context=[])
    replayed=dict(engine_sha256='raw-second',engine_canonical_sha256='stable')
    if fault=='different_state':root['root_state_canonical_sha256']='different'
    elif fault=='missing_mask':root.pop('root_state_canonical_sha256')
    else:replayed.pop('engine_canonical_sha256')
    with pytest.raises(ValueError,match='root engine state differs from replay'):
        roof.root_signature(root,replayed)


@pytest.mark.parametrize('field',['history','context'])
def test_root_signatures_keep_student_history_and_context_distinct(field):
    root=dict(root_state_sha256='raw',history={'inputs':[1]},context=[2])
    changed=deepcopy(root);changed[field]=[]
    replayed=dict(engine_sha256='raw',engine_canonical_sha256='stable')
    assert roof.root_signature(root,replayed)!=roof.root_signature(changed,replayed)


@pytest.fixture
def futures():
    result=[]
    for k in range(3):
        rows=[]
        for plant in (False,True):
            rows.append(dict(candidate=int(plant) if k!=1 else int(not plant),
                action=dict(type='plant',packet=2,row=1,col=3) if plant else dict(type='wait',ticks=60),
                common_future=100+k,leaf_frame=dict(terminal=False),
                outcome=dict(terminal=True,won=(k!=2) if plant else (k==2)),
                values=dict(v5=.2 if plant else .8,v6=.9 if plant else .1)))
        result.append(rows[::-1] if k==1 else rows)
    return result


def test_action_matching_survives_changed_ordinals_and_averages_three_truths(futures):
    candidates=roof.paired_candidates(futures)
    plant=next(c for c in candidates if c['action']['type']=='plant')
    wait=next(c for c in candidates if c['action']['type']=='wait')
    assert plant['truth']==pytest.approx(2/3)
    assert wait['truth']==pytest.approx(1/3)
    stats=roof.consistency([dict(world='held-out',candidates=candidates)])
    assert stats['pairs']==1
    assert stats['rates']==dict(v5=0.,v6=1.)
    assert roof.improvement_interval(stats,draws=100)['lower']==1.


@pytest.mark.parametrize('fault',['missing_action','duplicate_action','duplicate_future','unfinished_truth'])
def test_incomplete_or_duplicated_futures_cannot_be_averaged(futures,fault):
    if fault=='missing_action': futures[1].pop()
    elif fault=='duplicate_action': futures[1].append(deepcopy(futures[1][0]))
    elif fault=='duplicate_future':
        for row in futures[1]: row['common_future']=100
    else: futures[1][0]['outcome']['terminal']=False
    with pytest.raises(ValueError): roof.paired_candidates(futures)


def test_terminal_pairs_are_reported_but_cannot_be_scored_as_nonterminal(futures):
    futures[0][0]['leaf_frame']['terminal']=True
    futures[0][0]['leaf_frame']['result']=2
    roots=[dict(world='held-out',candidates=roof.paired_candidates(futures))]
    primary=roof.consistency(roots)
    assert primary['pairs']==0 and primary['terminal_excluded_pairs']==1
    assert primary['terminal_leaves']==1 and primary['roots']==1
    assert roof.consistency(roots,include_terminal=True)['pairs']==1
    interval=roof.improvement_interval(primary,draws=100)
    assert interval['lower'] is None and interval['point'] is None


def test_terminal_bypass_must_agree_with_actual_terminal_truth(futures):
    futures[0][0]['leaf_frame']=dict(terminal=True,result=1)
    with pytest.raises(ValueError,match='terminal leaf and continuation truth differ'):
        roof.paired_candidates(futures)


def test_terminal_leaf_bypasses_models():
    # Sentinels are deliberately not callable: a terminal must never hit a net.
    models=dict(v5=object(),v6=object())
    assert roof.score_leaf(dict(leaf_frame=dict(terminal=True,result=2)),[],models)==dict(v5=0.,v6=0.)


def test_undefined_bootstrap_denominator_is_not_a_zero_rate():
    stats=dict(pairs=1,rates=dict(v5=0.,v6=1.),
               worlds=dict(signal=dict(pairs=1,v5=0.,v6=1.),empty=dict(pairs=0,v5=0.,v6=0.)))
    interval=roof.improvement_interval(stats,draws=100)
    assert interval['undefined_resamples']>0
    assert interval['lower'] is None and interval['status']=='insufficient_evidence'


def test_missing_jobs_produce_an_incomplete_receipt_without_loading_models(tmp_path,monkeypatch):
    candidate=tmp_path/'model.pt';candidate.write_bytes(b'not a checkpoint')
    monkeypatch.setattr(roof,'OUT',tmp_path)
    plan=tmp_path/'plan.json';plan.write_text('{}')
    missing=[dict(k=0,index=0,horizon=2500)]
    monkeypatch.setattr(roof,'coverage',lambda:([],missing,missing))
    result=roof.evaluate(candidate,tmp_path/'scored.json')
    assert result['complete'] is False and result['missing_jobs']==missing
    assert result['engine_processes_started']==0
    assert json.loads((tmp_path/'collection_jobs.json').read_text())['jobs']==missing


def test_legacy_single_future_reader_rejects_an_independent_panel(tmp_path):
    import task78_gates as gates
    path=tmp_path/'export.json'
    path.write_text(json.dumps(dict(complete=True,training_eligible=False,shards=[])))
    with pytest.raises(ValueError,match='task78_roof_score'):
        gates.load_rows(path,tmp_path)
