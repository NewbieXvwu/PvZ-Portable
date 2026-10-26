"""A continuation ordinal is insufficient to identify an original task72b state."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import task78_task72_rebuild as rebuild


def test_reconstruction_binds_the_original_mac_static_checkpoint(monkeypatch):
    monkeypatch.setattr(rebuild,'sha',lambda path:rebuild.ORIGINAL_STATIC_SHA256
                        if path==rebuild.ORIGINAL_STATIC else 'other input')
    assert rebuild.inputs()['static']==dict(path=str(rebuild.ORIGINAL_STATIC),
                                           sha256=rebuild.ORIGINAL_STATIC_SHA256)
    assert rebuild.OUT.name=='task72_rebuild_mac_branch1'


def test_desktop_static_checkpoint_cannot_substitute_for_task72b(monkeypatch):
    monkeypatch.setattr(rebuild,'sha',lambda path:
        '2485841036cf3a8738174affb0dadc0201a925ca3ea0ebe6e3eb02e74621e314')
    with pytest.raises(ValueError,match='original Mac branch1'):
        rebuild.inputs()


@pytest.fixture
def signature():
    return dict(task=rebuild.TASK,seed=102,decision=113,tick=17083,wave=7,sun=135,
                balloons=2,balloon_x=[764,788],n_cells=2,cells=[[0,6],[1,7]],
                static_best=[0,6],chosen=dict(type='wait',ticks=60,until='timeout'),
                action_rejected=False,engine_state_sha256='state')


def test_same_task_future_and_ordinal_can_name_two_different_original_states():
    rows=rebuild.reference_rows()
    repeated=[row for row in rows if row['seed']==102 and row['decision']==113]
    assert len(repeated)==2
    assert len({rebuild.identity(row) for row in repeated})==2
    assert len({row['tick'] for row in repeated})==2


@pytest.mark.parametrize('field',['task','seed','decision','tick','wave','sun','balloons',
                                 'n_cells','cells','balloon_x','static_best','chosen'])
def test_exact_matching_rejects_changed_state_or_candidate_identity(signature,field):
    reference=deepcopy(signature)
    reference['chosen']=json.dumps(reference['chosen'],sort_keys=True)
    assert rebuild.matches(reference,signature)
    changed=deepcopy(signature)
    value=changed[field]
    changed[field]=value+1 if isinstance(value,int) else 'different'
    assert not rebuild.matches(reference,changed)


def install_trace(tmp_path,monkeypatch,signature,takeover):
    directory=tmp_path/f'n{takeover}/f102';directory.mkdir(parents=True)
    path=directory/'signatures.jsonl';path.write_text(json.dumps(signature)+'\n')
    plan=tmp_path/'plan.json'
    if not plan.exists():plan.write_text('{}')
    (directory/'complete.json').write_text(json.dumps(dict(
        plan_sha256=rebuild.sha(plan),takeover=takeover,future_seed=102,
        signatures_sha256=rebuild.sha(path),result=dict(terminal=True))))
    return path


def test_missing_trajectory_can_hide_an_ambiguous_match(signature,tmp_path,monkeypatch):
    monkeypatch.setattr(rebuild,'OUT',tmp_path)
    monkeypatch.setattr(rebuild,'reference_rows',lambda:[signature])
    monkeypatch.setattr(rebuild,'frozen',lambda:dict(jobs=[dict(takeover=n,future_seed=102) for n in (229,306)]))
    install_trace(tmp_path,monkeypatch,signature,229)
    report=rebuild.bind()
    assert len(report['bindings'][0]['matches'])==1
    assert report['mapping_complete'] is False
    install_trace(tmp_path,monkeypatch,signature,306)
    report=rebuild.bind()
    assert len(report['bindings'][0]['matches'])==2
    assert report['mapping_complete'] is False
    assert report['rescore_eligible'] is False


def test_unique_mapping_still_does_not_verify_the_original_probe_protocol(signature,tmp_path,monkeypatch):
    monkeypatch.setattr(rebuild,'OUT',tmp_path)
    monkeypatch.setattr(rebuild,'reference_rows',lambda:[signature])
    monkeypatch.setattr(rebuild,'frozen',lambda:dict(jobs=[dict(takeover=229,future_seed=102)]))
    path=install_trace(tmp_path,monkeypatch,signature,229)
    report=rebuild.bind()
    assert report['mapping_complete'] is True
    assert report['original_probe_protocol_verified'] is False
    assert report['task72b_evaluation_complete'] is False
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed identity'):rebuild.bind()


def test_legacy_replay_fails_before_opening_an_engine():
    from task78_truth import run_group
    with pytest.raises(RuntimeError,match='continuation decision is not a demo decision'):
        run_group(None)


def test_legacy_complete_marker_does_not_prove_an_original_protocol(tmp_path):
    from task78_truth import report
    path=tmp_path/'h2500.json'
    path.write_text(json.dumps(dict(complete=True,passed=True,results=[])))
    with pytest.raises(ValueError,match='protocol and v5 values must be verified'):
        report(path,dict(v5='/old',v6='/new'))
