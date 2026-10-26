"""Formal evaluation must not borrow the diagnostic model or its evaluation IDs."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import task78_formal_ab as formal


@pytest.fixture
def seed_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(formal,'OUT',tmp_path)
    directory=tmp_path/'registration_seed_audit';directory.mkdir()
    evidence=directory/'seed_audit.json';evidence.write_text('{"collision_count":0}')
    summary=directory/'seed_audit_summary.json'
    summary.write_text(json.dumps(dict(passed=True,evidence=str(evidence),
                                       evidence_sha256=formal.sha(evidence))))
    return summary,evidence,dict(broad_seed_audit_summary_sha256=formal.sha(summary))


def test_formal_runtime_reuses_registration_evidence_without_global_rescan(seed_receipt,monkeypatch):
    import task78_seed_audit
    summary,_,registration=seed_receipt
    monkeypatch.setattr(task78_seed_audit,'main',
                        lambda *a,**kw:pytest.fail('duplicate global seed scan'))
    assert formal.registered_seed_audit(registration)==formal.sha(summary)


@pytest.mark.parametrize('fault',['changed_summary','changed_evidence','failed_registration'])
def test_registration_receipt_reuse_still_rejects_changed_or_failed_evidence(seed_receipt,fault):
    summary,evidence,registration=seed_receipt
    if fault=='changed_summary':summary.write_text('{}')
    elif fault=='changed_evidence':evidence.write_text('{"collision_count":1}')
    else:
        audit=json.loads(summary.read_text());audit['passed']=False
        summary.write_text(json.dumps(audit))
        registration['broad_seed_audit_summary_sha256']=formal.sha(summary)
    with pytest.raises(ValueError,match='registration seed audit'):
        formal.registered_seed_audit(registration)


def test_absent_student_seed_list_is_unknown_instead_of_a_verified_empty_set():
    assert formal.declared_overlap(dict(labels='source_a,source_b'),{78100512},'student') is None
    assert formal.declared_overlap(dict(trained_on=[78100512]),{78100512},'v6')==[78100512]
    with pytest.raises(ValueError,match='metadata is missing'):
        formal.declared_overlap(dict(labels='source_a'),{78100512},'v6')


@pytest.fixture
def registration():
    return dict(mix=dict(old_episode=64,old_branch=192,old_decision=128,fog=64,roof=64),
        hyper=dict(formal.HYPER),new_export_sha256='corrected-192-task-export',
        source_splits={name:dict(fit=10,val=2) for name in
                       ('old_episode','old_branch','old_decision','fog','roof')})


@pytest.fixture
def receipt(registration):
    cp=dict(preregistered_corpus=True,mix=deepcopy(registration['mix']),dropped_sources=[],
        input_provenance=dict(original=dict(descriptor='/original/corpus.json',sha256='original'),
                              new=dict(sha256=registration['new_export_sha256'])),
        hyper=registration['hyper']|dict(device='cuda'),sources=deepcopy(registration['source_splits']),
        mem_available_gib=10.5)
    log=dict(complete=True,preregistered_corpus=True,mix=deepcopy(cp['mix']),
             sources=deepcopy(cp['sources']),wall_seconds=300)
    return cp,log


def test_original_loader_descriptor_receipt_needs_no_diagnostic_only_flag(registration,receipt):
    cp,log=receipt
    # original_sources returns descriptor/sha256; only the fallback adds its flag.
    assert 'preregistered_corpus' not in cp['input_provenance']['original']
    formal.validate_training(cp,log,registration)


def test_gpu_hour_limit_does_not_invent_a_cpu_training_limit(registration,receipt):
    cp,log=receipt
    cp['hyper']['device']='cpu';log['wall_seconds']=4000
    formal.validate_training(cp,log,registration)


@pytest.mark.parametrize('fault',[
    'diagnostic','wrong_mix','reconstructed_descriptor','missing_descriptor','old_export',
    'changed_hyper','missing_source','changed_split','low_memory','nan_memory','nan_wall',
    'unfinished','timeout','wrong_log_mix'])
def test_invalid_training_cannot_become_formal_by_changing_output_directory(registration,receipt,fault):
    cp,log=receipt
    if fault=='diagnostic':cp['preregistered_corpus']=False
    elif fault=='wrong_mix':cp['mix']={'old_episode':384,'fog':64,'roof':64}
    elif fault=='reconstructed_descriptor':cp['input_provenance']['original']['preregistered_corpus']=False
    elif fault=='missing_descriptor':cp['input_provenance']['original'].pop('descriptor')
    elif fault=='old_export':cp['input_provenance']['new']['sha256']='144-task-snapshot'
    elif fault=='changed_hyper':cp['hyper']['lr']=1e-5
    elif fault=='missing_source':cp['sources'].pop('old_branch')
    elif fault=='changed_split':cp['sources']['roof']['fit']=9
    elif fault=='low_memory':cp['mem_available_gib']=9.99
    elif fault=='nan_memory':cp['mem_available_gib']=float('nan')
    elif fault=='nan_wall':log['wall_seconds']=float('nan')
    elif fault=='unfinished':log['complete']=False
    elif fault=='timeout':log['wall_seconds']=3601
    elif fault=='wrong_log_mix':log['mix']={'old_episode':384,'fog':64,'roof':64}
    with pytest.raises(ValueError):formal.validate_training(cp,log,registration)


def test_shift_preserves_demo_base_seed_deck_takeover_and_other_semantics():
    original=[dict(cell_id='day',seeds=[78000000,78000001],deck=[0,1,2,3,4,5]),
              dict(cell_id='demo',seeds=[20261033],future_seeds=[78100000,78100001],
                   deck=[9,8,10,16,23,17,19,2,26,5],demo_decision=839)]
    before=deepcopy(original);fresh=formal.shifted_panel(original)
    assert original==before
    assert fresh[0]['seeds']==[78000512,78000513]
    assert fresh[1]['future_seeds']==[78100512,78100513]
    assert fresh[1]['seeds']==original[1]['seeds']
    for old,new in zip(original,fresh):
        key='future_seeds' if 'future_seeds' in old else 'seeds'
        assert {k:v for k,v in old.items() if k!=key}=={k:v for k,v in new.items() if k!=key}


def test_shift_rejects_original_block_exhaustion_and_cross_panel_collision():
    with pytest.raises(ValueError,match='reserved blocks'):
        formal.shifted_panel([dict(seeds=[78001999])])
    with pytest.raises(ValueError,match='overlap'):
        formal.shifted_panel([dict(seeds=[78000000,78000512])])


def test_missing_formal_candidate_stops_before_any_game(tmp_path,monkeypatch):
    root=tmp_path/'work';(root/'artifacts/task78').mkdir(parents=True)
    registration=tmp_path/'plan.json';registration.write_text('{}')
    monkeypatch.setattr(formal,'ROOT',root)
    monkeypatch.setattr(formal,'REGISTRATION',registration)
    monkeypatch.setattr(formal,'CANDIDATE',tmp_path/'missing/model.pt')
    monkeypatch.setattr(formal,'frozen_panel',lambda:[])
    monkeypatch.setattr(formal.subprocess,'run',lambda *a,**kw:pytest.fail('game started without formal candidate'))
    with pytest.raises(FileNotFoundError,match='restore original v5 corpus'):
        formal.run_arm('v6')


def test_empty_formal_panel_report_cannot_inherit_diagnostic_passes(tmp_path,monkeypatch):
    panel=[dict(cell_id=f'{terrain}_cell',task_id=f'{terrain}_task',terrain=terrain,
                pairs=32,seeds=[78000512+i*32+j for j in range(32)])
           for i,terrain in enumerate(('day','night','pool','fog','roof'))]
    monkeypatch.setattr(formal,'OUT',tmp_path)
    monkeypatch.setattr(formal,'frozen_panel',lambda:panel)
    monkeypatch.setattr(formal,'runtime_plan',lambda:pytest.fail('missing rows must not borrow a candidate'))
    result=formal.report()
    assert result['complete'] is False and result['all_game_gates_pass'] is False
    assert result['formal_candidate_verified'] is False
    assert result['preregistered_recipe_met'] is None
    assert all(g['status']=='incomplete' and g['passed'] is None for g in result['gates'].values())
