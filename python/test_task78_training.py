"""Corpus fidelity and sampling checks; no games or GPU allocations."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import task78_train as train


def test_sampler_balances_roots_despite_unequal_candidate_counts():
    keys=[('fog','world',1,2500)]+[('fog','world',2,2500)]*9
    weights=train.balanced_weights(keys,np.arange(10))
    assert weights[0]==pytest.approx(.5)
    assert weights[1:].sum()==pytest.approx(.5)


def test_sampler_balances_groups_worlds_and_horizons_without_held_out_rows():
    keys=[('a','world1',1,2500),('a','world1',1,4000),('a','world2',1,2500),
          ('b','world3',1,2500),('held','excluded',1,2500)]
    weights=train.balanced_weights(keys,np.arange(4))
    assert weights==pytest.approx([.125,.125,.25,.5])


def test_half_precision_cache_batches_convert_only_selected_rows():
    rows=train.Rows(torch.ones(4,2,dtype=torch.float16),torch.ones(4,3),torch.zeros(4),
                    np.arange(4),np.array([False,False,True,True]))
    b,s,y=rows.batch(np.array([0]),'cpu')
    assert b.dtype==s.dtype==y.dtype==torch.float32
    assert b.shape==(1,2) and rows.boards.dtype==torch.float16


def test_partial_export_never_becomes_a_training_corpus(tmp_path):
    data=tmp_path/'data';data.mkdir()
    (tmp_path/'export.json').write_text(json.dumps(dict(complete=False)))
    with pytest.raises(ValueError,match='partial/smoke'):
        train.new_sources(data)


def test_restored_original_blob_must_match_its_digest(tmp_path):
    path=tmp_path/'original.pt';path.write_bytes(b'original')
    entry=dict(path=str(path),sha256=train.sha(path))
    assert train.restored_file(entry)==path
    path.write_bytes(b'replacement')
    with pytest.raises(ValueError,match='changed'):
        train.restored_file(entry)


def test_a_subset_of_the_v5_branch_inputs_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(train,'sha',lambda _:train.BASELINE_SHA)
    monkeypatch.setattr(torch,'load',lambda *a,**k:dict(branches=['old_a.pt','old_b.pt']))
    (tmp_path/'corpus.json').write_text(json.dumps(dict(baseline_sha256=train.BASELINE_SHA,
        branches=[dict(source_path='old_a.pt')])) )
    with pytest.raises(ValueError,match='branch coverage'):
        train.original_sources(tmp_path/'init.pt',tmp_path/'decision',tmp_path/'episodes')


def test_original_v5_loaders_keep_legacy_boards_mapped_and_convert_context(tmp_path,monkeypatch):
    from pvz_value_net import SCALARS,CONTEXT
    fit=next(s for s in range(100) if not train.held(s))
    val=next(s for s in range(100) if train.held(s))
    decision=tmp_path/'decision';decision.mkdir()
    episode=tmp_path/'episodes';episode.mkdir()
    branch_entries=[]
    for name in ('a','b'):
        path=tmp_path/f'branch_{name}.pt'
        torch.save(dict(boards=torch.zeros(2,92,6,10,dtype=torch.float16),
            scalars=torch.ones(2,SCALARS+CONTEXT),won=torch.tensor([0.,1.]),
            seed=torch.tensor([fit,val]),config=dict(task_id='day_l7_original')),path)
        branch_entries.append(dict(path=str(path),sha256=train.sha(path),source_path=f'/original/{name}.pt'))
    decisions=[]
    for seed in (fit,val):
        path=decision/f'seed_{seed}.pt'
        torch.save(dict(boards=torch.zeros(1,92,6,10,dtype=torch.float16),
            scalars_v2=torch.ones(1,SCALARS+6+CONTEXT),won=torch.ones(1),
            seed=torch.tensor([seed]),config=dict(task_id='pool_original')),path)
        decisions.append(dict(path=str(path),sha256=train.sha(path)))
    np.save(episode/'boards.npy',np.zeros((2,92,6,10),np.float16))
    np.save(episode/'scalars.npy',np.ones((2,SCALARS+6+CONTEXT),np.float32))
    np.savez(episode/'rows.npz',labels=np.array([0.,1.],np.float32),
             seeds=np.array([fit,val]),cohorts=np.array(['day','pool']))
    (episode/'episodes.json').write_text(json.dumps(dict(rows=2,excluded_audit_seeds=[999])))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps([dict(seed=999)]))
    descriptor=dict(baseline_sha256=train.BASELINE_SHA,branches=branch_entries,
        decision_shards=decisions,decision_source_path='/original/decision',
        audit_manifest=dict(source_path='/original/audit.json',path=str(audit),sha256=train.sha(audit)),
        episode_files=[dict(path=str(p),sha256=train.sha(p)) for p in episode.iterdir()])
    (tmp_path/'corpus.json').write_text(json.dumps(descriptor))
    initial=dict(branches=[e['source_path'] for e in branch_entries],decision_data='/original/decision',
        audit_manifest='/original/audit.json',decision_seeds=2,trained_on=[fit,val],
        branch_fit_rows=2,branch_validation_rows=2,decision_fit_rows=1,decision_validation_rows=1,
        episode_fit_rows=1,episode_validation_rows=1)
    init=tmp_path/'model.pt';init.write_bytes(b'checkpoint fixture')
    real_sha=train.sha;real_load=torch.load
    monkeypatch.setattr(train,'sha',lambda p:train.BASELINE_SHA if Path(p)==init else real_sha(p))
    monkeypatch.setattr(torch,'load',lambda p,*a,**k:initial if Path(p)==init else real_load(p,*a,**k))
    sources,_,_=train.original_sources(init,decision,episode)
    assert set(sources)=={'old_branch','old_decision','old_episode'}
    b,s,y=sources['old_branch'].batch(np.array([0]),'cpu')
    assert b.dtype==torch.float32 and s.shape==(1,SCALARS+6+CONTEXT)
    assert bool((s[:,SCALARS:SCALARS+6]==0).all())
    assert sources['old_episode'].boards.dtype==torch.float16
    assert sources['old_branch'].data.parts[0]['boards'].dtype==torch.float16
