"""Archive forks retain full action coverage, history and precise supervision identity."""
import hashlib
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as forks
from pvz_frame_store import write_frames
from pvz_training_pack import fingerprint_paths, training_rows
from pack_training_rows import build_from_archives
from test_compact_loader import game_frames


def test_full_legal_coverage_and_future_pairing():
    obs={'legal_actions':{'plants':[{'packet':2,'row':1,'col':3},{'packet':0,'row':0,'col':2}],
                          'shovels':[{'row':2,'col':3},[1,0]]}}
    actions=forks.all_candidates(obs)
    assert len(actions)==5 and [a['type'] for a in actions]==['wait','plant','plant','shovel','shovel']
    seeds=[forks.future_seed('root',stage,k,search) for stage in ('screen','confirm')
           for k in range(16) for search in (False,True)]
    assert len(set(seeds))==64
    assert forks.future_seed('root','screen',0)==forks.future_seed('root','screen',0)
    with pytest.raises(ValueError,match='duplicate'):
        forks.all_candidates({'legal_actions':{'plants':[obs['legal_actions']['plants'][0]]*2}})


def test_selection_keeps_breach_and_flat_without_reading_terminal_candidate_labels():
    frames=[]
    for i in range(12):
        frames.append(dict(tick=i*60,mowers=[[0,1 if i<8 else 2]],terminal=i==11,
                           result=2 if i==11 else 0,targets=[]))
    frames[10]['targets']=[[-1,-1,-1,.01],[0,0,0,.015]]
    params=dict(pre_breach_decisions=1,flat_points_per_game=1,max_points_per_game=2,
                flat_spread_threshold=.02,flat_window_before_loss_ticks=6000)
    points=forks.select_points(frames,params)
    assert [p['decision'] for p in points]==[10,11]
    assert points[0]['reasons'][0]['kind']=='flat_archived_leaf'
    assert points[1]['tick']==600 and points[1]['reasons'][0]['breach']==11


def test_terminal_win_preserves_last_nonterminal_mowers(tmp_path,monkeypatch):
    import render_episode
    monkeypatch.setattr(render_episode,'_frame',lambda obs:dict(obs))
    monkeypatch.setattr(forks,'TEACHER',SimpleNamespace(MAX_ACTIONS=3))
    monkeypatch.setattr(forks,'usage',lambda *a,**k:None)
    obs=dict(terminal=True,result=1,tick=9000,wave=20,defenses=[])
    outcome,leaf=forks.finish_teacher(None,obs,{},SimpleNamespace(horizon=2500),1,
                                    tmp_path/'trace.gz',5,5,10000)
    assert outcome['won'] and outcome['mowers_used']==0 and leaf['value']==1


def test_exact_training_rows_preserve_history_and_wait_confidence(tmp_path):
    data=tmp_path/'data';archive=data/'seed_100'/'correction';archive.mkdir(parents=True)
    path=archive/'frames.npz';frames=game_frames(12);write_frames(path,frames)
    meta=archive/'meta.json';meta.write_text('{}')
    old=hashlib.sha1(f'{path.relative_to(data)}:{path.stat().st_size}'.encode()).hexdigest()[:16]
    assert fingerprint_paths(data,[path])==old
    full,*_=build_from_archives(data,{'correction'},wait_keep=1.,obs='v2',packet_head='identity')
    meta.write_text(json.dumps({'training_rows':[4,8]}))
    masked=fingerprint_paths(data,[path]);assert masked!=old
    for cache in (None,tmp_path/'cache'):
        selected,_,_,rows=build_from_archives(data,{'correction'},wait_keep=.01,
                     obs='v2',packet_head='identity',cache_dir=cache,label_weight={'correction':.875})
        assert rows.tolist()==[4,8]
        for column in (0,1):
            a=selected[column].batch(np.arange(2),'cpu')
            b=full[column].batch(np.array([3,7]),'cpu')
            assert torch.equal(a,b)
        assert selected[2].tolist()==[0.,0.]  # Explicit wait corrections are always retained.
        assert selected[5].tolist()==[.875,.875]
    meta.write_text(json.dumps({'training_rows':[4,7]}))
    assert fingerprint_paths(data,[path])!=masked
    meta.write_text(json.dumps({'training_rows':[4,99]}))
    with pytest.raises(ValueError,match='outside archive'):
        build_from_archives(data,{'correction'},obs='v2')


@pytest.mark.parametrize('rows',[[0],[2,2],[3,2],[True]])
def test_invalid_training_row_identity_rejected(tmp_path,rows):
    (tmp_path/'meta.json').write_text(json.dumps({'training_rows':rows}))
    with pytest.raises(ValueError,match='sorted unique positive'):
        training_rows(tmp_path/'frames.npz')


def test_confirmation_statistics_require_fresh_directional_evidence_and_use_magnitudes():
    from critical_forks_data import sign_p,bootstrap_difference
    assert sign_p([0]*16)==1.
    assert sign_p([1]*16)==1/65536
    assert sign_p([-1]*16)==1.
    balanced=bootstrap_difference([1]*8+[-1]*8)
    stronger=bootstrap_difference([10]*8+[-1]*8)
    assert balanced['mean_gain']==0 and .3<balanced['confidence']<.5
    assert stronger['mean_gain']==4.5 and stronger['confidence']>.95


def test_panel_rejects_missing_future_or_root_checks(tmp_path):
    from critical_forks_data import panel
    job=dict(id='root',decision=1,frames_sha256='archive',archive='/original',terrain='day')
    plan=dict(out=str(tmp_path),inputs={},screen_futures=2);forks.PLAN=plan
    folder=tmp_path/'roots/root';folder.mkdir(parents=True)
    actions=[dict(type='wait',ticks=60,until='timeout')]
    forks.save(folder/'root.json',dict(candidates=actions,student_action=actions[0],frame={'packets':[],'tick':0}))
    forks.save(folder/'screen-complete.json',dict(indices=[0],samples=2,labels=2))
    for k in range(2):
        path=folder/f'screen_0_{k}.json';trace=path.with_suffix('.jsonl.gz')
        with gzip.open(trace,'wt') as out:out.write('')
        row=dict(identity=forks.identity(job,actions[0],0,'screen',k),
                 outcome=dict(terminal=True,result=2,won=False,mowers_used=1,tick=100,wave=5,decisions=0),
                 checks={n:True for n in ('root','income','history','legal','terminal','strict_actions')},
                 trace_sha256=forks.sha(trace),v4_short=0.,short_leaf={'terminal':True},cpu_seconds=.1)
        forks.save(path,row)
    assert panel(plan,[job])[0]['labels']==2
    path=folder/'screen_0_1.json';row=forks.load(path);row['checks']={};forks.save(path,row)
    with pytest.raises(ValueError,match='missing or failed'):
        panel(plan,[job])
    path.unlink()
    with pytest.raises(ValueError,match='incomplete/unexpected'):
        panel(plan,[job])
