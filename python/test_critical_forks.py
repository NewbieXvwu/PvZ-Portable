"""Archive forks retain full action coverage, history and precise supervision identity."""
import hashlib
import gzip
import json
import os
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


def test_paired_evaluation_refuses_partial_and_keeps_unanimous_uncertainty():
    from task53_evaluate import summary
    evaluation=[dict(id=t,terrain=t,reset={'seed':i}) for i,t in enumerate(('day','night','pool'))]
    rows=[dict(id=j['id'],arm=arm,won=True) for j in evaluation for arm in ('incumbent','candidate')]
    result=summary(rows,evaluation)
    assert result['overall']['delta_rate']==0
    assert result['overall']['ci95'][0]<0<result['overall']['ci95'][1]
    with pytest.raises(ValueError,match='incomplete or duplicated'):
        summary(rows[:-1],evaluation)
    with pytest.raises(ValueError,match='incomplete or duplicated'):
        summary(rows+[rows[0]],evaluation)


def test_pipeline_wait_checks_process_identity_without_restarting():
    from task53_pipeline import alive
    pid=os.getpid();start=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    assert alive(pid,start)
    assert not alive(pid,'wrong-start-time')
    assert not alive(-1,start)


def test_confirmation_jobs_cannot_write_into_screen_namespace(tmp_path,monkeypatch):
    import critical_forks_data as data
    candidates=[dict(index=i,action={'type':'plant' if i else 'wait'},v4=.1,
                     means={m:float(i) for m in data.METRICS}) for i in (0,1)]
    group=dict(job={'id':'root'},candidates=candidates,student_index=0)
    monkeypatch.setattr(data,'screen_summary',lambda *args:([group],{}))
    selection=data.confirmation_plan(dict(out=str(tmp_path),confirm_futures=16,
                  protocol={'confirmation':{'mean_gap':{m:.1 for m in data.METRICS}}}),[])
    assert selection['jobs'][0]['stage']=='confirm'
    assert selection['jobs'][0]['indices']==[0,1] and selection['labels']==32
    candidates.append(dict(index=2,action={'type':'shovel'},v4=.1,
                           means={m:2. for m in data.METRICS}))
    selection=data.confirmation_plan(dict(out=str(tmp_path),confirm_futures=16,
                  protocol={'confirmation':{'mean_gap':{m:.1 for m in data.METRICS}}}),[])
    assert selection['jobs'][0]['indices']==[0,1]
    assert selection['unrepresented_global_bests'][0]['candidate']==2


def test_value_pairs_keep_metrics_futures_and_leaf_boundaries_separate():
    from critical_forks_data import terminal_pairs
    outcomes=[dict(won=True,wave=5,tick=2000,mowers_used=1),
              dict(won=False,wave=6,tick=1000,mowers_used=2),
              dict(won=False,wave=3,tick=500,mowers_used=1),
              dict(won=False,wave=5,tick=1000,mowers_used=1)]
    pairs=list(terminal_pairs(outcomes,[0,1,1,2],[0,0,1,0],[1000,1000,1000,1001],
                             dict(won=.125,wave=.5,tick=600,mowers_used=.5)))
    assert len(pairs)==3
    assert {p['metric']:p['winner_row'] for p in pairs}==dict(won=0,wave=1,mowers_used=0)
    assert all({p['winner_row'],p['loser_row']}=={0,1} for p in pairs)
    losses=[outcomes[1],dict(outcomes[2],tick=0)]
    pairs=list(terminal_pairs(losses,[0,1],[0,0],[1000,1000],dict(won=.125,wave=.5,tick=600,mowers_used=.5)))
    assert next(p for p in pairs if p['metric']=='tick')['winner_row']==0


def test_repro_checks_semantics_instead_of_gzip_timestamps(tmp_path):
    from task53_repro import semantic_compare
    fields=('identity','outcome','v4_short','short_leaf','post_action_frame','teacher_leaf',
            'terminal_engine_sha256','checks')
    a,b=tmp_path/'a.gz',tmp_path/'b.gz'
    for path,mtime in ((a,1),(b,2)):
        with gzip.GzipFile(filename=str(path),mode='wb',mtime=mtime) as stream:stream.write(b'{"decision":0}\n')
    original={k:1 for k in fields};repeated=dict(original)
    assert a.read_bytes()!=b.read_bytes()
    assert semantic_compare(original,repeated,a,b)['full_teacher_trace_equal']
    repeated['outcome']=0
    with pytest.raises(ValueError,match='outcome'):semantic_compare(original,repeated,a,b)
    repeated['outcome']=1
    with gzip.open(b,'wb') as stream:stream.write(b'{"decision":1}\n')
    with pytest.raises(ValueError,match='teacher_trace'):semantic_compare(original,repeated,a,b)


def test_round_budget_selects_whole_stratified_roots_and_rejects_empty_success():
    from task53_experiment import budget_roots
    jobs=[dict(id=str(i),terrain=t,reasons=[{'kind':kind}]) for i,(t,kind) in enumerate(
          [('day','before_loss_breach'),('night','before_loss_breach'),('pool','flat_archived_leaf'),('day','flat_archived_leaf')])]
    prices={'0':3.,'1':4.,'2':5.,'3':20.}
    selected,spent=budget_roots(jobs,prices,12.,3,round_index=1)
    assert selected and spent<=12. and len(selected)<=3
    assert all(j in jobs and 'indices' not in j and 'samples' not in j for j in selected)
    assert len({j['id'] for j in selected})==len(selected)
    with pytest.raises(ValueError,match='no complete'):
        budget_roots(jobs,prices,2.,3)


def test_longer_winning_completion_is_not_a_survival_correction(tmp_path,monkeypatch):
    import critical_forks_data as data
    job={'id':'root'}
    forks.save(tmp_path/'confirmation.json',{'jobs':[job]})
    plan=dict(out=str(tmp_path),protocol={'confirmation':{'alpha':.05,
              'mean_gap':dict(won=.125,wave=.5,tick=600,mowers_used=.5)}})
    candidates=[dict(index=i,action={'type':'wait'},plant_type=None,
                    outcomes=[dict(won=True,wave=20,tick=1000+2000*i,mowers_used=0) for _ in range(16)])
                for i in (0,1)]
    group=dict(job=job,student_index=0,candidates=candidates,labels=32)
    monkeypatch.setattr(data,'panel',lambda *args:[group])
    assert data.confirm_summary(plan)['corrections']==[]
    for candidate in candidates:
        for outcome in candidate['outcomes']:outcome['won']=False
    assert len(data.confirm_summary(plan)['corrections'])==1
