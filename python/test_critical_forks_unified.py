"""Shared fog/roof capabilities and production changes retained by task60."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as f
import critical_forks_data as data


def test_temporal_selection_respects_each_window_without_reading_outcomes():
    points=[dict(decision=i,tick=t,won=i%2) for i,t in enumerate((0,3900,4900,7900,9900))]
    assert [p['tick'] for p in f.pick_roots(points,3,8000)]==[0,3900,7900]
    assert [p['tick'] for p in f.pick_roots(points,3,10000)]==[0,4900,9900]
    assert f.pick_roots([],3)==[]
    assert len(f.pick_roots(points[:1],3))==1


def test_legacy_seeds_keep_namespaces_decision_origin_and_fresh_futures():
    job=dict(id='new',decision=197,legacy=dict(namespace='task46-B',cell='roof'))
    expected=lambda stage:int.from_bytes(hashlib.sha256(f'task46-B:roof:196:{stage}:1'.encode()).digest()[:4],'little')
    assert f.job_future(job,'screen',1)==expected('screen')
    assert f.job_future(job,'screen',1,True)==expected('screen-teacher-search')
    assert f.job_future(job,'confirm',1)!=f.job_future(job,'screen',1)
    assert f.job_future(dict(id='root'),'screen',1)==f.future_seed('root','screen',1)


def fixture():
    wait=dict(type='wait',ticks=60,until='timeout')
    plant=dict(type='plant',packet=1,row=2,col=3)
    group=dict(student_index=0,candidates=[
        dict(index=0,action=wait,means=dict(won=0,wave=8,tick=18000,mowers_used=1)),
        dict(index=1,action=plant,means=dict(won=0,wave=10,tick=22000,mowers_used=0))])
    trace=dict(searched=True,static_selected=[plant],candidates=[wait,plant],best_index=1,student_index=0,
        pre_verification_values=[.5,.515],final_values=[.5,.515],halving_counts=[3,3],
        required_halving_samples=3,chosen=wait,margin=.02)
    return group,trace


@pytest.mark.parametrize('cause', ['initial_margin_blocked','halving_pruned','leaf_ranking_not_top',
    'verification_rejected','candidate_generation_omission','outside_production_action_space',
    'executed','search_not_scheduled'])
def test_attribution_separates_each_production_stage(cause):
    group,trace=fixture()
    if cause=='halving_pruned':trace['halving_counts'][1]=1
    if cause=='leaf_ranking_not_top':trace['best_index']=0
    if cause=='verification_rejected':trace['pre_verification_values'][1]=.6
    if cause in ('candidate_generation_omission','outside_production_action_space'):
        trace['candidates']=trace['candidates'][:1]
    if cause=='outside_production_action_space':group['candidates'][1]['action']=dict(type='shovel',row=2,col=3)
    if cause=='executed':
        trace['pre_verification_values'][1]=trace['final_values'][1]=.6
        trace['chosen']=trace['candidates'][1]
    if cause=='search_not_scheduled':trace['searched']=False
    got=data.attribution(group,trace,dict(wave=1))['stages']['wave']
    assert got['observed_improvers']==got[cause]==1
    if cause!='outside_production_action_space':assert got['static_retained']==1


def test_paired_observation_bootstrap_requires_two_informative_reset_seeds():
    def group(seed,mode,value):
        g,_=fixture();g['job']=dict(reset=dict(level=31,seed=seed),archive=mode)
        g['candidates'][1].update(v4=value,short_all_nonterminal=True)
        g['candidates'][0].update(v4=.5,short_all_nonterminal=True)
        return g
    a,b=group(1,'human',.6),group(1,'xray',.4)
    assert data.agreement([a,b],'wave',1)['game_cluster_bootstrap_95'] is None
    assert data.agreement([a,group(2,'human',.4)],'wave',1)['game_cluster_bootstrap_95']==[0.,1.]


def test_legacy_repro_requires_full_physics_and_action_chain(tmp_path):
    from task53_repro import compare_legacy
    outcome=dict(won=False,terminal=True,result=2,wave=8,tick=100,mowers_used=0,decisions=1)
    original=dict(outcome=outcome,short_leaf={},teacher_leaf=dict(frame={},value=.4),
        v4_short_value=.4,terminal_state_sha256='hash',identity=dict(action={},future=1,search_seed_base=2,k=0))
    repeated=dict(outcome=outcome,short_leaf={},teacher_leaf=dict(frame={},value=.40000001),
        v4_short=.40000001,terminal_engine_sha256='hash',identity=dict(action={},future=1,search_future=2,k=0))
    a,b=tmp_path/'a.gz',tmp_path/'b.gz'
    for path in (a,b):
        with gzip.open(path,'wt') as stream:stream.write(json.dumps(dict(decision=0,tick=1,student={},chosen={})))
    assert compare_legacy(original,repeated,a,b)['full_teacher_trace_equal']
    changed=deepcopy(repeated);changed['outcome']['tick']=101
    with pytest.raises(ValueError,match='tick'):compare_legacy(original,changed,a,b)
    changed=deepcopy(repeated);changed['v4_short']=.41
    with pytest.raises(ValueError,match='v4_value'):compare_legacy(original,changed,a,b)
    with gzip.open(b,'wt') as stream:stream.write(json.dumps(dict(decision=0,tick=1,student={},chosen={'type':'plant'})))
    with pytest.raises(ValueError,match='full_teacher_trace'):compare_legacy(original,repeated,a,b)


@pytest.mark.parametrize('observed',[False,True])
def test_fast_play_observer_gets_each_decision_and_default_keeps_native_path(tmp_path,monkeypatch,observed):
    import search_play as teacher
    import render_episode
    import pvz_value_net
    import pvz_episode_stats
    class Env:
        _last_response=dict(decisions=1,mowers_left=0)
        native=0
        def reset(self,**kw):return dict(terminal=False,tick=0,defenses=[]),{}
        def rollout(self,*a,**kw):self.native+=1;return self.final()
        def final(self):return dict(terminal=True,result=2,wave=1,tick=60,wave_count=1,defenses=[])
        def step(self,action):return self.final(),0,True,False,dict(ok=True)
    env=Env();calls=[]
    observer=SimpleNamespace(before=lambda *a:calls.append('before'),after=lambda *a:calls.append('after')) if observed else None
    manifest=tmp_path/'tasks.json';manifest.write_text(json.dumps(dict(tasks=[dict(task_id='test',level=1,deck=[0,1,2,3,4,5])])) )
    args=teacher.parser().parse_args(['--rollout-policy','/model','--value','/value','--manifest',str(manifest),
        '--task-id','test','--out',str(tmp_path)])
    monkeypatch.setattr(teacher,'_W',dict(env=env))
    monkeypatch.setattr(teacher,'_fast_action',lambda *a:dict(type='wait',ticks=60))
    monkeypatch.setattr(teacher,'_final_zombies',lambda obs:[])
    monkeypatch.setattr(pvz_value_net,'observation_context',lambda obs:None)
    monkeypatch.setattr(pvz_episode_stats,'planted_slots',lambda details:{})
    monkeypatch.setattr(pvz_episode_stats,'planted_by_type',lambda *a:{})
    result=teacher._play(dict(args=args,seed=1,arm='fast'),observer)
    assert result['terminal'] and result['decisions']==1
    assert calls==(['before','after'] if observed else [])
    assert env.native==(0 if observed else 1)


@pytest.mark.parametrize('cached',[False,True])
def test_explicit_rejected_training_row_is_never_silently_dropped(tmp_path,cached):
    from test_compact_loader import game_frames
    from pvz_frame_store import write_frames
    from pack_training_rows import build_from_archives
    archive=tmp_path/'data/seed_1/correction';archive.mkdir(parents=True)
    frames=game_frames(8);frames[4]['action_rejected']=True
    write_frames(archive/'frames.npz',frames)
    f.save(archive/'meta.json',dict(training_rows=[4]))
    with pytest.raises(ValueError,match='lack valid labels'):
        build_from_archives(tmp_path/'data',{'correction'},obs='v2',
                            cache_dir=tmp_path/'cache' if cached else None)


def test_diagnostic_confirmation_retains_each_metric_challenger(tmp_path,monkeypatch):
    group,_=fixture();group['job']=dict(id='root');group['student_index']=0
    group['candidates'][0]['v4']=.5;group['candidates'][1]['v4']=.4
    group['candidates'].append(dict(index=2,action=dict(type='shovel'),v4=.6,
        means=dict(won=1,wave=5,tick=100,mowers_used=1)))
    monkeypatch.setattr(data,'screen_summary',lambda *args:([group],{}))
    selected=data.confirmation_plan(dict(out=str(tmp_path),confirm_futures=16,
        protocol=dict(confirmation=dict(selection='metric_bests'))),[])
    assert selected['jobs'][0]['indices']==[0,1,2]
    assert selected['jobs'][0]['stage']=='confirm'
    assert selected['labels']==48


def test_sampling_fresh_budget_gate_never_substitutes_partial_panel(tmp_path,monkeypatch):
    import critical_forks_sampling as sampling
    monkeypatch.setattr(sampling,'OUT',tmp_path)
    monkeypatch.setattr(sampling,'check_source_plan',lambda:dict(cells=[],spec={}))
    monkeypatch.setattr(sampling,'timing_samples',lambda plan:{})
    monkeypatch.setattr(sampling,'selections',lambda *args:[dict(point=dict(candidate_count=10))])
    monkeypatch.setattr(sampling,'timing_forecast',lambda *args:100_000)
    with pytest.raises(RuntimeError,match='no partial panel'):
        sampling.select()
    assert not (tmp_path/'selection-active.json').exists()
    assert len(f.load(tmp_path/'sampling-budget-failed.json')['options'])==4


def test_old_and_new_cpu_receipts_count_reaped_children_once(tmp_path):
    f.save(tmp_path/'runs/old/usage-worker.json',dict(cpu_seconds=10))
    f.save(tmp_path/'runs/old/usage-controller.json',dict(cpu_seconds=2))
    f.save(tmp_path/'runs/old/complete.json',dict(controller_children_cpu_seconds=15))
    f.save(tmp_path/'runs/new/usage-worker.json',dict(cpu_seconds=4))
    f.save(tmp_path/'runs/new/complete.json',dict(cpu_seconds=5))
    assert f.spent_cpu(tmp_path)==20


def test_diagnostic_report_retains_observation_arms_legal_coverage_and_attribution(tmp_path,monkeypatch):
    group,trace=fixture();trace['audit_schema']=2
    group.update(labels=2,job=dict(id='human',terrain='fog',observation='human',reset=dict(level=31,seed=1)))
    for i,c in enumerate(group['candidates']):
        c.update(plant_type=None if i==0 else 4,v4=.5,teacher_path_v4=.5,
                 short_all_nonterminal=True,outcomes=[dict(won=False,wave=c['means']['wave'],tick=c['means']['tick'],mowers_used=0)])
    second=deepcopy(group);second['job'].update(id='xray',observation='xray')
    monkeypatch.setattr(data,'screen_summary',lambda *args:([group,second],{}))
    monkeypatch.setattr(f,'load',lambda path:trace)
    result=data.diagnostic_report(dict(out=str(tmp_path),protocol=dict(confirmation=dict(mean_gap=dict(won=1/3,wave=1,tick=600,mowers_used=.5)))),[])
    assert set(result['by_level_observation'])=={'l31-human','l31-xray'}
    assert result['overall']['base_seed_units']==1 and result['overall']['labels']==4
    assert result['overall']['terminal_wins']==0
    assert result['overall']['available_root_types']=={'4':2}
    assert result['attribution_totals']['wave']['initial_margin_blocked']==2


def test_frozen_selection_resume_ignores_later_cpu_usage(tmp_path,monkeypatch):
    import critical_forks_sampling as pipeline
    monkeypatch.setattr(pipeline,'OUT',tmp_path/'artifacts')
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'check_source_plan',lambda:{'inputs':'unchanged'})
    trajectory=pipeline.OUT/'episodes'/'cell'/'trajectory.json'
    f.save(trajectory,dict(prefix=[1,2,3]))
    selection=pipeline.OUT/'selection-active.json'
    f.save(selection,dict(jobs=[],root_trajectory_sha256={'cell':f.sha(trajectory)}))
    summary=dict(root_selection_sha256=f.sha(selection),cpu_seconds_before_screen=100)
    f.save(pipeline.OUT/'sampling.json',summary)
    monkeypatch.setattr(f,'spent_cpu',lambda out:9999)
    assert pipeline.select()==summary
    f.save(trajectory,dict(prefix=[1,2,4]))
    with pytest.raises(ValueError,match='trajectory changed'):pipeline.select()


def test_timing_amendment_keeps_both_stages_and_requires_completion(tmp_path,monkeypatch):
    import critical_forks_sampling as pipeline
    monkeypatch.setattr(pipeline,'OUT',tmp_path)
    manifests=[]
    for stage,tick,cpu in [('pilot',5000,50),('pilot-end',10000,5)]:
        job=dict(cell=dict(id='cell',level=42),point=dict(decision=tick//60,tick=tick),stage=stage)
        name=stage+'-jobs.json';manifests.append(name);f.save(tmp_path/name,dict(jobs=[job]))
        folder=tmp_path/'episodes'/'cell'/f'root_{tick//60}'
        f.save(folder/'root.json',dict(root_frame=dict(packets=[[0,33]])))
        f.save(folder/f'{stage}_0_0.json',dict(identity=dict(action=dict(type='plant',packet=0)),cpu_seconds=cpu))
        f.save(folder/f'{stage}-complete.json',dict(labels=1))
    plan={'spec':{'timing_job_manifests':manifests}}
    times=pipeline.timing_samples(plan)
    assert times[(42,5000,'plant:33')]==[50]
    assert times[(42,10000,'plant:33')]==[5]
    (folder/'pilot-end-complete.json').unlink()
    with pytest.raises(ValueError,match='incomplete'):pipeline.timing_samples(plan)


def test_sparse_cpu_selection_preserves_named_and_reset_seed_coverage(tmp_path,monkeypatch):
    import critical_forks_sampling as pipeline
    monkeypatch.setattr(pipeline,'OUT',tmp_path)
    a=dict(id='a',level=42,seed=1);b=dict(id='b',level=42,seed=2)
    archive=dict(id='archive',level=42,seed=1,archive=True)
    named=dict(cell=archive,point=dict(decision=99),samples=2,stage='screen')
    monkeypatch.setattr(pipeline,'selections',lambda *args:[named])
    # Cheap root wins despite opposite, irrelevant outcome metadata; outside-window root excludef.
    f.save(tmp_path/'episodes/b/trajectory.json',dict(eligible=[
        dict(decision=1,tick=60,candidate_count=30,won=True),
        dict(decision=2,tick=120,candidate_count=20,won=False),
        dict(decision=3,tick=11000,candidate_count=1,won=True)]))
    plan=dict(cells=[a,b,archive],spec=dict(window_end_tick=10000))
    selected=pipeline.budget_selections(plan,{(42,0,'wait'):[10]})
    assert selected[0]==named
    assert [(j['cell']['seed'],j['point']['decision']) for j in selected]==[(1,99),(2,2)]
