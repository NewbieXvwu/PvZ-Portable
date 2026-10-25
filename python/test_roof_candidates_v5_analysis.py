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
    monkeypatch.setattr(audit.base, 'owned_game_processes', lambda: [])
    with pytest.raises(ValueError, match='one or two'):
        audit.base.require_game_capacity(3)
    audit.base.require_game_capacity(2)


def test_existing_task76_games_count_toward_its_two_slots(monkeypatch):
    import pytest
    monkeypatch.setattr(audit.base, 'owned_game_processes', lambda: [dict(pid=123)])
    with pytest.raises(RuntimeError, match='task76 slots occupied'):
        audit.base.require_game_capacity(2)
    audit.base.require_game_capacity(1)


def test_other_tasks_do_not_consume_task76_game_slots(monkeypatch):
    processes=[Path('/proc')/str(i) for i in (11,22,33)]
    executables={11:'/qc/pvz-portable',22:'/other/pvz-portable',33:'/another/pvz-portable-release'}
    monkeypatch.setattr(Path,'iterdir',lambda path:iter(processes))
    monkeypatch.setattr(audit.base.os,'readlink',lambda path:executables[int(path.parent.name)])
    assert audit.base.owned_game_processes()==[]
    audit.base.require_game_capacity(2)
    executables[11]=str(audit.base.OUT/'frozen/pvz-portable')
    assert [g['pid'] for g in audit.base.owned_game_processes()]==[11]


def test_holm_controls_the_whole_required_level_family():
    assert audit.holm_adjust({'a':.01,'b':.02,'c':.5})=={'a':.03,'b':.04,'c':.5}
    assert audit.holm_adjust({'a':.7,'b':.8})=={'a':1.,'b':1.}


def test_shared_reset_seeds_are_not_counted_as_independent_levels():
    # Eight gains sharing one reset seed yield one sign-flip unit, not eight.
    same=audit.seed_cluster_sign_test({1:8})
    separate=audit.seed_cluster_sign_test({i:1 for i in range(8)})
    assert same['exact_cluster_sign_flip_p']==1
    assert separate['exact_cluster_sign_flip_p']==2/256
    assert audit.seed_cluster_sign_test({1:0,2:0})['exact_cluster_sign_flip_p']==1


def audited_rows(arm='all_plants'):
    r=row([A,WAIT],[A,WAIT,B],[(A,.1),(WAIT,.05),(B,.3)],B)
    r.update(searched=True,frame={'tick':0},next_tick=60,decision_seed=7,
             rejected=False,terminal=True,policy=A)
    for entry in r['rounds'][0]:entry['future']=7000
    receipt=dict(recorded_decisions=1,recorded_searches=1,outcome={'decisions':1})
    return [r],receipt


def test_expanded_trace_audit_checks_every_required_action_and_future():
    import pytest
    rows,receipt=audited_rows()
    result=audit.audit_trace(rows,receipt,'all_plants')
    assert result['searched_roots']==1 and result['scored_halving_entries']==3
    rows[0]['evaluated_candidates'].remove(B)
    rows[0]['rounds'][0].pop()
    with pytest.raises(AssertionError,match='omitted'):
        audit.audit_trace(rows,receipt,'all_plants')


def test_trace_audit_rejects_unpaired_futures_and_missing_terminal():
    import pytest
    rows,receipt=audited_rows()
    rows[0]['rounds'][0][0]['future']=7001
    with pytest.raises(AssertionError,match='unpaired'):
        audit.audit_trace(rows,receipt,'all_plants')
    rows[0]['rounds'][0][0]['future']=7000
    rows[0]['terminal']=False
    with pytest.raises(AssertionError,match='terminal'):
        audit.audit_trace(rows,receipt,'all_plants')


class ProbeEnv:
    def __init__(self,unfinished=False):
        self._sun_production_history=[];self._sun_history_start_tick=0
        self.model=None;self.calls=[];self.unfinished=unfinished

    def restore_fast(self,root):
        assert root==17

    def state_hash(self):return 'root-state'

    def branch_snapshot(self,root,actions,retain_snapshots):
        assert root==17 and not retain_snapshots
        return [{'ok':True}]

    def load_value_net(self,path):self.model=path

    def rollout(self,action,until_tick,**options):
        assert options['snapshot_id']==17
        self.calls.append((audit.key(action),until_tick,self.model,options))
        if until_tick==0:
            return dict(terminal=not self.unfinished,result=1,wave=5,tick=9999)
        return dict(terminal=False,value=.25 if self.model=='v5.bin' else .75)


def setup_probe(monkeypatch):
    import search_play as teacher
    plan={'inputs':{},'teacher':{'rollout_sim_skip':61}}
    r={'tick':500,'production_candidates':[A,WAIT],'static_scores':[],
       'executed':A,'policy':A,'legal_actions':[A,WAIT]}
    monkeypatch.setattr(audit.base,'trace_rows',lambda *args:[r])
    monkeypatch.setattr(audit.base,'probe_targets',lambda *args:[0])
    monkeypatch.setattr(audit.base,'all_actions',lambda obs:[WAIT,A])
    monkeypatch.setattr(audit.base,'sha',lambda path:str(path))
    monkeypatch.setattr(audit,'history_sha',lambda state:'current-history')
    monkeypatch.setitem(teacher._W,'student_history',{'phase':'after-current-proposal'})
    monkeypatch.setattr(Path,'exists',lambda path:False)
    documents=[]
    monkeypatch.setattr(audit.base,'write',lambda path,document:documents.append(document))
    observer=audit.PairedProbe(plan,42,45000000,{'value':'v5.bin','v4_control':'v4.bin'})
    observer.decision=0;observer.fast_state={}
    return observer,documents


def test_probe_uses_eight_common_terminal_futures_and_explicit_matched_models(monkeypatch):
    observer,documents=setup_probe(monkeypatch);env=ProbeEnv()
    observer.probe(env,dict(tick=500,wave=1,sun=50,packets=[]),17,[A,WAIT],A)
    assert len(documents)==1 and documents[0]['root_restore_verified']
    actions=documents[0]['actions']
    assert len(actions)==2 and all(len(r['terminal'])==8 for r in actions)
    assert [r['future'] for r in actions[0]['terminal']]==[r['future'] for r in actions[1]['terminal']]
    assert all(r['leaf']['value']['2500']==[.25]*9 for r in actions)
    assert all(r['leaf']['v4_control']['2500']==[.75]*9 for r in actions)
    assert len(env.calls)==2*(8+9*4) and env.model=='v5.bin'
    assert all(c[3]['student_history']=={'phase':'after-current-proposal'} for c in env.calls)
    assert all(c[3]['score_value'] for c in env.calls if c[1])


def test_unfinished_terminal_future_writes_no_root_and_restores_model(monkeypatch):
    import pytest
    observer,documents=setup_probe(monkeypatch);env=ProbeEnv(unfinished=True)
    with pytest.raises(AssertionError,match='unfinished'):
        observer.probe(env,dict(tick=500),17,[A,WAIT],A)
    assert not documents and env.model=='v5.bin'


def test_terminal_evidence_validator_requires_full_action_coverage_and_replay(monkeypatch):
    import pytest
    observer,documents=setup_probe(monkeypatch);env=ProbeEnv()
    observer.probe(env,dict(tick=500,wave=1,sun=50,packets=[]),17,[A,WAIT],A)
    document=documents[0];baseline=audit.base.trace_rows(42,'production',45000000)[0]
    receipt={'outcome':{'terminal':True,'won':False,'seconds':3},
             'trace_sha256':document['identity']['trace_sha256']}
    replay={'complete_game_parity':True,'input_sha256':{},
            'outcome':audit.base.semantic_game(receipt['outcome'])}
    arguments=(document,replay,receipt,baseline,observer.plan,42,45000000,0)
    audit.validate_terminal_root(*arguments)
    replay['complete_game_parity']=False
    with pytest.raises(AssertionError):audit.validate_terminal_root(*arguments)
    replay['complete_game_parity']=True
    document['actions'].pop()
    with pytest.raises(AssertionError,match='coverage incomplete'):
        audit.validate_terminal_root(*arguments)


def test_historical_comparison_keeps_shovel_action_scope_separate():
    shovel=dict(type='shovel',row=0,col=0)
    document=dict(actions=[dict(action=a,terminal=[dict(won=(i<n)) for i in range(8)],
        leaf={'value':{'2500':[v]*9}}) for a,n,v in [(A,2,.3),(WAIT,0,.1),(shovel,8,.05)]])
    production=audit.metrics(document)
    all_legal=audit.metrics(document,representable=False)
    assert production['actions']==2 and production['pairs']==1 and production['accuracy']==1
    assert all_legal['actions']==3 and all_legal['pairs']==3 and all_legal['accuracy']==1/3
    assert audit.direction_score(1e-11,1)==.5
    assert audit.direction_score(-1e-11,-1)==.5


def test_replay_compares_full_json_frame_without_tuple_list_false_alarm(monkeypatch):
    import pytest
    import render_episode
    frame={'tick':0,'plants':[(0,1,33,300,300)],'v2':{'wave_timer':1800}}
    monkeypatch.setattr(render_episode,'_frame',lambda obs:frame)
    probe=object.__new__(audit.PairedProbe)
    probe.rows=[{'frame':{'tick':0,'plants':[[0,1,33,300,300]],'v2':{'wave_timer':1800}}}]
    probe.level=42;probe.seed=45000000
    documents=[]
    monkeypatch.setattr(audit.base,'write',lambda path,document:documents.append(document))
    probe.before(None,{}, {},0)
    assert not documents
    frame['v2']['wave_timer']=1799
    with pytest.raises(AssertionError,match='state differs at 0'):
        probe.before(None,{}, {},0)
    assert documents[0]['differing_fields']==['v2']
