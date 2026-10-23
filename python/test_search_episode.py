"""Teacher archive identity and controlled student-continuation counterfactuals."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
import episode_query as eq
import search_episode


def test_teacher_capture_does_not_dispatch_to_student(monkeypatch,tmp_path):
    calls=[]
    def fake(*args,**kwargs):
        calls.append((args,kwargs))
        return dict(task={'policy':'teacher','checkpoint':'weights'},outcome={'reason':'test'},
                    totals={},frames=[{'tick':0}],log=[])
    monkeypatch.setattr(search_episode,'collect_search',fake)
    monkeypatch.setattr(eq,'collect',lambda *a,**k: (_ for _ in ()).throw(AssertionError('student capture')))
    result=eq.capture('resource',39000000,31,'teacher',[0,9,16],tmp_path,4000,
                      checkpoint='weights',search_config={'manifest':'tasks','task_id':'fog'})
    assert result['meta']['task']['policy']=='teacher'
    assert calls[0][1]['arm']=='fast_search'
    assert calls[0][1]['manifest']=='tasks'


def test_teacher_whatif_uses_paired_continuation_baseline(monkeypatch,tmp_path):
    calls=[]
    def variant(meta,override,resource_dir,prefix=None):
        calls.append((override,prefix))
        return dict(final_wave=9,final_tick=100,wave_count=20,reason='loss',mowers_used=2,plants_eaten=3)
    monkeypatch.setattr(eq,'_run_variant',variant)
    original={'type':'plant','packet':1,'row':0,'col':0}
    prefix_action={'type':'wait','ticks':60}
    frames=[{'tick':0,'packets':[[0,0],[1,9]]},
            {'tick':60,'packets':[[0,0],[1,9]],'action':prefix_action},
            {'tick':60,'action':original}]
    meta=dict(task={'policy':'teacher','deck':[0,9]},outcome={'final_wave':8})
    text=eq._whatif(tmp_path,meta,frames,2,['wait:60'],'resource',False,None,12,None)
    assert calls==[({2:original},[prefix_action]),({2:{'type':'wait','ticks':60}},[prefix_action])]
    assert '同一学生续走' in text
    assert '最终第 9/20' in text  # compared with the paired baseline, not original teacher's wave 8


def test_teacher_variant_preserves_prefix_and_human_fog(monkeypatch):
    received=[]
    def fake(*args,**kwargs):
        received.append((args,kwargs))
        return {'frames':[{},{}], 'outcome':{'final_wave':8}}
    monkeypatch.setattr(eq,'collect',fake)
    task=dict(policy='teacher',level=31,seed=39000000,deck=[0,9,16],checkpoint='weights',fog_observation='human')
    prefix=[{'type':'wait','ticks':60}]
    eq._run_variant({'task':task},{1:prefix[0]},'resource',prefix)
    args,kwargs=received[0]
    assert args[3]=='student'
    assert kwargs['task_extra']['fog_observation']=='human'
    assert kwargs['prefix_actions']==prefix
    assert kwargs['checkpoint']=='weights'
