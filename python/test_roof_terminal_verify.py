import hashlib
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import terminal_candidate_audit as audit
import roof_terminal_verify as verify


def test_identity_rebuild_uses_preregistered_namespace_and_common_future(monkeypatch):
    plan=dict(spec=dict(namespace='fixture-ns'),sha256={'value':'x'})
    monkeypatch.setattr(audit,'PLAN',plan)
    cell=dict(id='l44-student-45000000',level=44,seed=1)
    point=dict(decision=196,tick=11039)
    action=dict(type='wait',ticks=60,until='timeout')
    got=verify.expected_identity(cell,point,3,action,1,plan['sha256'])
    def seed(stage):
        return int.from_bytes(hashlib.sha256(f'fixture-ns:l44-student-45000000:196:{stage}:1'.encode()).digest()[:4],'little')
    assert got['future']==seed('screen')
    assert got['search_seed_base']==seed('screen-teacher-search')
    assert (got['index'],got['k'],got['stage'],got['action'])==(3,1,'screen',action)
    assert got['inputs_sha256']==plan['sha256'] and got['root'] is point
    other=verify.expected_identity(cell,point,3,action,0,plan['sha256'])
    assert other['future']!=got['future']


def test_identity_rebuild_changes_with_namespace(monkeypatch):
    cell=dict(id='l44-student-45000000',level=44,seed=1)
    point=dict(decision=196,tick=11039)
    action=dict(type='wait',ticks=60,until='timeout')
    monkeypatch.setattr(audit,'PLAN',dict(spec=dict(namespace='a'),sha256={}))
    first=verify.expected_identity(cell,point,0,action,0,{})['future']
    monkeypatch.setattr(audit,'PLAN',dict(spec=dict(namespace='b'),sha256={}))
    assert verify.expected_identity(cell,point,0,action,0,{})['future']!=first
