"""Ranking denominators, independence units and CPU accounting must stay explicit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from fog_terminal_report import pair_counts,sign_p,holm,bootstrap_counts
import fog_terminal_diagnostic as diag


def test_truth_ties_excluded_value_ties_receive_half_credit():
    r=pair_counts([0,1,1],[.1,.1,.2])
    assert r['pairs']==2 and r['accuracy']==.75 and r['value_ties']==1
    assert pair_counts([0,.1],[0,1],1/3)['accuracy'] is None
    assert pair_counts([0,1],[1,0])['accuracy']==0


def test_sign_test_and_holm_cover_the_whole_test_family():
    assert sign_p([1]*16)==1/65536
    assert sign_p([-1]*16)==1
    assert sign_p([0]*16)==1
    tests=holm([{'p':.01},{'p':.04},{'p':.2}])
    assert [round(t['p_holm'],5) for t in tests]==[.03,.08,.2]


def test_bootstrap_resamples_episode_seeds_instead_of_candidate_pairs():
    groups=[dict(cell=dict(level=31,seed=i),pair_metrics={'wave':{'v4':{'correct':i+1,'pairs':2}}}) for i in (0,1)]
    base=bootstrap_counts(groups,'wave',iterations=200)
    more=[dict(cell=g['cell'],pair_metrics={'wave':{'v4':{'correct':g['pair_metrics']['wave']['v4']['correct']*100,'pairs':200}}}) for g in groups]
    assert bootstrap_counts(more,'wave',iterations=200)==base


def test_cpu_meter_uses_final_reaped_cpu_without_double_counting_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(diag,'OUT',tmp_path)
    diag.save(tmp_path/'runs/finished/usage-worker.json',{'cpu_seconds':10})
    diag.save(tmp_path/'runs/finished/usage-controller.json',{'cpu_seconds':5})
    diag.save(tmp_path/'runs/finished/complete.json',{'controller_children_cpu_seconds':20})
    diag.save(tmp_path/'runs/live/usage-worker.json',{'cpu_seconds':3})
    assert diag.spent_cpu()==23
