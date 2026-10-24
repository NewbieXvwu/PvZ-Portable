"""Task58 measurement gates: causality, chunk carry, dt and label boundaries."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_memory_probe import CONFIGS, MemoryProbe, make_probe
from memory_probe_data import labels
from memory_probe_metrics import auc, masks, stats
from memory_probe_train import objective
import memory_probe_train
import memory_probe_queue

torch.set_num_threads(1)


@pytest.mark.parametrize('kind', CONFIGS)
def test_causal_and_chunk_equivalent(kind):
    torch.manual_seed(58)
    model = MemoryProbe(kind, 8).eval()
    board = torch.randn(9, 2, 102, 6, 10)
    scalar = torch.randn(9, 2, 157)
    dt = torch.tensor([[0., 0.], [60., 30.], [0., 0.]] * 3)
    with torch.no_grad():
        full, _ = model(board, scalar, dt)
        first, state = model(board[:4], scalar[:4], dt[:4])
        second, _ = model(board[4:], scalar[4:], dt[4:], state)
        torch.testing.assert_close(full, torch.cat((first, second)), atol=1e-6, rtol=1e-5)
        changed = board.clone()
        changed[5:] += 10
        future, _ = model(changed, scalar, dt)
        torch.testing.assert_close(full[:5], future[:5])
        # A new episode's None state must reproduce the fresh call.
        repeated, _ = model(board, scalar, dt, None)
        torch.testing.assert_close(full, repeated)
    assert full.shape == (9, 2, 35, 6, 9)


def test_ssm_zero_dt_preserves_state_and_positive_dt_changes_it():
    model = MemoryProbe('spatial_ssm', 8)
    b, s = torch.randn(1, 2, 102, 6, 10), torch.randn(1, 2, 157)
    state = torch.randn(2, 16, 6, 10)
    _, zero = model(b, s, torch.zeros(1, 2), state)
    torch.testing.assert_close(zero, state)
    _, advanced = model(b, s, torch.full((1, 2), 600.), state)
    assert not torch.allclose(zero, advanced)
    advanced.square().mean().backward()
    assert model.log_rate.grad.isfinite().all()
    assert model.frequency.grad.isfinite().all()


def test_budget_alignment():
    counts = [sum(p.numel() for p in make_probe(k).parameters()) for k in CONFIGS]
    assert max(counts) / min(counts) < 1.05


def test_engine_label_onboard_and_multitype_semantics():
    def z(x, kind=0, onboard=True):
        return dict(row=1, x=x, type=kind, on_board=onboard)
    y = labels(dict(zombies=[z(759), z(760, 2), z(39, 2), z(600, 3, False), z(759, 2)]))
    assert y[0].sum() == 1
    assert y[1, 1, 8] == y[3, 1, 8] == 1
    assert y[1:].sum() == 2
    assert np.isin(y, [0, 1]).all()


def test_metric_ties_and_single_class():
    assert auc([0,1], [.5,.5]) == .5
    assert auc([0,1,1,0], [.1,.9,.9,.1]) == 1.
    assert auc([0,1], [.9,.1]) == 0.
    assert auc([1,1], [.2,.8]) is None
    assert stats([],[])['logloss'] is None
    assert stats([0,1],[.5,.5])['logloss'] == pytest.approx(np.log(2))
    assert stats([1,0],[2e-9,1e-9])['auc'] == 1.


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(58)
    y = rng.integers(0,2,200).astype(bool)
    p = rng.integers(0,7,200) / 6.
    comparisons = p[y,None] - p[None,~y]
    reference = np.mean((comparisons > 0) + .5 * (comparisons == 0))
    assert auc(y,p) == pytest.approx(reference)


def test_recency_and_event_masks_do_not_overlap_buckets():
    fog = np.ones((5,6,9), dtype=bool)
    data = dict(fog=fog, age=np.broadcast_to(np.array([-1,0,120,600,601])[:,None,None],fog.shape),
                tick=np.array([0,60,120,660,720]))
    m = masks(data,[dict(tick=120)])
    buckets = [m[k] for k in ('age_0_120','age_121_600','age_over_600','never_visible')]
    np.testing.assert_array_equal(sum(buckets),fog.astype(int))
    assert m['return_before'][:2].all() and not m['return_before'][2:].any()
    assert m['return_after'][2:].all() and not m['return_after'][:2].any()


def test_loss_ignores_padded_episodes_and_has_type_gradient():
    target = torch.zeros(2,2,35,6,9,dtype=torch.uint8)
    target[0,0,0,1,2] = target[0,0,3,1,2] = 1
    valid = torch.tensor([[True,False],[True,False]])[:,:,None,None]
    logits = torch.zeros(2,2,35,6,9,requires_grad=True)
    loss = objective(logits,target,valid)
    loss.backward()
    assert not logits.grad[:,1].any()
    assert logits.grad[0,0,3,1,2] < 0
    assert logits.grad[0,0,2,1,2] > 0


def test_child_waits_for_memory_before_loading_data_or_cuda(monkeypatch,tmp_path):
    readings = iter([9 * 1024**3, 11 * 1024**3])
    calls = []
    monkeypatch.setattr(memory_probe_train,'OUT',tmp_path)
    monkeypatch.setattr(memory_probe_train,'mem_available',lambda:next(readings))
    monkeypatch.setattr(memory_probe_train.time,'sleep',lambda seconds:calls.append(('sleep',seconds)))
    monkeypatch.setattr(memory_probe_train,'write_json',lambda path,value:calls.append(('gate',value['mem_available_bytes'])))
    def stop_before_gpu(*args):
        calls.append(('load',None))
        raise RuntimeError('test stops before allocating data or model')
    monkeypatch.setattr(memory_probe_train,'load_episodes',stop_before_gpu)
    with pytest.raises(RuntimeError,match='test stops'):
        memory_probe_train.train_one('current','human',999,20)
    assert calls == [('sleep',30),('gate',11 * 1024**3),('load',None)]


def test_lightweight_queue_covers_every_configuration_without_torch_import():
    import subprocess
    assert memory_probe_queue.CONFIGS == CONFIGS
    check = subprocess.run([sys.executable,'-c',
        'import sys; sys.path.insert(0,' + repr(str(ROOT/'scripts')) + '); '
        'import memory_probe_queue; assert "torch" not in sys.modules'],check=True)
    assert check.returncode == 0
