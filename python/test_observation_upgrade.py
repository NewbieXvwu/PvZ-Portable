"""task131: a wider input preserves the parent's function before training."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_rollout_policy import RolloutNet, warm_start_student


@pytest.mark.parametrize('obs', ['v1', 'v2', 'v2boss', 'v2bungee', 'v2sleep'])
@pytest.mark.parametrize('context,head', [(False, 'identity'), (True, 'slot')])
def test_widened_parent_preserves_all_outputs(obs, context, head):
    torch.set_num_threads(1)
    torch.manual_seed(131)
    old = RolloutNet(8, 'meanmax', context, obs, row_pool=True, aux=True, packet_head=head).eval()
    new = RolloutNet(8, 'meanmax', context, 'v2sleep', row_pool=True, aux=True, packet_head=head).eval()
    saved = copy.deepcopy(old.state_dict())
    warm_start_student(new, old)
    for name, weight in saved.items():
        actual = new.state_dict()[name]
        if name in ('trunk.0.weight', 'scalar.0.weight'):
            torch.testing.assert_close(actual[:, :weight.shape[1]], weight, rtol=0, atol=0)
            assert not actual[:, weight.shape[1]:].any()
        else:
            torch.testing.assert_close(actual, weight, rtol=0, atol=0)
    board = torch.randn(2, old.trunk[0].in_channels, 6, 10)
    scalars = torch.randn(2, old.scalar[0].in_features)
    wider_board = torch.zeros(2, new.trunk[0].in_channels, 6, 10)
    wider_scalars = torch.zeros(2, new.scalar[0].in_features)
    wider_board[:, :board.shape[1]] = board
    wider_scalars[:, :scalars.shape[1]] = scalars
    info = torch.zeros(2, 10, 4)
    info[:, :, 0] = torch.arange(10)
    packet = torch.tensor([1, 3])
    with torch.inference_mode():
        expected = old(board, scalars, packet, auxiliary=True, packet_info=info)
        actual = new(wider_board, wider_scalars, packet, auxiliary=True, packet_info=info)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-6)


def test_widening_rejects_semantically_different_inputs():
    with pytest.raises(ValueError, match='input upgrade'):
        warm_start_student(RolloutNet(8, obs='v2sleep'), RolloutNet(8, obs='v2fog'))
    with pytest.raises(ValueError, match='context'):
        warm_start_student(RolloutNet(8, context=True, obs='v2sleep'), RolloutNet(8, obs='v2'))


@pytest.mark.parametrize('new_fields', [False, True])
def test_sleep_cache_reuses_only_missing_field_v2_rows(tmp_path, monkeypatch, new_fields):
    import numpy as np
    import pvz_frame_store as store
    from test_bungee_inputs import frame

    first = frame()
    first.pop('bungee')
    if new_fields:
        first['sleep'] = [[0, 1, 1, 0]]
    second = copy.deepcopy(first)
    second.update(tick=160, action=dict(type='wait', ticks=60))
    path = tmp_path / 'seed_30000/example/frames.npz'
    store.write_frames(path, [first, second])
    old = store.cache_episode(path, tmp_path / 'cache', 'v2')
    source = store.frame_bytes(path)
    expected = store.encode_student_frames([first, second], source, 'v2sleep')
    original = store.encode_student_frames
    calls = []
    def encode(*args):
        calls.append(args[-1])
        return original(*args)
    monkeypatch.setattr(store, 'encode_student_frames', encode)
    upgraded = store.cache_episode(path, tmp_path / 'cache', 'v2sleep')
    assert calls == (['v2sleep'] if new_fields else [])
    with np.load(upgraded, allow_pickle=False) as actual:
        for name, array in expected.items():
            np.testing.assert_array_equal(actual[name], array)
    assert old.exists() and old != upgraded
    assert store.cache_episode(path, tmp_path / 'cache', 'v2sleep') == upgraded
