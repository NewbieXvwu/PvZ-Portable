"""Small CPU-only tests for the optional v3 student; no engine or corpus needed."""
import io
import numpy as np
import pytest
import torch

from pvz_student_tokens import frame_tokens, collate_tokens
from pvz_student_v3 import StudentV3, network_config
from pvz_recipe import validate


def frame():
    return dict(sun=250, tick=60, wave=1, scene=dict(wave_count=10, pool=True),
                packets=[[0, 0, True, 0, 100, 100], [2, 16, False, 20, 100, 25]],
                plants=[[0, 1, 0, 200, 300]], zombies=[[0, 1, 150.25, -.2, 180, True, False, 20]],
                mowers=[[0, 1]], lanes=list(range(6)), v2=dict(projectiles=[[0, 175.5, 20]]))


def test_v3_compact_wire_and_padding():
    f = frame()
    a = frame_tokens(f)
    z = np.flatnonzero(a['ids'][:, 0] == 4)[0]
    assert a['features'][z, 2] == np.float32(150.25/900)
    assert a['features'][z, 9] == np.float32(-.2)
    assert a['packet_index'][1] == -1
    rich = dict(f, tokens_v3=dict(schema_version=1, **{k: v.tolist() for k, v in a.items()}))
    b = frame_tokens(rich)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    f['zombies'] = []
    batch = collate_tokens([a, frame_tokens(f)])
    assert batch['mask'].sum(1).tolist() == [len(a['ids']), len(a['ids'])-1]


@pytest.mark.parametrize('mode,hybrid', [('dense', False), ('terrain', False), ('moe', False), ('dense', True)])
def test_v3_structures_forward_backward(mode, hybrid):
    torch.set_num_threads(1)
    a, b = frame_tokens(frame()), frame_tokens(frame())
    # Explicit link to the visible zombie exercises the fifth relation term.
    plant = int(np.flatnonzero(a['ids'][:, 0] == 3)[0])
    zombie = int(np.flatnonzero(a['ids'][:, 0] == 4)[0])
    a['target_indices'][plant] = zombie
    b['terrain'] = np.asarray(4)
    tokens = collate_tokens([a, b])
    net = StudentV3(12, dict(layers=2, heads=3, ff_width=16, ffn=mode, hybrid=hybrid), aux=True)
    from pvz_value_net import BOARD_CHANNELS
    board = torch.randn(2, BOARD_CHANNELS, 6, 10)
    out = net(board, None, torch.tensor([0, 2]), auxiliary=True, tokens=tokens)
    assert {k: list(v.shape) for k, v in out.items()} == dict(plant=[2], packet=[2, 10], cell=[2, 54], aux=[2, 6])
    assert (out['packet'][:, 1] == -1e9).all()
    loss = out['plant'].sum() + out['packet'][:, [0, 2]].sum() + out['cell'].square().mean() + out['aux'].sum()
    loss.backward()
    assert net.encoder[0].attention.qkv.weight.grad.abs().sum() > 0
    assert net.encoder[0].attention.target_bias.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if mode == 'moe':
        assert net.encoder[0].ff.router.weight.grad.abs().sum() > 0


def test_v3_padding_does_not_change_logits():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    net = StudentV3(12, dict(layers=1, heads=3, ff_width=16)).eval()
    a = frame_tokens(frame())
    f = frame()
    f['zombies'] *= 4
    with torch.no_grad():
        one = net(None, None, torch.tensor([0]), tokens=collate_tokens([a]))
        padded = net(None, None, torch.tensor([0, 0]), tokens=collate_tokens([a, frame_tokens(f)]))
    for k in one:
        torch.testing.assert_close(one[k][0], padded[k][0], atol=1e-6, rtol=1e-5)


def test_v3_recipe_opt_in():
    base = dict(schema_version=1, kind='training', data='/tmp/data', out='/tmp/out', labels=['test'])
    assert validate(base)['network'] == {}
    with pytest.raises(ValueError, match='requires'):
        validate(dict(base, obs='v3'))
    c = validate(dict(base, width=12, obs='v3', packet_head='identity',
                      network=dict(architecture='token', heads=3, layers=1, ff_width=16)))
    assert c['network']['layers'] == 1
    with pytest.raises(ValueError):
        network_config(dict(architecture='token', surprise=1), 192)
