"""Small independent C++ parity checks for complete exported v3 models."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from pvz_student_v3 import StudentV3
from pvz_student_tokens import collate_tokens, frame_tokens
from pvz_student_v3_export import student_graph, graph_feeds
from pvz_generic_forward import run_driver, reference
from pvz_value_net import BOARD_CHANNELS

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(os.environ.get('PVZ_GENERIC_DRIVER', ROOT/'build-studentv3net/generic-forward'))


def rich_frame():
    raw = np.zeros((69, 38), np.float32)
    raw[:, 3:6] = -1
    raw[1, 0] = 1
    raw[2:56, 0] = 2
    raw[2:56, 3] = np.arange(54)//9
    raw[2:56, 4] = np.arange(54)%9
    raw[56:66, 0] = 8
    raw[56:59, 1] = [1, 2, 17]
    raw[66, :6] = [3, 1, 0, 1, 2, 68]
    raw[67, :6] = [5, 2, 1, 1, 2, -1]  # scatter duplicate
    raw[68, :6] = [4, 3, 0, 1, 4, -1]
    raw[:, 6:] = np.random.default_rng(163).normal(0, .1, (69, 32))
    return dict(scene=dict(pool=True), v3=dict(version=3, tokens=raw.tolist()))


@pytest.mark.parametrize('mode,hybrid', [('dense',False),('terrain',False),('moe',False),('dense',True)])
def test_complete_v3_native_parity(tmp_path, mode, hybrid):
    torch.set_num_threads(1)
    torch.manual_seed(163)
    net = StudentV3(24, dict(layers=2, heads=3, ff_width=32, ffn=mode, hybrid=hybrid), token_input='obs_v3').eval()
    with torch.no_grad():
        for layer in net.encoder:
            layer.attention.target_bias.normal_(0, .05)
    a = frame_tokens(rich_frame(), token_input='obs_v3')
    b = dict(a, terrain=np.asarray(4))
    tokens = collate_tokens([a, b], capacity=72)
    board = torch.randn(2, BOARD_CHANNELS,6,10)
    packet = torch.tensor([0,2])
    with torch.no_grad(): expected = net(board, None, packet, tokens=tokens)
    module, graph = student_graph(net, capacity=72, batch=2)
    feeds = graph_feeds(tokens, packet, board)
    actual, _ = run_driver(module, graph, feeds, tmp_path, DRIVER)
    graph_expected = reference(module, graph, feeds)
    maximum = 0.
    for key in ('plant','packet','cell'):
        value = actual[key].flatten(1) if key != 'plant' else actual[key].squeeze(1)
        for other in (expected[key], graph_expected[key].squeeze(1) if key=='plant' else graph_expected[key]):
            error = (value-other).abs().max().item()
            maximum = max(maximum, error)
            torch.testing.assert_close(value, other, rtol=0, atol=1e-5)
    print(json.dumps(dict(mode=mode, hybrid=hybrid, max_abs_error=maximum)))
