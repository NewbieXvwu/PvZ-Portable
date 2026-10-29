"""Targeted standalone parity. Each pytest invocation selects a single -k case."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

from pvz_generic_forward import GraphSpec, export_graph, reference
from pvz_generic_examples import EXAMPLES, example, random_feeds

ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(os.environ.get('PVZ_GENERIC_DRIVER', ROOT/'build-cppnet/generic-forward'))


def run_cpp(module, graph, feeds, folder, iterations=1):
    folder = Path(folder)
    export_graph(module, graph, folder/'model.bin')
    with open(folder/'inputs.f32', 'wb') as f:
        for name, desc in graph.tensors.items():
            if desc['role'] == 1: f.write(feeds[name].numpy().astype('<f4').tobytes())
    result = subprocess.run([str(DRIVER), str(folder/'model.bin'), str(folder/'inputs.f32'),
                             str(folder/'outputs.f32'), str(iterations)], check=True, capture_output=True, text=True)
    raw = np.fromfile(folder/'outputs.f32', dtype='<f4')
    outputs, pos = {}, 0
    for n in graph.nodes:
        shape = graph.tensors[n['name']]['shape']
        size = int(np.prod(shape))
        outputs[n['name']] = torch.from_numpy(raw[pos:pos+size].copy().reshape(shape))
        pos += size
    assert pos == len(raw)
    return outputs, json.loads(result.stdout)


def compare(module, graph, feeds, tmp_path):
    expected = reference(module, graph, feeds, intermediates=True)
    actual, _ = run_cpp(module, graph, feeds, tmp_path)
    maximum = 0.
    for name, value in actual.items():
        error = (value-expected[name]).abs().max().item()
        maximum = max(maximum, error)
        torch.testing.assert_close(value, expected[name], rtol=0, atol=1e-5, msg=lambda msg: name+': '+msg)
    return maximum


@pytest.mark.parametrize('name', EXAMPLES)
def test_example(name, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(161)
    m, g = example(name)
    error = compare(m, g, random_feeds(g), tmp_path)
    print(json.dumps({'example': name, 'max_abs_error': error}))


def test_small_batch_and_modules(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(162)
    m, g = example('token_2_64', batch=2)
    feeds = random_feeds(g)
    feeds['valid'][1] = 0  # all-masked batch must produce finite zeros in attention
    assert compare(m, g, feeds, tmp_path) < 1e-5
    # Exercise RMSNorm, ReLU FFN, top-k + shared MoE with ties, 1xk conv and scatter duplicates.
    m = nn.Module()
    g = GraphSpec()
    g.tensor('x', [2, 4, 8]); g.tensor('route', [2, 4, 3])
    m.register_buffer('scale', torch.rand(8)); m.register_buffer('offset', torch.rand(8))
    g.node('rms', 'norm', ['x', g.weight(m, 'scale'), g.weight(m, 'offset')], [2, 4, 8], [1], [1e-5])
    weights = []
    for i in range(4):
        for name, layer in ((f'up{i}', nn.Linear(8, 12)), (f'down{i}', nn.Linear(12, 8))):
            m.add_module(name, layer)
            weights.extend([g.weight(m, name+'.weight'), g.weight(m, name+'.bias')])
    g.node('relu_ffn', 'ffn', ['rms', *weights[:4]], [2, 4, 8], [0])
    g.node('moe', 'moe', ['relu_ffn', 'route', *weights], [2, 4, 8], [0, 0, 3, 2, 1])
    feeds = {'x': torch.randn(2, 4, 8), 'route': torch.zeros(2, 4, 3)}
    assert compare(m, g, feeds, tmp_path) < 1e-5
    m, g = example('cnn64', batch=2)
    assert compare(m, g, random_feeds(g), tmp_path) < 1e-5
