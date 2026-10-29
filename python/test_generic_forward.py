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


def test_real_frames_and_legacy_actions(tmp_path):
    """One simulator; three layouts, occupied frames, greedy actions and short native rollouts."""
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_rollout_policy import RolloutNet, act, export_student, features, packet_features, identity_scalars
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from render_episode import _frame
    from pvz_generic_forward import rollout_graph
    binary = Path(os.environ.get('PVZ_EXECUTABLE', ROOT/'build-cppnet/pvz-portable'))
    resource = Path(os.environ.get('PVZ_RESOURCE_DIR', Path.home()/'.cache/pvz-research-resources'))
    if not binary.is_file() or not (resource/'main.pak').is_file(): pytest.skip('needs isolated build/resources')
    torch.set_num_threads(1)
    torch.manual_seed(161006)
    maximum, decisions = 0., 0
    configs = [dict(width=8), dict(width=13, pool='meanmax', context=True, obs='v2',
                                  packet_head='identity', plant_embedding_dim=7),
               dict(width=64, pool='meanmax', obs='v2timer', row_pool=True, packet_head='identity')]
    with EventWaitEnv(resource, executable=binary) as env:
        for config in configs:
            net = RolloutNet(**config).eval()
            with torch.no_grad(): net.plant.bias.fill_(.4)
            graph = rollout_graph(net)
            export_graph(net, graph, tmp_path/'generic.bin')
            export_student(net, tmp_path/'legacy.bin')
            obs, _ = env.reset(task=TaskSpec(level=9, seed=161006))
            while obs['tick'] < 6000:
                legal = obs['legal_actions']['plants']
                action = {'type': 'plant', **legal[0]} if legal else {'type': 'wait', 'ticks': 60}
                obs, *_ = env.step(action)
            history = StudentHistory(net.obs) if net.obs != 'v1' else None
            ctx = observation_context(obs) if net.context else None
            previous = previous_action = None
            for decision in range(4):
                frame = _frame(obs)
                frame['action'] = previous_action
                native = []
                for path in ('legacy.bin', 'generic.bin'):
                    env.load_rollout_net(tmp_path/path)
                    native.append(env.rollout_predict(previous['tick'] if previous else None,
                                  bool(previous_action and previous_action['type'] == 'plant'), student_history=history))
                old, new = native
                assert old['cell_packet'] == new['cell_packet']
                assert old['action'] == new['action']
                for key in ('plant', 'packet', 'cell'):
                    error = float(np.max(np.abs(np.asarray(old[key])-np.asarray(new[key]))))
                    maximum = max(maximum, error)
                    assert error < 1e-5
                board, scalars = history.features(frame, previous, ctx) if history else features(frame, previous, ctx)
                if net.packet_head == 'identity': scalars = identity_scalars(frame, scalars)
                selected = max(0, new['cell_packet'])
                feeds = dict(board=board[None], scalars=scalars[None], packet_info=packet_features(frame)[None],
                             selected_packet=torch.tensor([selected], dtype=torch.float32))
                maximum = max(maximum, compare(net, graph, feeds, tmp_path))
                with torch.inference_mode():
                    expected = net(board[None], scalars[None], torch.tensor([selected]), packet_info=feeds['packet_info'])
                for key in ('plant', 'packet', 'cell'):
                    if key == 'cell' and new['cell_packet'] < 0: continue
                    torch.testing.assert_close(torch.as_tensor(new[key]).reshape(-1), expected[key].reshape(-1), atol=1e-5, rtol=0)
                chosen = act(net, frame, previous, obs['legal_actions']['plants'], context=ctx, history=history)
                assert chosen == new['action']
                previous, previous_action = frame, chosen
                obs, *_ = env.step(chosen)
                decisions += 1
            root = env.snapshot_fast()
            hashes = []
            for path in ('legacy.bin', 'generic.bin'):
                env.load_rollout_net(tmp_path/path)
                env.rollout(None, obs['tick']+600, snapshot_id=root, seed=161007, student_history=history)
                hashes.append(env.state_hash())
            assert hashes[0] == hashes[1]
            env.release_snapshot(root)
    print(json.dumps(dict(real_decisions=decisions, max_abs_error=maximum, legacy_actions_equal=True,
                          short_native_rollouts_equal=True)))
