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


def test_v3_hf_real_frames_native(tmp_path):
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import features
    root = Path(os.environ.get('PVZ_TASK157_DATA', ROOT/'artifacts/student/task163/hf/transfer/task157_obs_v3_7e702892'))
    manifest = json.loads((root/'manifest.json').read_text())
    samples = [next(a for a in manifest['archives'] if a['split']=='train')]
    samples += [a for a in manifest['archives'] if a['split']=='terrain_sample']
    maximum = 0.
    torch.set_num_threads(1)
    for sample in samples:
        frames = read_frames(root/sample['path'])
        f = frames[len(frames)//2]
        source = frame_tokens(f, token_input='obs_v3')
        capacity = len(source['ids'])+5
        tokens = collate_tokens([source], capacity=capacity)
        board = features(f,None)[0][None]
        selected = torch.tensor([next(i for i,p in enumerate(source['packet_index']) if p>=0)])
        for mode,hybrid in [('dense',False),('terrain',False),('moe',False),('dense',True)]:
            torch.manual_seed(163)
            net = StudentV3(24, dict(layers=2,heads=3,ff_width=32,ffn=mode,hybrid=hybrid), token_input='obs_v3').eval()
            with torch.no_grad(): expected = net(board,None,selected,tokens=tokens)
            module,graph=student_graph(net,capacity)
            actual,_=run_driver(module,graph,graph_feeds(tokens,selected,board),tmp_path,DRIVER)
            for name in expected:
                value=actual[name].squeeze(1) if name=='plant' else actual[name]
                maximum=max(maximum,(value-expected[name]).abs().max().item())
                torch.testing.assert_close(value,expected[name],atol=1e-5,rtol=0)
    print(json.dumps(dict(real_frames=len(samples), max_abs_error=maximum)))


def test_v3_engine_adapter_and_actions(tmp_path):
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_rollout_policy import export_student, features, act
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from render_episode import _frame
    binary=ROOT/'build-studentv3net/pvz-portable'
    resource=Path(os.environ.get('PVZ_RESOURCE_DIR',Path.home()/'.cache/pvz-research-resources'))
    torch.set_num_threads(1)
    maximum=0.
    with EventWaitEnv(resource,executable=binary) as env:
        env.set_token_observation(True)
        obs,_=env.reset(task=TaskSpec(level=7,seed=163))
        obs,*_=env.step(dict(type='wait',ticks=120))
        f=_frame(obs)
        source=frame_tokens(f,token_input='obs_v3')
        tokens=collate_tokens([source])
        board=features(f,None)[0][None]
        for mode,hybrid in [('dense',False),('terrain',False),('moe',False),('dense',True)]:
            torch.manual_seed(163)
            net=StudentV3(24,dict(layers=2,heads=3,ff_width=32,ffn=mode,hybrid=hybrid),token_input='obs_v3').eval()
            with torch.no_grad(): net.plant.bias.fill_(1)
            export_student(net,tmp_path/'model.bin',token_capacity=256)
            env.load_rollout_net(tmp_path/'model.bin')
            native=env.rollout_predict()
            with torch.no_grad(): expected=net(board,None,torch.tensor([max(0,native['cell_packet'])]),tokens=tokens)
            for key in expected:
                value=np.asarray(native[key]).reshape(-1)
                if key=='cell' and native['cell_packet']<0: continue
                other=expected[key].detach().numpy().reshape(-1)
                error=float(np.max(np.abs(value-other)))
                maximum=max(maximum,error)
                np.testing.assert_allclose(value,other,atol=1e-5,rtol=0)
            assert native['action']==act(net,f,None,obs['legal_actions']['plants'])
    print(json.dumps(dict(engine_structures=4,max_abs_error=maximum)))
