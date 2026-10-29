"""Small CPU-only tests for the optional v3 student; no engine or corpus needed."""
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
    assert a['features'][z, 10] == np.float32(-.2/5)
    assert a['features'][z, 5] == 0  # Compact total hp must not masquerade as body hp.
    assert a['features'][z, 21] == np.float32(180/2000)
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


def test_v3_pipeline_pack_train_load_actions(tmp_path, monkeypatch):
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from pvz_frame_store import write_frames
    from pvz_training_pack import pack_options, archive_fingerprint
    from pack_training_rows import pack
    import train_rollout_policy as trainer
    from pvz_rollout_policy import load_student, act, ranked_plants, export_student
    from pvz_recipe import write_recipe

    torch.set_num_threads(1)
    data = tmp_path / 'data'
    labels = {'a', 'b'}
    expected = []
    for seed in range(4):
        for label in sorted(labels):
            folder = data / f'seed_{seed}' / label
            folder.mkdir(parents=True)
            frames = []
            for i in range(4):
                f = frame()
                f['tick'] = i*60
                f['sun'] = 200+seed*10+i
                f['zombies'] *= i+1
                # Include the native-array entry point in actual packed supervision.
                if seed == 0:
                    raw = frame_tokens(f)
                    f['tokens_v3'] = dict(schema_version=1, **{k: v.tolist() for k, v in raw.items()})
                if i:
                    f['action'] = dict(type='plant', packet=0, row=0, col=i) if i%2 else dict(type='wait', ticks=60)
                    f['targets'] = [[0, 0, 1, .7], [0, 0, 2, .3]]
                frames.append(f)
            write_frames(folder / 'frames.npz', frames)
            expected.extend(frame_tokens(f) for f in frames[:-1])
    options = pack_options(labels, obs='v3', packet_head='identity', cells=True, aux_horizon=2,
                           sources=archive_fingerprint(data, labels))
    # Exercise the real segmented packer even when a legacy cache directory is supplied.
    pack(data, options, cache_dir=tmp_path/'cache', quiet=True)
    loaded = trainer.load(data, labels, obs='v3', packet_head='identity', cells=True, aux_horizon=2)
    assert len(loaded) == 12
    store = loaded[-1]
    for i, sample in enumerate(expected):
        for k, array in sample.items():
            np.testing.assert_array_equal(store.sample(i)[k], array)
    reused = trainer.load(data, labels, obs='v3', packet_head='identity', cells=True, aux_horizon=2)
    for key, array in store.arrays.items():
        np.testing.assert_array_equal(array, reused[-1].arrays[key])

    recipe = dict(schema_version=1, kind='training', data=str(data), out=str(tmp_path/'out'),
                  labels=sorted(labels), obs='v3', packet_head='identity', width=12,
                  network=dict(architecture='token', layers=1, heads=3, ff_width=16, hybrid=True),
                  batch_size=4, validation_batch_size=4, epochs=1, max_steps=1, seed=5,
                  aux_weight=.1, aux_horizon=2, cell_soft_weight=.1,
                  validation=dict(seed=0, fraction=.25))
    path = tmp_path/'recipe.json'
    write_recipe(path, recipe)
    monkeypatch.setattr(sys, 'argv', ['train_rollout_policy.py', '--recipe', str(path)])
    # Isolate telemetry and process-priority side effects from the production machine.
    class QuietMetrics:
        def __init__(self, *a, **kw): pass
        def start(self): return self
        def update(self, **kw): pass
        def close(self): pass
    monkeypatch.setattr(trainer, 'Metrics', QuietMetrics)
    monkeypatch.setattr(trainer, 'set_oom_score_adj', lambda _: None)
    losses = []
    trainer.main(step_observer=lambda _, loss: losses.append(loss))
    assert len(losses) == 1 and np.isfinite(losses[0])
    net = load_student(tmp_path/'out'/'model.pt')
    assert net.network['hybrid']
    legal = [dict(type='plant', packet=0, row=0, col=1), dict(type='plant', packet=2, row=2, col=2)]
    ranked = ranked_plants(net, frame(), None, legal)
    assert len(ranked) == len(legal)
    assert all(np.isfinite(p) for p, _ in ranked)
    assert act(net, frame(), None, legal) in [*legal, dict(type='wait', ticks=60, until='timeout')]
    with pytest.raises(ValueError, match='input contract'):
        export_student(net, tmp_path/'native.bin')
    assert not (tmp_path/'native.bin').exists()


@pytest.mark.parametrize('obs,packet_head', [('v1', 'slot'), ('v2timer', 'identity')])
def test_v3_disabled_legacy_bitwise(tmp_path, obs, packet_head):
    from pvz_rollout_policy import RolloutNet, make_student, load_student, export_student
    from pvz_value_net import BOARD_CHANNELS, SCALARS
    from pvz_student_observation import OBS_EXTRA
    torch.set_num_threads(1)
    torch.manual_seed(19)
    legacy = RolloutNet(width=8, obs=obs, packet_head=packet_head).eval()
    torch.manual_seed(19)
    default = make_student(width=8, obs=obs, packet_head=packet_head, network={}).eval()
    for key, value in legacy.state_dict().items():
        assert torch.equal(value, default.state_dict()[key])
    board = torch.randn(2, BOARD_CHANNELS+OBS_EXTRA[obs][0], 6, 10)
    scalars = torch.randn(2, SCALARS+2+OBS_EXTRA[obs][1])
    info = torch.zeros(2, 10, 4) if packet_head == 'identity' else None
    pk = torch.tensor([0, 1])
    torch.save(dict(state_dict=legacy.state_dict(), obs=obs, packet_head=packet_head), tmp_path/'old.pt')
    loaded = load_student(tmp_path/'old.pt')
    with torch.no_grad():
        a = legacy(board, scalars, pk, packet_info=info)
        for model in (default, loaded):
            b = model(board, scalars, pk, packet_info=info)
            for key in a:
                assert torch.equal(a[key], b[key])
    export_student(legacy, tmp_path/'legacy.bin')
    export_student(default, tmp_path/'default.bin')
    export_student(loaded, tmp_path/'loaded.bin')
    assert (tmp_path/'legacy.bin').read_bytes() == (tmp_path/'default.bin').read_bytes() == (tmp_path/'loaded.bin').read_bytes()


def test_v3_rich_input_contract_and_empty_card_mask():
    raw = np.zeros((68, 38), np.float32)
    raw[:, 3:6] = -1
    raw[2:56, 0] = 2
    raw[2:56, 3] = np.arange(54)//9
    raw[2:56, 4] = np.arange(54)%9
    raw[56:66, 0] = 8
    raw[56, 1] = 1
    raw[66, :6] = [4, 2, 0, 0, 3, -1]
    raw[66, 6:] = np.arange(32)/32
    raw[67, :6] = [3, 1, 0, 0, 3, 5]
    f = dict(frame(), v3=dict(version=3, tokens=raw.tolist()))
    sample = frame_tokens(f, token_input='obs_v3')
    np.testing.assert_array_equal(sample['features'][66], raw[66, 6:])
    assert sample['target_indices'][67] == 5
    assert sample['packet_index'].tolist() == [56]+[-1]*9
    batch = collate_tokens([sample])
    assert batch['mask'].sum() == 59
    assert batch['cell_index'][0].tolist() == list(range(2, 56))
    with pytest.raises(ValueError, match='backfill'):
        frame_tokens(frame(), token_input='obs_v3')


def test_v3_window_jobs_resume_and_manifest_eval(tmp_path, monkeypatch):
    import json
    import os
    import signal
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import prepare_student_v3_jobs as prep
    import train_rollout_policy as trainer
    import eval_window_policy as evaluator
    from pvz_frame_store import write_frames
    from test_student_v3_native import rich_frame
    source=tmp_path/'source'
    archives=[]
    for i, split in enumerate(['train']*4+['seen','unseen','terrain_sample']):
        path=source/'frames'/f'seed_{i}'/'fixture'/'frames.npz'
        path.parent.mkdir(parents=True)
        frames=[]
        for t in range(3):
            f=frame()
            f.pop('v2')
            f['v3']=rich_frame()['v3']
            f['tick']=t*60
            if t: f['action']=dict(type='plant',packet=0,row=1,col=2) if t==1 else dict(type='wait',ticks=60)
            frames.append(f)
        write_frames(path,frames)
        (path.parent/'meta.json').write_text(json.dumps(dict(context=dict(level=7))))
        archives.append(dict(path=str(path.relative_to(source)),split=split,seed=i,level=7))
    (source/'manifest.json').write_text(json.dumps(dict(input_revision='fixture',archives=archives)))
    matrix=json.loads((prep.ROOT/'configs/task163_student_v3_jobs.json').read_text())
    matrix.update(seeds=[1],epochs=1,internal_validation=dict(seed=0,fraction=.25))
    for structure in matrix['structures']:
        structure.update(width=12,batch_size=2)
        if structure['obs']=='v3': structure['network'].update(heads=3,layers=1,ff_width=16)
    prepared=prep.prepare(source,tmp_path/'state',matrix)
    assert len(prepared['jobs'])==7 and prepared['enqueued'] is False
    job=json.loads((tmp_path/'state/jobs/task163_token128_2_s1.json').read_text())
    class QuietMetrics:
        def __init__(self,*a,**kw): pass
        def start(self): return self
        def update(self,**kw): pass
        def close(self): pass
    monkeypatch.setattr(trainer,'Metrics',QuietMetrics)
    monkeypatch.setattr(trainer,'set_oom_score_adj',lambda _:None)
    command=['train',*job['command'][2:],'--device','cpu']
    monkeypatch.setattr(sys,'argv',command)
    with pytest.raises(SystemExit) as stopped:
        trainer.main(step_observer=lambda step,loss: os.kill(os.getpid(),signal.SIGUSR1) if step==1 else None)
    assert stopped.value.code==75
    out=Path(job['out'])
    saved=torch.load(out/'resume.pt',weights_only=False,map_location='cpu')
    assert saved['token_input']=='obs_v3' and saved['completed_steps']==1
    monkeypatch.setattr(sys,'argv',[*command,'--resume',str(out/'resume.pt')])
    trainer.main()
    monkeypatch.setattr(sys,'argv',['evaluate',*job['evaluate'][2:]])
    evaluator.main()
    metrics=json.loads((out/'heldout.json').read_text())['groups']
    assert metrics['all']['rows']==6
    assert all(metrics[k]['rows']==2 for k in ('seen','unseen','terrain_sample'))
    assert metrics['seen/day']['rows']==metrics['unseen/day']['rows']==2
