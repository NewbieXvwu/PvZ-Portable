"""Small lossless sidecar and real collection-switch tests."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from pvz_research_data import SparseTokenWriter, SparseTokens, episode_tokens, capture_frame, SIDECAR_NAME
from pvz_student_tokens import frame_tokens
from pvz_frame_store import write_frames, frame_bytes
from test_student_v3 import frame
from test_student_v3_native import rich_frame


def test_sidecar_lossless_target_signed_zero_and_frame_binding(tmp_path):
    f=dict(frame(),v3=rich_frame()['v3'])
    sample=frame_tokens(f,token_input='obs_v3')
    sample['features'][0,31]=-0.
    writer=SparseTokenWriter(tmp_path)
    writer.append(sample)
    path=tmp_path/'frames.npz'
    compact=dict(f);compact.pop('v3')
    write_frames(path,[compact])
    before=path.read_bytes()
    writer.finish(tmp_path/SIDECAR_NAME,source_frame=path)
    assert path.read_bytes()==before
    store=episode_tokens(path)
    for name,value in sample.items():
        actual=store.sample(0)[name]
        assert value.dtype==actual.dtype and value.tobytes()==actual.tobytes()
    changed=dict(compact,sun=999)
    write_frames(path,[changed])
    with pytest.raises(ValueError,match='different compact'):
        episode_tokens(path)


@pytest.mark.parametrize('demo', [False, True])
def test_sidecar_collection_switch_keeps_compact_bytes(tmp_path,monkeypatch,demo):
    from argparse import Namespace
    from pvz_rollout_policy import RolloutNet,export_student
    import search_play as search
    torch.set_num_threads(1)
    net=RolloutNet(8).eval()
    model=tmp_path/'model.bin';export_student(net,model)
    (tmp_path/'checkpoint.pt').write_bytes(b'fixture; net supplied through worker state')
    manifest=tmp_path/'tasks.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(task_id='fixture',level=7,deck=[0,1,2,3,4,5])])))
    demo_archive = None
    if demo:
        from human_demo import Recorder
        from pvz_event_env import EventWaitEnv
        from pvz_env import TaskSpec
        from pvz_tasks import profile_for_deck
        demo_archive=tmp_path/'demo'
        deck=(0,1,2,3,4,5)
        with EventWaitEnv(Path.home()/'.cache/pvz-research-resources',executable=ROOT/'build-studentv3net/pvz-portable') as env:
            recorder=Recorder(env,demo_archive,'fixture',deck,TaskSpec(level=7,seed=157,playthrough=2,profile=profile_for_deck(deck)))
            recorder.step(dict(type='wait',ticks=60))
            recorder.close()
    monkeypatch.setattr(search,'MAX_ACTIONS',4)
    collected=[]
    for enabled in (False,True):
        target=tmp_path/str(enabled)
        args=Namespace(manifest=manifest,task_id='fixture',resource_dir=Path.home()/'.cache/pvz-research-resources',
            executable=ROOT/'build-studentv3net/pvz-portable',obs_v3=enabled,frames=target,frames_label='fixture',
            rollout_policy=tmp_path/'checkpoint.pt',dagger_beta=1.,demo_archive=demo_archive,demo_decision=2)
        search._W.clear();search._W.update(fast=net,rollout_file=str(model))
        search._play(dict(args=args,arm='fast',seed=157))
        folder=target/'seed_157/fixture_fast'
        collected.append(folder/'frames.npz')
        assert (folder/SIDECAR_NAME).exists()==enabled
        if enabled: assert episode_tokens(folder/'frames.npz').meta['rows']==5
        search._close_worker_env()
    assert frame_bytes(collected[0])==frame_bytes(collected[1])
    with np.load(collected[0]) as a,np.load(collected[1]) as b:
        assert a.files==b.files
        for name in a.files:
            assert a[name].tobytes()==b[name].tobytes()


def test_sidecar_pack_sparse_resume_train_and_evaluate(tmp_path, monkeypatch):
    """Exact selected decision rows survive attachments, interrupted packing and training."""
    import os
    import signal
    import sqlite3
    from argparse import Namespace
    import prepare_decision_data as prepare
    import train_rollout_policy as trainer
    import eval_window_policy as evaluate
    from pvz_training_pack import archive_fingerprint, pack_options, pack_path
    from pack_training_rows import pack
    from pvz_recipe import write_recipe
    from pvz_research_data import decision_tokens

    torch.set_num_threads(1)
    data=tmp_path/'data'
    sources=tmp_path/'sources'
    sources.mkdir()
    connection=sqlite3.connect(sources/'sources.sqlite')
    connection.execute('CREATE TABLE sources (source TEXT PRIMARY KEY, rows BLOB)')
    expected=[]
    for seed in range(4):
        path=data/f'seed_{seed}'/'fixture'/'frames.npz'
        path.parent.mkdir(parents=True)
        frames=[]; samples=[]
        for i in range(4):
            f=dict(frame(),tick=i*60)
            if i: f['action']=dict(type='plant',packet=0,row=0,col=1) if i==1 else dict(type='wait',ticks=60)
            frames.append(f)
            rich=dict(f,v3=rich_frame()['v3'])
            sample=frame_tokens(rich,token_input='obs_v3')
            sample['features'][0,31]=np.float32(seed+i/8)
            samples.append(sample)
        write_frames(path,frames)
        meta=dict(training_rows=[1,3],context=dict(level=7))
        (path.parent/'meta.json').write_text(json.dumps(meta))
        legacy=archive_fingerprint(data,{'fixture'})
        oldrich=archive_fingerprint(data,{'fixture'},'obs_v3')
        writer=SparseTokenWriter(path.parent)
        for sample in samples: writer.append(sample)
        writer.finish(path.parent/SIDECAR_NAME,source_frame=path)
        assert archive_fingerprint(data,{'fixture'})==legacy
        assert archive_fingerprint(data,{'fixture'},'obs_v3')!=oldrich
        source=str(path.relative_to(data))
        raw=sources/'raw'/source
        raw.parent.mkdir(parents=True)
        os.link(path,raw)
        os.link(path.parent/SIDECAR_NAME,raw.parent/SIDECAR_NAME)
        connection.execute('INSERT INTO sources VALUES (?,?)',(source,np.array([1,3],np.int32).tobytes()))
        expected.extend([samples[0],samples[2]])
    connection.commit(); connection.close()
    args=Namespace(out=tmp_path/'decision_tokens',sources=sources,token_input='obs_v3')
    with pytest.raises(SystemExit) as interrupted:
        prepare.tokens(args,lambda:True)
    assert interrupted.value.code==75
    with (args.out/'values.bin').open('ab') as stream: stream.write(b'partial uncommitted row')
    prepare.tokens(args,lambda:False)
    sparse=decision_tokens(args.out,8,'obs_v3')
    for i,sample in enumerate(expected):
        for name,value in sample.items(): assert sparse.sample(i)[name].tobytes()==value.tobytes()
    with pytest.raises(ValueError,match='rows/input'): decision_tokens(args.out,9,'obs_v3')
    with pytest.raises(ValueError,match='rows/input'): decision_tokens(args.out,8,'compact')
    options=pack_options({'fixture'},obs='v3',packet_head='identity',token_input='obs_v3',
        sources=archive_fingerprint(data,{'fixture'},'obs_v3'))
    pack(data,options,quiet=True)
    packed=trainer.load(data,{'fixture'},obs='v3',packet_head='identity',token_input='obs_v3')
    for i,sample in enumerate(expected):
        for name,value in sample.items(): assert packed[-1].sample(i)[name].tobytes()==value.tobytes()
    # Existing compact pack and row plan are reusable; no dense rich copy is required.
    compact=pack_options({'fixture'},obs='v1',packet_head='identity',sources=archive_fingerprint(data,{'fixture'}))
    pack(data,compact,quiet=True)
    frozen=pack_path(data,compact)
    recipe=tmp_path/'recipe.json'; out=tmp_path/'trained'
    write_recipe(recipe,dict(schema_version=1,kind='training',data=str(data),out=str(out),labels=['fixture'],
        obs='v3',token_input='obs_v3',packet_head='identity',width=12,
        network=dict(architecture='token',layers=1,heads=3,ff_width=16),
        batch_size=2,epochs=1,max_steps=2,seed=1,validation=dict(seed=0,fraction=.25)))
    class QuietMetrics:
        def __init__(self,*a,**kw): pass
        def start(self): return self
        def update(self,**kw): pass
        def close(self): pass
    monkeypatch.setattr(trainer,'Metrics',QuietMetrics)
    monkeypatch.setattr(trainer,'set_oom_score_adj',lambda _:None)
    command=['train','--recipe',str(recipe),'--pack',str(frozen),'--tokens',str(args.out),
        '--merge-dir',str(tmp_path/'merged'),'--checkpoint-seconds','60','--device','cpu']
    monkeypatch.setattr(sys,'argv',command)
    with pytest.raises(SystemExit) as yielded:
        trainer.main(step_observer=lambda step,loss:os.kill(os.getpid(),signal.SIGUSR1))
    assert yielded.value.code==75
    saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
    assert saved['token_contract']['path']==str(args.out.resolve())
    monkeypatch.setattr(sys,'argv',command+['--resume',str(out/'resume.pt')])
    trainer.main()
    monkeypatch.setattr(sys,'argv',['eval','--recipe',str(recipe),'--pack',str(frozen),'--tokens',str(args.out),
        '--merge-dir',str(tmp_path/'merged'),'--out',str(out),'--batch-size','2'])
    evaluate.main()
    assert json.loads((out/'heldout.json').read_text())['groups']['all']['rows']==2
