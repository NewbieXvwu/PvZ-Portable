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


def test_sidecar_collection_switch_keeps_compact_bytes(tmp_path,monkeypatch):
    from argparse import Namespace
    from pvz_rollout_policy import RolloutNet,export_student
    import search_play as search
    torch.set_num_threads(1)
    net=RolloutNet(8).eval()
    model=tmp_path/'model.bin';export_student(net,model)
    manifest=tmp_path/'tasks.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(task_id='fixture',level=7,deck=[0,1,2,3,4,5])])))
    monkeypatch.setattr(search,'MAX_ACTIONS',4)
    collected=[]
    for enabled in (False,True):
        target=tmp_path/str(enabled)
        args=Namespace(manifest=manifest,task_id='fixture',resource_dir=Path.home()/'.cache/pvz-research-resources',
            executable=ROOT/'build-studentv3net/pvz-portable',obs_v3=enabled,frames=target,frames_label='fixture',
            rollout_policy=tmp_path/'checkpoint.pt',dagger_beta=1.)
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
