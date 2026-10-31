"""Prepared deploy18 collection wiring; tiny isolated engine, no production services."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]


def test_obsrec_queue_forwards_only_explicit_flag(tmp_path):
    import curriculum_queue as q
    args = SimpleNamespace(python=Path(sys.executable), rollout_policy=tmp_path/'model',
        value=tmp_path/'value', static_value=tmp_path/'static', executable=tmp_path/'engine',
        resource_dir=tmp_path, frames=tmp_path/'frames', out=tmp_path/'out', workers=1)
    for demo in (None, 'demonstration'):
        old = q.command_for(args, tmp_path/'manifest', 'task', 0, 1, 'label', demo_archive=demo)
        args.obs_v3 = True
        new = q.command_for(args, tmp_path/'manifest', 'task', 0, 1, 'label', demo_archive=demo)
        assert new.count('--obs-v3') == 1
        assert [a for a in new if a != '--obs-v3'] == old
        args.obs_v3 = False
        assert q.command_for(args, tmp_path/'manifest', 'task', 0, 1, 'label', demo_archive=demo) == old


@pytest.mark.parametrize('demo', [False, True])
def test_obsrec_compact_bytes_and_teacher_roots(tmp_path, monkeypatch, demo):
    import torch
    import search_play as s
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from pvz_rollout_policy import RolloutNet, export_student
    from pvz_frame_store import frame_bytes
    from pvz_research_data import episode_tokens
    exe = ROOT.parent/'studentv3net/build-studentv3net/pvz-portable'
    if not exe.is_file():
        pytest.skip('existing isolated task163 engine unavailable')
    torch.set_num_threads(1)
    model = RolloutNet(8).eval()
    weights = tmp_path/'model.bin'; export_student(model, weights)
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'fixture model in worker state')
    deck = (0, 1, 2, 3, 4, 5)
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(task_id='fixture',level=7,deck=deck)])))
    resource = Path.home()/'.cache/pvz-research-resources'
    archive = None
    if demo:
        from human_demo import Recorder
        archive = tmp_path/'demo'
        with EventWaitEnv(resource, executable=exe) as env:
            recorder = Recorder(env, archive, 'fixture', deck,
                TaskSpec(level=7,seed=157,playthrough=2,profile=profile_for_deck(deck)))
            recorder.step(dict(type='wait',ticks=60)); recorder.close()
    monkeypatch.setattr(s, 'MAX_ACTIONS', 3)
    monkeypatch.setattr(EventWaitEnv, 'load_value_net', lambda *a,**k:None)
    searches = []
    def search(env, obs, action, args, seed, stats):
        assert ('v3' in obs) == args.obs_v3
        assert 'v3' not in env.observe()  # never serialize tokens inside CNN teacher rollouts
        searches.append(args.obs_v3)
        return action
    monkeypatch.setattr(s, '_search', search)
    outputs = []
    for enabled in (False, True):
        args = SimpleNamespace(manifest=manifest,task_id='fixture',resource_dir=resource,
            executable=exe,obs_v3=enabled,frames=tmp_path/str(enabled),frames_label='fixture',
            rollout_policy=checkpoint,dagger_beta=1.,demo_archive=archive,demo_decision=2)
        s._W.clear(); s._W.update(fast=model,rollout_file=str(weights),value_files=['fixture']*2)
        try:
            result = s._play(dict(args=args,arm='fast_search',seed=157))
        finally:
            s._close_worker_env()
        path = args.frames/'seed_157/fixture_fast_search/frames.npz'
        outputs.append(path)
        attachment = episode_tokens(path)
        meta = json.loads((path.parent/'meta.json').read_text())
        if enabled:
            assert attachment.meta['rows'] == 4
            assert result['obs_v3_bytes'] == (path.parent/'tokens_v3.npz').stat().st_size
            assert meta['observation_attachment']['bytes'] == result['obs_v3_bytes']
        else:
            assert attachment is None and 'obs_v3_bytes' not in result
    assert searches == [False]*3+[True]*3
    assert frame_bytes(outputs[0]) == frame_bytes(outputs[1])
    with np.load(outputs[0]) as a, np.load(outputs[1]) as b:
        assert a.files == b.files
        assert all(a[name].tobytes() == b[name].tobytes() for name in a.files)


def test_obsrec_block_codec_preserves_float32_and_targets():
    from pvz_token_blocks import encode, decode
    from pvz_student_tokens import frame_tokens
    raw = np.zeros((67,38),np.float32); raw[:,3:6] = -1
    raw[1,0] = 1; raw[2:56,0] = 2; raw[56:66,0] = 8
    raw[56,1] = 1; raw[66,:6] = [3,1,0,0,0,2]
    raw[0,37] = -0.; raw[66,6:10] = [.1,-.2,np.inf,np.nan]
    sample = frame_tokens(dict(scene={},v3=dict(version=3,tokens=raw)),token_input='obs_v3')
    compressed, _ = encode([sample]*3)
    store = decode(compressed)
    for i in (2,0,1):
        for key, expected in sample.items():
            actual = store.sample(i)[key]
            assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
