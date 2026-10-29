"""Bounded CPU fixtures for task162's sixteen-job decision study."""
import json
from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def test_decision_architectures_forward_backward_and_load(tmp_path):
    from pvz_rollout_policy import create_student, load_student, act, ranked_plants, WAIT
    from pvz_student_tokens import frame_tokens, collate_tokens
    from pvz_value_net import BOARD_CHANNELS, SCALARS
    from pvz_student_observation import OBS_EXTRA
    torch.set_num_threads(1)
    frame = dict(sun=100, wave=0, tick=0, scene={}, lanes=list(range(5)), plants=[], zombies=[],
                 mowers=[], packets=[[0, 0, True, 0, 100, 100]])
    tokens = collate_tokens([frame_tokens(frame) for _ in range(5)])
    board = torch.randn(5, BOARD_CHANNELS + OBS_EXTRA['v2timer'][0], 6, 10)
    scalar = torch.zeros(5, SCALARS + 2 + 9)
    scalar[1, SCALARS+2] = 1; scalar[2, SCALARS+3] = 1
    scalar[3, SCALARS+4] = 1; scalar[4, SCALARS+5] = 1
    info = torch.zeros(5, 10, 4)
    for factory in ('resnet6', 'TerrainExperts', 'MoEPolicy', 'token_small', 'token_medium', 'cnn_entity'):
        arch = 'pvz_decision_models:' + factory
        net = create_student(arch, width=12, obs='v2timer', packet_head='identity', pool='meanmax', row_pool=True)
        extra = dict(tokens=tokens) if getattr(net, 'requires_tokens', False) else {}
        out = net(board, scalar, torch.zeros(5, dtype=torch.long), packet_info=info, **extra)
        sum(value.square().mean() for value in out.values()).backward()
        assert all(torch.isfinite(value).all() for value in out.values())
        assert out['cell'].shape == (5, 54)
        net.eval()
        path = tmp_path / (factory + '.pt')
        torch.save(dict(state_dict=net.state_dict(), arch=arch, width=12, obs='v2timer', packet_head='identity',
                        pool='meanmax', row_pool=True), path)
        loaded = load_student(path)
        with torch.no_grad():
            expected = net(board, scalar, torch.zeros(5, dtype=torch.long), packet_info=info, **extra)
            actual = loaded(board, scalar, torch.zeros(5, dtype=torch.long), packet_info=info, **extra)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        from pvz_student_observation import StudentHistory
        legal = [dict(type='plant', packet=0, row=0, col=col) for col in (0, 1)]
        history = StudentHistory('v2timer')
        assert len(ranked_plants(loaded, frame, None, legal, history=history)) == 2
        assert act(loaded, frame, None, legal, history=history) in [*legal, dict(WAIT)]


def test_decision_sparse_tokens_preserve_exact_r42_rows(tmp_path):
    import sqlite3
    from pvz_frame_store import write_frames
    from pvz_student_tokens import frame_tokens
    from pvz_research_data import SparseTokens
    from prepare_decision_data import tokens
    source = 'seed_1/example/frames.npz'
    frame = dict(sun=100, wave=0, tick=0, scene={}, lanes=list(range(5)), plants=[[0, 1, 0, 50, 100]], zombies=[],
                 mowers=[[0, 1]], packets=[[0, 0, True, 0, 100, 100]])
    frames = [{**frame, 'tick': i*60} for i in range(4)]
    path = tmp_path / 'sources/raw' / source
    path.parent.mkdir(parents=True)
    write_frames(path, frames)
    db = sqlite3.connect(tmp_path / 'sources/sources.sqlite')
    db.execute('CREATE TABLE sources (source TEXT, rows BLOB)')
    db.execute('INSERT INTO sources VALUES (?, ?)', (source, np.array([1, 3], np.int32).tobytes()))
    db.commit(); db.close()
    args = Namespace(sources=tmp_path / 'sources', out=tmp_path / 'tokens')
    tokens(args, lambda: False)
    store = SparseTokens(args.out)
    for i, row in enumerate((1, 3)):
        expected = frame_tokens(frames[row-1])
        for key, value in expected.items():
            np.testing.assert_array_equal(store.sample(i)[key], value)
    # A restart truncates only uncommitted tails, then reuses the completed cursor.
    with (args.out / 'values.bin').open('ab') as handle:
        handle.write(b'broken tail')
    tokens(args, lambda: False)
    assert SparseTokens(args.out).meta['rows'] == 2


def test_decision_queue_honors_priority_and_dependencies(tmp_path):
    from gpu_window_runner import Runner
    queue = tmp_path / 'queue'; queue.mkdir()
    args = Namespace(queue=queue, production_log=tmp_path / 'prod.log', memory_max_gib=5)
    for name, priority, deps in [('later', 50, []), ('first', 10, []), ('blocked', 0, [str(tmp_path/'missing')])]:
        (queue / (name + '.json')).write_text(json.dumps(dict(id=name, priority=priority, host_gib=5, gpu_gib=4,
            recipe=str(tmp_path / 'recipe'), out=str(tmp_path/name), command=['python'], depends_on=deps)))
    assert [job['id'] for job, state in Runner(args).candidates()] == ['first', 'later']


def test_decision_split_excludes_complete_decks_and_freezes_sources(tmp_path):
    from prepare_decision_data import split
    data = tmp_path / 'raw'
    sources = []
    for kind in range(10):
        source = f'seed_{kind:02d}/example/frames.npz'
        path = data / source; path.parent.mkdir(parents=True)
        path.write_bytes(b'archive')
        (path.parent / 'meta.json').write_text(json.dumps(dict(context=dict(level=7), task=dict(deck=[kind]))))
        sources.append(source)
    packs = tmp_path / 'packs'; packs.mkdir()
    np.savez(packs/'segment.npz', source=sources, source_archive=np.arange(10, dtype=np.int32), source_row=np.ones(10, np.int32))
    np.savez(packs/'index.npz', rows=10, segments=['segment.npz'])
    merged = tmp_path / 'merge'; merged.mkdir()
    kinds = np.full((10, 10), -1, np.int8); kinds[:, 0] = np.arange(10)
    np.save(merged/'packet_col0.npy', kinds)
    out = tmp_path / 'split'
    split(Namespace(out=out, pack=packs/'index.npz', data=data, merge_dir=merged), lambda: False)
    result = json.loads((out/'ready.json').read_text())
    assert result['unseen_rows'] == 1 and len(result['heldout_decks']) == 1
    assert np.load(out/'unseen.npy').sum() == 1
    assert (out/'raw'/sources[0]).stat().st_ino == (data/sources[0]).stat().st_ino


def test_decision_microbatch_preserves_weighted_objective(tmp_path, monkeypatch):
    import test_identity_training as fixtures
    import train_rollout_policy as trainer
    from pack_training_rows import packed_load
    from pvz_recipe import write_recipe
    # Isolate gradient accumulation from Adam's amplification of near-zero roundoff.
    monkeypatch.setattr(torch.optim, 'AdamW', lambda params, **kw:
                        torch.optim.SGD(params, lr=kw['lr'], weight_decay=kw['weight_decay']))
    fixtures.IdentityTrainingTest().fixture(tmp_path/'data')
    packed_load(tmp_path/'data', {'example'}, obs='v2', packet_head='identity', cells=True, rebuild=True)
    base = dict(schema_version=1, kind='training', data=str(tmp_path/'data'), labels=['example'],
        width=8, seed=7, obs='v2', packet_head='identity', epochs=1, max_steps=2, batch_size=16,
        packet_balance=.5, cell_soft_weight=3, spread_weight=8)
    losses, parameters = [], []
    for micro in (None, 4):
        out = tmp_path/str(micro)
        path = tmp_path/(str(micro)+'.json')
        write_recipe(path, dict(base, out=str(out), micro_batch_size=micro))
        monkeypatch.setattr(sys, 'argv', ['train', '--recipe', str(path)])
        recorded = []
        trainer.main(step_observer=lambda _, loss: recorded.append(loss))
        losses.append(recorded)
        parameters.append(torch.load(out/'model.pt', weights_only=True)['state_dict'])
    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-6, atol=1e-6)
    for name, value in parameters[0].items():
        torch.testing.assert_close(value, parameters[1][name], rtol=1e-4, atol=1e-5)


def test_decision_manifest_enqueues_two_pinned_seeds_per_arm(tmp_path):
    from queue_decision_study import enqueue
    from pvz_recipe import write_recipe
    state = tmp_path/'state'; (state/'corpus').mkdir(parents=True)
    (state/'corpus/corpus.json').write_text(json.dumps(dict(rows=23631235, pack='/frozen/index.npz', merge_dir='/frozen/merge')))
    recipe = tmp_path/'base.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', obs='v2timer', packet_head='identity',
        data=str(tmp_path/'data'), labels=['example'], out=str(tmp_path/'old')))
    manifest = json.loads((ROOT/'configs/decision_r42.json').read_text())
    revision = 'a' * 40
    result = enqueue(state, recipe, revision, ROOT, sys.executable, manifest)
    jobs = [json.loads(path.read_text()) for path in (state/'queue').glob('*.json')]
    trained = [job for job in jobs if job.get('kind') != 'prepare']
    assert len(trained) == result['training_jobs'] == 16 and len(jobs) == 18
    assert all(job['revision'] == revision and job['code_root'] == str(ROOT) for job in jobs)
    for arm in manifest['arms']:
        group = [job for job in trained if job['id'].startswith('r42_'+arm['name']+'_s')]
        assert {job['seed'] for job in group} == {1, 2}
        assert all('--deck-holdout' in job['command'] for job in group)
        assert all(('--tokens' in job['command']) == bool(arm.get('tokens')) for job in group)
