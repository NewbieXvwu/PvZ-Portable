import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from pack_training_rows import assert_payload_equal, packed_load
from pvz_frame_store import read_frames, write_frames
from pvz_llm_demo_rows import demo_paths, import_demo
from pvz_recipe import training_row_options, validate
from pvz_training_pack import archive_fingerprint, pack_options, segment_options, segment_path
from pvz_training_rows import DemoRowWeights
from test_compact_loader import game_frames
from train_rollout_policy import load
from pvz_training_rows import source_factor


def imported_archives(root):
    imported = root / 'import'
    (imported / 'receipts').mkdir(parents=True)
    rows = []
    frames = game_frames(9)
    for frame in frames:
        frame.pop('targets', None)
    frames[0]['action'] = None
    frames[3]['action_rejected'] = True
    frames[4]['action'] = None
    frames[4]['input_action'] = dict(type='shovel', row=0, col=0)
    for tag, result in zip('abcdef', (1, 1, 2, 0, 1, 1)):
        sha = tag * 64
        source = imported / 'sha256' / sha
        source.mkdir(parents=True)
        write_frames(source / 'frames.npz', frames)
        (source / 'meta.json').write_text(json.dumps(dict(
            label_source='llm_gpt_pro', verified=True, task=dict(id='same_task', seed=103),
            outcome=dict(result=result), context=dict(level=42, multiplier=2, roster=[0]))))
        receipt = dict(archive_sha256=sha, verified=tag != 'e')
        (imported / 'receipts' / f'{sha}.json').write_text(json.dumps(receipt))
        rows.append(dict(archive_sha256=sha, seed=103, result=result, verify='通过'))
        if receipt['verified']:
            import_demo(root / 'converted', source, sha, results=(0, 1, 2))
    (imported / 'merged_progress.json').write_text(json.dumps(dict(archives=rows + rows[:1])))
    return root / 'converted', frames


@pytest.mark.parametrize('cached', [False, True])
def test_task152_only_verified_wins_add_exact_decisions_and_compose_weights(tmp_path, cached):
    imported, frames = imported_archives(tmp_path)
    data = tmp_path / 'data'
    base = data / 'seed_101' / 'base' / 'frames.npz'
    base.parent.mkdir(parents=True)
    write_frames(base, game_frames(9))
    (base.parent / 'meta.json').write_text(json.dumps(dict(label_source='teacher')))
    original = packed_load(data, {'base'}, obs='v2', wait_keep=.1)
    options = pack_options({'base'}, obs='v2', wait_keep=.1,
                           sources=archive_fingerprint(data, {'base'}))
    source = segment_path(data, segment_options(options, 'base', options['sources']))
    stamp = source.stat().st_mtime_ns
    default = validate(dict(schema_version=1, kind='training', data=str(data), out=str(tmp_path / 'out'),
                            labels=['base'], obs='v2', wait_keep=.1))
    request = training_row_options(default)
    assert pack_options(**request, sources=options['sources']) == options
    assert_payload_equal(original, packed_load(data, **request))
    assert not (data / '.llm_demo_rows').exists()
    enabled = packed_load(data, {'base'}, obs='v2', wait_keep=.1, llm_demo_import=imported,
                          cache_dir=tmp_path / 'cache' if cached else None)
    assert len(enabled[2]) == len(original[2]) + 18
    assert source.stat().st_mtime_ns == stamp
    plan, observed = DemoRowWeights(4, 2), []
    def observe(path, rows):
        plan.observe(path, rows)
        observed.append((path, rows.copy()))
    weighted = load(data, {'base'}, obs='v2', wait_keep=.1, llm_demo_import=imported,
                    source_weight={'llm_gpt_pro': .5}, row_observer=observe)
    plan.apply(weighted[5])
    offset = 0
    for path, rows in observed:
        count = len(rows)
        if path.parent.name == 'base':
            assert weighted[5][offset:offset + count].numpy().tobytes() == original[5].numpy().tobytes()
            assert torch.equal(enabled[0].batch(np.arange(offset, offset + count), 'cpu'),
                               original[0].batch(np.arange(count), 'cpu'))
        else:
            assert rows.tolist() == [1, 2, 5, 6, 7, 8]
            assert read_frames(path) == frames
            assert weighted[5][offset:offset + count].tolist() == [2.] * count
            assert enabled[5][offset:offset + count].tolist() == [1.] * count
        offset += count
    assert len(observed) == 4
    assert plan.rows['llm_demo'] == 18
    assert len(demo_paths(data, imported, [0, 1, 2])) == 5


def test_task152_recipe_rejects_invalid_results():
    for results in ([], [3], [True], [1, 1], [[1]], '1'):
        with pytest.raises(ValueError, match='llm_demo_results'):
            validate(dict(schema_version=1, kind='training', data='/data', out='/out',
                          labels=['base'], llm_demo_results=results))


def test_clean_b_only_uses_verified_demo_and_recorded_r52_or_gptbc_prior(tmp_path, monkeypatch):
    imported, _ = imported_archives(tmp_path)
    data = tmp_path / 'shared'
    checkpoints = dict(old='/runs/round_000051/student/model.pt',
                       r52='/runs/round_000052/student/model.pt',
                       gpt='/artifacts/student/gptbc_s1/model.pt', unknown='/student/current/model.pt')
    for label, checkpoint in checkpoints.items():
        path = data / 'seed_207' / label
        path.mkdir(parents=True)
        write_frames(path / 'frames.npz', game_frames(9))
        (path / 'meta.json').write_text(json.dumps(dict(label_source='teacher', task=dict(checkpoint=checkpoint))))
    labels = set(checkpoints)
    assert source_factor(dict(label_source='teacher', task=dict(policy='round_000052')), dict(default=0., teacher_r52=1.)) == 1.
    packed_load(data, labels, obs='v2', llm_demo_import=imported)
    from pvz_training_pack import pack_path
    pinned = pack_path(data, pack_options(labels, obs='v2',
        sources=archive_fingerprint(data, labels, imported), llm_demo_import=imported))
    import train_rollout_policy as trainer
    monkeypatch.setattr(trainer, 'archive_fingerprint', lambda *a, **kw: pytest.fail('do not reselect the round pack'))
    spans = []
    b = load(data, labels, obs='v2', llm_demo_import=imported, pack_file=pinned,
        source_weight=dict(default=0., teacher_r52=1., teacher_gptbc=1., llm_demo=1.),
        row_observer=lambda path, rows: spans.append((path.parent.name, len(rows))))
    a = load(data, labels, obs='v2', llm_demo_import=imported, pack_file=pinned, source_weight=dict(llm_demo=0.))
    offset = 0
    for label, count in spans:
        allowed = label in ('r52', 'gpt', 'llm_demo_decisions')
        assert (b[5][offset:offset + count] > 0).all().item() is allowed
        if not allowed:
            assert not b[5][offset:offset + count].any()
        assert (a[5][offset:offset + count] > 0).all().item() is (label != 'llm_demo_decisions')
        offset += count


def test_clean_b_demo_conversion_filters_unmasked_fog_and_empty_packet_plant(tmp_path):
    imported_archives(tmp_path)
    source = tmp_path / 'import/sha256' / ('a' * 64)
    meta = json.loads((source / 'meta.json').read_text())
    meta['task']['level'] = 32
    (source / 'meta.json').write_text(json.dumps(meta))
    target = tmp_path / 'new'
    assert import_demo(target, source, 'a' * 64) is None
    meta['environment_task'] = dict(fog_observation='human_v2')
    (source / 'meta.json').write_text(json.dumps(meta))
    frames = read_frames(source / 'frames.npz')
    frames[5]['packets'][6][1] = -1
    write_frames(source / 'frames.npz', frames)
    converted = import_demo(target, source, 'a' * 64)
    selection = json.loads((converted.parent / 'meta.json').read_text())['training_rows']
    assert selection == [1, 2, 5, 7, 8]


def test_clean_b_unified_import_exports_demo_rows_after_verification(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import import_llm_run as importer
    imported_archives(tmp_path)
    source = tmp_path / 'import/sha256' / ('a' * 64)
    meta = json.loads((source / 'meta.json').read_text())
    meta['task'].update(level=42, deck=[1, 33, 0, 2, 3, 4])
    meta['outcome'].update(won=True, final_wave=10)
    meta.update(schema='episode_archive_v2', schema_version=2, operations=1, frame_count=9)
    (source / 'meta.json').write_text(json.dumps(meta))
    (source / 'ticks.jsonl').write_text('{}\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"tasks": []}')
    engine = tmp_path / 'engine'
    engine.write_bytes(b'engine')
    monkeypatch.setattr(importer, 'verify_copy', lambda *args: dict(verified=True))
    result = importer.run(source, tmp_path / 'dest', manifest, tmp_path, engine,
        executor_factory=ThreadPoolExecutor, verify_workers=1, llm_demo_data=tmp_path / 'demos')
    assert result['rows'][0]['verify'] == '通过'
    paths = demo_paths(tmp_path / 'view', tmp_path / 'demos')
    assert len(paths) == 1
    assert json.loads((paths[0].parent / 'meta.json').read_text())['llm_demo_sha256']
