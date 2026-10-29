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
from pvz_llm_demo_rows import DEPLOY14_ENGINE, demo_paths
from pvz_recipe import training_row_options, validate
from pvz_training_pack import archive_fingerprint, pack_options, segment_options, segment_path
from pvz_training_rows import DemoRowWeights
from test_compact_loader import game_frames
from train_rollout_policy import load


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
        receipt = dict(archive_sha256=sha, verified=tag != 'e',
                       context=dict(engine=DEPLOY14_ENGINE if tag != 'f' else 'other_engine'))
        (imported / 'receipts' / f'{sha}.json').write_text(json.dumps(receipt))
        rows.append(dict(archive_sha256=sha, seed=103, result=result, verify='通过'))
    (imported / 'merged_progress.json').write_text(json.dumps(dict(archives=rows + rows[:1])))
    return imported, frames


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
    assert len(enabled[2]) == len(original[2]) + 12
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
    assert len(observed) == 3
    assert plan.rows['llm_demo'] == 12
    assert len(demo_paths(data, imported, [0, 1, 2])) == 4


def test_task152_recipe_rejects_invalid_results():
    for results in ([], [3], [True], [1, 1], [[1]], '1'):
        with pytest.raises(ValueError, match='llm_demo_results'):
            validate(dict(schema_version=1, kind='training', data='/data', out='/out',
                          labels=['base'], llm_demo_results=results))
