"""task132: tiny packed-row checks only; never starts a trainer or game."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from pack_training_rows import assert_payload_equal, pack, pack_monolithic
from pvz_frame_store import write_frames
from pvz_recipe import validate, training_row_options, load_recipe
from pvz_training_pack import archive_fingerprint, pack_options, pack_path
from pvz_training_rows import DemoRowWeights
from test_compact_loader import game_frames
from train_rollout_policy import read_pack


def recipe(**updates):
    return validate(dict(schema_version=1, kind='training', data='/data', out='/out',
                         labels=['human', 'continuation', 'ordinary_demo_name', 'llm', 'counterfactual', 'legacy'],
                         **updates))


@pytest.mark.parametrize('segmented', [False, True])
def test_defaults_keep_pack_tensors_and_only_provenance_targets_change(tmp_path, segmented):
    base = recipe()
    assert base['demo_row_weight'] == base['demo_start_row_weight'] == 1.0
    assert training_row_options(base) == training_row_options(recipe(demo_row_weight=4, demo_start_row_weight=2))
    data = tmp_path / 'frames'
    metadata = {
        'human': dict(label_source='human'),
        'continuation': dict(label_source='teacher', demo_archive='/raw/demo', demo_decision=0),
        'ordinary_demo_name': dict(label_source='teacher', task=dict(task_id='demo_roof_49')),
        'llm': dict(label_source='llm_agent'),
        'counterfactual': dict(label_source='counterfactual', demo_archive='/raw/demo'),
        'legacy': dict(demo_archive='/raw/demo'),  # historical missing source defaults to teacher
    }
    for label, meta in metadata.items():
        path = data / 'seed_101' / label / 'frames.npz'
        path.parent.mkdir(parents=True)
        write_frames(path, game_frames(9))
        (path.parent / 'meta.json').write_text(json.dumps(meta))
    options = pack_options(base['labels'], obs='v2', wait_keep=.5,
                           label_weight={'human': 1.25}, sources=archive_fingerprint(data, base['labels']))
    path = pack(data, options) if segmented else pack_monolithic(data, options, out=tmp_path / 'full.npz')
    stamp = path.stat().st_mtime_ns
    source_weight = {'human': .5, 'teacher': .75}
    reference = read_pack(path, data, source_weight=source_weight)
    default = DemoRowWeights()
    unchanged = read_pack(path, data, default.observe, source_weight)
    default.apply(unchanged[5])
    assert_payload_equal(reference, unchanged)
    # Default provenance hook itself never touches metadata.
    with patch.object(Path, 'read_text', side_effect=AssertionError('default metadata read')):
        default.observe(data / 'missing' / 'frames.npz', np.arange(3))
    assert not default.spans

    plan, expected = DemoRowWeights(4, 2), []
    def observer(source, rows):
        plan.observe(source, rows)
        factor = 4 if source.parent.name == 'human' else 2 if source.parent.name in ('continuation', 'legacy') else 1
        expected.extend([factor] * len(rows))
    weighted = read_pack(path, data, observer, source_weight)
    plan.apply(weighted[5])
    torch.testing.assert_close(weighted[5], reference[5] * torch.tensor(expected), rtol=0, atol=0)
    assert_payload_equal(reference, (*weighted[:5], reference[5], *weighted[6:]))
    assert path.stat().st_mtime_ns == stamp
    assert plan.metadata()['participates_in_cap'] is False
    assert sum(plan.rows.values()) < len(weighted[5])


def test_each_factor_is_independent_and_invalid_recipe_values_are_rejected(tmp_path):
    for human, continuation in ((4, 1), (1, 2), (0, 2)):
        plan = DemoRowWeights(human, continuation)
        plan.observe(tmp_path / 'human' / 'frames.npz', [1, 5], dict(label_source='human'))
        plan.observe(tmp_path / 'teacher' / 'frames.npz', [3], dict(label_source='teacher', demo_archive='/demo'))
        plan.observe(tmp_path / 'ordinary' / 'frames.npz', [2], dict(label_source='teacher'))
        weights = torch.tensor([2., 3., 4., 5.])  # already includes packet/spread/level factors
        plan.apply(weights)
        assert weights.tolist() == [2*human, 3*human, 4*continuation, 5]
    for name in ('demo_row_weight', 'demo_start_row_weight'):
        for value in (-1, True, float('nan'), float('inf'), '4'):
            with pytest.raises(ValueError):
                recipe(**{name: value})
