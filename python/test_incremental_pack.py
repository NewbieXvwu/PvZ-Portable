"""task49: incremental labels preserve source order, tensors and two epochs exactly."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from pack_training_rows import assert_payload_equal, pack, pack_monolithic, verify
from pvz_frame_store import write_frames
from pvz_recipe import write_recipe
from pvz_training_pack import archive_fingerprint, pack_options
from test_compact_loader import game_frames
import train_rollout_policy as trainer


def fixture(data, labels):
    for seed in (100, 101, 102):
        for label in labels:
            target = data / f'seed_{seed}' / label / 'frames.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            write_frames(target, game_frames(18))
            (target.parent / 'meta.json').write_text(json.dumps({'context': {'multiplier': 1., 'roster': [0, 1]}}))


def options(data, labels, **kwargs):
    return pack_options(labels, sources=archive_fingerprint(data, labels), **kwargs)


def test_incremental_reuses_segments_and_parallel_matches_full():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        data = root / 'data'
        fixture(data, ['b'])
        settings = dict(wait_keep=.1, obs='v2', cells=True, aux_horizon=2,
                        packet_head='identity', with_context=True, label_weight={'b': 2.5})
        first = pack(data, options(data, ['b'], **settings), root / 'cache')
        segments = list((data / 'packs/segments').glob('*.npz'))
        stamp = [(p, p.stat().st_mtime_ns, p.read_bytes()) for p in segments]
        fixture(data, ['a', 'c'])  # a interleaves BEFORE b in every seed
        contract = options(data, ['a', 'b', 'c'], **settings)
        index = pack(data, contract, root / 'cache', precompute_workers=2)
        assert all(p.stat().st_mtime_ns == t and p.read_bytes() == content for p, t, content in stamp)
        full = pack_monolithic(data, contract, out=root / 'full.npz')
        assert_payload_equal(trainer.read_pack(full), trainer.read_pack(index))
        verify(data, contract, index)
        # Old pinned snapshot still reads the same data after a larger index is published.
        assert len(trainer.read_pack(first)[2]) < len(trainer.read_pack(index)[2])
        # Changing b's weight only creates b's new segment.
        before = set((data / 'packs/segments').glob('*.npz'))
        pack(data, options(data, ['a', 'b', 'c'], **{**settings, 'label_weight': {'b': 3.5}}), root / 'cache')
        assert len(set((data / 'packs/segments').glob('*.npz')) - before) == 1


def train_pair(root, data, contract, device='cpu'):
    """The real trainer, identical RNG/kernel/split/optimizer, full two-epoch traces and weights."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    index = pack(data, contract, root / 'cache')
    full = pack_monolithic(data, contract, root / 'cache', out=root / 'full.npz')
    assert_payload_equal(trainer.read_pack(full), trainer.read_pack(index))
    traces, models, reports = [], [], []
    try:
        for name, path in [('full', full), ('incremental', index)]:
            out = root / name
            recipe = dict(schema_version=1, kind='training', data=str(data), labels=contract['labels'],
                          out=str(out), device=device, threads=1, epochs=2, width=8, seed=7,
                          wait_keep=contract['wait_keep'], obs=contract['obs'],
                          packet_head='identity', pool='meanmax', row_pool=True,
                          packet_balance=.5, cell_soft_weight=3., spread_weight=8.,
                          label_weight=contract['label_weight'], lr_schedule='cosine',
                          source_row_report=True, batch_size=32, validation_batch_size=32,
                          validation=dict(seed=0, fraction=.34))
            recipe_path = root / f'{name}.json'
            write_recipe(recipe_path, recipe)
            losses = []
            def load(_data, **kwargs):
                return trainer.read_pack(path, data, kwargs.get('row_observer'))
            with patch.object(sys, 'argv', ['trainer', '--recipe', str(recipe_path)]), \
                 patch.object(trainer, 'load', load):
                trainer.main(step_observer=lambda step, loss: losses.append(loss))
            traces.append(losses)
            models.append(torch.load(out / 'model.pt', map_location='cpu', weights_only=False))
            reports.append((out / 'source_rows.json').read_bytes())
        assert traces[0] == traces[1] and traces[0]
        assert len(models[0]['history']) == len(models[1]['history']) == 2
        for key, value in models[0]['state_dict'].items():
            assert torch.equal(value.view(torch.uint8), models[1]['state_dict'][key].view(torch.uint8)), key
        assert reports[0] == reports[1]
        for a, b in zip(models[0]['history'], models[1]['history']):
            assert {k: v for k, v in a.items() if k != 'seconds'} == {k: v for k, v in b.items() if k != 'seconds'}
        return dict(passed=True, epochs=2, steps=len(traces[0]), max_loss_error=0,
                    model_tensors=len(models[0]['state_dict']), source_rows_identical=True,
                    rows=len(trainer.read_pack(index)[2]), device=device)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def test_two_epoch_training_is_bitwise_identical():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        fixture(root / 'data', ['a', 'b'])
        train_pair(root, root / 'data', options(root / 'data', ['a', 'b'], wait_keep=.1,
                                               obs='v2', cells=True, packet_head='identity'))
