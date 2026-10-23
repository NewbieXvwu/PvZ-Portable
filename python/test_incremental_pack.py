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
from verify_training_segments import train_pair
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



def test_two_epoch_training_is_bitwise_identical():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        fixture(root / 'data', ['a', 'b'])
        train_pair(root, root / 'data', options(root / 'data', ['a', 'b'], wait_keep=.1,
                                               obs='v2', cells=True, packet_head='identity'))
