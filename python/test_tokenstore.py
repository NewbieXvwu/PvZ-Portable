"""Small CPU tests of sparse blocks in the existing immutable segment pipeline."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pvz_student_tokens import TokenBuilder, frame_tokens
from pvz_research_data import SparseTokenWriter
from pvz_frame_store import write_frames
from pvz_training_pack import pack_options, archive_fingerprint
from pack_training_rows import pack
from test_compact_loader import game_frames
import train_rollout_policy as trainer


def test_tokenstore_attachment_merge_batch_and_reuse(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / 'data'
    expected = []
    for seed in range(3):
        for label in ('a', 'b'):
            folder = data / f'seed_{seed}' / label
            folder.mkdir(parents=True)
            frames = game_frames(38 if seed else 3)
            path = folder / 'frames.npz'
            write_frames(path, frames)
            writer = SparseTokenWriter(folder)
            for i, frame in enumerate(frames):
                sample = frame_tokens(frame)
                sample['features'][0, 31] = -0.0
                sample['target_indices'][-1] = 2
                writer.append(sample)
                if i + 1 < len(frames):
                    expected.append(sample)
            writer.finish(folder / 'tokens_v3.npz', source_frame=path)
    options = pack_options({'a', 'b'}, obs='v3', token_input='obs_v3', packet_head='identity',
                           sources=archive_fingerprint(data, {'a', 'b'}, 'obs_v3'))
    index = pack(data, options, quiet=True)
    with patch('pvz_token_blocks.decode', side_effect=AssertionError('merge decoded tokens')):
        loaded = trainer.read_pack(index, data)
    store = loaded[-1]
    assert set(store.arrays) == {'offsets', 'data', 'rows'}
    assert isinstance(store.arrays['data'], np.memmap)
    for i, sample in enumerate(expected):
        actual = store.sample(i)
        assert all(actual[k].tobytes() == v.tobytes() for k, v in sample.items())
    indices = torch.tensor([len(expected)-1, 0, 36, 2, 0])
    batch = store.batch(indices)
    assert batch['features'][1, 0, 31].signbit()
    assert batch['ids'].shape[0] == len(indices)
    # No segment payload is read on reuse, and tiny budgets evict decoded sparse blocks.
    with patch.object(trainer, 'read_pack', side_effect=AssertionError('reloaded payload')):
        with np.load(index) as stored:
            segments = [index.parent / name for name in stored['segments'].tolist()]
        reused = trainer.read_segments(segments, data)[-1]
    reused.cache_bytes = 1
    assert reused.sample(0)['features'].tobytes() == expected[0]['features'].tobytes()
    assert not reused.cache
    # Row options still have no new fields for CNN packs.
    assert 'token_storage' not in pack_options({'a'})


def test_tokenstore_partial_and_full_blocks():
    builder = TokenBuilder()
    samples = [frame_tokens(f) for f in game_frames(35)]
    for sample in samples:
        builder.append(sample)
    store = builder.finish()
    assert store.arrays['rows'].tolist() == list(range(32)) + [0, 1, 2]
    assert store.arrays['offsets'][-1] == len(store.arrays['data'])
    for i in (0, 31, 32, 34):
        assert store.sample(i)['features'].tobytes() == samples[i]['features'].tobytes()
