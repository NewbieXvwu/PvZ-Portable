"""task49: incremental labels preserve source order, tensors and two epochs exactly."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
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


def test_incremental_reuses_segments_and_parallel_matches_full(capsys):
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
        # Tiny fixtures use well below 2 GiB for both children. Exercise the real spawn
        # branch even when unrelated production jobs make the conservative cap choose 1.
        original_read = Path.read_text
        def meminfo(path, *args, **kwargs):
            if path == Path('/proc/meminfo'):
                return 'MemAvailable: 67108864 kB\n'
            return original_read(path, *args, **kwargs)
        with patch.object(Path, 'read_text', meminfo):
            index = pack(data, contract, root / 'cache', precompute_workers=2)
        summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
        assert summaries[-1]['workers'] == 2 and summaries[-1]['built'] == 2
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


def _merge_oracle(paths):
    """The old read_segments semantics, written out plainly: load everything, concat per chunk."""
    import torch
    parts, chunks = [], []
    for index, path in enumerate(paths):
        parts.append(trainer.read_pack(path))
        with np.load(path, allow_pickle=False) as stored:
            sources = stored['source'].tolist()
            counts = np.bincount(stored['source_archive'], minlength=len(sources))
        start = 0
        for source, count in zip(sources, counts):
            stop = start + int(count)
            chunks.append((source, index, start, stop))
            start = stop
    chunks.sort(key=lambda chunk: str(Path(chunk[0]).parent))

    def rows_of(column):
        return np.concatenate([np.asarray(parts[index][column])[start:stop]
                               for _, index, start, stop in chunks])

    def csr_of(column, prefix):
        store = [part[column] for part in parts]
        counts = np.concatenate([np.diff(store[index].offsets[start:stop + 1])
                                 for _, index, start, stop in chunks])
        plane_base = np.cumsum([0] + [len(part[column].planes) for part in parts])
        plane_of = np.concatenate([
            np.where((np.full(stop - start, -1, np.int32) if store[index].plane_of is None
                      else store[index].plane_of[start:stop]) >= 0,
                     (np.full(stop - start, -1, np.int32) if store[index].plane_of is None
                      else store[index].plane_of[start:stop]) + plane_base[index], -1).astype(np.int32)
            for _, index, start, stop in chunks])
        return dict(indices=np.concatenate([store[index].indices[
                                                store[index].offsets[start]:store[index].offsets[stop]]
                                            for _, index, start, stop in chunks]),
                    values=np.concatenate([store[index].values[
                                               store[index].offsets[start]:store[index].offsets[stop]]
                                           for _, index, start, stop in chunks]),
                    offsets=np.concatenate([[0], np.cumsum(counts.astype(np.int64))]),
                    planes=[plane for part in parts for plane in part[column].planes],
                    plane_of=plane_of if any(part[column].planes for part in parts) else None)

    def entries_of(column, field):
        return np.concatenate([getattr(parts[index][column], field)[
            parts[index][column].offsets[start]:parts[index][column].offsets[stop]]
            for _, index, start, stop in chunks])

    extra = []
    for column in range(9, len(parts[0])):
        if isinstance(parts[0][column], trainer.PacketStore):
            extra.append(('packets', [np.concatenate([parts[index][column].columns[c][start:stop]
                                                      for _, index, start, stop in chunks])
                                      for c in range(4)]))
        else:
            extra.append(('dense', rows_of(column)))
    return dict(boards=csr_of(0, 'board'), scalars=csr_of(1, 'scalar'),
                plant=rows_of(2), packet=rows_of(3), cell=rows_of(4), weight=rows_of(5),
                seed=rows_of(6),
                value_counts=np.concatenate([parts[index][7].counts[start:stop]
                                             for _, index, start, stop in chunks]),
                value_slots=entries_of(7, 'slots'), value_values=entries_of(7, 'values'),
                cand=None if parts[0][8] is None else (
                    np.concatenate([parts[index][8].counts[start:stop]
                                    for _, index, start, stop in chunks]),
                    entries_of(8, 'entries')),
                extra=extra,
                tensors=(torch.from_numpy(rows_of(2)), torch.from_numpy(rows_of(3)),
                         torch.from_numpy(rows_of(4)), torch.from_numpy(rows_of(5))))


def _assert_bitwise(result, oracle):
    for store, expected in ((result[0], oracle['boards']), (result[1], oracle['scalars'])):
        assert store.indices.dtype == expected['indices'].dtype
        assert np.array_equal(store.indices, expected['indices'])
        assert np.array_equal(store.values, expected['values'])
        assert np.array_equal(store.offsets, expected['offsets'])
        assert np.array_equal(store.plane_of, expected['plane_of'])
        assert len(store.planes) == len(expected['planes'])
        for (cells, values), (want_cells, want_values) in zip(store.planes, expected['planes']):
            assert np.array_equal(cells, want_cells) and np.array_equal(values, want_values)
    for index, name in ((2, 'plant'), (3, 'packet'), (4, 'cell'), (5, 'weight')):
        assert result[index].dtype == oracle['tensors'][index - 2].dtype
        assert np.array_equal(result[index].numpy(), oracle['tensors'][index - 2].numpy())
    assert np.array_equal(result[6], oracle['seed'])
    assert np.array_equal(result[7].counts, oracle['value_counts'])
    assert np.array_equal(result[7].slots, oracle['value_slots'])
    assert np.array_equal(result[7].values, oracle['value_values'])
    if oracle['cand'] is not None:
        assert np.array_equal(result[8].counts, oracle['cand'][0])
        assert np.array_equal(result[8].entries, oracle['cand'][1])
    for offset, (kind, payload) in enumerate(oracle['extra']):
        actual = result[9 + offset]
        if kind == 'packets':
            for column, want in zip(actual.columns, payload):
                assert np.array_equal(np.asarray(column), want)
        else:
            assert np.array_equal(actual.numpy(), payload)


def test_read_segments_per_segment_copy_is_bitwise_including_spill(tmp_path):
    """task61: the two-pass merge matches load-all-and-concat on every raw column,
    with plain arrays and with _spilled_array targets, and observes rows in the same order."""
    data = tmp_path / 'data'
    fixture(data, ['a', 'b'])
    contract = options(data, ['a', 'b'], packet_head='identity', aux_horizon=3)
    packed = pack(data, contract, tmp_path / 'cache', precompute_workers=2)
    with np.load(packed, mmap_mode='r') as stored:
        paths = [packed.parent / name for name in stored['segments'].tolist()]
    oracle = _merge_oracle(paths)
    observed = []
    for kwargs in ({}, {'data': data}):     # spill root = corpus root, like the trainer
        observer = ((lambda path, rows: observed.append((str(path.relative_to(data)), rows.copy())))
                    if kwargs else None)
        result = trainer.read_segments(paths, row_observer=observer, **kwargs)
        _assert_bitwise(result, oracle)
    want = []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as stored:
            sources = stored['source'].tolist()
            counts = np.bincount(stored['source_archive'], minlength=len(sources))
            source_row = np.asarray(stored['source_row'])
        start = 0
        for source, count in zip(sources, counts):
            if count:
                want.append((source, source_row[start:start + count]))
            start += count
    want.sort(key=lambda item: str(Path(item[0]).parent))
    assert [(a, b.tolist()) for a, b in observed] == [(a, b.tolist()) for a, b in want]


ROUND2_PACK = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/frames/packs/0d95f2ff1c6803b6.npz')
ROUND2_SUBSET_HASHES = {
    # sha256[:16] of every merged column, first 40 segments (2560231 rows), recorded from the
    # OLD concatenate implementation (2026-10-24, peak RSS 3299 MB) before the task61 rewrite.
    'board_indices': '81dd3aa1501d74e5', 'board_values': '80ff70734e3ef9a6',
    'board_offsets': '52656171bdd25b6f', 'board_plane_of': '39e372160e41ef8a',
    'board_planes_cells': '86a4ce41f36a5906', 'board_planes_values': '10231d223a7b9953',
    'scalar_indices': '780e12c1363ae130', 'scalar_values': 'c0b776e72b2619d9',
    'scalar_offsets': '949926976d4e47d5', 'scalar_plane_of': None,
    'plant': '78e7bab786a734a2', 'packet': '0abc52115bacb67b', 'cell': '2ae1869a8d51015f',
    'weight': '68b169251374bbe0', 'seed': '4c6a39979f571f3b',
    'value_counts': '6a932229cf65eeac', 'value_slots': '3097b11bd997da2d',
    'value_values': 'cb97e4e9c61912bd', 'cand_counts': '24e4935fe8d0ff2f',
    'cand_entries': 'f1e1e5361bb45931', 'packet_col0': 'c4bbb2655da6f209',
    'packet_col1': 'abacaa64ad3b417e', 'packet_col2': '469b90d7c1f9fe37',
    'packet_col3': '5e662c79934c7564',
}


def _column_hashes(result):
    import hashlib
    import torch

    def sha(array):
        if isinstance(array, torch.Tensor):
            array = array.numpy()
        if array is None:
            return None
        return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()[:16]

    return {
        'board_indices': sha(result[0].indices), 'board_values': sha(result[0].values),
        'board_offsets': sha(result[0].offsets), 'board_plane_of': sha(result[0].plane_of),
        'board_planes_cells': sha(np.concatenate([p[0] for p in result[0].planes])
                                  if result[0].planes else np.zeros(0)),
        'board_planes_values': sha(np.concatenate([p[1] for p in result[0].planes])
                                   if result[0].planes else np.zeros(0)),
        'scalar_indices': sha(result[1].indices), 'scalar_values': sha(result[1].values),
        'scalar_offsets': sha(result[1].offsets), 'scalar_plane_of': sha(result[1].plane_of),
        'plant': sha(result[2]), 'packet': sha(result[3]), 'cell': sha(result[4]),
        'weight': sha(result[5]), 'seed': sha(result[6]),
        'value_counts': sha(result[7].counts), 'value_slots': sha(result[7].slots),
        'value_values': sha(result[7].values),
        'cand_counts': sha(result[8].counts), 'cand_entries': sha(result[8].entries),
        'packet_col0': sha(result[9].columns[0]), 'packet_col1': sha(result[9].columns[1]),
        'packet_col2': sha(result[9].columns[2]), 'packet_col3': sha(result[9].columns[3]),
    }


def test_read_segments_matches_concatenating_reference_on_round2_subset(tmp_path):
    if not ROUND2_PACK.is_file():
        pytest.skip('round-2 pack not on this machine')
    with np.load(ROUND2_PACK, mmap_mode='r') as stored:
        names = stored['segments'].tolist()
    paths = [ROUND2_PACK.parent / name for name in sorted(names)[:40]]
    expected = {k: v for k, v in ROUND2_SUBSET_HASHES.items()}
    for kwargs in ({}, {'data': tmp_path}):
        got = _column_hashes(trainer.read_segments(paths, **kwargs))
        assert got == expected
