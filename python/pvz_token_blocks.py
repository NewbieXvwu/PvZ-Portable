"""Experimental lossless row-block codec for the task157 storage design.

Not selected by production training. Same numeric columns as task162 decision tokens;
32 feature columns remain float32 and signed zero/target links are preserved.
"""
import struct
import zlib
import numpy as np
from pvz_research_data import RICH_FIELDS, SparseTokens

HEADER = struct.Struct('<4sIII')


def encode(samples):
    columns = {key: [] for key in RICH_FIELDS}
    token_count = feature_count = 0
    for key in ('offsets', 'feature_offsets'):
        columns[key].append(np.zeros(1, np.int64))
    for sample in samples:
        features = np.asarray(sample['features'], np.float32)
        rr, cc = np.nonzero(features.view(np.uint32))
        token_count += len(features); feature_count += len(rr)
        row = dict(ids=sample['ids'], counts=np.bincount(rr, minlength=len(features)),
                   columns=cc, values=features[rr, cc], offsets=[token_count], feature_offsets=[feature_count],
                   packet_index=sample['packet_index'], cell_index=sample['cell_index'],
                   terrain=sample['terrain'], target_indices=sample['target_indices'])
        for key, value in row.items():
            columns[key].append(np.asarray(value, dtype=RICH_FIELDS[key][0]).reshape(-1))
    raw = HEADER.pack(b'TKB1', len(samples), token_count, feature_count)
    raw += b''.join(np.concatenate(columns[key]).tobytes() for key in RICH_FIELDS)
    return zlib.compress(raw, level=3), len(raw)


def decode(compressed):
    raw = zlib.decompress(compressed)
    magic, rows, tokens, features = HEADER.unpack_from(raw)
    if magic != b'TKB1':
        raise ValueError('unsupported experimental token block')
    counts = dict(ids=tokens, counts=tokens, columns=features, values=features,
                  offsets=rows+1, feature_offsets=rows+1, packet_index=rows,
                  cell_index=rows, terrain=rows, target_indices=tokens)
    store = SparseTokens.__new__(SparseTokens)
    store.meta = dict(rows=rows, token_input='obs_v3', index_kind='decision')
    store.arrays = {}
    offset = HEADER.size
    for key, (dtype, shape) in RICH_FIELDS.items():
        count = counts[key]*int(np.prod(shape) if shape else 1)
        array = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
        store.arrays[key] = array.reshape(-1, *shape)
        offset += array.nbytes
    return store
