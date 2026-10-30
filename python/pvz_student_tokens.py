"""Public student token contract v1; compact frames and obs-v3 producers share arrays.

Missing compact fields stay zero (never extrapolated). Indices are local to a frame.
See task163 for the wire format and feature slots. No trajectories or policy rules.
"""
from __future__ import annotations

import numpy as np
import torch

TOKEN_KINDS = dict(global_=0, profile=1, cell=2, plant=3, zombie=4, projectile=5,
                   defense=6, grid_item=7, seed_packet=8, zombie_roster=9, lane=10, lane_enemy=11)
FEATURE_COUNT = 32
TOKEN_SCHEMA = 1


def terrain_id(scene):
    # Five actual backgrounds, including L35's night-grass exception (engine flags).
    return 4 if scene.get('roof') else 3 if scene.get('fog') else 2 if scene.get('pool') else 1 if scene.get('night') else 0


def frame_tokens(frame, context=None, token_input='compact'):
    if token_input == 'obs_v3':
        from pvz_token_observation import graph_inputs
        source = graph_inputs(frame)
        raw = source['tokens'][0]
        cards = np.arange(56, 66, dtype=np.int32)
        cards[raw[56:66, 1] == 0] = -1
        return dict(ids=raw[:, :5].astype(np.int16), features=raw[:, 6:].copy(),
                    target_indices=raw[:, 5].astype(np.int32), packet_index=cards,
                    cell_index=np.arange(2, 56, dtype=np.int32),
                    terrain=np.asarray(terrain_id(frame['scene']), np.int64))
    if token_input != 'compact':
        raise ValueError('unknown token input format')
    if 'tokens_v3' in frame:
        raw = frame['tokens_v3']
        if raw['schema_version'] != TOKEN_SCHEMA:
            raise ValueError('unsupported tokens_v3 schema_version')
        return {key: np.asarray(raw[key], dtype=dtype) for key, dtype in (
            ('ids', np.int16), ('features', np.float32), ('target_indices', np.int32),
            ('packet_index', np.int32), ('cell_index', np.int32), ('terrain', np.int64))}
    ids, values = [], []
    packet_index = np.full(10, -1, np.int32)
    cell_index = np.zeros(54, np.int32)

    def add(kind, category=-1, variant=-1, row=-1, col=-1, features=()):
        at = len(ids)
        ids.append([kind, max(0, min(127, category + 1)), max(0, min(127, variant + 1)), row, col])
        value = np.zeros(FEATURE_COUNT, np.float32)
        value[:len(features)] = features
        values.append(value)
        return at

    scene = frame['scene']
    add(0, features=(frame['sun']/1000, frame['wave']/max(1, scene.get('wave_count') or 1),
                     frame['tick']/60000, (scene.get('wave_count') or 0)/30,
                     float(context[0]*3/10) if context is not None else 0,
                     *(float(scene.get(k, False)) for k in ('night', 'pool', 'fog', 'roof')),
                     len(frame['plants'])/40, len(frame['zombies'])/40,
                     len((frame.get('v2') or {}).get('projectiles', []))/50, 0, 0,
                     (frame.get('sun_income_rate') or 0)/50))
    add(1)  # Compact frames do not expose the player profile.
    lanes = set(frame.get('lanes', range(5)))
    for row in range(6):
        for col in range(9):
            cell_index[row*9+col] = add(2, row=row, col=col,
                features=(row/5, col/8, 0, 0, 0, 0, 0, 0, float(row in lanes)))
    for row in sorted(lanes):
        add(10, row=row)
    for row, col, kind, hp, max_hp in frame['plants']:
        add(3, kind, row=row, col=col, features=(row/5, col/8, hp/max(1, max_hp), hp/3000, max_hp/3000))
    for zi, zombie in enumerate(frame['zombies']):
        row, kind, x, vx, hp, on_board, eating, chilled = zombie[:8]
        if not on_board:
            continue
        col = max(0, min(8, int((x-40)//80)))
        f = np.zeros(32, np.float32)
        # Preserve the old rich-token slots; compact hp is total, not body hp.
        f[[0, 1, 2, 10, 11, 17, 20, 21]] = (row/5, col/8, x/900, vx/5, chilled/1200, float(eating), 1, hp/2000)
        if len(zombie) > 8:
            f[13] = zombie[8]/1200
        if len(zombie) > 10:
            f[24] = zombie[10]
        detail = (frame.get('v2') or {}).get('zombies', [])
        if zi < len(detail):
            has_object, helm, shield = detail[zi][:3]
            f[[5, 16, 22, 23]] = ((hp-helm-shield)/2000, float(has_object), helm/2000, shield/2000)
        add(4, kind, row=row, col=col, features=f)
    for row, x, damage in (frame.get('v2') or {}).get('projectiles', []):
        col = max(0, min(8, int((x-40)//80)))
        f = np.zeros(32, np.float32)
        f[[0, 1, 2, 9]] = (row/5, col/8, x/900, damage/500)
        add(5, row=row, col=col, features=f)
    for row, state in frame['mowers']:
        add(6, row=row, features=(row/5, state/5))
    for row, col, kind in frame.get('graves', []):
        add(7, kind, row=row, col=col, features=(row/5, col/8))
    for slot, kind, active, cooldown, refresh, cost in frame['packets']:
        if 0 <= slot < 10 and kind >= 0:
            packet_index[slot] = add(8, kind, features=(0, cost/500, cooldown/3000, refresh/3000, float(active)))
    if context is not None:
        for kind in np.flatnonzero(context[1:]):
            add(9, int(kind), features=(1,))
    return dict(ids=np.asarray(ids, np.int16), features=np.asarray(values, np.float32),
                target_indices=np.full(len(ids), -1, np.int32), packet_index=packet_index,
                cell_index=cell_index, terrain=np.asarray(terrain_id(scene), np.int64))


def collate_tokens(samples, device='cpu', capacity=None):
    """Pad only the current minibatch; no quadratic relations stored on disk."""
    required = max(len(s['ids']) for s in samples)
    count = required if capacity is None else capacity
    if count < required:
        raise ValueError(f'token capacity {count} < {required}; truncation is unsupported')
    batch = len(samples)
    ids = np.zeros((batch, count, 5), np.int64)
    ids[:, :, 3:] = -1
    features = np.zeros((batch, count, FEATURE_COUNT), np.float32)
    targets = np.full((batch, count), -1, np.int64)
    mask = np.zeros((batch, count), bool)
    for i, s in enumerate(samples):
        n = len(s['ids'])
        ids[i, :n], features[i, :n], targets[i, :n] = s['ids'], s['features'], s['target_indices']
        mask[i, :n] = ~((s['ids'][:, 0] == 8) & (s['ids'][:, 1] == 0))
    result = dict(ids=ids, features=features, target_indices=targets, mask=mask)
    result.update({key: np.stack([s[key] for s in samples]).astype(np.int64)
                   for key in ('packet_index', 'cell_index', 'terrain')})
    return {k: torch.from_numpy(v).to(device) for k, v in result.items()}


class DenseTokenStore:
    """Reader for the earlier dense-feature prototype packs."""
    token_fields = dict(ids=(np.int16, (5,)), features=(np.float32, (32,)), target_indices=(np.int32, ()))
    row_fields = dict(packet_index=(np.int32, (10,)), cell_index=(np.int32, (54,)), terrain=(np.int64, ()))

    def __init__(self, arrays):
        self.arrays = arrays

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    @classmethod
    def from_arrays(cls, stored):
        return cls({k: np.asarray(stored['token_'+k]) for k in ('offsets', *cls.token_fields, *cls.row_fields)})

    def packed_arrays(self):
        return {'token_'+k: v for k, v in self.arrays.items()}

    def sample(self, i):
        lo, hi = self.arrays['offsets'][i:i+2]
        return {**{k: self.arrays[k][lo:hi] for k in self.token_fields},
                **{k: self.arrays[k][i] for k in self.row_fields}}

    def batch(self, indices, device='cpu'):
        return collate_tokens([self.sample(int(i)) for i in indices], device)


class TokenStore:
    """32-row sparse compressed blocks inside the ordinary training segment.

    offsets holds each row's block byte address (plus the final end address),
    rows its ordinal within that block. Blocks never cross source archives,
    allowing the existing archive-order merger to copy bytes without decoding.
    """
    token_fields = dict(data=(np.uint8, ()))
    row_fields = dict(rows=(np.uint8, ()))

    def __init__(self, arrays, cache_bytes=16 * 1024**2):
        from collections import OrderedDict
        self.arrays = arrays
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.cached_bytes = 0

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    @classmethod
    def from_arrays(cls, stored):
        if 'token_data' not in stored:
            return DenseTokenStore.from_arrays(stored)
        return cls({k: np.asarray(stored['token_'+k]) for k in ('offsets', 'data', 'rows')})

    def packed_arrays(self):
        return {'token_'+k: v for k, v in self.arrays.items()}

    def sample(self, i):
        from pvz_token_blocks import decode
        import struct
        at = int(self.arrays['offsets'][i])
        block = self.cache.pop(at, None)
        if block is None:
            data = self.arrays['data']
            size, = struct.unpack_from('<I', data, at)
            block = decode(data[at+4:at+4+size])
            self.cached_bytes += block.nbytes
        self.cache[at] = block
        while self.cached_bytes > self.cache_bytes and self.cache:
            _, old = self.cache.popitem(last=False)
            self.cached_bytes -= old.nbytes
        return block.sample(int(self.arrays['rows'][i]))

    def batch(self, indices, device='cpu'):
        return collate_tokens([self.sample(int(i)) for i in indices], device)


class TokenBuilder:
    """Compress at most 32 rows at a time; never build corpus-wide dense tokens."""
    def __init__(self):
        import tempfile
        self.file = tempfile.TemporaryFile()
        from array import array
        self.offsets = array('q')
        self.rows = bytearray()
        self.pending = []

    def append(self, sample):
        self.pending.append(sample)
        if len(self.pending) == 32:
            self.flush()

    def flush(self):
        """Also called at each source archive boundary, including short archives."""
        if not self.pending:
            return
        import struct
        from pvz_token_blocks import encode
        compressed, _ = encode(self.pending)
        at = self.file.tell()
        self.offsets.extend([at] * len(self.pending))
        self.rows.extend(range(len(self.pending)))
        self.file.write(struct.pack('<I', len(compressed)))
        self.file.write(compressed)
        self.pending.clear()

    def finish(self):
        self.flush()
        self.offsets.append(self.file.tell())
        self.file.seek(0)
        data = np.fromfile(self.file, dtype=np.uint8)
        self.file.close()
        return TokenStore(dict(offsets=np.asarray(self.offsets, np.int64), data=data,
                               rows=np.frombuffer(self.rows, dtype=np.uint8)))
