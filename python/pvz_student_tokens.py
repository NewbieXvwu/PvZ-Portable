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


def frame_tokens(frame, context=None):
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
                     len(frame['plants'])/40, len(frame['zombies'])/40, 0, 0, 0,
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
    for zombie in frame['zombies']:
        row, kind, x, vx, hp, on_board, eating, chilled = zombie[:8]
        if not on_board:
            continue
        col = max(0, min(8, int((x-40)//80)))
        f = np.zeros(32, np.float32)
        f[[0, 1, 2, 5, 9, 10, 13]] = (row/5, col/8, x/900, hp/2000, vx, chilled/2000, float(eating))
        add(4, kind, row=row, col=col, features=f)
    for row, x, damage in (frame.get('v2') or {}).get('projectiles', []):
        col = max(0, min(8, int((x-40)//80)))
        f = np.zeros(32, np.float32)
        f[[0, 1, 2, 9]] = (row/5, col/8, x/900, damage/500)
        add(5, row=row, col=col, features=f)
    for row, state in frame['mowers']:
        add(6, row=row, features=(row/5, state/5))
    for slot, kind, active, cooldown, refresh, cost in frame['packets']:
        if 0 <= slot < 10 and kind >= 0:
            packet_index[slot] = add(8, kind, features=(0, cost/500, cooldown/3000, refresh/3000, float(active)))
    if context is not None:
        for kind in np.flatnonzero(context[1:]):
            add(9, int(kind), features=(1,))
    return dict(ids=np.asarray(ids, np.int16), features=np.asarray(values, np.float32),
                target_indices=np.full(len(ids), -1, np.int32), packet_index=packet_index,
                cell_index=cell_index, terrain=np.asarray(terrain_id(scene), np.int64))


def collate_tokens(samples, device='cpu'):
    """Pad only the current minibatch; no quadratic relations stored on disk."""
    count = max(len(s['ids']) for s in samples)
    batch = len(samples)
    ids = np.zeros((batch, count, 5), np.int64)
    ids[:, :, 3:] = -1
    features = np.zeros((batch, count, FEATURE_COUNT), np.float32)
    targets = np.full((batch, count), -1, np.int64)
    mask = np.zeros((batch, count), bool)
    for i, s in enumerate(samples):
        n = len(s['ids'])
        ids[i, :n], features[i, :n], targets[i, :n], mask[i, :n] = s['ids'], s['features'], s['target_indices'], True
    result = dict(ids=ids, features=features, target_indices=targets, mask=mask)
    result.update({key: np.stack([s[key] for s in samples]).astype(np.int64)
                   for key in ('packet_index', 'cell_index', 'terrain')})
    return {k: torch.from_numpy(v).to(device) for k, v in result.items()}
