"""Privileged-state identity for task78, masking the engine's uninitialised field.

LawnApp.cpp emits grid_items[*].state from GridItem::mGridItemState, but no file
under src/ ever assigns that member: a cold replay reads whatever the allocator left
there. Two replays of one frozen archive therefore differ in exactly that field and
nowhere else (archive 17 frame 650, 2026-10-26: 2052151040 vs -1613670656). The
compact frame every policy sees keeps only row/col/type, so the value never reaches a
policy or a label.

state_digest masks that one field and hashes everything else, which is what root
identity comparisons use. The unmasked _debug_state_sha256 stays recorded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python')]

from pvz_common import canonical_digest

UNINITIALISED_GRID_ITEM_FIELDS = ('state',)


def masked_state(state):
    """Deep copy of a privileged state with the never-written GridItem fields blanked."""
    masked = json.loads(json.dumps(state))
    for item in masked.get('grid_items') or []:
        if isinstance(item, dict):
            for field in UNINITIALISED_GRID_ITEM_FIELDS:
                if field in item:
                    item[field] = None
    return masked


def state_digest(state):
    return canonical_digest(masked_state(state))


def verified_root_digest(root, replayed):
    """Return one reproducible identity after proving this label matches its replay.

    A legacy label has no masked digest. An exact raw match still proves its
    identity, and lets us use the independently replayed masked digest when present.
    """
    raw = root.get('root_state_sha256')
    masked = root.get('root_state_canonical_sha256')
    expected_masked = replayed.get('engine_canonical_sha256')
    if raw and raw == replayed.get('engine_sha256'):
        return expected_masked or raw
    if masked and expected_masked and masked == expected_masked:
        return masked
    raise ValueError('root engine state differs from replay')
