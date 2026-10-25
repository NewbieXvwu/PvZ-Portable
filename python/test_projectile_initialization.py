"""Serialized projectiles must not inherit portal state from reused heap storage."""
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from pvz_env import PvZEnv, TaskSpec, profile_for_deck

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/home/newbiexvwu/.cache/pvz-research-resources'))


def projectile_portals(state):
    """Read each live projectile's existing SAVE4 tail, without excluding hash fields."""
    assert state[:7] == b'PVZSB1\n'
    size = struct.unpack_from('<Q', state, 7)[0]
    board = state[15:15 + size]
    assert board[:12] == b'PVZP_SAVE4\0\0'
    offset = 24
    while offset < len(board):
        kind, length = struct.unpack_from('<II', board, offset)
        payload = board[offset + 8:offset + 8 + length]
        offset += 8 + length
        if kind != 4:
            continue
        version, field_id, blob_size = struct.unpack_from('<3I', payload)
        assert version == 1 and field_id == 1 and blob_size == len(payload) - 12
        free, used, count, next_key, capacity = struct.unpack_from('<5I', payload, 12)
        assert used <= capacity
        cursor, portals = 32, []
        for _ in range(used):
            identifier, item_size = struct.unpack_from('<II', payload, cursor)
            cursor += 8
            end = cursor + item_size
            while cursor < end:
                field, field_size = struct.unpack_from('<II', payload, cursor)
                cursor += 8
                if field == 100:
                    portals.append(struct.unpack_from('<i', payload, cursor + field_size - 4)[0])
                cursor += field_size
            assert cursor == end
        assert len(portals) == count
        return portals
    raise AssertionError('missing projectile chunk')


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'needs native engine/resources')
class ProjectileInitializationTest(unittest.TestCase):
    def test_spawned_projectiles_have_no_previous_portal(self):
        deck = (0, 1, 2, 3, 4, 5)
        with tempfile.TemporaryDirectory() as folder, PvZEnv(RESOURCE, executable=BINARY) as env:
            env.reset(deck=deck, task=TaskSpec(level=7, seed=6901,
                profile=profile_for_deck(deck), preplanted=tuple((0, row, 2) for row in range(5))))
            path = Path(folder) / 'state.bin'
            for _ in range(100):
                env.step(dict(type='wait', ticks=100))
                self.assertTrue(env._command('STATE_BYTES ' + json.dumps(str(path)))['ok'])
                portals = projectile_portals(path.read_bytes())
                if portals:
                    self.assertEqual(portals, [-1] * len(portals))
                    break
            else:
                self.fail('fixture produced no live projectile within 10000 ticks')


if __name__ == '__main__':
    unittest.main()
