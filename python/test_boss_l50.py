import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import profile_for_deck
from pvz_student_observation import StudentHistory


RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')


def _frame(boss=None):
    return {
        'tick': 100,
        'wave': 1,
        'sun': 100,
        'scene': {'wave_count': 30},
        'packets': [],
        'lanes': [0, 1, 2, 3, 4],
        'plants': [],
        'mowers': [],
        'zombies': [[0, 0, 80.0, 0.35, 200.0, True, False, 1000, 0, 1]],
        'v2': {'wave_timer': 2250, 'huge_wave_countdown': 0, 'wave_health': 100,
               'wave_start_health': 200, 'sun_countdown': 0, 'sun_pending': 0,
               'plants': [], 'zombies': [[0, 0, 0]], 'projectiles': []},
        'boss': boss,
    }


def test_v2boss_adds_only_visible_boss_channels():
    plain = _frame()
    baseline_board, baseline_scalars = StudentHistory('v2').features(plain, None)
    boss = {'active': True, 'stage': 2, 'head': True, 'action': 'summon',
            'target_row': 3, 'target_col': 4,
            'projectile': {'type': 'fire', 'row': 1, 'x': 455.0, 'y': 120.0}}
    boss_board, boss_scalars = StudentHistory('v2boss').features({**plain, 'boss': boss}, None)
    assert tuple(boss_board.shape) == (105, 6, 10)
    assert tuple(boss_scalars.shape) == (124,)
    np.testing.assert_array_equal(boss_board[:92], baseline_board[:92])
    np.testing.assert_array_equal(boss_board[92:101], baseline_board[92:])
    np.testing.assert_array_equal(boss_scalars[:115], baseline_scalars[:115])
    np.testing.assert_array_equal(boss_scalars[115:122], baseline_scalars[115:])
    assert boss_board[92 + 9, 0, 0] == 1
    assert boss_board[92 + 10, 1].sum() == 1
    assert boss_board[92 + 12, 3, 4] == 1
    np.testing.assert_allclose(boss_scalars[122:124], [1.0, 0.2])


@engine
def test_l50_reaches_terminal_with_boss_observation():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        obs, *_ = env.reset(deck=(), task=TaskSpec(
            level=50, seed=85_000_000, conveyor_start='original',
            profile=profile_for_deck(())))
        assert obs['boss']['active']
        for _ in range(120):
            if obs['terminal']:
                break
            obs, *_ = env.step({'type': 'wait', 'ticks': 1000})
        assert obs['terminal'], obs['tick']
        assert obs['result'] in (1, 2)
        assert obs['wave_count'] > 1
