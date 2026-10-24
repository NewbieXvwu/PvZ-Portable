"""Task66: fog coverage as the renderer paints it, not as the cell value says.

The pre-task66 rule hides a zombie whenever its grid cell's fog value is non-zero, but
Board::DrawFog paints a 210x190 cel whose own alpha fades to zero at the edges, so the leftmost
fog column is nearly transparent even at value 200.  These tests pin the replacement quantity:
the composite alpha over the zombie's body.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/home/newbiexvwu/.cache/pvz-research-resources'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')

TICKS = 9000


def reset(env, level=31, seed=660031):
    deck = deck_for_level(level)
    return env.reset(deck=deck, task=TaskSpec(level=level, seed=seed,
                                             profile=profile_for_deck(deck)))[0]


def zombies(state):
    return [(z['id'], z['row'], round(z['x'], 3), z['fog_opacity']) for z in state['zombies']
            if z['on_board']]


@engine
def test_fog_opacity_is_available_headless_and_deterministic():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env)
        env.step(dict(type='wait', ticks=TICKS))
        first = env.privileged_state()
        repeat = env.privileged_state()
        assert first['zombies'] and all('fog_opacity' in z for z in first['zombies'])
        assert zombies(first) == zombies(repeat)   # same state, same numbers
        assert all(0 <= z['fog_opacity'] <= 1000 for z in first['zombies'])

        reset(env)                                  # a second run of the same task
        env.step(dict(type='wait', ticks=TICKS))
        again = env.privileged_state()
        assert zombies(first) == zombies(again)


@engine
def test_fog_opacity_follows_the_drawn_cel_not_the_cell_value():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env)
        # The first ~200 ticks are the fog blowing in (mFogBlownCountDown), when it still
        # covers the whole board; skip them and measure the settled fog.
        env.step(dict(type='wait', ticks=1000))
        clear, fogged = [], []
        for _ in range(60):
            env.step(dict(type='wait', ticks=250))
            state = env.privileged_state()
            assert state['fog_amount'] is not None
            for zombie in state['zombies']:
                if not zombie['on_board'] or zombie['x'] > 780:
                    continue
                col = max(0, min(8, int((zombie['x'] - 40) // 80)))
                cell = state['fog_amount'][zombie['row']][col]
                # x < 450 puts the whole sampling box (body + padding, ~106 px to the right of
                # mPosX) left of the first fog quad, which starts at 570 for level 31.
                if zombie['x'] < 450:
                    assert cell == 0
                    clear.append(zombie['fog_opacity'])
                elif col == 6:
                    fogged.append(zombie['fog_opacity'])
        # Nothing is drawn there, so nothing can be covered.
        assert clear and max(clear) == 0
        # The leftmost fog column is the one the user reported: some of its zombies are under
        # less than half the fog, which the cell rule called hidden.
        assert fogged and min(fogged) < 500
        assert max(fogged) > 800


@engine
def test_fog_diagnostics_never_reach_the_public_observation():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        observation = reset(env)
        observation, *_ = env.step(dict(type='wait', ticks=6000))
        assert 'fog_opacity' not in observation
        assert all('fog_opacity' not in zombie for zombie in observation['zombies'])
        assert 'fog_offset' not in observation and 'fog_look' not in observation
        state = env.privileged_state()
        assert 'fog_offset' in state and 'fog_look' in state
        assert all('fog_opacity' in zombie and 'fog_cell_hidden' in zombie
                   for zombie in state['zombies'])
