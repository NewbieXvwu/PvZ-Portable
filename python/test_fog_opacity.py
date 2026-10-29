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
from pvz_env import PvZEnv, TaskSpec, fog_diagnostics
from pvz_tasks import deck_for_level, profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')

TICKS = 9000


def reset(env, level=31, seed=660031, fog=None, preplanted=()):
    deck = deck_for_level(level)
    return env.reset(deck=deck, task=TaskSpec(level=level, seed=seed,
                                             profile=profile_for_deck(deck),
                                             fog_observation=fog, preplanted=preplanted))[0]


def zombies(state):
    fog = fog_diagnostics(state)
    return [(z['id'], z['row'], round(z['x'], 3), fog[z['id']]['opacity']) for z in state['zombies']
            if z['on_board']]


@engine
def test_fog_opacity_is_available_headless_and_deterministic():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env)
        env.step(dict(type='wait', ticks=TICKS))
        first = env.privileged_state()
        repeat = env.privileged_state()
        diagnostics = fog_diagnostics(first)
        assert first['zombies'] and all(z['id'] in diagnostics for z in first['zombies'])
        assert zombies(first) == zombies(repeat)   # same state, same numbers
        assert all(0 <= entry['opacity'] <= 1000 for entry in diagnostics.values())

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
            fog = fog_diagnostics(state)
            for zombie in state['zombies']:
                if not zombie['on_board'] or zombie['x'] > 780:
                    continue
                col = max(0, min(8, int((zombie['x'] - 40) // 80)))
                cell = state['fog_amount'][zombie['row']][col]
                # x < 450 puts the whole sampling box (body + padding, ~106 px to the right of
                # mPosX) left of the first fog quad, which starts at 570 for level 31.
                if zombie['x'] < 450:
                    assert cell == 0
                    clear.append(fog[zombie['id']]['opacity'])
                elif col == 6:
                    fogged.append(fog[zombie['id']]['opacity'])
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
        assert 'fog_zombies' not in observation and 'fog_offset' not in observation
        assert all('fog_opacity' not in zombie for zombie in observation['zombies'])
        state = env.privileged_state()
        assert 'fog_offset' in state and 'fog_look' in state
        diagnostics = fog_diagnostics(state)
        assert diagnostics and all(set(entry) == {'id', 'opacity', 'head', 'body', 'box',
                                              'cell_hidden'}
                                   for entry in diagnostics.values())


# The threshold the user's labelling produced: 20 of 23 frames visible, highest visible 0.900,
# lowest invisible 0.950, cut at the midpoint.
THRESHOLD = 925


@engine
def test_human_v2_hides_exactly_what_the_threshold_says():
    """Every zombie in the public frame is exactly the ones under 0.925 of fog."""
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env, fog='human_v2')
        shown_ticks = hidden_ticks = 0
        for _ in range(60):
            observation, *_ = env.step(dict(type='wait', ticks=250))
            state = env.privileged_state()
            visible = {zombie['id'] for zombie in observation['zombies']}
            fog = fog_diagnostics(state)
            for zombie in state['zombies']:
                if not zombie['on_board']:
                    continue
                shown = zombie['id'] in visible
                assert shown == (fog[zombie['id']]['opacity'] < THRESHOLD), zombie
                if shown:
                    shown_ticks += 1
                else:
                    hidden_ticks += 1
        # Both sides of the cut have to be exercised, or the assertion above is vacuous.
        assert shown_ticks and hidden_ticks


@engine
def test_human_v2_shows_what_the_cell_rule_hides():
    """The same tick, same zombies: the cell rule hides one the eye can see."""
    results = {}
    for mode in ('human', 'human_v2'):
        with PvZEnv(RESOURCE, executable=BINARY) as env:
            reset(env, fog=mode)
            env.step(dict(type='wait', ticks=9000))
            observation = env.observe()
            state = env.privileged_state()
            fog = fog_diagnostics(state)
            results[mode] = ({zombie['id'] for zombie in observation['zombies']},
                             {zombie['id']: fog[zombie['id']]['opacity']
                              for zombie in state['zombies'] if zombie['on_board']})
    old_visible, opacities = results['human']
    new_visible, _ = results['human_v2']
    assert new_visible > old_visible
    for zombie_id in new_visible - old_visible:
        assert opacities[zombie_id] < THRESHOLD


@engine
def test_human_v2_applies_the_same_rule_to_projectiles():
    """A pea inside the fog wall is hidden by the cell rule and shown by the opacity rule."""
    preplanted = ((0, 0, 0), (0, 1, 0))       # peashooters on the two lawn rows
    runs = {}
    for mode in ('human', 'human_v2'):
        with PvZEnv(RESOURCE, executable=BINARY) as env:
            reset(env, fog=mode, preplanted=preplanted)
            ticks = []
            for _ in range(12):
                observation, *_ = env.step(dict(type='wait', ticks=250))
                state = env.privileged_state()
                if state['projectiles']:
                    ticks.append((observation['tick'],
                                  sorted(round(p['x'], 1) for p in observation['projectiles']),
                                  sorted(round(p['x'], 1) for p in state['projectiles'])))
            runs[mode] = ticks
    hidden_by_cell = [(tick, public, every) for tick, public, every in runs['human']
                      if len(public) < len(every)]
    assert hidden_by_cell, 'the cell rule should hide at least one pea in the fog'
    shown_by_opacity = {tick: public for tick, public, _ in runs['human_v2']}
    for tick, public, every in hidden_by_cell:
        assert len(shown_by_opacity[tick]) > len(public)
        # Whatever it was, it was inside the fog band the cell rule masks.
        assert any(x > 570 for x in every)


@engine
def test_fog_family_only_changes_what_is_reported():
    """Masking is presentation: the same task and actions give the same state hash."""
    hashes = {}
    for mode in ('human', 'human_v2', 'xray'):
        with PvZEnv(RESOURCE, executable=BINARY) as env:
            reset(env, fog=mode)
            env.step(dict(type='wait', ticks=6000))
            hashes[mode] = env.state_hash()
    assert len(set(hashes.values())) == 1
