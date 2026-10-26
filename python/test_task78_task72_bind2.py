"""The original probe rounded balloon x; truncation is what broke 68 of the 94 states."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import task78_task72_bind2 as bind2


def frame(*xs):
    zombies = [[0, 0, 500.0, 0.3, 55, True, False, 0, 0, 1]]
    zombies += [[2, 16, x, 0.28, 270, True, False, 0, 0, 2] for x in xs]
    return dict(tick=15498, zombies=zombies)


@pytest.fixture
def reference():
    return dict(task="demo_fog_33_cactus_snow_m200", seed=101, decision=14, tick=15498,
                wave=7, sun=135, balloons=2, balloon_x=[746, 764], n_cells=2,
                cells=[[0, 6], [1, 7]], static_best=[0, 6],
                chosen=json.dumps(dict(ticks=60, type="wait", until="timeout")))


def test_balloon_ints_uses_python_round_half_even():
    assert bind2.balloon_ints(frame(746.5, 763.734)) == [746, 764]
    assert bind2.balloon_ints(frame(787.556)) == [788]
    assert bind2.balloon_ints(dict(zombies=[[0, 0, 1.0, 0, 0, True, False, 0, 0, 1]])) == []


def test_rounded_balloons_bind_while_truncated_balloons_do_not(reference):
    row = dict(reference, balloon_x=[745, 763], action_rejected=False,
               chosen=dict(ticks=60, type="wait", until="timeout"),
               frame=frame(745.6, 763.734))
    assert bind2.same(reference, row) is True
    assert bind2.legacy_same(reference, row) is False


def test_identity_still_requires_every_other_field(reference):
    base = dict(reference, action_rejected=False,
                chosen=dict(ticks=60, type="wait", until="timeout"),
                frame=frame(745.6, 763.734))
    for field, value in (("tick", 15499), ("sun", 136), ("cells", [[0, 6], [1, 8]]),
                         ("static_best", [1, 7]), ("n_cells", 3), ("balloons", 1)):
        assert bind2.same(reference, dict(base, **{field: value})) is False
    assert bind2.same(reference, dict(base, chosen=dict(type="wait", ticks=60, until="timeout"),
                                      action_rejected=True)) is False
    assert bind2.same(reference, dict(base, chosen=dict(type="plant", row=0, col=6))) is False


def test_balloon_count_must_match(reference):
    row = dict(reference, action_rejected=False,
               chosen=dict(ticks=60, type="wait", until="timeout"),
               frame=frame(745.6))
    assert bind2.same(reference, row) is False
