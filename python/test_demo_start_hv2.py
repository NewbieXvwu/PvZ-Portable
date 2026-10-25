"""Task70 B6: demo start cells replayed under human_v2 instead of the recording-time x-ray.

The demo tasks predate the fog families, so their archives record fog_observation: null and
every demo_start cell was x-ray for the student.  fog_observation is presentation only: the
restore may override it with the manifest's value while the board stays identical.  These
tests pin that on a synthetic archive recorded by this engine (the user demo archives live
only on the recording machine).  Board identity is asserted on the privileged logic state,
not on state_hash: under production qc load the reset particle race (task70 A2 probe) splits
the save-image hash across runs of the SAME binary and mode, while logic fields cannot move.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from human_demo import Recorder, replay
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/home/newbiexvwu/.cache/pvz-research-resources'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))

engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')

DECK = tuple(deck_for_level(31))
# A clear-side and a fog-side peashooter, so both sides of the masking are on the board.
PREPLANTED = ((0, 0, 2), (0, 1, 7))


def record_archive(tmp: Path) -> Path:
    """Record a small archive the way the demo recorder does (fog mode unset = null)."""
    archive = tmp / 'demo_fog_31_snow_m200'
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        task = TaskSpec(level=31, seed=660031, profile=profile_for_deck(DECK),
                        preplanted=PREPLANTED)
        recorder = Recorder(env, archive, 'demo_fog_31_snow_m200', DECK, task, 'synthetic')
        recorder.step(dict(type='wait', ticks=400))
        recorder.step(dict(type='wait', ticks=400))
        recorder.close()
    return archive


def restored_state(archive, mode=None):
    """Restore at decision 2, reseed like a start cell, settle the fog, return states."""
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        # strict_hashes=False: the relaxed verification (tick/ok/outcome) is load-robust;
        # board identity across modes is asserted on the privileged state below.
        obs, _, _ = replay(env, archive, decision=2, strict_hashes=False, fog_observation=mode)
        env.reseed(44_000_001)
        # decision 2 stops within the first seconds; run well past the fog blow-in (~1000)
        obs, *_ = env.step(dict(type='wait', ticks=2400))
        state = env.privileged_state()
        state.pop('fog_plants', None)
        return obs, state


@engine
def test_fog_mode_override_keeps_board_and_masks_public_observation(tmp_path):
    archive = record_archive(tmp_path)
    meta = json.loads((archive / 'meta.json').read_text())
    # the recording is the pre-B6 situation: no fog_observation in the task
    assert meta['environment_task'].get('fog_observation') is None

    public, logic = {}, {}
    for mode in (None, 'human_v2'):
        obs, state = restored_state(archive, mode)
        public[mode] = {(p['row'], p['col']): p for p in obs['plants']}
        logic[mode] = json.dumps(state, sort_keys=True)
    # presentation only: the board is identical, the report is not
    assert logic[None] == logic['human_v2']
    assert set(public[None]) == set(public['human_v2'])
    assert set(public[None]) == {(0, 2), (1, 7)}, 'both preplanted plants must survive the restore'
    for key, real in public[None].items():
        seen = public['human_v2'][key]
        assert seen['row'] == real['row'] and seen['col'] == real['col'] and seen['type'] == real['type']
        if key == (1, 7):        # fog side: internals are the unknown sentinel
            assert seen['health'] == -1 and seen['state'] == -1
        else:                    # clear side: real values
            assert seen['health'] == real['health'] and seen['state'] == real['state']


@engine
def test_replay_without_override_keeps_recording_mode(tmp_path):
    """No override -> the archive's own null mode, exactly as before B6."""
    archive = record_archive(tmp_path)
    with_override = restored_state(archive, None)[1]
    without_override = restored_state(archive)[1]
    assert with_override == without_override
