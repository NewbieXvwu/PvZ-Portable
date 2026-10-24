"""Generic temporal copies retain only actually observed zombie channels."""
import copy

import numpy as np
import torch

from pvz_student_observation import StudentHistory


def frame(tick, visible):
    return dict(tick=tick, wave=1, sun=50, plants=[],
                zombies=[(0, 2, 700, .2, 200, True, False, 0, 0, 1)] if visible else [],
                mowers=[], packets=[], scene={'wave_count': 20},
                fog_amount=[[255] * 9 for _ in range(6)])


def test_stack_retains_visible_channels_without_extrapolating_and_expires():
    history = StudentHistory('v2fogstack4')
    first = history.features(frame(0, True), None)[0]
    assert first[56, 0, 8] == 1
    at60 = history.features(frame(60, False), {'tick': 0})[0]
    assert at60[54:88].sum() == 0
    assert at60[102 + 2, 0, 8] == 1  # exactly the previous visible location
    assert at60[102:136].sum() == 1
    before = copy.deepcopy(history)
    assert torch.equal(at60, history.features(frame(60, False), {'tick': 0})[0])
    history = before
    for tick in (120, 180):
        result = history.features(frame(tick, False), {'tick': tick - 60})[0]
        assert result[102:].sum() == 1
    expired = history.features(frame(240, False), {'tick': 180})[0]
    assert expired[102:].sum() == 0


def test_stack_addition_leaves_current_features_and_scalars_identical():
    current = StudentHistory('v2fog')
    stacked = StudentHistory('v2fogstack4')
    for tick, visible in ((0, True), (1, False), (60, False), (121, True)):
        a = current.features(frame(tick, visible), None)
        b = stacked.features(frame(tick, visible), None)
        np.testing.assert_array_equal(a[0].numpy().view(np.uint32), b[0][:102].numpy().view(np.uint32))
        assert torch.equal(a[1], b[1])
