"""task157 富输入交接给固定形状前向图的契约。"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvz_token_observation import graph_inputs


def test_graph_inputs_keep_indices_and_mask_empty_slots():
    tokens = np.zeros((68, 38), np.float32)
    tokens[:, 3:6] = -1
    tokens[0, 0], tokens[1, 0] = 0, 1
    tokens[2:56, 0] = 2
    tokens[2:56, 3] = np.arange(54) // 9
    tokens[2:56, 4] = np.arange(54) % 9
    tokens[56:66, 0] = 8
    tokens[56:62, 1] = np.arange(1, 7)
    tokens[66, :6] = [4, 21, 0, 2, 4, 24]
    tokens[67, :6] = [3, 4, 0, 1, 2, -1]
    tokens[67, 6:] = np.arange(32) / 32
    inputs = graph_inputs(dict(v3=dict(version=3, tokens=tokens.tolist())), capacity=72)
    for value in inputs.values():
        assert value.dtype == np.float32 and value.flags.c_contiguous
    np.testing.assert_array_equal(inputs["tokens"][0, :68], tokens)
    np.testing.assert_array_equal(inputs["token_meta"][0, 66], [4, 2, 4])
    np.testing.assert_array_equal(inputs["coords"][0, 67], [1, 2])
    assert inputs["tokens"][0, 66, 5] == 24  # target keeps its absolute token index
    np.testing.assert_array_equal(inputs["valid"][0, 56:], [1] * 6 + [0] * 4 + [1] * 2 + [0] * 4)
    np.testing.assert_array_equal(inputs["tokens"][0, 68:, 3:6], -np.ones((4, 3)))
    with pytest.raises(ValueError, match="truncation"):
        graph_inputs(dict(v3=dict(version=3, tokens=tokens)), capacity=67)


def test_graph_inputs_require_backfill_for_legacy_frames():
    with pytest.raises(ValueError, match="backfill"):
        graph_inputs(dict(plants=[], zombies=[]))
