import struct
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

from pvz_rollout_policy import RolloutNet, export_student, load_student
from pvz_frame_store import cache_episode, row_damage_labels, write_frames

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_rollout_policy import load


def test_rowpool_checkpoint_and_export_without_aux():
    torch.manual_seed(17)
    net = RolloutNet(13, "meanmax", True, "v2", row_pool=True, aux=True).eval()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.pt"
        binary = Path(directory) / "model.bin"
        torch.save({"state_dict": net.state_dict(), "pool": "meanmax", "context": True,
                    "obs": "v2", "row_pool": True}, checkpoint)
        loaded = load_student(checkpoint)
        assert loaded.row_pool and loaded.aux is None
        export_student(net, binary)
        data = binary.read_bytes()
        assert data[:28] == b"PVZRN005" + struct.pack("<IIIII", 13, 1, 1, 3, 1)
        expected_size = 28 + 4 * sum(value.numel() for name, value in net.state_dict().items()
                                    if not name.startswith("aux."))
        assert len(data) == expected_size
        export_student(loaded, binary)
        assert binary.read_bytes() == data


def test_three_frame_aux_labels_and_sampled_cache():
    base = dict(sun=50, wave=0, scene={"wave_count": 10}, packets=[], zombies=[], lanes=list(range(5)))
    frames = [dict(base, tick=0, plants=[[0, 0, 0, 100, 100], [1, 1, 1, 100, 100]], mowers=[[2, 1], [3, 1]]),
              dict(base, tick=60, plants=[[1, 1, 1, 100, 100]], mowers=[[2, 2], [3, 1]],
                   action={"type": "wait", "ticks": 60}),
              dict(base, tick=120, plants=[], mowers=[], action={"type": "wait", "ticks": 60})]
    one = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(row_damage_labels(frames, 1), one)
    two = one.copy()
    two[0] = np.maximum(one[0], one[1])
    np.testing.assert_array_equal(row_damage_labels(frames, 2), two)
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / "data"
        archive = data / "seed_17" / "teacher" / "frames.jsonl"
        archive.parent.mkdir(parents=True)
        write_frames(archive, frames)
        cache = Path(directory) / "cache"
        old = cache_episode(archive, cache)
        for horizon, expected in ((1, one), (2, two)):
            cached = cache_episode(archive, cache, aux_horizon=horizon)
            assert cached != old
            with np.load(cached) as stored:
                assert stored["aux_version"] == 1
                assert stored["aux_horizon"] == horizon
                np.testing.assert_array_equal(stored["aux"], expected[:-1])
            for fraction in (1.0, 0.5):
                direct = load(data, {"teacher"}, wait_keep=fraction, dense_boards=True, aux_horizon=horizon)
                actual = load(data, {"teacher"}, wait_keep=fraction, cache_dir=cache,
                              dense_boards=True, aux_horizon=horizon)
                torch.testing.assert_close(direct[-1], actual[-1], rtol=0, atol=0)
                np.testing.assert_array_equal(actual[-1], expected[1:2] if fraction < 1 else expected[:-1])
        binary = archive.with_suffix(".npz")
        write_frames(binary, frames)
        assert cache_episode(binary, cache, aux_horizon=1) != binary


def test_aux_excludes_executed_shovel():
    before = dict(plants=[[1, 2, 0, 100, 100]], mowers=[])
    after = dict(plants=[], mowers=[], action={"type": "shovel", "row": 1, "col": 2})
    assert not row_damage_labels([before, after], 1).any()
