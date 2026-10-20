import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
import pytest

from pvz_rollout_policy import RolloutNet, export_student, load_student
from pvz_frame_store import (cache_aux, cache_episode, encode_student_frames, encoder_signature,
                             frame_bytes, row_damage_labels, write_frames)

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


@pytest.mark.parametrize("obs", ["v1", "v2"])
def test_three_frame_aux_labels_and_sampled_cache(obs):
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
        old = cache_episode(archive, cache, obs)
        modified = old.stat().st_mtime_ns
        for horizon, expected in ((1, one), (2, two)):
            cached = cache_aux(archive, cache, horizon)
            assert cached != old
            with np.load(cached) as stored:
                assert set(stored.files) == {"rows", "aux", "digest"}
                assert stored["aux"].dtype == np.uint8
                np.testing.assert_array_equal(stored["rows"], [1, 2])
                np.testing.assert_array_equal(stored["aux"], expected[:-1])
            timestamp = cached.stat().st_mtime_ns
            assert cache_aux(archive, cache, horizon) == cached
            assert cached.stat().st_mtime_ns == timestamp
            for fraction, workers in ((1.0, 1), (0.5, 2)):
                direct = load(data, {"teacher"}, wait_keep=fraction, dense_boards=True, obs=obs, aux_horizon=horizon)
                actual = load(data, {"teacher"}, wait_keep=fraction, cache_dir=cache,
                              dense_boards=True, obs=obs, precompute_workers=workers, aux_horizon=horizon)
                torch.testing.assert_close(direct[-1], actual[-1], rtol=0, atol=0)
                np.testing.assert_array_equal(actual[-1], expected[1:2] if fraction < 1 else expected[:-1])
            assert cache_episode(archive, cache, obs) == old
            assert old.stat().st_mtime_ns == modified
            assert set(cached.parent.glob("*.npz")) == {cached}
        binary = archive.with_suffix(".npz")
        source = frame_bytes(archive)
        if obs == "v1":
            write_frames(binary, frames)
        else:
            np.savez_compressed(binary, frames=np.frombuffer(source, dtype=np.uint8), encoder=encoder_signature(obs),
                                **encode_student_frames(frames, source, obs))
        with patch("pvz_frame_store.encode_frames", side_effect=AssertionError("unexpected feature encoding")):
            assert cache_episode(binary, cache, obs) == binary
            before = load(data, {"teacher"}, cache_dir=cache, obs=obs, dense_boards=True)
            actual = load(data, {"teacher"}, cache_dir=cache, obs=obs, dense_boards=True, aux_horizon=1)
            for plain, auxiliary in zip(before[:6], actual[:6]):
                torch.testing.assert_close(plain, auxiliary, rtol=0, atol=0)
            np.testing.assert_array_equal(actual[-1], one[:-1])
            assert cache_episode(binary, cache, obs) == binary
        side = cache_aux(binary, cache, 1)
        with np.load(side) as stored:
            arrays = dict(stored)
        arrays["rows"] = arrays["rows"][::-1]
        np.savez_compressed(side, **arrays)
        with pytest.raises(AssertionError):
            load(data, {"teacher"}, cache_dir=cache, obs=obs, aux_horizon=1)


def test_aux_excludes_executed_shovel():
    before = dict(plants=[[1, 2, 0, 100, 100]], mowers=[])
    after = dict(plants=[], mowers=[], action={"type": "shovel", "row": 1, "col": 2})
    assert not row_damage_labels([before, after], 1).any()
