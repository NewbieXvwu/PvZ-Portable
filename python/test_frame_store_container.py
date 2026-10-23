"""The archive container: LZMA, lossless for every dtype, and packs stay mmap-able.

pvz_frame_store._save writes ZIP_LZMA (59.6% of the old zlib container on a 12-archive sample,
because the frames JSON is the bulk).  This pins the two things that must stay true for that to be
safe: np.load reads it without a codec switch, and every array comes back with the same dtype and
values.  It also pins the exception: training packs are written ZIP_STORED by
scripts/pack_training_rows.py so np.load(mmap_mode="r") can map them, and must not be LZMA.
"""
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pvz_frame_store import frame_bytes, save_arrays  # noqa: E402


def sample() -> dict:
    return {
        "frames": np.frombuffer(b'{"tick":0}\n{"tick":60}\n', dtype=np.uint8),
        "encoder": np.array("5c3448262051", dtype="<U64"),
        "boards": np.zeros((3, 4, 2), dtype=np.float16),
        "scalars": np.linspace(0, 1, 7, dtype=np.float32),
        "rows": np.array([1, 2], dtype=np.int64),
        "hints": np.array([True, False, True]),
    }


class ContainerTest(unittest.TestCase):
    def test_archive_is_lzma_and_round_trips_every_dtype(self):
        arrays = sample()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frames.npz"
            save_arrays(path, **arrays)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual({info.compress_type for info in archive.infolist()}, {zipfile.ZIP_LZMA})
                self.assertEqual(sorted(info.filename for info in archive.infolist()),
                                 sorted(name + ".npy" for name in arrays))
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(sorted(stored.files), sorted(arrays))
                for name, array in arrays.items():
                    self.assertEqual(stored[name].dtype, array.dtype, name)
                    self.assertTrue(np.array_equal(stored[name], array), name)
            self.assertEqual(frame_bytes(path), arrays["frames"].tobytes())

    def test_zero_dimensional_string_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cache.npz"
            save_arrays(path, digest=np.array("abc", dtype="<U64"), rows=np.array([1], dtype=np.int64))
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(str(stored["digest"]), "abc")
                self.assertEqual(stored["digest"].shape, ())

    def test_stored_packs_stay_uncompressed(self):
        """scripts/pack_training_rows.py uses np.savez so a pack can be mapped; LZMA must not leak in."

        numpy 2.5 still hands npz members back as ndarrays, so what this pins is the container:
        ZIP_STORED members are the ones a future numpy (or a direct zipfile reader) can map.
        """
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pack.npz"
            np.savez(path, rows=np.arange(10, dtype=np.int64),
                     board_values=np.arange(10, dtype=np.float32))
            with zipfile.ZipFile(path) as archive:
                self.assertEqual({info.compress_type for info in archive.infolist()}, {zipfile.ZIP_STORED})
            with np.load(path, mmap_mode="r", allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["board_values"], np.arange(10, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
