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
    def test_historical_encoder_and_packet_signatures_stay_frozen(self):
        import hashlib
        import inspect
        from unittest.mock import patch
        import pvz_frame_store as store
        import pvz_rollout_policy as policy

        expected = {
            'v1': '5c34482620514a7434e638058f9b75d05e51a8aa9c18b634a8f80e4dd424f008',
            'v2lite': 'b6020e4131d35d7ff9ef0fb0996f877e6f3ffd2122045a65bd5e87be2c065aa5',
            'v2': '5031d93614db9ecff271dcaa6aa8645d7a799452f170e5ebf699cc2fb8c1df91',
        }
        store.encoder_signature.cache_clear()
        self.assertEqual({obs: store.encoder_signature(obs) for obs in expected}, expected)
        packet_source = (inspect.getsource(policy.packet_features) + inspect.getsource(store.encode_packets)
                         + str((policy.PACKETS, policy.PlantTypes)))
        self.assertEqual(hashlib.sha256(packet_source.encode()).hexdigest(),
                         'c84d95592bfecadbb64fe2726fa22798defe3ff845c2d40b3f120eefe26821ce')
        # Old bytes must hit both caches without invoking the encoder or writer.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'frames.jsonl'
            source.write_bytes(b'{"tick":0}\n')
            key = hashlib.sha256(str(source.resolve()).encode()).hexdigest() + '.npz'
            caches = root / 'cache'
            for subdir, signature in [('v2', expected['v2']), ('packets', packet_source)]:
                target = caches / subdir / key
                target.parent.mkdir(parents=True)
                digest = hashlib.sha256(signature.encode() + source.read_bytes()).hexdigest()
                np.savez(target, digest=digest)
            with patch.object(store, '_save', side_effect=AssertionError('rewrote a valid historical cache')):
                self.assertEqual(store.cache_episode(source, caches, 'v2'), caches / 'v2' / key)
                self.assertEqual(store.cache_packets(source, caches), caches / 'packets' / key)

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
