"""Tests for the shared digest/path helpers in ``pvz_common``."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from pvz_common import (
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    REPLAY_FORMAT_VERSION,
    TASK_VERSION,
    TRAINING_SEED,
    VALUE_RANGE,
    canonical_digest,
    git_metadata,
    is_generated_artifact,
    sha256_bytes,
    sha256_file,
)


class SharedDigestTests(unittest.TestCase):
    def test_sha256_bytes_matches_the_reference_implementation(self) -> None:
        self.assertEqual(sha256_bytes(b"pvz-portable"), hashlib.sha256(b"pvz-portable").hexdigest())

    def test_sha256_file_streams_the_same_digest_as_sha256_bytes(self) -> None:
        payload = bytes(range(256)) * 5000  # crosses the 1 MiB read chunk boundary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(payload)

            self.assertEqual(sha256_file(path), sha256_bytes(payload))

    def test_canonical_digest_ignores_key_order_but_not_content(self) -> None:
        self.assertEqual(canonical_digest({"b": 1, "a": 2}), canonical_digest({"a": 2, "b": 1}))
        self.assertNotEqual(canonical_digest({"a": 1}), canonical_digest({"a": 2}))
        self.assertEqual(len(canonical_digest({"a": 1})), 64)

    def test_generated_artifacts_are_recognised(self) -> None:
        for path in ("artifacts/adventure2_level7/x.json", "experiment_manifest.json",
                     "working_tree.patch", "replays/search_seed_1.jsonl.gz", "traces/a.jsonl"):
            with self.subTest(path=path):
                self.assertTrue(is_generated_artifact(path))
        for path in ("python/pvz_env.py", "src/LawnApp.cpp", "README.md", "docs/experiment_manifest.md"):
            with self.subTest(path=path):
                self.assertFalse(is_generated_artifact(path))

    def test_git_metadata_reports_a_revision_or_gives_up_cleanly(self) -> None:
        revision, dirty = git_metadata(Path(__file__).resolve().parent.parent)

        self.assertTrue(revision is None or len(revision) == 40)
        self.assertTrue(dirty is None or isinstance(dirty, bool))

    def test_git_metadata_returns_nothing_outside_a_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(git_metadata(Path(directory)), (None, None))

    def test_version_constants_are_the_documented_values(self) -> None:
        self.assertEqual(ENV_PROTOCOL_VERSION, 4)
        self.assertEqual(REPLAY_FORMAT_VERSION, 5)
        # Bumped 2 -> 3 with the derived lane features: the environment now emits
        # ``sun_income_rate``, so the observation schema is not the old one.
        self.assertEqual(OBSERVATION_VERSION, 3)
        self.assertEqual(TASK_VERSION, 2)
        self.assertEqual(TRAINING_SEED, 17)
        self.assertEqual(VALUE_RANGE, (-1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
