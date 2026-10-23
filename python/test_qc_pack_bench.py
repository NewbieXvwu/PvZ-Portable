import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import bench_qc_pack as pack


class PackEvidenceTest(unittest.TestCase):
    def test_duplicate_path_cannot_replace_missing_archive(self):
        with self.assertRaisesRegex(ValueError, 'exactly'):
            pack.validate_sources(['a', 'a'], ['a', 'b'])

    def test_same_count_wrong_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly'):
            pack.validate_sources(['a', 'c'], ['a', 'b'])

    def test_interrupted_cache_cannot_be_reused_as_cold_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack.require_cold(root)
            (root / '.feature_cache').mkdir()
            with self.assertRaisesRegex(ValueError, 'untimed'):
                pack.require_cold(root)

    def fixture(self, root, missing_rows=False, invalid_frame=False):
        view = root / 'frames'; labels = ['block0_fast_search', 'block1_fast_search']; paths = []
        for i in range(16):
            relative = f'seed_{100+i}/{labels[i//8]}/frames.npz'
            folder = (view / relative).parent; folder.mkdir(parents=True)
            (folder / 'meta.json').write_text(json.dumps(dict(task=dict(seed=100+i))))
            paths.append(relative)
        target = view / 'packs/test.npz'; target.parent.mkdir()
        ids = np.arange(16); rows = np.ones(16, dtype=np.int64)
        if missing_rows:
            ids[-1] = 0
        if invalid_frame:
            rows[-1] = 2
        np.savez(target, meta=np.asarray(json.dumps(dict(labels=labels))), source=np.asarray(paths),
                 rows=np.int64(16), source_archive=ids, source_row=rows)
        stage = dict(view=str(view), labels=labels, expected_sources=paths,
                     expected_frame_counts={p: 2 for p in paths})
        return target, stage

    def test_all_sources_listed_but_one_has_no_rows_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            file, stage = self.fixture(Path(tmp), missing_rows=True)
            with self.assertRaisesRegex(ValueError, 'all 16'):
                pack.audit_pack(file, stage)

    def test_packed_source_frame_must_exist_in_frozen_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            file, stage = self.fixture(Path(tmp), invalid_frame=True)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                pack.audit_pack(file, stage)

    def test_valid_pack_keeps_provenance_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            file, stage = self.fixture(Path(tmp))
            evidence = pack.audit_pack(file, stage)
            self.assertEqual(evidence['source_archives'], 16)
            self.assertEqual(evidence['rows'], 16)
            self.assertEqual(evidence['sha256'], pack.sha(file))


if __name__ == '__main__':
    unittest.main()
