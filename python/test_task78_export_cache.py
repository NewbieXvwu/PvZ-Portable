import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task78_export_cache import digest, ready, seal


class ExportReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.raw = self.base / 'raw'
        self.raw.mkdir()
        self.manifest = self.base / 'sources.json'
        self.export = self.base / 'export.json'
        self.model = self.base / 'model.bin'
        self.label = self.raw / 'candidate_0.json'
        for name in ('frames.npz', 'meta.json', 'engine', 'model.bin'):
            (self.base / name).write_text('{}')
        for name in ('candidate_0.json', 'root.json', 'summary.json'):
            (self.raw / name).write_text('{}')
        self.manifest.write_text(json.dumps(dict(archives=[dict(
            archive=str(self.base), frames_path=str(self.base / 'frames.npz'))])))
        (self.base / 'replay.json').write_text(json.dumps(dict(executable=str(self.base / 'engine'))))
        complete = self.raw / 'complete.json'
        complete.write_text(json.dumps(dict(identity=dict(models=dict(
            value=dict(path=str(self.model)))))))
        shards = []
        for horizon in (2500, 4000, 6000):
            target = self.base / f'shard_{horizon}.bin'
            target.write_bytes(b'shard')
            shards.append(dict(archive_index=24, horizon=horizon, path=str(target),
                provenance={str(p): digest(p) for p in (
                    complete, self.label, self.raw / 'root.json')}))
        self.export.write_text(json.dumps(dict(complete=True, training_eligible=False,
            source_manifest_sha256=digest(self.manifest), index_offset=24, shards=shards)))

    def seal(self):
        return seal(self.manifest, self.base, self.export, 24)

    def ready(self):
        return ready(self.manifest, self.base, self.export, 24)

    def test_changed_raw_label_invalidates_completed_export(self):
        self.seal()
        self.assertTrue(self.ready())
        self.label.write_text('[]')  # Same byte count; content changed.
        self.assertFalse(self.ready())

    def test_replaced_model_with_same_size_and_mtime_invalidates_export(self):
        self.seal()
        before = self.model.stat()
        replacement = self.base / 'replacement.bin'
        replacement.write_text('[]')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(self.model)
        self.assertFalse(self.ready())

    def test_partial_export_cannot_be_sealed(self):
        document = json.loads(self.export.read_text())
        document['shards'].pop()
        self.export.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            self.seal()
        self.assertFalse(self.ready())

    def test_added_candidate_invalidates_completed_export(self):
        self.seal()
        (self.raw / 'candidate_9.json').write_text('{}')
        self.assertFalse(self.ready())


if __name__ == '__main__':
    unittest.main()
