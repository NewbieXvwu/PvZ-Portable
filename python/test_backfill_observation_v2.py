import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import backfill_observation_v2 as backfill
from pvz_frame_store import read_frames, write_frames


class BackfillObservationTest(unittest.TestCase):
    def test_labels_unchanged_and_mismatch_skips_whole_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/seed_1/teacher/frames.npz"
            source.parent.mkdir(parents=True)
            original = [{"tick": 0, "label": {"type": "plant", "packet": 2, "row": 0, "col": 0},
                         "targets": [[2, 0, 0, .8]], "verification": {"samples": 9}}]
            write_frames(source, original)
            before = source.read_bytes()
            (source.parent / "meta.json").write_text(json.dumps({"task": {"seed": 1}}))
            kw = {"level": 7, "deck": [0, 1, 2, 3, 4, 5], "task_extra": {}}
            for mismatch in (False, True):
                args = SimpleNamespace(out=root / str(mismatch), manifest="test", task_id="test")
                obs = {"tick": int(mismatch), "v2": {"plants": [], "zombies": []}}
                env = SimpleNamespace(reset=lambda **kwargs: (obs, {}))
                with patch.object(backfill, "_ARGS", args), patch.object(backfill, "_ENV", env), \
                        patch.object(backfill, "load_training_task", return_value={}), \
                        patch.object(backfill, "training_task_kwargs", return_value=kw), \
                        patch.object(backfill, "_frame", side_effect=lambda value: value), \
                        patch("render_episode._frame", side_effect=lambda value: value):
                    result = backfill.backfill(source)
                self.assertEqual(result["mismatch"], mismatch)
                destination = args.out / "frames/seed_1/teacher/frames.npz"
                self.assertEqual(destination.exists(), not mismatch)
                self.assertEqual(source.read_bytes(), before)
                if not mismatch:
                    actual = read_frames(destination)
                    self.assertEqual(actual[0].pop("v2"), obs["v2"])
                    self.assertEqual(actual, original)


if __name__ == "__main__":
    unittest.main()
