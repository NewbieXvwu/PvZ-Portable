import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import search_play
import pvz_rollout_policy as policy
from pvz_frame_store import cache_episode, encode_student_frames, encoder_signature, read_frames, write_frames
from pvz_student_observation import StudentHistory
from pvz_value_net import _featurize_reference
from pack_training_rows import packed_load

RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))
BINARY = ROOT / "build/pvz-portable"


class ObservationEncodingTest(unittest.TestCase):
    def frame(self):
        return {"tick": 100, "wave": 1, "sun": 100, "scene": {"wave_count": 30},
                "plants": [[0, 0, 4, 300, 300], [1, 0, 2, 300, 300], [2, 0, 1, 300, 300]],
                "zombies": [[0, 3, 80.0, .35, 200, True, False, 1000, 0, 1]],
                "packets": [], "mowers": [], "lanes": [0, 1, 2, 3, 4]}

    def raw(self):
        return {"wave_timer": 2250, "huge_wave_countdown": 750, "wave_health": 100,
                "wave_start_health": 200, "sun_countdown": 612, "sun_pending": 50,
                "plants": [[0, 0, 4, 0, 750, 0, 0, 0], [1, 0, 2, 0, 0, 50, 0, 0],
                           [2, 0, 1, 0, 0, 0, 1250, 2500]],
                "zombies": [[1, 185, 0]], "projectiles": [[0, 80.0, 20], [0, 80.0, 20]]}

    def test_new_fields_leave_v1_and_v2lite_bitwise_unchanged(self):
        old = self.frame()
        new = {**old, "v2": self.raw()}
        for left, right in zip(policy.features(old, None), policy.features(new, None)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        histories = [StudentHistory(), StudentHistory()]
        for tick in (100, 250):
            old["tick"] = new["tick"] = tick
            left, right = histories[0].extra(old), histories[1].extra(new)
            expected = np.zeros((4, 6, 10), np.float32)
            expected[:3, 0, 0] = [.5, .35, .35]
            expected[3, :3, 0] = (tick - 100) / 1500.0
            np.testing.assert_array_equal(left[0].view(np.uint32), expected.view(np.uint32))
            for actual, original in zip(right, left):
                np.testing.assert_array_equal(actual.view(np.uint32), original.view(np.uint32))

    def test_v2_values_and_old_frame_availability(self):
        frame = self.frame()
        missing, scalars = StudentHistory("v2").extra(frame)
        self.assertFalse(missing[3:].any())
        self.assertFalse(scalars.any())
        frame["v2"] = self.raw()
        board, scalars = StudentHistory("v2").extra(frame)
        np.testing.assert_array_equal(board[:, 0, 0], np.array([.5, .5, .5, 1, .5, .5, .5, 0, .2], np.float32))
        self.assertEqual(board[6, 1, 0], .5)
        self.assertEqual(board[7, 2, 0], .5)
        np.testing.assert_array_equal(scalars, [.5, 1, .5, .5, .25, 0, 1])
        frame["zombies"][0][3] = -.35
        board, _ = StudentHistory("v2").extra(frame)
        self.assertEqual(board[2, 0, 0], -.5)


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / "main.pak").is_file(), "requires simulator and game resources")
class StudentObservationTest(unittest.TestCase):
    def test_live_archive_replay(self):
        output = ROOT / "artifacts/v2lite"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as directory:
            directory = Path(directory)
            # Keep this encoder fixture alive long enough to observe pole vaulters. The
            # untrained network can now spend its sun replacing existing plants, so its
            # survival time is no longer a reliable way to arrange feature coverage.
            task = {"task_id": "test", "level": 7, "seeds": [30000], "deck": [0, 1, 2, 3, 4, 5],
                    "preplanted": [[0, row, col] for row in range(5) for col in (0, 1)]}
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"tasks": [task]}))
            torch.manual_seed(0)
            net = policy.RolloutNet(8, "meanmax")
            net.plant.bias.data.fill_(2)
            checkpoint = directory / "model.pt"
            torch.save({"state_dict": net.state_dict(), "pool": "meanmax"}, checkpoint)
            self.assertEqual(policy.load_student(checkpoint).obs, "v1")
            self.assertEqual(encoder_signature(), "5c34482620514a7434e638058f9b75d05e51a8aa9c18b634a8f80e4dd424f008")
            recorded_count = {}
            for version in ("v1", "v2lite", "v2"):
                search_play._W.clear()
                torch.set_num_threads(1)
                torch.manual_seed(0)
                search_play._W.update(fast=policy.RolloutNet(8, "meanmax", obs=version).eval())
                search_play._W["fast"].plant.bias.data.fill_(2)
                search_play._W["rollout_file"] = str(directory / f"{version}.bin")
                policy.export_student(search_play._W["fast"], search_play._W["rollout_file"])
                args = SimpleNamespace(manifest=str(manifest), task_id="test", resource_dir=str(RESOURCE),
                                       frames=str(directory / version), frames_label="test",
                                       rollout_policy=str(checkpoint), dagger_beta=1.0)
                recorded = []
                decision = [0]
                deciding = [False]
                original_action = search_play._fast_action
                original_features = policy.features if version == "v1" else StudentHistory.features

                def capture(*args, **kwargs):
                    board, scalars = original_features(*args, **kwargs)
                    if deciding[0]:
                        recorded.append((decision[0], board.clone(), scalars.clone()))
                    return board, scalars

                def step(*args):
                    deciding[0] = True
                    action = original_action(*args)
                    deciding[0] = False
                    decision[0] += 1
                    return action

                target = policy if version == "v1" else StudentHistory
                try:
                    with patch.object(target, "features", capture), patch.object(search_play, "_fast_action", step):
                        result = search_play._play({"seed": 30000, "arm": "fast", "args": args})
                finally:
                    search_play._W.pop("env").close()
                archive = directory / version / "seed_30000/test_fast/frames.npz"
                if version != "v1":
                    with np.load(archive) as stored:
                        raw_frames = stored["frames"]
                    np.savez_compressed(archive, frames=raw_frames)
                frames = read_frames(archive)
                history = StudentHistory(version)
                replay = [history.features(frame, frames[index - 1] if index else None)
                          for index, frame in enumerate(frames[:-1])]
                for index, board, scalars in recorded:
                    expected_board, expected_scalars = replay[index]
                    torch.testing.assert_close(board, expected_board[:board.shape[0]], rtol=0, atol=0)
                    torch.testing.assert_close(scalars, expected_scalars[:len(scalars)], rtol=0, atol=0)
                    reference, reference_scalars = _featurize_reference(frames[index])
                    self.assertTrue(torch.equal(board[:92], reference))
                    self.assertTrue(torch.equal(scalars[:113], reference_scalars))
                self.assertGreater(len(recorded), 10)
                recorded_count[version] = {"decisions": result["decisions"], "compared": len(recorded)}
                if version != "v1":
                    torch.save({"state_dict": search_play._W["fast"].state_dict(), "pool": "meanmax", "obs": version}, checkpoint)
                    self.assertEqual(policy.load_student(checkpoint).obs, version)
                    self.assertTrue(any(bool(board[95].any()) for _, board, _ in recorded))
                    self.assertTrue(any(bool(board[93].any()) for _, board, _ in recorded))
                    binary_cache = cache_episode(archive, directory / "cache", version)
                    text = archive.with_suffix(".jsonl")
                    write_frames(text, frames)
                    with np.load(binary_cache) as cached:
                        encoded = encode_student_frames(read_frames(text), text.read_bytes(), version)
                        for name in encoded:
                            np.testing.assert_array_equal(cached[name], encoded[name])
                    for context in (False, True):
                        packed = packed_load(directory / version, {"test_fast"}, .1,
                                             with_context=context, obs=version,
                                             cache_dir=directory / "cache", rebuild=True)
                        self.assertGreater(len(packed[2]), 10)
                    policy.export_student(search_play._W["fast"], directory / "student.bin")
                    from pvz_event_env import EventWaitEnv
                    with self.assertRaisesRegex(ValueError, "exported C\\+\\+"):
                        EventWaitEnv(RESOURCE).load_rollout_net(checkpoint)
                    value = directory / "value.pt"
                    from pvz_value_net import ValueNet
                    torch.save({"state_dict": ValueNet().state_dict()}, value)
                    search_play._init_worker(str(value), str(checkpoint))
                    self.assertTrue(Path(search_play._W["rollout_file"]).is_file())
            print(json.dumps(recorded_count), flush=True)


if __name__ == "__main__":
    unittest.main()
