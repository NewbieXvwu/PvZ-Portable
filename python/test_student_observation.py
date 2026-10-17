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
from pvz_frame_store import cache_episode, encode_v2lite_frames, encoder_signature, read_frames, write_frames
from pvz_student_observation import StudentHistory
from pvz_value_net import _featurize_reference
from train_rollout_policy import load

RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))
BINARY = ROOT / "build/pvz-portable"


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / "main.pak").is_file(), "requires simulator and game resources")
class StudentObservationTest(unittest.TestCase):
    def test_live_archive_replay(self):
        output = ROOT / "artifacts/v2lite"
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as directory:
            directory = Path(directory)
            task = {"task_id": "test", "level": 7, "seeds": [30000], "deck": [0, 1, 2, 3, 4, 5]}
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
            for version in ("v1", "v2lite"):
                search_play._W.clear()
                torch.set_num_threads(1)
                torch.manual_seed(0)
                search_play._W.update(fast=policy.RolloutNet(8, "meanmax", obs=version).eval(), cpp_rollouts=False)
                search_play._W["fast"].plant.bias.data.fill_(2)
                args = SimpleNamespace(manifest=str(manifest), task_id="test", resource_dir=str(RESOURCE),
                                       frames=str(directory / version), frames_label="test", frames_format="npz",
                                       rollout_policy=str(checkpoint), dagger_beta=1.0, max_actions=3000)
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
                if version == "v2lite":
                    with np.load(archive) as stored:
                        raw_frames = stored["frames"]
                    np.savez_compressed(archive, frames=raw_frames)
                frames = read_frames(archive)
                history = StudentHistory()
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
                if version == "v2lite":
                    torch.save({"state_dict": search_play._W["fast"].state_dict(), "pool": "meanmax", "obs": version}, checkpoint)
                    self.assertEqual(policy.load_student(checkpoint).obs, version)
                    self.assertTrue(any(bool(board[95].any()) for _, board, _ in recorded))
                    self.assertTrue(any(bool(board[93].any()) for _, board, _ in recorded))
                    binary_cache = cache_episode(archive, directory / "cache", version)
                    text = archive.with_suffix(".jsonl")
                    write_frames(text, frames)
                    with np.load(binary_cache) as cached:
                        encoded = encode_v2lite_frames(read_frames(text), text.read_bytes())
                        for name in encoded:
                            np.testing.assert_array_equal(cached[name], encoded[name])
                    for context in (False, True):
                        expected = load(directory / version, {"test_fast"}, .1, with_context=context, obs=version)
                        actual = load(directory / version, {"test_fast"}, .1, cache_dir=directory / "cache",
                                      with_context=context, obs=version, precompute_workers=2)
                        for left, right in zip(expected, actual):
                            if isinstance(left, torch.Tensor):
                                torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
                            else:
                                self.assertEqual(left, right)
                    with self.assertRaisesRegex(ValueError, "v2lite"):
                        policy.export_student(search_play._W["fast"], directory / "bad.bin")
                    from pvz_event_env import EventWaitEnv
                    with self.assertRaisesRegex(ValueError, "v2lite"):
                        EventWaitEnv(RESOURCE).load_rollout_net(checkpoint)
                    with self.assertRaisesRegex(ValueError, "v2lite"):
                        search_play._play({"seed": 30000, "arm": "fast_search", "args": args})
                    with self.assertRaisesRegex(ValueError, "v2lite"):
                        search_play._rollout_policy(None, {}, {}, 0)
                    value = directory / "value.pt"
                    from pvz_value_net import ValueNet
                    torch.save({"state_dict": ValueNet().state_dict()}, value)
                    search_play._init_worker(str(value), str(checkpoint))
                    self.assertFalse(search_play._W["cpp_rollouts"])
            print(json.dumps(recorded_count), flush=True)


if __name__ == "__main__":
    unittest.main()
