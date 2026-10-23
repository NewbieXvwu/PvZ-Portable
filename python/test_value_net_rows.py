"""The value-net row store must reproduce the dense featurization exactly (task32).

`train_value_net.load_games` used to keep every training sample as a dense (92, 6, 10) float16 board
(11 040 bytes each, 5.3 GB for the 477k samples of the 16.6 GB run). It now appends them to the CSR row
store. This test featurizes the same frames a second time and demands the same bits back.
"""
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
from pvz_frame_store import write_frames  # noqa: E402
from pvz_value_net import featurize  # noqa: E402
from train_rollout_policy import CompactBoards  # noqa: E402
from train_value_net import FRAME_STRIDE, load_games  # noqa: E402


def frames(count: int) -> list:
    out = []
    for index in range(count):
        out.append({
            "tick": 100 + 60 * index, "wave": 1 + index // 10, "sun": 50 + 7 * index,
            "scene": {"wave_count": 20},
            "plants": [[row, col, 4, 300 - 7 * index, 300]
                       for row in range(2) for col in range(min(3, 1 + index // 5))],
            "zombies": [[row, 3, 60.0 + 25.0 * ((index + row) % 12), .35, 200 - 5 * index, True,
                         index % 3 == 0, index % 5 == 0] for row in range(3)],
            "mowers": [[0, 1], [1, 1], [2, 2]],
            "lanes": [0, 1, 2, 3, 4],
            "packets": [[slot, slot, slot < 4, 0 if slot < 4 else 500, 750, 100] for slot in range(7)],
        })
    return out


class ValueNetRowStoreTest(unittest.TestCase):
    def test_boards_match_a_second_featurization(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            kept = []
            for seed in (11, 12):
                archive = directory / f"seed_{seed}" / "fast"
                archive.mkdir(parents=True)
                game = frames(40)
                write_frames(archive / "frames.jsonl", game)
                (archive / "meta.json").write_text(json.dumps({"outcome": {"won": seed == 11},
                                                                "task": {"seed": seed}}))
                kept.append(game)
            games, train = load_games(directory, training=True)
            self.assertIsInstance(train["boards"], CompactBoards)
            self.assertEqual(len(games), 2)
            expected = [featurize(frame) for game in kept
                        for frame in [game[index] for index in range(0, len(game) - 1, FRAME_STRIDE)]]
            rows = len(expected)
            self.assertEqual(train["boards"].shape[0], rows)
            for start in range(0, rows, 7):
                batch = torch.arange(start, min(start + 7, rows))
                actual = train["boards"].batch(batch, "cpu")
                reference = torch.stack([expected[index][0] for index in batch.tolist()]).half().float()
                self.assertTrue(torch.equal(actual.view(torch.int32), reference.view(torch.int32)))
                self.assertTrue(torch.equal(train["scalars"][batch],
                                            torch.stack([expected[index][1] for index in batch.tolist()])))
            self.assertTrue(torch.equal(train["labels"], torch.tensor([1.0] * 13 + [0.0] * 13)))


if __name__ == "__main__":
    unittest.main()
