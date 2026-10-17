from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_rollout_policy import CompactBoards


class CompactBoardsTest(unittest.TestCase):
    def test_exact_batches(self):
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
        for channels in (92, 96):
            boards = np.zeros((7, channels, 6, 10), dtype=np.float16)
            boards[1, 0, 0, 0] = -0.0
            boards[2, 1, 2, 3] = .1234
            boards[3, -1, -1, -1] = -3.25
            boards[4] = .3
            boards[6, -1, 0, 0] = np.nextafter(np.float16(0), np.float16(1))
            store = CompactBoards(boards.shape[1:])
            store.append(boards[:3])
            store.append(boards[3:])
            store.finish()
            self.assertEqual(store.shape, boards.shape)
            self.assertEqual(store.indices.dtype, np.uint16)
            self.assertEqual(store.values.dtype, np.float16)
            self.assertEqual(store.offsets.dtype, np.int64)
            self.assertEqual(store.nbytes, store.nnz * 4 + 8 * (len(boards) + 1))
            for device in devices:
                for indices in ([6, 0, 3, 2, 1, 3], [0, 5], []):
                    actual = store.batch(torch.tensor(indices, dtype=torch.int64), device).cpu()
                    expected = torch.from_numpy(boards[indices]).float()
                    self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
