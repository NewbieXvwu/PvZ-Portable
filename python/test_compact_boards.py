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
            boards[5, 0, 0, 0] = np.float16(np.nan)   # a NaN payload must survive too
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

    def test_batch_with_planes_matches_dense(self):
        # task61: the per-identity scatter loop became one fused gather+scatter; the result
        # must stay exactly the dense board with each row's plane written over it.
        rng = np.random.default_rng(7)
        rows, channels = 40, 92
        cells_total = channels * 6 * 10
        boards = np.zeros((rows, channels, 6, 10), dtype=np.float16)
        planes = {}
        store = CompactBoards(boards.shape[1:])
        for group in range(4):
            cells = np.sort(rng.choice(cells_total, size=20 + 5 * group, replace=False)).astype(np.int64)
            values = (rng.random(len(cells)) + 0.5).astype(np.float16)
            planes[group] = (cells, values)
            covered = np.zeros(cells_total, dtype=bool)
            covered[cells] = True
            chunk = boards[group * 10:(group + 1) * 10]
            chunk[:, 0, 0, 0] = 1.0                       # sparse content alongside the plane
            chunk[:, 1, 1, 1] = -0.0                      # stored zero, still distinct from plane cells
            store.append(chunk, covered=covered, plane=planes[group])
        store.finish()
        flat = boards.reshape(rows, -1)
        for group in range(4):
            cells, values = planes[group]
            flat[group * 10:(group + 1) * 10][:, cells] = values
        dense = flat.reshape(boards.shape)
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            for indices in ([0, 39, 5, 5, 17], list(range(rows)), []):
                actual = store.batch(torch.tensor(indices, dtype=torch.int64), device).cpu()
                expected = torch.from_numpy(dense[indices]).float()
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))


if __name__ == "__main__":
    unittest.main()
