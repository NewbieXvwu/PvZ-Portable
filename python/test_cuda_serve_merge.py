"""A merged wave must answer every lane exactly as if that request had run alone.

The server holds `--gather-ms` to put several workers' continuations into one wave, which is what makes
the seam worth building: a wave costs its ~50 rounds, not its lanes (~/pvz-logs/cuda_status.md round 9).
That only holds if packing is a pure addressing change.  It was not: results were read back at each
request's own offset instead of its position in the packed wave, so the second slice in a merged wave got
the first slice's continuations -- and the search then scored candidates with another game's futures
(two merged games lost that each won alone, 2026-10-18).

Build the server first (it is not a CMake target):
  /usr/local/cuda/bin/nvcc -arch=sm_120 -O3 -std=c++17 --expt-relaxed-constexpr -I. -lcublas \
      -o build/rollout_cuda cuda/RolloutCuda.cu
and have a corpus (`scripts/cuda_root_corpus.py`) plus `artifacts/cuda/student/v14.pvzrn`
(`pvz_rollout_policy.export_student`).
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pvz_cuda_rollout import CudaRolloutClient, CudaServerError, FIELDS, RECORD, launch_server
from pvz_cuda_root import MAGIC, RECORD_ROOT, RECORD_TAG, VERSION, _records

BINARY = Path(os.environ.get("PVZ_CUDA_BINARY", ROOT / "build/rollout_cuda"))
CORPUS = Path(os.environ.get("PVZ_CUDA_CORPUS",
                             ROOT / "artifacts/cuda/corpus_l7_z15_v8/roots.cudaroot"))
NET = Path(os.environ.get("PVZ_CUDA_NET", ROOT / "artifacts/cuda/student/v14.pvzrn"))

# Distinct roots and distinct future seeds, so a swapped readback cannot look like a matching answer.
FIRST = [(root, 1600000000 + 7 * root) for root in (0, 1, 2)]
SECOND = [(root, 1700000000 + 7 * root) for root in (3, 4, 5, 0)]
HORIZON = 2500


@unittest.skipUnless(BINARY.is_file() and CORPUS.is_file() and NET.is_file(),
                     "requires build/rollout_cuda, a .cudaroot corpus and a .pvzrn student net")
class CudaServeMergeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pvz-cuda-serve-")
        self.addCleanup(self.directory.cleanup)
        self.processes = []
        self.addCleanup(self._stop)

    def _stop(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=30)

    def _serve(self, gather_ms: int, lanes: int = 64) -> str:
        """Keep each request below the server's half-full dispatch threshold so gathering is exercised."""
        socket_path = str(Path(self.directory.name) / f"serve-{gather_ms}.sock")
        log = open(socket_path + ".log", "w+")
        self.addCleanup(log.close)
        process = launch_server(BINARY, NET, socket_path, lanes=lanes, gather_ms=gather_ms,
                                start_timeout_ms=60000, log=log)
        self.addCleanup(process.stdout.close)
        self.processes.append(process)
        return socket_path

    def _answer(self, socket_path: str, groups: list[list[tuple[int, int]]]) -> list[list[dict]]:
        clients = []
        for pairs in groups:
            client = CudaRolloutClient(socket_path)
            client.connect()
            self.addCleanup(client.close)
            clients.append(client)
        for client, pairs in zip(clients, groups):
            client.begin(CORPUS, [(root, seed, None) for root, seed in pairs], horizon=HORIZON,
                         max_actions=3000)
        return [client.answer() for client in clients]

    def test_merge_does_not_move_a_lane(self):
        quiet = self._serve(gather_ms=0)
        alone_first, alone_second = self._answer(quiet, [FIRST, SECOND])

        busy = self._serve(gather_ms=600)
        merged_first, merged_second = self._answer(busy, [FIRST, SECOND])

        import numpy as np
        for name, (alone, alone_leaf), (merged, merged_leaf) in (("first", alone_first, merged_first),
                                    ("second", alone_second, merged_second)):
            self.assertEqual(len(alone), len(merged))
            for index, (reference, packed) in enumerate(zip(alone, merged)):
                self.assertEqual(reference, packed,
                                 f"lane {index} of the {name} request changed when it shared a wave")
            np.testing.assert_array_equal(alone_leaf, merged_leaf)
        self.assertNotEqual(alone_first[0][0], alone_second[0][0], "the two requests ran the same continuation")
        # Observe actual merging. A previous lanes=4 fixture dispatched the first 3-lane request
        # immediately at the half-full threshold, so its green check proved nothing about merging.
        client = CudaRolloutClient(busy)
        self.addCleanup(client.close)
        self.assertEqual(client.status()["waves"], 1)

    def test_a_request_spanning_waves_preserves_every_leaf(self):
        pairs = FIRST + SECOND + FIRST
        wide = self._answer(self._serve(gather_ms=0), [pairs])[0]
        sliced = self._answer(self._serve(gather_ms=0, lanes=4), [pairs])[0]
        self.assertEqual(wide[0], sliced[0])
        import numpy as np
        np.testing.assert_array_equal(wide[1], sliced[1])

    def test_empty_request_is_not_sent(self):
        client = CudaRolloutClient(self._serve(gather_ms=0))
        client.connect()
        self.addCleanup(client.close)
        self.assertEqual(client.run(CORPUS, [], horizon=HORIZON), ([], None))
        with self.assertRaises(CudaServerError):
            client.run(CORPUS, [(9999, 1, None)], horizon=HORIZON, max_actions=3000)

    def test_forced_wait_finishes_in_initialization(self):
        client = CudaRolloutClient(self._serve(gather_ms=0))
        self.addCleanup(client.close)
        records, leaves = client.run(CORPUS, [(0, 123, {"type": "wait", "ticks": 5})],
                                     horizon=1, max_actions=1)
        self.assertEqual(records[0]["decisions"], 1)
        self.assertEqual(records[0]["ticks"], 5)
        self.assertEqual(records[0]["errorFlags"], 0)
        self.assertEqual(leaves.shape, (1, 5668))

    def test_all_connected_workers_waiting_skips_gather_delay(self):
        client = CudaRolloutClient(self._serve(gather_ms=2000))
        client.connect()
        self.addCleanup(client.close)
        started = time.monotonic()
        records, _ = client.run(CORPUS, [(0, 123, {"type": "wait", "ticks": 1})],
                                horizon=1, max_actions=1)
        self.assertEqual(records[0]["errorFlags"], 0)
        self.assertLess(time.monotonic() - started, 1.0,
                        "the only connected worker spent the whole gather window waiting for itself")


class RootArrayGrowthTest(unittest.TestCase):
    """A root must continue the same whether or not its corpus was big enough to grow the arrays.

    The round loop reaches the root and wave-table arrays through `Arguments`, which `DeviceOpen` fills
    from the allocation standing at the time (256 records).  `GrowRoots` frees and reallocates both, so a
    corpus above 256 roots left the captured pointers pointing at freed memory: the *host* writes the new
    buffer, the *kernels* read the old mapping.  On the 425-root z15 corpus that was an illegal access in
    UpdateWaves for root 35 and a wave table taken from whatever bytes the reused address happened to hold
    for most of the rest (compute-sanitizer, 2026-10-18).  Corpora of 256 roots or fewer never grow, which
    is why the phase-1 comparison over the z13 corpus did not show it.
    """

    LARGE = Path(os.environ.get("PVZ_CUDA_CORPUS_LARGE",
                                ROOT / "artifacts/cuda/corpus_l7_z15_v8/roots.cudaroot"))
    # Present in both corpora, spread far enough apart that a shifted read cannot hide.
    ROOTS = (0, 35, 200, 255)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pvz-cuda-grow-")
        self.addCleanup(self.directory.cleanup)

    def _prefix(self, keep: int) -> Path:
        """The large corpus truncated to its level record plus its first ``keep`` roots, so those roots
        keep the same device indices in both files."""
        out = bytearray()
        roots = 0
        for kind, payload in _records(self.LARGE.read_bytes()):
            if kind == RECORD_ROOT:
                if roots >= keep:
                    continue
                roots += 1
            out += RECORD_TAG.pack(MAGIC, VERSION, kind) + payload
        path = Path(self.directory.name) / f"prefix-{keep}.cudaroot"
        path.write_bytes(bytes(out))
        return path

    def _batch(self, corpus: Path, label: str) -> tuple[bytes, bytes]:
        pairs = Path(self.directory.name) / f"pairs-{label}.txt"
        pairs.write_text("".join(f"{root} {1600000000 + 7 * root}\n" for root in self.ROOTS))
        out = Path(self.directory.name) / f"records-{label}.bin"
        leaf = Path(self.directory.name) / f"leaf-{label}.bin"
        run = subprocess.run([str(BINARY), "--roots", str(corpus), "--net", str(NET),
                              "--pairs", str(pairs), "--horizon", str(HORIZON),
                              "--max-actions", "3000", "--out", str(out), "--leaf-out", str(leaf)],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, f"{label}: {run.stdout}\n{run.stderr}")
        return out.read_bytes(), leaf.read_bytes()

    def test_growing_the_root_array_does_not_move_a_lane(self):
        if not (BINARY.is_file() and NET.is_file() and self.LARGE.is_file()):
            self.skipTest("requires build/rollout_cuda, a .pvzrn student net and a large corpus")
        # A corpus of exactly kGrowFloor roots is allocated once and never reallocated, so its lanes read
        # the array through the pointer DeviceOpen captured while it was still current.
        small = self._prefix(keep=256)
        total = sum(1 for kind, _ in _records(self.LARGE.read_bytes()) if kind == RECORD_ROOT)
        self.assertGreater(total, 256, f"{self.LARGE} holds {total} roots: too few to grow the array")

        alone, grown = self._batch(small, "alone"), self._batch(self.LARGE, "grown")
        self.assertEqual(alone[0], grown[0], "a record changed because the corpus needed a bigger array")
        self.assertEqual(alone[1], grown[1], "a leaf encoding changed with the corpus size")

        # Not vacuous: four different roots, four different continuations, none of them an error lane.
        records = [RECORD.unpack_from(grown[0], index * RECORD.size) for index in range(len(self.ROOTS))]
        self.assertEqual(len({record for record in records}), len(self.ROOTS),
                         "the four roots produced the same answer")
        for root, record in zip(self.ROOTS, records):
            fields = dict(zip(FIELDS, record[:len(FIELDS)]))
            self.assertEqual(fields["errorFlags"], 0, f"root {root} failed its import")
            self.assertGreater(fields["ticks"], 0, f"root {root} simulated nothing")


@unittest.skipUnless(BINARY.is_file() and CORPUS.is_file() and NET.is_file(),
                     "requires build/rollout_cuda, a .cudaroot corpus and a .pvzrn student net")
class PrivatePathTest(unittest.TestCase):
    """The one-block-per-lane path must be a scheduling change, not a different simulation.

    PVZ_CUDA_PRIVATE_LIMIT switches the wave between the SoA kernels and PrivateRoundKernel
    (state in shared memory, one block per lane).  The search uses the private path for small
    waves and the stage-0 benchmark uses SoA, so a divergence here would silently split the two
    sets of numbers that the gates compare.
    """

    def test_private_and_soa_paths_agree(self):
        pairs = FIRST + SECOND
        with tempfile.TemporaryDirectory(prefix="pvz-cuda-private-") as directory:
            pair_file = Path(directory) / "pairs.txt"
            pair_file.write_text("".join(f"{root} {seed}\n" for root, seed in pairs))
            answers = []
            for limit in ("0", "4096"):
                out = Path(directory) / f"results-{limit}.bin"
                run = subprocess.run(
                    [str(BINARY), "--net", str(NET), "--roots", str(CORPUS), "--pairs", str(pair_file),
                     "--horizon", str(HORIZON), "--lanes", str(len(pairs)), "--bench", "full",
                     "--out", str(out)],
                    env={**os.environ, "PVZ_CUDA_PRIVATE_LIMIT": limit},
                    capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                answers.append(out.read_bytes())
        self.assertEqual(answers[0], answers[1], "the private path changed a continuation")


if __name__ == "__main__":
    unittest.main()
