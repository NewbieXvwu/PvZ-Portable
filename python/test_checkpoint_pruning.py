"""Retention policy for `trained` checkpoints.

Every update used to write a fresh ~43 MB checkpoint and nothing ever removed the
older ones, so a single 500k-decision run accumulated 70+ of them (3.1 GB). These
checks pin down what survives pruning and what must never be touched.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pvz_research import TRAINED_CHECKPOINT_KEEP, prune_trained_checkpoints


def make(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"checkpoint")
    return path


def trained(update: int, stamp: int | None = None) -> str:
    return f"update_{update:06d}_trained_{update if stamp is None else stamp}.pt"


class PruneTrainedCheckpoints(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.run_dir = Path(holder.name)

    def names(self, pattern: str = "*.pt") -> list[str]:
        return sorted(path.name for path in self.run_dir.glob(pattern))

    def test_keeps_the_newest_n(self) -> None:
        for update in range(10):
            make(self.run_dir, trained(update))

        removed = prune_trained_checkpoints(self.run_dir, 3, set())

        self.assertEqual(len(removed), 7)
        self.assertEqual(self.names(), [trained(7), trained(8), trained(9)])

    def test_ordering_is_chronological_not_mtime(self) -> None:
        # The probe reads the last 8 updates; the filename embeds the update
        # number, so lexicographic order must decide, not filesystem times.
        make(self.run_dir, trained(0, stamp=999_999_999_999))
        make(self.run_dir, trained(1, stamp=1))

        prune_trained_checkpoints(self.run_dir, 1, set())

        self.assertEqual(self.names(), [trained(1, stamp=1)])

    def test_milestones_are_never_pruned(self) -> None:
        make(self.run_dir, "update_000000_initial_1.pt")
        make(self.run_dir, "update_000050_boundary_2.pt")
        make(self.run_dir, "update_000025_evaluated_3.pt")
        make(self.run_dir, "update_000075_evaluated_4.pt")
        for update in range(20):
            make(self.run_dir, trained(update))

        prune_trained_checkpoints(self.run_dir, 1, set())

        self.assertEqual(self.names("update_*_initial_*.pt"), ["update_000000_initial_1.pt"])
        self.assertEqual(self.names("update_*_boundary_*.pt"), ["update_000050_boundary_2.pt"])
        self.assertEqual(self.names("update_*_evaluated_*.pt"),
                         ["update_000025_evaluated_3.pt", "update_000075_evaluated_4.pt"])
        self.assertEqual(self.names("update_*_trained_*.pt"), [trained(19)])

    def test_protected_survives_even_when_oldest(self) -> None:
        oldest = make(self.run_dir, trained(0))
        for update in range(1, 6):
            make(self.run_dir, trained(update))

        prune_trained_checkpoints(self.run_dir, 1, {oldest})

        self.assertEqual(self.names(), [trained(0), trained(5)])

    def test_noop_when_at_or_below_keep(self) -> None:
        for update in range(3):
            make(self.run_dir, trained(update))

        self.assertEqual(prune_trained_checkpoints(self.run_dir, 8, set()), [])
        self.assertEqual(len(self.names()), 3)

    def test_zero_keep_retains_only_protected(self) -> None:
        keep_me = make(self.run_dir, trained(0))
        for update in range(1, 5):
            make(self.run_dir, trained(update))

        removed = prune_trained_checkpoints(self.run_dir, 0, {keep_me})

        self.assertEqual(len(removed), 4)
        self.assertEqual(self.names(), [trained(0)])

    def test_other_runs_are_out_of_scope(self) -> None:
        sibling = self.run_dir / "run_2"
        sibling.mkdir()
        for update in range(5):
            make(self.run_dir, trained(update))
            make(sibling, trained(update))

        prune_trained_checkpoints(self.run_dir, 1, set())

        self.assertEqual(len(self.names()), 1)
        self.assertEqual(len(sorted(path.name for path in sibling.glob("*.pt"))), 5)

    def test_default_covers_the_late_policy_probe_window(self) -> None:
        # scripts/research_late_policy_probe.py reads update_history[-8:]; a
        # smaller default would silently break that diagnostic.
        self.assertGreaterEqual(TRAINED_CHECKPOINT_KEEP, 8)


if __name__ == "__main__":
    unittest.main()
