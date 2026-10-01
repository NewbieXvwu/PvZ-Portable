"""`scripts/prune_research_checkpoints.py` 的保留判定。

它把 `pvz_research.py` 的策略应用到**已跑完**的 run 上（策略本身只在保存时生效）。
这里钉住两件事：保留数跟策略一致，以及**被指针引用的检查点绝不进删除列表**。

第二条是有来历的：曾经有一个 shell 版本"只保留最新一个 trained"，
而 `resume.json` 指向的可能是更早的某个快照 —— 那种写法会直接把续跑链删断。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import prune_research_checkpoints as tool  # noqa: E402


def trained(update: int, stamp: int | None = None) -> str:
    return f"update_{update:06d}_trained_{update if stamp is None else stamp}.pt"


class PruneTool(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.run_dir = Path(holder.name)
        (self.run_dir / "runs" / "run_1").mkdir(parents=True)

    def add(self, name: str) -> Path:
        path = self.run_dir / "runs" / "run_1" / name
        path.write_bytes(b"weights")
        return path

    def write_json(self, name: str, payload: dict) -> None:
        (self.run_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_protected_paths_reads_both_pointer_files(self) -> None:
        self.write_json("training_state.json", {"checkpoint": "runs/run_1/a.pt"})
        self.write_json("resume.json", {"checkpoint": "runs/run_1/b.pt", "sha256": "x"})
        names = {p.name for p in tool.protected_paths(self.run_dir)}
        self.assertEqual(names, {"a.pt", "b.pt"})

    def test_protected_paths_walks_nested_structures(self) -> None:
        self.write_json("training_state.json",
                        {"runs": [{"checkpoint": "runs/run_1/deep.pt"}],
                         "note": "not-a-checkpoint"})
        names = {p.name for p in tool.protected_paths(self.run_dir)}
        self.assertEqual(names, {"deep.pt"})

    def test_missing_or_broken_pointer_files_do_not_crash(self) -> None:
        self.assertEqual(tool.protected_paths(self.run_dir), set())
        (self.run_dir / "resume.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(tool.protected_paths(self.run_dir), set())

    def test_plan_keeps_the_newest_n(self) -> None:
        for update in range(12):
            self.add(trained(update))
        snapshots, doomed, _ = tool.plan(self.run_dir, 8)
        self.assertEqual(len(snapshots), 12)
        self.assertEqual(sorted(p.name for p in doomed), [trained(u) for u in range(4)])

    def test_resume_target_survives_even_when_it_is_not_the_newest(self) -> None:
        """核心回归：resume.json 指向一个更老的 trained，它不能被删。"""
        for update in range(12):
            self.add(trained(update))
        self.write_json("resume.json", {"checkpoint": f"runs/run_1/{trained(0)}"})

        _, doomed, protected = tool.plan(self.run_dir, 8)
        self.assertIn(trained(0), {p.name for p in protected})
        self.assertNotIn(trained(0), {p.name for p in doomed})
        # 其余三个仍然该删
        self.assertEqual(sorted(p.name for p in doomed), [trained(u) for u in (1, 2, 3)])

    def test_keep_zero_deletes_every_snapshot_except_pointers(self) -> None:
        for update in range(4):
            self.add(trained(update))
        self.write_json("resume.json", {"checkpoint": f"runs/run_1/{trained(1)}"})

        _, doomed, _ = tool.plan(self.run_dir, 0)
        self.assertEqual(sorted(p.name for p in doomed),
                         [trained(0), trained(2), trained(3)])

    def test_only_trained_snapshots_are_candidates(self) -> None:
        self.add("update_000005_evaluated_1.pt")
        self.add("update_000005_boundary_2.pt")
        self.add("update_000000_initial_3.pt")
        for update in range(4):
            self.add(trained(update))

        _, doomed, _ = tool.plan(self.run_dir, 1)
        self.assertEqual(sorted(p.name for p in doomed), [trained(u) for u in (0, 1, 2)])

    def test_keep_matches_the_live_policy_by_default(self) -> None:
        from pvz_research import TRAINED_CHECKPOINT_KEEP
        self.assertEqual(tool.TRAINED_CHECKPOINT_KEEP, TRAINED_CHECKPOINT_KEEP)


if __name__ == "__main__":
    unittest.main()
