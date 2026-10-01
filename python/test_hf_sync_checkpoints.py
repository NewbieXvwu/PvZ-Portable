"""`scripts/hf_sync.py` 到底该传哪些检查点。

2026-10-01 的第一次真跑暴露：早先版本只挑最新的 `evaluated` 节点上传，
而四个 `reward_r*_v2` run 的 `resume.json` 全都指向 **`boundary`** 检查点 ——
也就是说下载回来的归档**接不上**，跨机续跑这个核心目的直接落空。

这里的检查点集合必须满足两条：续跑要的那个一定在里面，纯历史不要在里面。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import hf_sync  # noqa: E402


def cp(update: int, phase: str, stamp: int | None = None) -> str:
    return f"update_{update:06d}_{phase}_{update if stamp is None else stamp}.pt"


class Collect(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.run_dir = Path(holder.name)
        (self.run_dir / "runs" / "run_1").mkdir(parents=True)

    def add(self, name: str) -> Path:
        path = self.run_dir / "runs" / "run_1" / name
        path.write_bytes(b"weights")
        return path

    def resume(self, target: str | None) -> None:
        payload = {"checkpoint": target} if target else {}
        (self.run_dir / "resume.json").write_text(json.dumps(payload), encoding="utf-8")

    def collect(self) -> tuple[list[str], list[str]]:
        checkpoints, _, warnings = hf_sync._collect(self.run_dir)
        names = sorted(p.name for p in checkpoints)
        return names, warnings

    def test_every_analysis_phase_is_included(self) -> None:
        self.add(cp(0, "initial"))
        for update in (10, 20, 30, 40, 50):
            self.add(cp(update, "evaluated"))
        self.add(cp(60, "boundary"))
        self.resume(cp(60, "boundary"))

        names, _ = self.collect()
        self.assertEqual(len([n for n in names if "_evaluated_" in n]), 5)
        self.assertIn(cp(0, "initial"), names)
        self.assertIn(cp(60, "boundary"), names)

    def test_only_the_newest_trained_survives(self) -> None:
        for update in range(1, 40):
            self.add(cp(update, "trained"))
        self.resume(cp(39, "trained"))

        names, _ = self.collect()
        trained = [n for n in names if "_trained_" in n]
        self.assertEqual(trained, [cp(39, "trained")])

    def test_resume_target_is_included_even_when_it_is_not_the_newest(self) -> None:
        """核心回归：resume.json 指向 boundary，而最新的评估节点是另一个文件。"""
        self.add(cp(34, "evaluated", stamp=100))
        self.add(cp(34, "boundary", stamp=200))
        self.resume(cp(34, "boundary", stamp=200))

        names, _ = self.collect()
        self.assertIn(cp(34, "boundary", stamp=200), names)

    def test_resume_target_outside_the_normal_phase_set_is_still_included(self) -> None:
        self.add(cp(7, "trained", stamp=1))
        self.add(cp(9, "trained", stamp=2))
        self.add(cp(5, "resumed", stamp=3))
        # 指向一个更老的 trained，正常规则不会带它。
        self.resume(cp(7, "trained", stamp=1))

        names, _ = self.collect()
        self.assertIn(cp(7, "trained", stamp=1), names)

    def test_missing_resume_target_warns_instead_of_pretending_ok(self) -> None:
        self.add(cp(1, "trained"))
        self.resume("runs/run_1/update_000099_boundary_999.pt")

        _, warnings = self.collect()
        self.assertTrue(any("不存在" in w for w in warnings), warnings)

    def test_no_duplicates_when_resume_target_is_already_covered(self) -> None:
        self.add(cp(60, "boundary"))
        self.resume(cp(60, "boundary"))

        names, _ = self.collect()
        self.assertEqual(names.count(cp(60, "boundary")), 1)

    def test_resume_target_reads_absent_or_broken_json_as_none(self) -> None:
        self.assertIsNone(hf_sync._resume_target(self.run_dir))
        (self.run_dir / "resume.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(hf_sync._resume_target(self.run_dir))
        (self.run_dir / "resume.json").write_text('{"checkpoint": 7}', encoding="utf-8")
        self.assertIsNone(hf_sync._resume_target(self.run_dir))

    def test_unknown_phase_is_kept_rather_than_silently_dropped(self) -> None:
        """阶段名对不上正则时宁可多传，不要悄悄丢证据。"""
        self.add("update_000005_mystery_123.pt")
        names, _ = self.collect()
        self.assertIn("update_000005_mystery_123.pt", names)


if __name__ == "__main__":
    unittest.main()
