"""`scripts/lint_skills.py` 的判据边界。

这个闸门存在的理由是一条实测结论：2026-10-03 的 S3 实验里，模型产出了 4 条
「可复用的判断」，其中只有 1 条有反事实证据。没有判据时，要么写噪音、要么不写。

这里钉住的是**闸门真的会拒**的那些情况 —— 尤其是「证据是编的」。
证据真伪的判据是：`frames.jsonl` 第 N 行记着原局当时**真实做出**的动作，
所以证据里的 `before` 必须与它一致。存档是跑出来的，不是写出来的。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import lint_skills  # noqa: E402

POTATO = {"type": "plant", "packet": 4, "row": 3, "col": 6}
WAIT = {"type": "wait", "ticks": 60}


def write_archive(root: Path, actions: list) -> Path:
    """造一个最小存档：第 0 帧无动作，第 N 帧是 actions[N-1]。"""
    archive = root / "ep"
    archive.mkdir(parents=True)
    with (archive / "frames.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"tick": 0, "action": None}) + "\n")
        for i, action in enumerate(actions, 1):
            fh.write(json.dumps({"tick": i * 60, "action": action}) + "\n")
    return archive


def write_skill(root: Path, name: str, body: str, evidence: str | None = None) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    text = f"---\nname: {name}\ndescription: 演示用\n---\n\n{body}\n"
    if evidence is not None:
        text += f"\n## 证据\n\n```pvz-evidence\n{evidence}\n```\n"
    (skill / "SKILL.md").write_text(text, encoding="utf-8")
    return skill


class LintSkills(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        # 决策 1..4 的真实动作：第 3 个（决策 3）是那一枚土豆雷。
        self.archive = write_archive(self.root, [WAIT, POTATO, WAIT, WAIT])
        self.skills = self.root / "skills"
        self.skills.mkdir()

    def lint(self, skill: Path) -> tuple[list[str], list[str]]:
        return lint_skills.lint_skill(skill, check_archives=True)

    # ---------------------------------------------------------------- 通过

    def test_two_verified_points_pass(self) -> None:
        skill = write_skill(
            self.skills,
            "potato",
            "# 规则\n\n土豆雷要早种。\n",
            f"kind: mechanism\n"
            f"claim: 土豆雷种下到能炸要 1620 tick\n"
            f"falsifier: 若提前量不够也能炸到有效目标，这条不成立\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:3 | 通关\n"
            f"point: {self.archive} | 1 | wait:60 | wait:120 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertEqual(fails, [])

    def test_procedure_needs_no_evidence_points(self) -> None:
        skill = write_skill(
            self.skills,
            "flow",
            "# 流程\n\n先落盘，再回放。\n",
            "kind: procedure\nrationale: 跳到最后一步就没有可回放的靶子\n",
        )
        fails, _ = self.lint(skill)
        self.assertEqual(fails, [])

    def test_rationale_may_wrap_onto_continuation_lines(self) -> None:
        skill = write_skill(
            self.skills,
            "wrapped",
            "# 流程\n\n正文。\n",
            "kind: procedure\nrationale: 第一句\n  第二句接着写\n  第三句\n",
        )
        fails, _ = self.lint(skill)
        self.assertEqual(fails, [])
        fields = lint_skills._parse_evidence((skill / "SKILL.md").read_text(encoding="utf-8"))
        self.assertIn("第二句接着写", fields["rationale"])
        self.assertIn("第三句", fields["rationale"])

    # ---------------------------------------------- 拒：三问里的第 2 问

    def test_single_point_is_rejected(self) -> None:
        """一个点上的成功可能是运气 —— S3 实测 (3,3) 通而 (3,4) 更差。"""
        skill = write_skill(
            self.skills,
            "one",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:3 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("至少要有 2 个" in f for f in fails), fails)

    def test_same_decision_twice_does_not_count_as_two_points(self) -> None:
        skill = write_skill(
            self.skills,
            "dup",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:3 | 通关\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:4 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("只落在 1 个决策点" in f for f in fails), fails)

    # ------------------------------------------------ 拒：证据是编的

    def test_before_must_match_the_recorded_action(self) -> None:
        """这条是闸门存在的理由：存档里那一帧的动作是原局跑出来的，改不了。"""
        skill = write_skill(
            self.skills,
            "fabricated",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.archive} | 2 | plant:4:3:3 | plant:4:3:6 | 通关\n"
            f"point: {self.archive} | 1 | wait:60 | wait:120 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("与存档对不上" in f for f in fails), fails)

    def test_out_of_range_decision_is_rejected(self) -> None:
        skill = write_skill(
            self.skills,
            "oob",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.archive} | 999 | wait:60 | wait:120 | 通关\n"
            f"point: {self.archive} | 1 | wait:60 | wait:120 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("越界" in f for f in fails), fails)

    def test_missing_archive_is_rejected(self) -> None:
        skill = write_skill(
            self.skills,
            "gone",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.root}/nope | 1 | wait:60 | wait:120 | 通关\n"
            f"point: {self.root}/nope | 2 | wait:60 | wait:120 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("存档不存在" in f for f in fails), fails)

    def test_check_archives_off_only_checks_shape(self) -> None:
        """存档不在本机时（另一台机器写的 skill）应该只查结构。"""
        skill = write_skill(
            self.skills,
            "elsewhere",
            "# 规则\n\n正文。\n",
            "kind: mechanism\nclaim: a\nfalsifier: b\n"
            "point: /nope/one | 1 | wait:60 | wait:120 | 通关\n"
            "point: /nope/two | 2 | wait:60 | wait:120 | 通关\n",
        )
        fails, _ = lint_skills.lint_skill(skill, check_archives=False)
        self.assertEqual(fails, [])

    # ------------------------------------------------------ 拒：结构与必填

    def test_missing_evidence_block_is_rejected(self) -> None:
        skill = write_skill(self.skills, "bare", "# 规则\n\n正文。\n")
        fails, _ = self.lint(skill)
        self.assertTrue(any("没有 ```pvz-evidence 块" in f for f in fails), fails)

    def test_procedure_without_rationale_is_rejected(self) -> None:
        skill = write_skill(self.skills, "flowless", "# 流程\n\n正文。\n", "kind: procedure\n")
        fails, _ = self.lint(skill)
        self.assertTrue(any("rationale" in f for f in fails), fails)

    def test_falsifier_must_differ_from_claim(self) -> None:
        skill = write_skill(
            self.skills,
            "echo",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: 土豆雷要早种\nfalsifier: 土豆雷要早种\n"
            f"point: {self.archive} | 1 | wait:60 | wait:120 | 通关\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:3 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("一模一样" in f for f in fails), fails)

    def test_unknown_kind_is_rejected(self) -> None:
        skill = write_skill(self.skills, "weird", "# 规则\n\n正文。\n", "kind: guess\n")
        fails, _ = self.lint(skill)
        self.assertTrue(any("kind" in f for f in fails), fails)

    def test_name_must_match_directory(self) -> None:
        skill = self.skills / "wrongdir"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: something-else\ndescription: x\n---\n\n正文。\n", encoding="utf-8"
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("与目录名" in f for f in fails), fails)

    def test_broken_point_arity_is_reported_not_crashed(self) -> None:
        skill = write_skill(
            self.skills,
            "short",
            "# 规则\n\n正文。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.archive} | 1 | wait:60\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:3 | 通关\n",
        )
        fails, _ = self.lint(skill)
        self.assertTrue(any("5 段" in f for f in fails), fails)

    # ------------------------------------------------------------ 提示

    def test_absolute_path_in_body_is_a_note_not_a_failure(self) -> None:
        skill = write_skill(
            self.skills,
            "abs",
            "# 规则\n\n存档放在 /tmp/ep 下面。\n",
            f"kind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {self.archive} | 1 | wait:60 | wait:120 | 通关\n"
            f"point: {self.archive} | 2 | plant:4:3:6 | plant:4:3:3 | 通关\n",
        )
        fails, notes = self.lint(skill)
        self.assertEqual(fails, [])
        self.assertTrue(any("绝对路径" in n for n in notes), notes)


if __name__ == "__main__":
    unittest.main()
