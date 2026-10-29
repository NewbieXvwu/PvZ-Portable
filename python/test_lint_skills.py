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
import probe  # noqa: E402

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


class ProbeGate(unittest.TestCase):
    """probe 条件的验收边界。

    这一层是"教师写的失败模式能不能变成自动筛子"的开关，所以必须钉住**会拒**的
    情况：条件在自家证据上抓不到（比主张窄）、反例被误标（泛化过头）、
    用了词表外的字段（发明新判据）。
    """

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.skills = self.root / "skills"
        self.skills.mkdir()

    def _archive(self, frames: list) -> Path:
        """写一份带完整字段的存档（probe 要读植物/僵尸/阳光/波次）。"""
        archive = self.root / "ep"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "meta.json").write_text(
            json.dumps({"task": {"deck": [0, 1, 2, 3, 4, 5], "policy": "scripted"}}),
            encoding="utf-8")
        with (archive / "frames.jsonl").open("w", encoding="utf-8") as fh:
            for fr in frames:
                fh.write(json.dumps(fr) + "\n")
        return archive

    def _frames(self) -> list:
        """决策 1、2 都在僵尸贴脸（x=100）的那条路上种向日葵 → 该命中；
        决策 3 在一条没有僵尸的路上种向日葵 → 反例，不该命中。

        注意僵尸要放在**决策前**的那一帧上（策略当时看到的就是它）。
        """
        zombie = [1, 0, 100.0, 0.0, 200, True, False, 0, 0, 0]

        def frame(tick, plants, zombies, action=None):
            return {"tick": tick, "wave": 1, "sun": 100, "plants": plants,
                    "zombies": zombies, "action": action,
                    "row_terrain": [1] * 6, "lanes": list(range(6))}
        return [
            frame(0, [], [zombie]),                                     # 0 开局
            frame(60, [[1, 0, 1, 300, 300]], [zombie],                  # 1 送菜
                  {"type": "plant", "packet": 1, "row": 1, "col": 0}),
            frame(120, [[1, 0, 1, 300, 300], [1, 1, 1, 300, 300]],      # 2 再送一次
                  [zombie], {"type": "plant", "packet": 1, "row": 1, "col": 1}),
            frame(180, [[1, 0, 1, 300, 300], [1, 1, 1, 300, 300],       # 3 反例
                        [3, 0, 1, 300, 300]], [],
                  {"type": "plant", "packet": 1, "row": 3, "col": 0}),
        ]

    def _skill(self, name: str, probe: str | None, points: str, counter: str = "") -> Path:
        skill = self.skills / name
        skill.mkdir(parents=True, exist_ok=True)
        fm_probe = f"probe:\n{probe}\n" if probe else ""
        text = (f"---\nname: {name}\ndescription: 演示用\n{fm_probe}---\n\n"
                f"# 规则\n\n正文。\n\n## 证据\n\n```pvz-evidence\n"
                f"kind: mechanism\nclaim: a\nfalsifier: b\n{points}{counter}\n```\n")
        (skill / "SKILL.md").write_text(text, encoding="utf-8")
        return skill

    def test_probe_passes_when_points_hit_and_counterpoints_do_not(self) -> None:
        archive = self._archive(self._frames())
        skill = self._skill("ok", "  action_type: plant\n  plant_role: [producer]\n"
                                  "  lane_front_zombie_x_max: 260",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关\n"
                            f"point: {archive} | 2 | plant:1:1:1 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, notes = lint_skills.lint_skill(skill, check_archives=True)
        self.assertEqual(fails, [], fails)
        self.assertTrue(any("probe" in n and "通过" in n for n in notes), notes)

    def test_probe_is_rejected_when_own_evidence_is_not_caught(self) -> None:
        """条件号称描述决策 1，却抓不到它 → 拒（比主张窄）。"""
        archive = self._archive(self._frames())
        skill = self._skill("narrow", "  action_type: plant\n  plant_role: [wall]",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("probe" in f for f in fails), fails)

    def test_probe_is_rejected_when_counterexample_is_flagged(self) -> None:
        """条件泛化过头，把反例也标红 → 拒。"""
        archive = self._archive(self._frames())
        skill = self._skill("wide", "  action_type: plant\n  plant_role: [producer]",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("误判" in f for f in fails), fails)

    def test_probe_with_unknown_field_is_rejected(self) -> None:
        """词表外的字段 = 发明新判据，整条拒。"""
        archive = self._archive(self._frames())
        skill = self._skill("invented", "  action_type: plant\n  my_own_score_max: 5",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("不认识" in f for f in fails), fails)

    def test_skill_without_probe_is_still_valid(self) -> None:
        archive = self._archive(self._frames())
        skill = self._skill("noprobe", None,
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关\n"
                            f"point: {archive} | 3 | plant:1:3:0 | wait:60 | 通关")
        fails, notes = lint_skills.lint_skill(skill, check_archives=True)
        self.assertEqual(fails, [], fails)
        self.assertFalse(any("probe" in n for n in notes), notes)

    # ---------------------------------------------- 跨路 / 列位置（S7 的覆盖缺口）

    def _frames_lane_choice(self, deck_packet: int = 3) -> list:
        """3 路上有僵尸，动作落在别的路上。

        决策 1、2 把防御植物种在**没僵尸**的 2 路（错路）→ 该命中；
        决策 3 种在**有僵尸**的 3 路（对路）→ 反例，不该命中。
        决策 4 是 wait：它没有落点，带行字段的条件判不了它，**不算命中**。
        """
        zombie = [3, 0, 600.0, 0.0, 200, True, False, 0, 0, 0]

        def frame(tick, plants, action=None):
            return {"tick": tick, "wave": 1, "sun": 100, "plants": plants,
                    "zombies": [zombie], "action": action,
                    "row_terrain": [1] * 6, "lanes": list(range(6))}
        plant = lambda row, col: {"type": "plant", "packet": deck_packet,
                                  "row": row, "col": col}
        return [
            frame(0, []),
            frame(60, [[2, 1, 3, 300, 300]], plant(2, 1)),            # 1 错路
            frame(120, [[2, 1, 3, 300, 300], [2, 3, 3, 300, 300]],    # 2 错路
                  plant(2, 3)),
            frame(180, [[2, 1, 3, 300, 300], [2, 3, 3, 300, 300],     # 3 对路
                        [3, 1, 3, 300, 300]], plant(3, 1)),
            frame(240, [[2, 1, 3, 300, 300], [2, 3, 3, 300, 300],     # 4 wait
                        [3, 1, 3, 300, 300]], {"type": "wait", "ticks": 60}),
        ]

    def test_cross_lane_probe_passes(self) -> None:
        """「种在了没敌人的路，而别的路有敌人」——S7 那条同行拦截机制的判据。

        `plant_role: [defense]` 必须同时抓到坚果墙（wall 类）和土豆雷（地雷类），
        所以类别是集合不是单标签。packet 3 → 坚果墙，就是被单标签漏掉的那个。
        """
        archive = self._archive(self._frames_lane_choice())
        skill = self._skill("crosslane",
                            "  action_type: plant\n  plant_role: [defense]\n"
                            "  lane_zombie_count_max: 0\n  other_lane_zombie_count_min: 1",
                            f"point: {archive} | 1 | plant:3:2:1 | wait:60 | 通关\n"
                            f"point: {archive} | 2 | plant:3:2:3 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, notes = lint_skills.lint_skill(skill, check_archives=True)
        self.assertEqual(fails, [], fails)
        self.assertTrue(any("probe" in n and "通过" in n for n in notes), notes)

    def test_wait_action_does_not_satisfy_lane_conditions(self) -> None:
        """wait 没有落点，带行字段的条件判不了它。

        如果写成"跳过这条约束"，每条带行字段的条件都会把 wait 全标红 ——
        筛出来的"可疑局"里会有一大半是等待动作。
        """
        archive = self._archive(self._frames_lane_choice())
        frames = self._frames_lane_choice()
        hits = probe.evaluate(frames, [0, 1, 2, 3, 4, 5],
                              {"lane_zombie_count_max": 0})
        self.assertEqual([h["decision"] for h in hits], [1, 2])
        self.assertNotIn(4, [h["decision"] for h in hits])
        self.assertTrue(archive.exists())

    def test_column_probe_passes(self) -> None:
        """「开局把阳光花在外侧列」——S7 那条屋顶经济机制的判据（列位置）。"""
        zombie_free = []

        def frame(tick, plants, action=None):
            return {"tick": tick, "wave": 0, "sun": 50, "plants": plants,
                    "zombies": zombie_free, "action": action,
                    "row_terrain": [1] * 6, "lanes": list(range(6))}
        plant = lambda col: {"type": "plant", "packet": 2, "row": 2, "col": col}
        frames = [
            frame(0, []),
            frame(60, [[2, 6, 2, 300, 300]], plant(6)),      # 1 外侧 → 命中
            frame(120, [[2, 6, 2, 300, 300], [2, 8, 2, 300, 300]],
                  plant(8)),                                  # 2 外侧 → 命中
            frame(180, [[2, 6, 2, 300, 300], [2, 8, 2, 300, 300],
                        [2, 2, 2, 300, 300]], plant(2)),      # 3 内侧 → 反例
        ]
        archive = self._archive(frames)
        skill = self._skill("outercol",
                            "  action_type: plant\n  plant_role: [other]\n"
                            "  action_col_min: 3\n  wave_max: 0",
                            f"point: {archive} | 1 | plant:2:2:6 | wait:60 | 通关\n"
                            f"point: {archive} | 2 | plant:2:2:8 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, notes = lint_skills.lint_skill(skill, check_archives=True)
        self.assertEqual(fails, [], fails)
        self.assertTrue(any("probe" in n and "通过" in n for n in notes), notes)

    def test_probe_threshold_must_be_an_integer(self) -> None:
        """`sun_max: 50` 和 `sun_max: "50"` 在 Python 里比较结果不同 —— 静默走错路。"""
        archive = self._archive(self._frames())
        skill = self._skill("strnum", "  action_type: plant\n  sun_max: 不是数字",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3")
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("必须是整数" in f for f in fails), fails)

    # ------------------------------------------------------------ 反例点

    def test_counterpoints_may_repeat(self) -> None:
        """反例和证据点一样是列表字段。只能写一条的话，写的人会把两条挤进一行。"""
        archive = self._archive(self._frames())
        skill = self._skill("twocounter",
                            "  action_type: plant\n  plant_role: [producer]\n"
                            "  lane_front_zombie_x_max: 260",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关\n"
                            f"point: {archive} | 2 | plant:1:1:1 | wait:60 | 通关",
                            f"\ncounterpoint: {archive} | 3\ncounterpoint: {archive} | 2")
        # 决策 2 是证据点也是反例 → 反例被误标，闸门该拒它。
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("误判" in f for f in fails), fails)
        self.assertFalse(any("写了两遍" in f for f in fails), fails)

    def test_counterpoint_needs_archive_and_decision(self) -> None:
        archive = self._archive(self._frames())
        skill = self._skill("prosecounter", "  action_type: plant",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关\n"
                            f"point: {archive} | 2 | plant:1:1:1 | wait:60 | 通关",
                            "\ncounterpoint: 有时候这样种也没事")
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("2 段" in f for f in fails), fails)

    def test_counterpoint_archive_must_exist(self) -> None:
        archive = self._archive(self._frames())
        skill = self._skill("ghostcounter", "  action_type: plant",
                            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关\n"
                            f"point: {archive} | 2 | plant:1:1:1 | wait:60 | 通关",
                            f"\ncounterpoint: {self.root}/nope | 1")
        fails, _ = lint_skills.lint_skill(skill, check_archives=True)
        self.assertTrue(any("counterpoint 的存档不存在" in f for f in fails), fails)

    def test_counterpoint_in_prose_is_not_evidence(self) -> None:
        """正文里写一句「counterpoint: ...」不算反例 —— 只认 ```pvz-evidence 块里的。"""
        archive = self._archive(self._frames())
        skill = self.skills / "prosecounterpoint"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: prosecounterpoint\ndescription: 演示用\n"
            "probe:\n  action_type: plant\n---\n\n"
            "# 规则\n\n正文里随口提一句 counterpoint: 这里不算证据。\n\n"
            "## 证据\n\n```pvz-evidence\nkind: mechanism\nclaim: a\nfalsifier: b\n"
            f"point: {archive} | 1 | plant:1:1:0 | wait:60 | 通关\n"
            f"point: {archive} | 2 | plant:1:1:1 | wait:60 | 通关\n```\n",
            encoding="utf-8")
        ok, report, has_probe = probe.check_skill(str(skill))
        self.assertTrue(has_probe)
        self.assertFalse(ok)
        self.assertIn("没有反例点", report)


if __name__ == "__main__":
    unittest.main()
