"""回放学生时，「用哪个任务、哪副卡组」必须来自任务清单本身。

清单里的 level / deck / 波数上限 / 僵尸倍率 / 预种植物**只能从清单取**，
手抄到命令行上一定会漂（第 49 关的卡组就这么漂过：[0,1,2,3,4,5,33] vs 清单 [0,1,3,4,7,33]）。
卡槽错位的后果不是"差一点"：学生说「用第 2 槽」，本机就种出另一个植物，而这一局会照常跑完、
给出一个结果，然后被当成证据 —— 所以缺卡组必须是硬错，不是警告。
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import render_episode  # noqa: E402


TASK = {
    "task_id": "demo_task",
    "level": 49,
    "zombie_count_multiplier": 1.5,
    "wave_cap": 5,
    "sun_start": 50,
    "preplanted": [[1, 0, 0], [0, 2, 3]],
    "playthrough": 2,
    "deck": [0, 1, 3, 4, 7, 33],
    "seeds": [61216, 61217, 61218],
}


class LoadTrainingTask(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.manifest = Path(holder.name) / "train.json"
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "tasks": [TASK, {"task_id": "other",
                                                              "level": 7,
                                                              "deck": [0, 1],
                                                              "seeds": []}]}),
            encoding="utf-8")

    def test_finds_the_task_by_id(self) -> None:
        self.assertEqual(render_episode.load_training_task(self.manifest, "demo_task"),
                         TASK)

    def test_missing_task_lists_what_is_available(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            render_episode.load_training_task(self.manifest, "nope")
        self.assertIn("demo_task", str(ctx.exception))
        self.assertIn("other", str(ctx.exception))

    def test_kwargs_come_from_the_manifest(self) -> None:
        kw = render_episode.training_task_kwargs(TASK, seed=None)
        self.assertEqual(kw["level"], 49)
        self.assertEqual(kw["deck"], [0, 1, 3, 4, 7, 33])
        self.assertEqual(kw["seed"], 61216)          # 不给 seed 就取清单里第一个
        self.assertEqual(kw["task_extra"]["wave_cap"], 5)
        self.assertEqual(kw["task_extra"]["zombie_count_multiplier"], 1.5)
        self.assertEqual(kw["task_extra"]["preplanted"], ((1, 0, 0), (0, 2, 3)))

    def test_explicit_seed_wins(self) -> None:
        self.assertEqual(render_episode.training_task_kwargs(TASK, seed=99)["seed"], 99)

    def test_first_aid_is_off_by_default_and_carried_when_set(self) -> None:
        """坚果包扎术是 profile 字段（不是 TaskSpec 字段）：缺省关，清单里写了才开。

        旧任务没有这个键 —— 它必须解析成 False，否则所有历史任务的 reset 报文都会变。
        """
        self.assertFalse(render_episode.training_task_kwargs(TASK)["first_aid_owned"])
        self.assertNotIn("first_aid_owned", render_episode.training_task_kwargs(TASK)["task_extra"])
        self.assertTrue(render_episode.training_task_kwargs(dict(TASK, first_aid_owned=True))["first_aid_owned"])

    def test_profile_for_deck_carries_first_aid(self) -> None:
        from pvz_tasks import profile_for_deck
        self.assertFalse(profile_for_deck((0, 1)).first_aid_owned)
        self.assertTrue(profile_for_deck((0, 1), True).first_aid_owned)
        # 打开包扎术不改卡组推导出来的其它东西（槽位数、升级植物所有权）
        self.assertEqual(profile_for_deck((0, 43), True).owned_upgrade_plants, (43,))
        self.assertEqual(profile_for_deck((0, 43), True).seed_slot_count, 2)

    def test_no_preplanted_means_no_key(self) -> None:
        """预种植物是空的就不传 —— `TaskSpec` 的默认值就是空，传个空元组只是噪音。"""
        task = dict(TASK, preplanted=[])
        self.assertNotIn("preplanted", render_episode.training_task_kwargs(task)["task_extra"])

    def test_fog_mode_replays_from_archive_without_changing_legacy_tasks(self) -> None:
        from episode_query import _replay_task_extra
        self.assertNotIn('fog_observation', _replay_task_extra(TASK))
        for mode in ('human', 'xray'):
            saved = dict(TASK, fog_observation=mode)
            self.assertEqual(_replay_task_extra(saved)['fog_observation'], mode)
            self.assertEqual(render_episode.training_task_kwargs(saved)['task_extra']['fog_observation'], mode)


class ReplayRefusesTheWrongDeck(unittest.TestCase):
    """学生回放不带检查点或卡组时必须报错，而不是悄悄用别的卡组跑完。"""

    def test_student_without_checkpoint_or_deck_is_rejected(self) -> None:
        for kwargs in ({}, {"checkpoint": "/nonexistent/model.pt"}):
            with self.assertRaises(ValueError) as ctx:
                render_episode.collect("/nonexistent", 1, 49, "student", **kwargs)
            self.assertIn("deck", str(ctx.exception))

    def test_only_the_student_policy_remains(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            render_episode.collect("/nonexistent", 1, 7, "scripted")
        self.assertIn("student", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
