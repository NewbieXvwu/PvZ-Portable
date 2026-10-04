"""回放训练出来的模型时，「用哪个任务、哪副卡组」必须来自训练清单本身。

这个文件存在的理由是一条 2026-10-04 的实测：`capture --policy ppo` 不给卡组时
用的是 `deck_for_level()`（**脚本基线**的卡组），而训练用的是任务清单里的
`deck`（`train_pvz_ppo.py:64` 是 `env.reset(deck=task["deck"], ...)`）。
第 49 关两者就不一样：脚本给 `[0,1,2,3,4,5,33]`，T7 清单给 `[0,1,3,4,7,33]`。

卡槽错位的后果不是"差一点"：模型说「用第 2 槽」，本机就种出另一个植物；
而且它看到的观测本身就是另一副卡槽（分布外输入）。这一局会照常跑完、
给出一个结果，然后被当成证据 —— 所以这里必须是硬错，不是警告。

清单里的 level / deck / 波数上限 / 僵尸倍率 / 预种植物**只能从清单取**，
手抄到命令行上一定会漂（第 49 关的卡组就是这么漂的）。
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
from scripted_baseline import deck_for_level  # noqa: E402

T7_MANIFEST = ROOT / "experiments" / "t7" / "bridge_level7_v1" / "train.json"

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

    def test_no_preplanted_means_no_key(self) -> None:
        """预种植物是空的就不传 —— `TaskSpec` 的默认值就是空，传个空元组只是噪音。"""
        task = dict(TASK, preplanted=[])
        self.assertNotIn("preplanted", render_episode.training_task_kwargs(task)["task_extra"])


class ReplayRefusesTheWrongDeck(unittest.TestCase):
    """`policy=ppo` 不带卡组时必须报错，而不是悄悄用脚本卡组跑完。"""

    def test_ppo_without_checkpoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            render_episode.collect("/nonexistent", 1, 49, "ppo")
        self.assertIn("checkpoint", str(ctx.exception))

    def test_ppo_without_deck_is_rejected(self) -> None:
        """检查点在加载之前就该被拦下 —— 所以传个不存在的路径也能测到这条。"""
        with self.assertRaises(ValueError) as ctx:
            render_episode.collect("/nonexistent", 61216, 49, "ppo",
                                   checkpoint="/nonexistent/ck.pt")
        message = str(ctx.exception)
        self.assertIn("deck", message)
        self.assertIn("task", message.lower())


@unittest.skipUnless(T7_MANIFEST.exists(), f"任务清单不在本机：{T7_MANIFEST}")
class ManifestDisagreesWithScriptedDeck(unittest.TestCase):
    """把"训练卡组 ≠ 脚本卡组"钉住。

    如果哪天有人把 `deck_for_level()` 改成跟清单一致，这条会失败 —— 那时该做的
    是**核对清单**再决定改哪边，而不是顺手把断言删掉。
    """

    def test_roof_deck_differs_from_scripted_default(self) -> None:
        task = render_episode.load_training_task(T7_MANIFEST, "train_roof_4")
        self.assertEqual(task["deck"], [0, 1, 3, 4, 7, 33])
        self.assertEqual(list(deck_for_level(49)), [0, 1, 2, 3, 4, 5, 33])
        self.assertNotEqual(list(deck_for_level(49)), list(task["deck"]))

    def test_day_deck_matches_this_time(self) -> None:
        """白天关两者恰好一致 —— 所以这个坑在白天任务上看不出来，只在屋顶现形。"""
        task = render_episode.load_training_task(T7_MANIFEST, "train_full_level7_v1")
        self.assertEqual(list(deck_for_level(7)), list(task["deck"]))

    def test_aided_task_carries_preplanted(self) -> None:
        task = render_episode.load_training_task(T7_MANIFEST,
                                                 "train_bridge7_v1_aid_cap10")
        kw = render_episode.training_task_kwargs(task)
        self.assertEqual(len(kw["task_extra"]["preplanted"]), 10)
        self.assertEqual(kw["task_extra"]["wave_cap"], 10)


if __name__ == "__main__":
    unittest.main()
