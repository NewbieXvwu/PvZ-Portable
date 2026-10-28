"""task139 的纯逻辑测试：循环排序、建局把关、下传与计数规则。

不启动引擎（真实引擎只由 selfcheck 与任务书指定的冒烟跑）。
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_TEMP = tempfile.TemporaryDirectory(prefix="llmplay3-test-")
os.environ["PVZ_LLMPLAY_SESSION"] = str(Path(_TEMP.name) / "budget.json")
os.environ["LLMPLAY_OUT"] = str(Path(_TEMP.name) / "out")

from llmplay import workflow  # noqa: E402
from llmplay.delivery import _robust  # noqa: E402
from llmplay.game import Game, validate_deck  # noqa: E402

PLAN = workflow.plan()
TASKS = {task_id: task for task_id, task in workflow.ordered_tasks()}
A_IDS = [task_id for task_id, task in TASKS.items() if task.get("layer") == "A"]


def deck(index):
    """9 张、彼此差 6 张以上、地形全覆盖（带睡莲 16 与花盆 33）。"""
    return [index * 3, index * 3 + 1, index * 3 + 2, index * 3 + 3,
            40 + index, 41 + index, 42 + index, 16, 33]


def entry(task_id, seed, cards, *, cycle=1, won=True, robust=False, abandoned=False,
          strategy="推进", review="好用", reason="试卡"):
    return dict(task_id=task_id, seed=seed, deck=list(cards), cycle=cycle, won=won, robust=robust,
                abandoned=abandoned, strategy=strategy, deck_review=review, deck_reason=reason,
                valid=True, archive="archives/x%d" % seed, robustness={}, outcome=dict(final_tick=1))


def seeds_of(task_id, cycle=1, count=40, run_id="run"):
    return workflow.seeds_for(task_id, cycle, run_id)[:count]


def fill_cycle(cycle, *, skip_layer=None, robust=True):
    """造出一整圈（或指定层的空缺）的合格存档，用来验证层序与下传。"""
    entries = []
    for task_id in A_IDS:
        pool = seeds_of(task_id, cycle, 40)
        # 每圈用不同的卡组：A 层要求新一圈的卡组和之前所有圈差≥3 张。
        base = (cycle - 1) * 4
        chosen = [deck(base + index) for index in range(4)]
        for index, cards in enumerate(chosen):
            for offset in (0, 1):
                entries.append(entry(task_id, pool[index * 4 + offset], cards, cycle=cycle,
                                     robust=robust and index < 2, strategy="思路%d" % index))
        if skip_layer == "B" and robust:
            continue
        for index, cards in enumerate(chosen[:2]):
            for offset in (0, 1):
                entries.append(entry(task_id, pool[20 + index * 4 + offset], cards, cycle=cycle,
                                     strategy="思路%d" % index))
        for index, cards in enumerate(chosen[:2]):
            for name in ("经济优先", "防线优先", "速攻清场"):
                for offset in (0, 1):
                    entries.append(entry(task_id, pool[24 + index * 8 + len(name) + offset], cards,
                                         cycle=cycle, strategy=name))
            for multiplier, tag in ((1.5, "m150"), (2.0, "m200")):
                e_task = "llmplay3_e_l%d_%s" % (TASKS[task_id]["level"], tag)
                for offset in (0, 1):
                    entries.append(entry(e_task, pool[32 + offset], cards, cycle=cycle, strategy=name))
    batch = workflow.batch_for_cycle(cycle)
    for task_id, task in TASKS.items():
        if task.get("layer") != "C" or int(task.get("batch") or 1) != batch:
            continue
        pool = seeds_of(task_id, cycle, 5)
        for seed in pool[:2]:
            entries.append(entry(task_id, seed, task["deck"], cycle=cycle))
    return entries


class CycleOrderTest(unittest.TestCase):
    def test_empty_progress_starts_at_cycle_one_layer_a(self):
        document = workflow.derive([], "run")
        self.assertEqual((document["cycle"], document["layer"]), (1, "A"))
        self.assertFalse(document["complete"])
        item = workflow.next_task(document)
        self.assertEqual(item["action"], "new_deck")
        self.assertEqual(item["task_id"], A_IDS[0])

    def test_next_task_never_returns_none_and_wraps_into_next_cycle(self):
        document = workflow.derive(fill_cycle(1), "run")
        self.assertEqual(document["cycles"]["1"]["complete"], True)
        self.assertEqual(document["cycle"], 2)
        self.assertEqual(document["layer"], "A")
        item = workflow.next_task(document)
        self.assertIsNotNone(item)
        self.assertEqual(item["cycle"], 2)

    def test_layer_stays_blocked_until_previous_layer_done(self):
        entries = [item for item in fill_cycle(1) if item["task_id"] != A_IDS[0]]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["layer"], "A")
        self.assertFalse(document["layers"]["B"]["upstream"])

    def test_b_and_d_need_the_decks_passed_down(self):
        entries = fill_cycle(1, robust=False)
        document = workflow.derive(entries, "run")
        self.assertEqual(document["layer"], "B")
        item = workflow.next_task(document)
        self.assertEqual(item["action"], "improve_robustness")
        self.assertTrue(item["archive"])


class GateTest(unittest.TestCase):
    def test_later_layers_are_refused_while_layer_a_is_open(self):
        document = workflow.derive([], "run")
        challenge = next(task for task in TASKS.values() if task.get("layer") == "C")
        with self.assertRaises(ValueError) as caught:
            workflow.check(document, challenge["task_id"], challenge["deck"])
        self.assertIn("当前", str(caught.exception))
        with self.assertRaises(ValueError):
            workflow.check(document, "llmplay3_e_l42_m150", deck(0))

    def test_new_deck_must_differ_by_three_cards(self):
        task_id = A_IDS[0]
        seed = seeds_of(task_id, 1)[0]
        document = workflow.derive([entry(task_id, seed, deck(0))], "run")
        near = list(deck(0))
        near[0] = near[0] + 100
        with self.assertRaises(ValueError):
            workflow.check(document, task_id, near)
        workflow.check(document, task_id, deck(0))     # 同卡组继续打同一个种子是允许的
        workflow.check(document, task_id, deck(1))     # 差 6 张，允许

    def test_challenge_deck_cannot_be_swapped(self):
        entries = fill_cycle(1)
        document = workflow.derive(entries, "run")
        self.assertEqual(document["layer"], "A")      # 还差第 1 圈 A 层的两套
        task = next(task for task in TASKS.values()
                    if task.get("layer") == "C" and int(task["batch"]) == 1)
        with self.assertRaises(ValueError):
            workflow.check(document, task["task_id"], deck(5))

    def test_duplicate_strategy_in_layer_d_is_refused(self):
        entries = fill_cycle(1)
        document = workflow.derive(entries, "run")
        while document["layer"] != "D":
            break
        # 用只差 D 层的一圈来观察 D 层：把 D 的打法删掉
        trimmed = [item for item in entries if item["strategy"] not in ("经济优先", "防线优先", "速攻清场")]
        document = workflow.derive(trimmed, "run")
        self.assertEqual(document["layer"], "D")
        task_id = A_IDS[0]
        passed = document["detail"]["B"][task_id]["items"]
        key = sorted(passed)[0]
        cards = passed[key]["deck"]
        with self.assertRaises(ValueError):
            workflow.check(document, task_id, cards, "思路0")
        workflow.check(document, task_id, cards, "经济优先")


class CountTest(unittest.TestCase):
    def test_two_abandons_drop_the_pair(self):
        task_id = A_IDS[0]
        seed = seeds_of(task_id, 1)[0]
        entries = [entry(task_id, seed, deck(0), won=False, abandoned=True) for _ in range(2)]
        entries.append(entry(task_id, seed, deck(0)))
        document = workflow.derive(entries, "run")
        decks = document["detail"]["A"][task_id]["decks"]
        self.assertEqual(decks.get(workflow.deck_key(deck(0)), {}).get("wins", 0), 0)

    def test_one_abandon_keeps_the_win(self):
        task_id = A_IDS[0]
        seed = seeds_of(task_id, 1)[0]
        entries = [entry(task_id, seed, deck(0), won=False, abandoned=True),
                   entry(task_id, seed + 1, deck(0))]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["detail"]["A"][task_id]["decks"][workflow.deck_key(deck(0))]["wins"], 1)

    def test_five_losses_with_diagnosis_complete_the_challenge(self):
        entries = fill_cycle(1)
        document = workflow.derive(entries, "run")
        batch = workflow.batch_for_cycle(1)
        task_id = next(task["task_id"] for task in TASKS.values()
                       if task.get("layer") == "C" and int(task["batch"]) == batch)
        pool = seeds_of(task_id, 1, 5)
        lost = [item for item in entries if item["task_id"] != task_id]
        lost += [entry(task_id, seed, TASKS[task_id]["deck"], won=False, abandoned=True) for seed in pool]
        document = workflow.derive(lost, "run")
        self.assertFalse(document["detail"]["C"][task_id]["complete"])
        self.assertEqual(workflow.next_task(document)["action"], "diagnose")
        folder = Path(os.environ["LLMPLAY_OUT"]) / "diagnosis"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (task_id + ".json")).write_text(json.dumps(
            dict(task_id=task_id, waves=[8], missing="第 8 波后排水路缺群伤")), encoding="utf-8")
        document = workflow.derive(lost, "run")
        # 诊断一写完这一圈就齐了：当前圈自动推进到第 2 圈。
        self.assertEqual(document["cycles"]["1"]["complete"], True)
        self.assertEqual(document["cycle"], 2)
        (folder / (task_id + ".json")).unlink()

    def test_robust_flag_requires_checkpoint_ten_and_five_wins(self):
        report = dict(policy_sha256="x", checkpoints={"10": dict(wins=5, k=8, reached=True)})
        self.assertTrue(_robust(dict(task=dict(level=49), robustness=report)))
        report["checkpoints"]["10"]["wins"] = 4
        self.assertFalse(_robust(dict(task=dict(level=49), robustness=report)))


class ExportTest(unittest.TestCase):
    def test_review_is_required_before_export_in_layer_a(self):
        with self.assertRaises(ValueError):
            workflow.archive_requirements("llmplay3_a_l42_m100", "投手", "")
        workflow.archive_requirements("llmplay3_a_l42_m100", "投手", "好用")
        workflow.archive_requirements("llmplay3_c_l7_hold", "", "")       # C 层不要求评价
        workflow.archive_requirements("llmplay3_a_l42_m100", "", "", abandoned=True)

    def test_custom_deck_roundtrip_keeps_abandonment_and_rejects_more_actions(self):
        game = Game("demo_roof_49_doom_m200", seed=81000000,
                    deck=[1, 33, 32, 34, 2, 3], reason="试双投手", strategy="投手", deck_review="经济较省")
        game.abandon_reason = "换成蘑菇试试"
        path = Path(_TEMP.name) / "run.json"
        game.save(path)
        restored = Game.load(path)
        self.assertEqual(restored.deck, game.deck)
        self.assertEqual(restored.deck_reason, game.deck_reason)
        self.assertEqual(restored.deck_review, game.deck_review)
        self.assertEqual(restored.abandon_reason, game.abandon_reason)
        with self.assertRaises(RuntimeError):
            restored.wait(1)
        game.close()
        restored.close()

    def test_deck_hard_constraints_still_apply(self):
        validate_deck(dict(level=42), [1, 33, 32, 34, 2, 38])
        for level, cards in [(22, [0, 1, 2, 3, 4, 5]), (42, [0, 1, 2, 3, 4, 5]),
                             (2, [0, 1]), (2, [0, 1, 2, 3, 4, 4])]:
            with self.assertRaises(ValueError):
                validate_deck(dict(level=level), cards)

    def test_task_family_covers_every_terrain_and_has_120_seeds(self):
        for task_id, task in TASKS.items():
            self.assertGreaterEqual(len(PLAN["tasks"][task_id]), 120, task_id)
            if task.get("layer") == "C":
                validate_deck(dict(level=task["level"]), task["deck"])

    def test_boss_checkpoint_fix_is_kept(self):
        class Env:
            def snapshot_fast(self): return "snapshot"
            def restore_fast(self, snapshot): pass
            def observe(self): return dict(tick=6000, wave=0, result=1)
            def reseed(self, seed): pass
            def release_snapshot(self, snapshot): pass

        class Trial:
            tick, terminal, ops = 6000, True, []
            def __init__(self): self.env = Env()
            def _start(self): pass
            def _reset_bookkeeping(self, obs): self.obs = obs
            def close(self): pass

        def policy(game): game.wait(30)

        parent = Game("demo_boss_50_orig_m100", seed=81032000, tick=6100)
        from unittest.mock import patch
        with patch.object(parent, "_start"), patch.object(parent, "_persist_best"), \
                patch("llmplay.game.Game", return_value=Trial()):
            report = parent.robustness(policy)
        self.assertTrue(report["passed"])
        self.assertTrue(_robust(dict(task=dict(level=50), robustness=report)))
        parent.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
