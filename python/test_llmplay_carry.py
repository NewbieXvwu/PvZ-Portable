"""task146 的纯逻辑测试：前次进度（carry）接上、重试任务进第 2 批。

不启动引擎：只验记录层（读 carry、补诊断、种子与卡组沿用、重试任务归属）。
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_TEMP = tempfile.TemporaryDirectory(prefix="llmplay-carry-test-")
os.environ["PVZ_LLMPLAY_SESSION"] = str(Path(_TEMP.name) / "budget.json")
os.environ["LLMPLAY_OUT"] = str(Path(_TEMP.name) / "out")

from llmplay import build_package, delivery, workflow  # noqa: E402

PLAN = workflow.plan()
TASKS = {task_id: task for task_id, task in workflow.ordered_tasks()}
A_ID = PLAN["order"][0]
RETRY = [task_id + "_retry" for task_id in ("llmplay3_c_l14_d02", "llmplay3_c_l14_d03",
                                            "llmplay3_c_l24_d02", "llmplay3_c_l42_d01",
                                            "llmplay3_c_l48_d02", "llmplay3_c_l48_d03")]


def entry(task_id, seed, **extra):
    row = dict(task_id=task_id, seed=seed, cycle=1, deck=[1, 2, 3, 4, 16, 33], won=True,
               robust=False, abandoned=False, result=1, strategy="推进", deck_reason="试",
               deck_review="好用", workflow_layer="A", opening=[], attempt_id="a%d" % seed,
               archive="archives/%d" % seed, valid=True)
    row.update(extra)
    return row


def prior(archives, *, cycle=1, detail=None):
    return dict(schema="llmplay_progress_v2", run_id="prior-run", cycle=cycle,
                archives=list(archives), detail=detail or dict(C={}))


class CarryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="carry-out-"))
        os.environ["LLMPLAY_OUT"] = str(self.root)
        self.source = Path(tempfile.mkdtemp(prefix="carry-src-")) / "progress.json"

    def write(self, document):
        self.source.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return self.source

    def test_records_become_the_starting_point(self):
        self.write(build_package.merge_carry(
            prior([entry(A_ID, 139000001)]), prior([entry(A_ID, 139000002)])))
        document = delivery.carry(self.source)
        self.assertEqual(document["run_id"], "prior-run")      # 沿用上一份运行ID
        progress = delivery.progress()
        self.assertEqual((progress["cycle"], progress["layer"]), (1, "A"))
        self.assertEqual(len(progress["archives"]), 2)
        self.assertEqual(progress["layers"]["A"]["completed"], 0)   # 一套卡组不够 A 层

    def test_used_seeds_are_not_offered_again(self):
        seed = workflow.seeds_for(A_ID, 1, "prior-run")[0]
        self.write(prior([entry(A_ID, seed)]))
        delivery.carry(self.source)
        offered = delivery.progress()["tasks"][A_ID]["available_seeds"]
        self.assertNotIn(seed, offered)
        self.assertTrue(offered)

    def test_previous_cycle_decks_block_reuse(self):
        # 圈号随记录一起接上，所以第 1 圈的卡组会挡掉第 2 圈 A 层「差<3 张」的重复卡组。
        deck = [1, 2, 3, 4, 16, 33]
        self.write(prior([entry(A_ID, 139000002, deck=deck)]))
        delivery.carry(self.source)
        archives = delivery.progress()["archives"]
        self.assertEqual([row["cycle"] for row in archives], [1])
        decks = {workflow.deck_key(deck): dict(deck=sorted(deck), wins=2, robust_wins=0,
                                               strategy="思路", review="好用")}
        self.assertEqual(workflow._select(decks, [sorted(deck)], PLAN["requirements"]["A"]), [])

    def test_second_call_does_nothing(self):
        self.write(prior([entry(A_ID, 139000003)]))
        self.assertIsNotNone(delivery.carry(self.source))
        self.assertIsNone(delivery.carry(self.source))
        self.assertEqual(len(delivery.progress()["archives"]), 1)

    def test_tasks_outside_the_plan_are_dropped(self):
        self.write(prior([entry(A_ID, 139000004), entry("old_family_l1", 1)]))
        delivery.carry(self.source)
        self.assertEqual([row["task_id"] for row in delivery.progress()["archives"]], [A_ID])

    def test_c_diagnosis_is_restored_as_a_file(self):
        task_id = "llmplay3_c_l14_d02"
        report = dict(waves=[8, 9], missing="第 8 波后排水路缺群伤", note="")
        self.write(prior([], detail=dict(C={task_id: dict(task_id=task_id, diagnosis=report)})))
        delivery.carry(self.source)
        path = self.root / "diagnosis" / "1" / (task_id + ".json")
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["waves"], [8, 9])
        self.assertIn((task_id, 1), workflow.diagnoses())

    def test_missing_fields_are_filled(self):
        row = delivery._carry_entry(dict(task_id=A_ID, seed=7))
        self.assertEqual((row["cycle"], row["won"], row["robust"], row["abandoned"]), (1, False, False, False))
        self.assertEqual((row["deck"], row["opening"], row["valid"]), ([], [], True))


class RetryTaskTest(unittest.TestCase):
    """第 1 圈 5 个种子全输的卡组：第 2 圈换新种子再给一次机会，原任务不动。"""

    def test_retry_tasks_are_in_batch_two_with_fresh_seeds(self):
        for task_id in RETRY:
            origin = TASKS[task_id[:-len("_retry")]]
            retry = TASKS[task_id]
            self.assertEqual((retry["level"], retry["deck"], retry["layer"]),
                             (origin["level"], origin["deck"], "C"))
            self.assertEqual(retry["batch"], 2)
            self.assertEqual(workflow.batch_for_cycle(2), 2)
            self.assertEqual(retry["retry_of"], origin["task_id"])
            self.assertFalse(set(PLAN["tasks"][task_id]) & set(PLAN["tasks"][origin["task_id"]]))

    def test_originals_keep_their_batch(self):
        for task_id in RETRY:
            self.assertEqual(TASKS[task_id[:-len("_retry")]]["batch"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
