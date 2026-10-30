"""task139 的纯逻辑测试：循环排序、建局把关、下传与计数规则。

不启动引擎（真实引擎只由 selfcheck 与任务书指定的冒烟跑）。
"""
import json
import os
import sys
import tempfile
import unittest
import uuid
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
          strategy="推进", review="好用", reason="试卡", layer="A", opening=None):
    return dict(task_id=task_id, seed=seed, deck=list(cards), cycle=cycle, won=won, robust=robust,
                abandoned=abandoned, strategy=strategy, deck_review=review, deck_reason=reason,
                workflow_layer=layer, opening=opening or [], result=0 if abandoned else 1 if won else 2,
                attempt_id=uuid.uuid4().hex, valid=True, archive="archives/x%d" % seed,
                robustness={}, outcome=dict(final_tick=1, final_wave=8))


def seeds_of(task_id, cycle=1, count=40, run_id="run"):
    return workflow.seeds_for(task_id, cycle, run_id)[:count]


def fill_cycle(cycle, *, skip_layer=None, robust=True):
    """造出一整圈（或指定层的空缺）的合格存档，用来验证层序与下传。"""
    entries = []
    for task_id in A_IDS:
        task = TASKS[task_id]
        if not task.get("first_cycle", 1) <= cycle <= task.get("last_cycle", cycle):
            continue
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
            for style, name in enumerate(workflow.STRATEGIES):
                for offset in (0, 1):
                    entries.append(entry(task_id, pool[8 + index * 6 + style * 2 + offset], cards,
                                         cycle=cycle, strategy=name, layer="D", opening=[style, 1, style]))
            for multiplier, tag in ((1.25, "m125"), (1.5, "m150")):
                e_task = task_id.replace("_a_", "_e_").replace("_m100", "_" + tag)
                for offset in (0, 1):
                    entries.append(entry(e_task, seeds_of(e_task, cycle)[index * 2 + offset], cards,
                                         cycle=cycle, strategy=name, layer="E"))
    batch = workflow.batch_for_cycle(cycle)
    for task_id, task in TASKS.items():
        if (task.get("layer") != "C" or int(task.get("batch") or 1) != batch
                or not task.get("first_cycle", 1) <= cycle <= task.get("last_cycle", cycle)):
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
        todo = workflow.next_task(document, claim=False)
        item = todo["main"]
        self.assertEqual(item["action"], "new_deck")
        self.assertEqual(item["task_id"], A_IDS[0])
        self.assertEqual(todo["challenge"]["layer"], "C")
        self.assertEqual(todo["challenge_quota"]["total"], len(document["detail"]["C"]))

    def test_next_task_never_returns_none_and_wraps_into_next_cycle(self):
        document = workflow.derive(fill_cycle(1), "run")
        self.assertEqual(document["cycles"]["1"]["complete"], True)
        self.assertEqual(document["cycle"], 2)
        self.assertEqual(document["layer"], "A")
        self.assertEqual({row["level"] for row in document["detail"]["A"].values()},
                         set(range(41, 51)) | {2, 6, 8, 12, 16, 18, 22, 26, 27, 28, 29, 32, 33, 34, 36, 38, 39, 40})
        item = workflow.next_task(document, claim=False)["main"]
        self.assertIsNotNone(item)
        self.assertEqual(item["cycle"], 2)
        old_id = "llmplay3_c_l42_d01_retry"
        completed = [entry(old_id, seed, TASKS[old_id]["deck"], cycle=2)
                     for seed in seeds_of(old_id, 2)[:2]]
        document = workflow.derive(fill_cycle(1) + completed, "run")
        carried = document["detail"]["C"]["llmplay5_c_l42_d01_retry"]
        self.assertTrue(carried["complete"])
        self.assertEqual(carried["carried_completion"]["task_id"], old_id)

    def test_layer_stays_blocked_until_previous_layer_done(self):
        entries = [item for item in fill_cycle(1) if item["task_id"] != A_IDS[0]]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["layer"], "A")
        self.assertFalse(document["layers"]["B"]["upstream"])

    def test_b_and_d_need_the_decks_passed_down(self):
        entries = fill_cycle(1, robust=False)
        document = workflow.derive(entries, "run")
        self.assertEqual(document["layer"], "B")
        item = workflow.next_task(document, claim=False)["main"]
        self.assertEqual(item["action"], "improve_robustness")
        self.assertTrue(item["archive"])

    def test_main_can_finish_but_c_quota_blocks_next_cycle(self):
        entries = [item for item in fill_cycle(1) if TASKS[item["task_id"]]["layer"] != "C"]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["cycle"], 1)
        self.assertTrue(all(document["layers"][layer]["complete"] for layer in workflow.MAIN_LAYERS))
        todo = workflow.next_task(document, claim=False)
        self.assertIsNone(todo["main"])                       # 主线本圈已完成
        self.assertEqual(todo["item"]["layer"], "C")           # 于是队列直接给 C 轨的项
        self.assertEqual(todo["item"]["action"], "play")
        self.assertEqual(todo["challenge"]["action"], "play")

    def test_c_does_not_block_d_or_e(self):
        entries = [item for item in fill_cycle(1) if TASKS[item["task_id"]]["layer"] != "C"
                   and item["workflow_layer"] != "D"]
        self.assertEqual(workflow.derive(entries, "run")["layer"], "D")
        entries = [item for item in fill_cycle(1) if TASKS[item["task_id"]]["layer"] not in ("C", "E")]
        self.assertEqual(workflow.derive(entries, "run")["layer"], "E")


class ClaimTest(unittest.TestCase):
    """多子代理并发领任务：各领各的、过期自动释放、全被领走时返回 busy 不阻塞。"""

    def setUp(self):
        workflow._claims_path().unlink(missing_ok=True)

    def tearDown(self):
        workflow._claims_path().unlink(missing_ok=True)

    def test_two_agents_get_different_items(self):
        document = workflow.derive([], "run")
        first = workflow.next_task(document)["item"]
        second = workflow.next_task(document)["item"]
        self.assertNotEqual((first["task_id"], first["seed"]), (second["task_id"], second["seed"]))
        self.assertTrue(first["claim"] and second["claim"])
        data = json.loads(workflow._claims_path().read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)
        self.assertTrue(all(row["expires"] > row["claimed_at"] for row in data.values()))

    def test_expired_claim_is_released_automatically(self):
        document = workflow.derive([], "run")
        item = workflow.next_task(document, ttl=-1)["item"]
        again = workflow.next_task(document)["item"]
        self.assertEqual((again["task_id"], again["seed"]), (item["task_id"], item["seed"]))

    def test_all_claimed_returns_busy_without_waiting(self):
        document = workflow.derive([], "run")
        for _ in range(400):
            item = workflow.next_task(document)["item"]
            if item.get("busy"):
                break
        self.assertTrue(item.get("busy"))
        self.assertEqual(item["action"], "busy")

    def test_c_agent_gets_challenge_items_first_and_falls_back_to_main(self):
        document = workflow.derive([], "run")
        item = workflow.next_task(document, track="C")["item"]
        self.assertEqual(item["layer"], "C")
        # C 轨本圈做完后（这里清空 C 轨待办来模拟），C 子代理自动接主线
        document["detail"]["C"] = {}
        item = workflow.next_task(document, track="C")["item"]
        self.assertIn(item["layer"], workflow.MAIN_LAYERS)

    def test_release_puts_the_slot_back(self):
        document = workflow.derive([], "run")
        item = workflow.next_task(document)["item"]
        workflow.release(item["task_id"], item["seed"])
        again = workflow.next_task(document)["item"]
        self.assertEqual((again["task_id"], again["seed"]), (item["task_id"], item["seed"]))


class GateTest(unittest.TestCase):
    def test_later_layers_are_refused_while_layer_a_is_open(self):
        document = workflow.derive([], "run")
        challenge = next(task for task in TASKS.values()
                         if task.get("layer") == "C" and int(task.get("batch") or 1) == 1)
        seed = workflow.seeds_for(challenge["task_id"], 1, "run")[0]
        context = workflow.check(document, challenge["task_id"], challenge["deck"], seed=seed)
        self.assertEqual(context, dict(cycle=1, workflow_layer="C"))
        with self.assertRaises(ValueError):
            workflow.check(document, "llmplay3_e_l42_m150", deck(0), seed=seed)

    def test_new_deck_must_differ_by_three_cards(self):
        task_id = A_IDS[0]
        seed = seeds_of(task_id, 1)[0]
        document = workflow.derive([entry(task_id, seed, deck(0))], "run")
        near = list(deck(0))
        near[0] = near[0] + 100
        with self.assertRaises(ValueError):
            workflow.check(document, task_id, near, seed=seed)
        near[1] += 100
        with self.assertRaises(ValueError):
            workflow.check(document, task_id, near, seed=seed)
        workflow.check(document, task_id, deck(0), seed=seed)   # 同卡组继续打是允许的
        workflow.check(document, task_id, deck(1), seed=seed)   # 差 6 张，允许

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
        # 用只差 D 层的一圈来观察 D 层：把 D 的打法删掉
        trimmed = [item for item in entries if item["strategy"] not in ("经济优先", "防线优先", "速攻清场")]
        document = workflow.derive(trimmed, "run")
        self.assertEqual(document["layer"], "D")
        task_id = A_IDS[0]
        passed = document["detail"]["B"][task_id]["items"]
        key = sorted(passed)[0]
        cards = passed[key]["deck"]
        seed = seeds_of(task_id, 1)[24]
        with self.assertRaises(ValueError):
            workflow.check(document, task_id, cards, "思路0", seed=seed)
        workflow.check(document, task_id, cards, "经济优先", seed=seed)


    def test_layer_d_allows_the_second_seed_of_the_same_strategy(self):
        """同一打法赢 1 个种子后必须还能继续用它赢第 2 个，否则 D 层永远做不完。"""
        entries = [item for item in fill_cycle(1) if item["strategy"] not in ("经济优先", "防线优先", "速攻清场")]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["layer"], "D")
        task_id = A_IDS[0]
        key = sorted(document["detail"]["B"][task_id]["items"])[0]
        cards = document["detail"]["B"][task_id]["items"][key]["deck"]
        seed = seeds_of(task_id, 1)[24]
        entries = entries + [entry(task_id, seed, cards, strategy="经济优先")]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["detail"]["D"][task_id]["items"][key]["done"], [])
        workflow.check(document, task_id, cards, "经济优先", seed=seed)   # 还差 1 个种子，允许继续
        entries[-1].update(workflow_layer="D", opening=[1, 2, 1])
        entries = entries + [entry(task_id, seed + 1, cards, strategy="经济优先", layer="D", opening=[1, 2, 1])]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["detail"]["D"][task_id]["items"][key]["done"], ["经济优先"])
        with self.assertRaises(ValueError):
            workflow.check(document, task_id, cards, "经济优先", seed=seed)   # 已经够 2 胜，换打法

    def test_cannot_pick_easy_task_or_use_seed_outside_window_or_old_family(self):
        document = workflow.derive([], "run")
        item = workflow.next_task(document, claim=False)["main"]
        with self.assertRaises(ValueError):        # 别的任务的种子块不能用
            workflow.check(document, A_IDS[-1], deck(0), seed=item["seed"])
        with self.assertRaises(ValueError):        # 窗口外（下一圈）的种子不能用
            workflow.check(document, item["task_id"], deck(0), seed=item["seed"] + 1000)
        with self.assertRaises(ValueError):        # 旧任务族不算本轮
            workflow.check(document, "demo_roof_49_doom_m200", deck(0), seed=item["seed"])
        workflow.check(document, item["task_id"], deck(0), seed=item["seed"])

    def test_previous_cycle_decks_are_rejected_before_play(self):
        document = workflow.derive(fill_cycle(1), "run")
        with self.assertRaises(ValueError):
            workflow.check(document, A_IDS[0], deck(0))

    def test_d_rejects_vague_strategy_labels(self):
        entries = [item for item in fill_cycle(1) if item["workflow_layer"] != "D"]
        document = workflow.derive(entries, "run")
        item = workflow.next_task(document, claim=False)["main"]
        for label in ("", "打法1", "随便打", "经济优先2"):
            with self.assertRaises(ValueError):
                workflow.check(document, item["task_id"], item["deck"], label)


class PressureExitTest(unittest.TestCase):
    def test_next_task_moves_on_after_five_losses_and_diagnosis(self):
        from llmplay import delivery
        for tag in ("m125", "m150", "m200"):
            with self.subTest(multiplier=tag), tempfile.TemporaryDirectory() as directory:
                previous_out = os.environ["LLMPLAY_OUT"]
                os.environ["LLMPLAY_OUT"] = directory
                try:
                    task_id = "llmplay3_e_l41_" + tag
                    # 只留下 E 的一个待办组；m200 模拟 carry 里的旧 2.0 倍成绩。
                    entries = fill_cycle(1)
                    cards = deck(0)
                    target = "llmplay3_e_l41_" + ("m125" if tag == "m200" else tag)
                    entries = [row for row in entries if not (
                        row["task_id"] == target and row["deck"] == cards)]
                    pool = seeds_of(task_id, count=5)
                    entries += [entry(task_id, seed, cards, won=False, layer="E") for seed in pool[:4]]
                    if tag != "m200":
                        before = workflow.next_task(workflow.derive(entries, "run"), claim=False)["item"]
                        self.assertEqual((before["task_id"], before["action"]), (task_id, "play"))
                    entries.append(entry(task_id, pool[4], cards, won=False, layer="E"))
                    delivery._publish(Path(directory), entries, delivery.budget.session(), run_id="run")
                    item = delivery.next_task()["item"]
                    self.assertEqual((item["task_id"], item["action"]), (task_id, "diagnose"))
                    path = delivery.diagnose(task_id, deck=item["deck"], waves=[8], missing="第 8 波缺群伤")
                    self.assertEqual(json.loads(path.read_text())["deck"], sorted(cards))
                    next_item = delivery.next_task()["item"]
                    self.assertEqual((next_item["cycle"], next_item["layer"]), (2, "A"))
                finally:
                    os.environ["LLMPLAY_OUT"] = previous_out


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
        lost += [entry(task_id, seed, TASKS[task_id]["deck"], won=False, layer="C") for seed in pool]
        document = workflow.derive(lost, "run")
        self.assertFalse(document["detail"]["C"][task_id]["complete"])
        self.assertEqual(workflow.next_task(document, claim=False)["challenge"]["action"], "diagnose")
        folder = Path(os.environ["LLMPLAY_OUT"]) / "diagnosis"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (task_id + ".json")).write_text(json.dumps(
            dict(task_id=task_id, cycle=1, waves=[8], missing="第 8 波后排水路缺群伤")), encoding="utf-8")
        document = workflow.derive(lost, "run")
        # 诊断一写完这一圈就齐了：当前圈自动推进到第 2 圈。
        self.assertEqual(document["cycles"]["1"]["complete"], True)
        self.assertEqual(document["cycle"], 2)
        (folder / (task_id + ".json")).unlink()

    def test_abandonment_does_not_complete_challenge_or_refresh_seed(self):
        document = workflow.derive([], "run")
        item = workflow.next_task(document, claim=False)["challenge"]
        entries = [entry(item["task_id"], seed, item["deck"], won=False, abandoned=True)
                   for seed in seeds_of(item["task_id"], count=5)]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["detail"]["C"][item["task_id"]]["attempted"], [])
        self.assertEqual(workflow.next_task(document, claim=False)["challenge"]["seed"], item["seed"])
        first = workflow.next_task(workflow.derive([], "run"), claim=False)["main"]
        abandoned = entry(first["task_id"], first["seed"], deck(0), won=False, abandoned=True)
        document = workflow.derive([abandoned], "run")
        self.assertEqual(workflow.next_task(document, claim=False)["main"]["seed"], first["seed"])
        unfinished = dict(abandoned, abandoned=False, result=0)
        document = workflow.derive([unfinished], "run")
        self.assertEqual(workflow.next_task(document, claim=False)["main"]["seed"], first["seed"])

    def test_reexporting_one_abandonment_is_one_attempt(self):
        task_id = A_IDS[0]
        seed = seeds_of(task_id)[0]
        abandoned = entry(task_id, seed, deck(0), won=False, abandoned=True)
        entries = [abandoned, dict(abandoned, archive="archives/another"), entry(task_id, seed, deck(0))]
        document = workflow.derive(entries, "run")
        self.assertEqual(document["detail"]["A"][task_id]["decks"][workflow.deck_key(deck(0))]["wins"], 1)

    def test_d_only_counts_declared_categories_and_distinct_actual_openings(self):
        entries = [item for item in fill_cycle(1) if item["workflow_layer"] != "D"]
        document = workflow.derive(entries, "run")
        item = workflow.next_task(document, claim=False)["main"]
        for style in ("经济优先", "防线优先", "打法3"):
            for seed in seeds_of(item["task_id"], count=2):
                entries.append(entry(item["task_id"], seed, item["deck"], layer="D",
                                     strategy=style, opening=[1, 1, 2]))
        document = workflow.derive(entries, "run")
        group = document["detail"]["D"][item["task_id"]]["items"][workflow.deck_key(item["deck"])]
        self.assertEqual(group["done"], ["经济优先"])

    def test_early_diagnosis_is_refused(self):
        from unittest.mock import patch
        from llmplay import delivery
        with patch.object(delivery, "_read_progress", return_value=dict(archives=[], run_id="run")):
            with self.assertRaises(ValueError):
                delivery.diagnose(next(task_id for task_id in TASKS if TASKS[task_id]["layer"] == "C"),
                                  waves=[8], missing="后排缺群伤")

    def test_diagnosis_is_bound_to_cycle_and_old_reports_survive(self):
        task_id = next(task_id for task_id in TASKS if TASKS[task_id]["layer"] == "C")
        entries = [entry(task_id, seed, TASKS[task_id]["deck"], won=False, layer="C")
                   for seed in seeds_of(task_id, count=5)]
        folder = Path(os.environ["LLMPLAY_OUT"]) / "diagnosis"
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder / "cycle1-test.json", folder / "cycle4-test.json"]
        try:
            paths[1].write_text(json.dumps(dict(task_id=task_id, cycle=4, waves=[8], missing="缺群伤")))
            self.assertFalse(workflow.derive(entries, "run")["detail"]["C"][task_id]["complete"])
            paths[0].write_text(json.dumps(dict(task_id=task_id, cycle=1, waves=[8], missing="缺群伤")))
            self.assertTrue(workflow.derive(entries, "run")["detail"]["C"][task_id]["complete"])
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

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
                    deck=[1, 33, 32, 34, 2, 3], reason="试双投手", strategy="投手", deck_review="经济较省",
                    cycle=7, workflow_layer="D", _gate=False)
        game.abandon_reason = "换成蘑菇试试"
        path = Path(_TEMP.name) / "run.json"
        game.save(path)
        restored = Game.load(path)
        self.assertEqual(restored.deck, game.deck)
        self.assertEqual(restored.deck_reason, game.deck_reason)
        self.assertEqual(restored.deck_review, game.deck_review)
        self.assertEqual(restored.abandon_reason, game.abandon_reason)
        self.assertEqual((restored.cycle, restored.workflow_layer), (7, "D"))
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

        parent = Game("demo_boss_50_orig_m100", seed=81032000, tick=6100, _gate=False)
        from unittest.mock import patch
        with patch.object(parent, "_start"), patch.object(parent, "_persist_best"), \
                patch("llmplay.game.Game", return_value=Trial()):
            report = parent.robustness(policy)
        self.assertTrue(report["passed"])
        self.assertTrue(_robust(dict(task=dict(level=50), robustness=report)))
        parent.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
