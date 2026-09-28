"""Tests for the search audit: sibling ranking, budget consistency, candidate recall.

These run against the stateful fake simulator from ``test_search_teacher``, so the
orderings the audit measures are computed over real branch results rather than over
whatever a stub felt like returning.
"""

from __future__ import annotations

import unittest
from typing import Any

from pvz_search import SearchTeacher
from pvz_search_audit import (
    budget_consistency,
    candidate_recall,
    leaf_score,
    pairwise_agreement,
    sibling_ranking,
    summarize,
)
from test_search_teacher import _FakeSimulator, _board, _observation


class _StubValueModel:
    """Scores a leaf by how many plants stand on the board.

    Deliberately crude: the audit's job is to compare orderings, so the tests only
    need a scorer whose ordering they can predict by hand.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def predict(self, observation: dict[str, Any]) -> float:
        return self.scale * float(len(observation["plants"]))


class _NarrowGenerator:
    """Proposes a single wait, so the audit can observe a pure generation loss."""

    def actions(self, observation: dict[str, Any], limit: int, root: bool, remaining_ticks: int,
                allow_instant: bool = True) -> list[dict[str, Any]]:
        return [{"type": "wait", "ticks": 60}]


class PairwiseAgreementTests(unittest.TestCase):
    def test_identical_orderings_agree_on_every_pair(self) -> None:
        agree, pairs = pairwise_agreement([3.0, 2.0, 1.0], [0.9, 0.5, 0.1])

        self.assertEqual((agree, pairs), (3, 3))

    def test_reversed_orderings_disagree_on_every_pair(self) -> None:
        agree, pairs = pairwise_agreement([3.0, 2.0, 1.0], [0.1, 0.5, 0.9])

        self.assertEqual((agree, pairs), (0, 3))

    def test_ties_are_excluded_rather_than_scored(self) -> None:
        """A constant scorer must not look accurate just because ties are 'close enough'."""
        agree, pairs = pairwise_agreement([3.0, 2.0, 1.0], [0.5, 0.5, 0.5])

        self.assertEqual((agree, pairs), (0, 0))

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equally long"):
            pairwise_agreement([1.0], [1.0, 2.0])


class LeafScoreTests(unittest.TestCase):
    def test_a_terminal_leaf_is_scored_by_its_outcome_not_by_the_value_model(self) -> None:
        model = _StubValueModel(scale=100.0)

        self.assertEqual(leaf_score(_observation(_board(), terminal=True, result=1), model), 1.0)
        self.assertEqual(leaf_score(_observation(_board(), terminal=True, result=0), model), -1.0)

    def test_a_value_model_prediction_is_clamped_like_the_search_clamps_it(self) -> None:
        planted = _board(plants=((0, 1, 0),))

        self.assertEqual(leaf_score(_observation(planted), _StubValueModel(scale=100.0)), 1.0)
        self.assertEqual(leaf_score(_observation(planted), _StubValueModel(scale=-100.0)), -1.0)

    def test_without_a_checkpoint_the_bootstrap_evaluator_is_used(self) -> None:
        observation = _observation(_board())

        self.assertEqual(leaf_score(observation, None),
                         SearchTeacher._bootstrap_leaf_value(observation))


class BudgetConsistencyTests(unittest.TestCase):
    def _advice(self, candidates: list[tuple[dict[str, Any], float]], action: dict[str, Any]) -> Any:
        class _Advice:
            pass

        advice = _Advice()
        advice.candidates = candidates
        advice.action = action
        return advice

    def test_only_shared_candidates_are_compared(self) -> None:
        shared = {"type": "wait", "ticks": 60}
        left = {"type": "wait", "ticks": 150}
        right = {"type": "wait", "ticks": 300}
        shallow = self._advice([(shared, 1.0), (left, 0.0)], shared)
        deep = self._advice([(shared, 0.5), (right, 0.9)], right)

        report = budget_consistency(shallow, deep)

        self.assertEqual(report["shared_candidates"], 1)
        self.assertEqual(report["pairwise_pairs"], 0)
        self.assertFalse(report["top1_agrees"])
        self.assertEqual(report["deep_top1"], right)

    def test_agreement_is_measured_on_the_shared_ranking(self) -> None:
        first = {"type": "wait", "ticks": 60}
        second = {"type": "wait", "ticks": 150}
        third = {"type": "wait", "ticks": 300}
        shallow = self._advice([(first, 0.9), (second, 0.5), (third, 0.1)], first)
        deep = self._advice([(first, 3.0), (second, 2.0), (third, 1.0)], first)

        report = budget_consistency(shallow, deep)

        self.assertEqual(report["shared_candidates"], 3)
        self.assertEqual((report["pairwise_agreements"], report["pairwise_pairs"]), (3, 3))
        self.assertEqual(report["pairwise_accuracy"], 1.0)
        self.assertTrue(report["top1_agrees"])


class CandidateRecallTests(unittest.TestCase):
    def test_the_generated_set_covers_every_legal_action_when_it_can(self) -> None:
        env = _FakeSimulator()
        teacher = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        observation = _observation(_board())

        report = candidate_recall(teacher, _StubValueModel(), observation, teacher.advice(observation))

        # 3 packets x 4 placements, no plants to shovel, and the three wait durations.
        self.assertEqual(report["legal_actions"], 3 * 4 + 3)
        self.assertEqual(report["generated_actions"], report["legal_actions"])
        self.assertTrue(report["recall"]["generated"]["contains_best"])
        self.assertEqual(report["recall"]["generated"]["recall@1"], 1.0)
        self.assertEqual(report["screened_actions"], teacher.root_candidate_limit)

    def test_a_narrow_generator_is_reported_as_a_generation_loss(self) -> None:
        """No simulation budget can recover an action the generator never proposed."""
        env = _FakeSimulator()
        teacher = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        teacher.candidate_generator = _NarrowGenerator()
        observation = _observation(_board())

        report = candidate_recall(teacher, _StubValueModel(), observation, teacher.advice(observation))

        self.assertEqual(report["generated_actions"], 1)
        self.assertFalse(report["recall"]["generated"]["contains_best"])
        self.assertFalse(report["recall"]["screened"]["contains_best"])
        # The lone wait is legal and does appear in the full ranking -- it is just far
        # from the top, because the plant-count scorer ranks every plant above it.
        self.assertGreater(report["recall"]["generated"]["best_rank_inside"], 1)

    def test_recall_never_exceeds_one(self) -> None:
        env = _FakeSimulator()
        teacher = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        observation = _observation(_board())

        report = candidate_recall(teacher, _StubValueModel(), observation, teacher.advice(observation))

        for name in ("generated", "screened"):
            for key, value in report["recall"][name].items():
                if key.startswith("recall@"):
                    self.assertLessEqual(value, 1.0)
                    self.assertGreaterEqual(value, 0.0)


class SiblingRankingTests(unittest.TestCase):
    def test_the_comparison_covers_the_deep_searches_own_candidates(self) -> None:
        env = _FakeSimulator()
        teacher = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        deep = SearchTeacher(env, simulation_budget=256, candidate_limit=4)
        observation = _observation(_board())

        report = sibling_ranking(teacher, deep, _StubValueModel(), observation)

        self.assertIsNotNone(report)
        deep_advice = deep.advice(observation)
        self.assertEqual(report["candidates"], len(deep_advice.candidates))
        self.assertEqual(report["scored"] + report["terminal_actions"], report["candidates"])
        self.assertEqual(report["pairwise_pairs"], 0, "a plant-count scorer ties every plant child")

    def test_a_constant_scorer_cannot_claim_agreement(self) -> None:
        """The guard against a value model that outputs one number for every state."""
        env = _FakeSimulator()
        teacher = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        deep = SearchTeacher(env, simulation_budget=256, candidate_limit=4)

        report = sibling_ranking(teacher, deep, _StubValueModel(scale=0.0), _observation(_board()))

        self.assertIsNotNone(report)
        self.assertEqual(report["pairwise_pairs"], 0)
        self.assertIsNone(report["pairwise_accuracy"])

    def test_a_single_candidate_has_no_ordering_to_measure(self) -> None:
        env = _FakeSimulator()
        teacher = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        teacher.candidate_generator = _NarrowGenerator()
        deep = SearchTeacher(env, simulation_budget=64, candidate_limit=2)
        deep.candidate_generator = _NarrowGenerator()

        self.assertIsNone(sibling_ranking(teacher, deep, _StubValueModel(), _observation(_board())))


class SummarizeTests(unittest.TestCase):
    def test_an_empty_audit_reports_zero_states_for_every_requested_mode(self) -> None:
        summary = summarize([], frozenset({"budget", "sibling", "recall"}))

        self.assertEqual(summary["states"], 0)
        for key in ("budget_consistency", "sibling_ranking", "candidate_recall"):
            self.assertEqual(summary[key]["states"], 0)

    def test_only_the_requested_modes_appear_in_the_summary(self) -> None:
        record = {"budget": {"top1_agrees": True, "pairwise_accuracy": 1.0, "shared_candidates": 3},
                  "sibling": {"pairwise_accuracy": 0.5, "top1_agrees": False, "scored": 4,
                              "deep_spread": 0.3}}

        summary = summarize([record], frozenset({"budget"}))

        self.assertIn("budget_consistency", summary)
        self.assertNotIn("sibling_ranking", summary)
        self.assertNotIn("candidate_recall", summary)

    def test_states_that_could_not_be_measured_are_excluded_from_the_means(self) -> None:
        records = [
            {"sibling": {"pairwise_accuracy": 1.0, "top1_agrees": True, "scored": 4, "deep_spread": 1.0}},
            {"sibling": {"pairwise_accuracy": None, "top1_agrees": False, "scored": 1, "deep_spread": 0.0}},
        ]

        summary = summarize(records, frozenset({"sibling"}))

        self.assertEqual(summary["sibling_ranking"]["mean_pairwise_accuracy"], 1.0)
        self.assertEqual(summary["sibling_ranking"]["top1_agreement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
