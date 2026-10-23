"""Student-ranked candidates must enter the same teacher search on every backend/schedule."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_play


class StudentCandidatesTest(unittest.TestCase):
    def test_k2_deduplicates_and_preserves_labels_and_stage_order(self):
        plants = [{"type": "plant", "packet": 0, "row": 0, "col": col} for col in range(4)]
        ranked = [(1.0 - col / 10, plant) for col, plant in enumerate(plants)]
        records = []
        for backend, speculate in [("cpp", False), ("cuda", False), ("cuda", True)]:
            args = SimpleNamespace(per_card=1, min_gap=0, student_candidates=2, shovel_candidates=0,
                                   halving=True, samples=3, margin=.02,
                                   cuda_speculative_halving=speculate, rollout_backend=backend)
            env = Mock()
            env.snapshot_fast.return_value = 7
            env.branch_snapshot.side_effect = lambda root, actions, **kw: [
                {"ok": True, "observation": {"static": 1.0 - a["col"] / 10}} for a in actions]
            stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
            batches = []

            def continuations(env, obs, root, jobs, args, counters):
                return [{"terminal": False, "score": 0.1 if a["type"] == "wait" else 0.4 + a["col"] / 10}
                        for a, seed in jobs]

            def values(handles):
                scores = [handle["score"] for handle in handles]
                batches.append(scores)
                return scores

            with patch.dict(sys.modules, {"pvz_rollout_policy": SimpleNamespace(
                    ranked_plants=lambda *a, **kw: ranked)}), \
                    patch.dict(search_play._W, {"cuda": object() if backend == "cuda" else None,
                        "fast": object(), "student_frame": {"tick": 60}}, clear=True), \
                    patch("render_episode._frame", side_effect=lambda obs: obs.copy()), \
                    patch.object(search_play, "_static_scores", side_effect=lambda env, root, actions, income: [(1.0-a["col"]/10,a) for a in actions]), \
                    patch.object(search_play, "_continuations", side_effect=continuations), \
                    patch.object(search_play, "_leaf_values", side_effect=values):
                action = search_play._search(env, {"tick": 60, "legal_actions": {"plants": plants}},
                                             plants[0], args, 123, stats)
            self.assertEqual(stats["student_candidate_added"], 2)
            self.assertEqual([a for a, value in stats["last_targets"]],
                             [plants[0], search_play.NOOP, plants[1], plants[2]])
            self.assertEqual(stats["student_candidate_best"], 1)
            self.assertEqual(stats["student_candidate_override"], 1)
            records.append(json.dumps({"action": action, "targets": stats["last_targets"],
                                       "batches": batches}, sort_keys=True).encode())
            env.restore_fast.assert_called_with(7)
            env.release_snapshot.assert_called_once_with(7)
        self.assertEqual(len(set(records)), 1)


class VerifySamplesBackendTest(unittest.TestCase):
    """--verify-samples must ride the same backend as the search (task4)."""

    def test_verification_futures_are_one_continuation_batch(self):
        args = SimpleNamespace(samples=3, verify_samples=9, margin=0.02, halving=True,
                               cuda_speculative_halving=False, rollout_backend="cuda")
        candidates = [{"type": "plant", "packet": 1, "row": 0, "col": 0},
                      {"type": "plant", "packet": 1, "row": 0, "col": 1}]
        stats = {"overrides": 0, "override_kinds": Counter()}
        seen = {}

        def continuations(env, obs, root, jobs, args, counters):
            seen["jobs"] = list(jobs)
            return [0.9] * len(jobs)

        with patch.dict(search_play._W, {"cuda": object()}, clear=True), \
                patch.object(search_play, "_continuations", side_effect=continuations), \
                patch.object(search_play, "_leaf_values", side_effect=lambda handles: list(handles)):
            action = search_play._verified_choice(
                Mock(), {"tick": 60}, 7, candidates, [0.1, 0.5], [3, 3], 1, 0,
                {"type": "wait"}, args, 123, stats)
        self.assertEqual(len(seen["jobs"]), 18)
        self.assertEqual([job[1] for job in seen["jobs"]],
                         [123 * 1000 + future for future in range(3, 12)] * 2)
        self.assertEqual([job[0] for job in seen["jobs"]], [candidates[0]] * 9 + [candidates[1]] * 9)
        self.assertEqual(action["col"], 1)


class CudaDumpOptionsTest(unittest.TestCase):
    """CUDA_DUMP carries the v2lite history and never a TRACE suffix.

    The 2026-10-19 merge gave _student_options a required trace_path and left cuda_dump
    calling it with one argument, so the GPU search arm died on its first decision.  The
    C++ CUDA_DUMP parser only knows V2H, so the right call passes None.
    """

    def test_cuda_dump_emits_v2h_without_trace(self):
        from collections import deque
        from pvz_env import PvZEnv
        from pvz_student_observation import StudentHistory

        env = object.__new__(PvZEnv)
        env._reset_done = True
        env._rollout_version = 2
        env._sun_history_start_tick = 0
        env._sun_production_history = deque([(10, 25)])
        sent = {}

        def command(line):
            sent["line"] = line
            return {"ok": True}

        env._command = command
        history = StudentHistory("v2lite")
        history.plants[(1, 2, 0)] = 30
        history.wave, history.wave_tick = 1, 5
        env.cuda_dump("/tmp/root.cudaroot", write_level=True, student_history=history)
        line = sent["line"]
        self.assertIn("CUDA_DUMP", line)
        self.assertIn(" V2H 1 5 1 1 2 0 30", line)
        self.assertNotIn("TRACE", line)


if __name__ == "__main__":
    unittest.main()
