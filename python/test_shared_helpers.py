"""Tests for the shared digest/path helpers, the label-diagnostic state key, and the benchmark summary."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from benchmark_pvz_agent import DEFAULT_MAX_ACTIONS, summarize
from pvz_common import (
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    REPLAY_FORMAT_VERSION,
    TASK_VERSION,
    TRAINING_SEED,
    VALUE_RANGE,
    canonical_digest,
    git_metadata,
    is_generated_artifact,
    sha256_bytes,
    sha256_file,
)
from pvz_search_diagnostics import visible_state_key


class SharedDigestTests(unittest.TestCase):
    def test_sha256_bytes_matches_the_reference_implementation(self) -> None:
        self.assertEqual(sha256_bytes(b"pvz-portable"), hashlib.sha256(b"pvz-portable").hexdigest())

    def test_sha256_file_streams_the_same_digest_as_sha256_bytes(self) -> None:
        payload = bytes(range(256)) * 5000  # crosses the 1 MiB read chunk boundary
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(payload)

            self.assertEqual(sha256_file(path), sha256_bytes(payload))

    def test_canonical_digest_ignores_key_order_but_not_content(self) -> None:
        self.assertEqual(canonical_digest({"b": 1, "a": 2}), canonical_digest({"a": 2, "b": 1}))
        self.assertNotEqual(canonical_digest({"a": 1}), canonical_digest({"a": 2}))
        self.assertEqual(len(canonical_digest({"a": 1})), 64)

    def test_generated_artifacts_are_recognised(self) -> None:
        for path in ("artifacts/adventure2_level7/x.json", "experiment_manifest.json",
                     "working_tree.patch", "replays/search_seed_1.jsonl.gz", "traces/a.jsonl"):
            with self.subTest(path=path):
                self.assertTrue(is_generated_artifact(path))
        for path in ("python/pvz_env.py", "src/LawnApp.cpp", "README.md", "docs/experiment_manifest.md"):
            with self.subTest(path=path):
                self.assertFalse(is_generated_artifact(path))

    def test_git_metadata_reports_a_revision_or_gives_up_cleanly(self) -> None:
        revision, dirty = git_metadata(Path(__file__).resolve().parent.parent)

        self.assertTrue(revision is None or len(revision) == 40)
        self.assertTrue(dirty is None or isinstance(dirty, bool))

    def test_git_metadata_returns_nothing_outside_a_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(git_metadata(Path(directory)), (None, None))

    def test_version_constants_are_the_documented_values(self) -> None:
        self.assertEqual(ENV_PROTOCOL_VERSION, 3)
        self.assertEqual(REPLAY_FORMAT_VERSION, 5)
        self.assertEqual(OBSERVATION_VERSION, 2)
        self.assertEqual(TASK_VERSION, 2)
        self.assertEqual(TRAINING_SEED, 17)
        self.assertEqual(VALUE_RANGE, (-1.0, 1.0))


def _diagnostic_observation(**overrides: object) -> dict:
    base = {
        "level": 7,
        "wave": 3,
        "tick": 900,
        "sun": 175,
        "zombie_count_multiplier": 1.0,
        "night": False, "pool": False, "fog": False, "roof": False,
        "plants": [{"type": 0, "row": 2, "col": 3, "health": 300, "max_health": 300}],
        "zombies": [{"type": 0, "row": 2, "x": 700.0, "body_health": 200,
                     "helm_health": 0, "shield_health": 0}],
        "packets": [{"type": 0, "imitater_type": -1, "active": True,
                     "cooldown": 0, "refresh_time": 3000}],
        "defenses": [{"row": 0, "state": 1}],
    }
    base.update(overrides)
    return base


class VisibleStateKeyTests(unittest.TestCase):
    def test_the_key_is_stable_for_the_same_state(self) -> None:
        source = _diagnostic_observation()

        self.assertEqual(visible_state_key(source), visible_state_key(dict(source)))

    def test_the_key_ignores_sub_tick_and_sub_sun_detail(self) -> None:
        """The key groups *similar* states, so it deliberately buckets tick and sun."""
        source = _diagnostic_observation()

        nudged = {**source, "tick": source["tick"] + 1, "sun": source["sun"] + 1}

        self.assertEqual(visible_state_key(source), visible_state_key(nudged))

    def test_the_key_changes_when_the_board_changes(self) -> None:
        source = _diagnostic_observation()
        variants = {
            "no plants": {**source, "plants": []},
            "moved zombie": {**source, "zombies": [{**source["zombies"][0], "row": 4}]},
            "later wave": {**source, "wave": 4},
            "night": {**source, "night": True},
        }

        for label, variant in variants.items():
            with self.subTest(variant=label):
                self.assertNotEqual(visible_state_key(source), visible_state_key(variant))

    def test_the_key_is_json_encoded_for_use_as_a_cluster_id(self) -> None:
        key = visible_state_key(_diagnostic_observation())

        self.assertTrue(key.startswith("{"))
        self.assertNotIn(" ", key)


def _episode_record(**overrides: object) -> dict:
    record = {
        "seed": 30000, "replay_id": "x.jsonl.gz", "won": True, "result": 1, "terminal": True,
        "wave": 10, "wave_count": 10, "tick": 9000, "actions": 40, "ticks_advanced": 9000,
        "reset_seconds": 0.5, "ticks_per_second": 1800.0, "actions_per_second": 8.0,
        "plants_eaten": 0, "mower_triggers": 0, "seconds": 5.0,
    }
    record.update(overrides)
    return record


class BenchmarkSummaryTests(unittest.TestCase):
    def test_summarize_rejects_an_empty_episode_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty episode set"):
            summarize([])

    def test_win_rate_is_bracketed_by_the_wilson_interval(self) -> None:
        records = [_episode_record(seed=30000 + index, won=index < 3, result=1 if index < 3 else 0)
                   for index in range(4)]

        summary = summarize(records)

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["wins"], 3)
        self.assertEqual(summary["win_rate"], 0.75)
        low, high = summary["wilson_95"]
        self.assertLess(low, summary["win_rate"])
        self.assertGreater(high, summary["win_rate"])
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_losses_are_bucketed_by_the_wave_they_failed_on(self) -> None:
        records = [_episode_record(won=True), _episode_record(won=False, wave=5),
                   _episode_record(won=False, wave=5), _episode_record(won=False, wave=2)]

        summary = summarize(records)

        self.assertEqual(summary["failure_wave_distribution"], {"2": 1, "5": 2})

    def test_totals_and_means_are_aggregated(self) -> None:
        records = [_episode_record(plants_eaten=2, mower_triggers=1, seconds=4.0, actions=10),
                   _episode_record(plants_eaten=6, mower_triggers=3, seconds=6.0, actions=30)]

        summary = summarize(records)

        self.assertEqual(summary["plants_eaten_total"], 8)
        self.assertEqual(summary["mower_triggers_total"], 4)
        self.assertEqual(summary["mean_plants_eaten"], 4.0)
        self.assertEqual(summary["mean_actions"], 20.0)
        self.assertEqual(summary["wall_seconds_total"], 10.0)
        self.assertEqual(summary["wall_seconds_mean"], 5.0)

    def test_search_metrics_are_only_summarised_when_present(self) -> None:
        plain = summarize([_episode_record(), _episode_record()])

        self.assertNotIn("search_mean_margin", plain)
        self.assertNotIn("search_simulations_total", plain)

        searched = summarize([
            _episode_record(search_simulations=100, search_mean_margin=0.5, search_mean_entropy=1.0,
                            search_mean_candidate_count=4, search_mean_elapsed_ticks=900,
                            search_mean_simulations_per_decision=2.5),
            _episode_record(search_simulations=200, search_mean_margin=1.5, search_mean_entropy=0.0,
                            search_mean_candidate_count=6, search_mean_elapsed_ticks=300,
                            search_mean_simulations_per_decision=3.5),
        ])

        self.assertEqual(searched["search_simulations_total"], 300)
        self.assertEqual(searched["search_mean_margin"], 1.0)
        self.assertEqual(searched["search_mean_candidate_count"], 5.0)
        self.assertEqual(searched["search_mean_elapsed_ticks"], 600.0)

    def test_a_single_episode_summarises_to_itself(self) -> None:
        summary = summarize([_episode_record(won=False, result=0, wave=3, actions=17)])

        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["mean_actions"], 17)
        self.assertEqual(summary["failure_wave_distribution"], {"3": 1})

    def test_the_default_decision_cap_is_documented(self) -> None:
        self.assertEqual(DEFAULT_MAX_ACTIONS, 2000)


if __name__ == "__main__":
    unittest.main()
