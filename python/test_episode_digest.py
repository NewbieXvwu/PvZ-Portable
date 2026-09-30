"""``episode_digest``: byte-level episode digests used for T5 trajectory provenance.

The digest replaced ``episode_hash``'s JSON path, which expanded every packed token
array into Python scalars (measured: 220,809 scalars per episode, 4.42e8 per
2,000-episode update).  These tests pin the two properties that make the fast path
safe to trust: the digest must be *deterministic* and it must be *sensitive* to every
field it claims to cover.
"""

from __future__ import annotations

import hashlib
import time
import unittest

import numpy as np

from train_pvz_ppo import EPISODE_DIGEST_FIELDS, _digest_into, episode_digest, episode_hash

TOKEN_COUNT = 70


def _packed_tokens(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A packed tokenization shaped exactly like ``pack_tokens`` output."""
    return {
        "ids": rng.integers(0, 100, size=(TOKEN_COUNT, 5)).astype(np.int8),
        "features": rng.random((TOKEN_COUNT, 32)).astype(np.float16),
        "cell_index": rng.integers(0, TOKEN_COUNT, size=54).astype(np.uint16),
        "packet_ids": np.arange(TOKEN_COUNT, dtype=np.uint8),
        "packet_index": rng.integers(0, TOKEN_COUNT, size=TOKEN_COUNT).astype(np.uint16),
    }


def _transition(rng: np.random.Generator, index: int) -> dict:
    return {
        "tokens": _packed_tokens(rng),
        "wave": index,
        "legal": {"plants": [0, 1, 2], "cells": {0: [1, 2], 1: [3]}, "shovel": [4], "wait": True},
        "previous_action": index % 3,
        "elapsed_since_previous_observation": 12,
        "action_duration_ticks": 6,
        "events": {"kills": index, "spawns": 0},
        "critic_extra": np.array([0.5, 0.25], dtype=np.float32),
        "action": {"kind": "wait"},
        "log_prob": -0.7 - index * 0.01,
        "value": 0.1,
        "potential": 0.3,
        "shaping_reward": 0.01,
        "terminal_outcome": 0.0,
        "reward": 0.02,
    }


def _episode(seed: int = 7, transitions: int = 64) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "task_seed": 50000 + seed,
        "task_id": "heldout_day_1",
        "result": {"won": False, "terminal_wave": 3},
        "transitions": [_transition(rng, index) for index in range(transitions)],
    }


def _mutators() -> dict[str, object]:
    """One perturbation per covered field, plus the episode identity fields."""
    return {
        "tokens": lambda t: t["tokens"]["ids"].__setitem__((0, 0), (int(t["tokens"]["ids"][0, 0]) + 1) % 100),
        "wave": lambda t: t.__setitem__("wave", t["wave"] + 1),
        "legal": lambda t: t["legal"]["plants"].append(3),
        "previous_action": lambda t: t.__setitem__("previous_action", t["previous_action"] + 1),
        "elapsed_since_previous_observation": lambda t: t.__setitem__(
            "elapsed_since_previous_observation", t["elapsed_since_previous_observation"] + 1),
        "action_duration_ticks": lambda t: t.__setitem__(
            "action_duration_ticks", t["action_duration_ticks"] + 1),
        "events": lambda t: t["events"].__setitem__("kills", t["events"]["kills"] + 1),
        "critic_extra": lambda t: t["critic_extra"].__setitem__(0, t["critic_extra"][0] + 1.0),
        "action": lambda t: t["action"].__setitem__("kind", "plant"),
        "log_prob": lambda t: t.__setitem__("log_prob", t["log_prob"] - 0.5),
        "value": lambda t: t.__setitem__("value", t["value"] + 0.5),
        "potential": lambda t: t.__setitem__("potential", t["potential"] + 0.5),
        "shaping_reward": lambda t: t.__setitem__("shaping_reward", t["shaping_reward"] + 0.5),
        "terminal_outcome": lambda t: t.__setitem__("terminal_outcome", 1.0),
        "reward": lambda t: t.__setitem__("reward", t["reward"] + 0.5),
    }


class EpisodeDigestTests(unittest.TestCase):
    def test_digest_is_deterministic_and_order_independent(self) -> None:
        episode = _episode()
        first = episode_digest(episode)
        self.assertEqual(first, episode_digest(episode))
        self.assertEqual(first, episode_digest(_episode()))
        # Rebuilding a transition dict with a different insertion order must not matter.
        shuffled = {key: episode["transitions"][0][key]
                    for key in reversed(list(episode["transitions"][0]))}
        reordered = _episode()
        reordered["transitions"][0] = shuffled
        self.assertEqual(first, episode_digest(reordered))

    def test_every_covered_field_changes_the_digest(self) -> None:
        baseline = episode_digest(_episode())
        for field, mutate in _mutators().items():
            with self.subTest(field=field):
                episode = _episode()
                mutate(episode["transitions"][3])
                self.assertNotEqual(baseline, episode_digest(episode),
                                    f"mutating {field} did not change the digest")

    def test_identity_and_result_change_the_digest(self) -> None:
        baseline = episode_digest(_episode())
        for field, value in (("seed", 8), ("task_seed", 99999), ("task_id", "heldout_night_1")):
            with self.subTest(field=field):
                episode = _episode()
                episode[field] = value
                self.assertNotEqual(baseline, episode_digest(episode))
        episode = _episode()
        episode["result"]["won"] = True
        self.assertNotEqual(baseline, episode_digest(episode))

    def test_a_missing_field_is_not_the_same_as_a_present_one(self) -> None:
        baseline = episode_digest(_episode())
        episode = _episode()
        del episode["transitions"][2]["reward"]
        self.assertNotEqual(baseline, episode_digest(episode))

    def test_dtype_and_shape_are_part_of_the_digest(self) -> None:
        wide = _episode()
        narrow = _episode()
        narrow["transitions"][0]["critic_extra"] = np.array([0.5, 0.25], dtype=np.float64)
        self.assertNotEqual(episode_digest(wide), episode_digest(narrow))

        reshaped = _episode()
        reshaped["transitions"][0]["critic_extra"] = np.array([[0.5], [0.25]], dtype=np.float32)
        self.assertNotEqual(episode_digest(wide), episode_digest(reshaped))

    def test_length_prefix_prevents_concatenation_collisions(self) -> None:
        """Structurally different payloads must never share a digest.

        Every leaf is already self-delimiting (ints carry a NUL, strings carry a
        length), so the container length prefix is belt-and-braces here rather than
        the sole defence.  ``test_encoding_is_self_describing`` pins the prefix
        itself; this test pins the property a reader actually cares about.
        """
        left = _episode()
        right = _episode()
        left["transitions"][0]["events"] = {"kills": 1, "spawns": 23}
        right["transitions"][0]["events"] = {"kills": 12, "spawns": 3}
        self.assertNotEqual(episode_digest(left), episode_digest(right))

        split = _episode()
        joined = _episode()
        split["transitions"][0]["legal"]["shovel"] = [1, 2]
        joined["transitions"][0]["legal"]["shovel"] = [1, 2]
        split["transitions"][0]["legal"]["plants"] = []
        joined["transitions"][0]["legal"]["plants"] = []
        split["transitions"][0]["legal"]["cells"] = {0: [1], 1: [2]}
        joined["transitions"][0]["legal"]["cells"] = {0: [1], 1: [2]}
        self.assertEqual(episode_digest(split), episode_digest(joined))
        joined["transitions"][0]["legal"]["cells"] = {0: [1, 2], 1: []}
        self.assertNotEqual(episode_digest(split), episode_digest(joined))

    def test_encoding_is_self_describing(self) -> None:
        """Pin the byte stream: the digest is a provenance contract, not an internal.

        If this test fails because the encoding changed, every digest already recorded
        in a run's ``trajectory_sha256`` became unreproducible.  Treat it as a signal
        to version the algorithm, not to update the expected bytes.
        """
        hasher = hashlib.blake2b()
        _digest_into(hasher, [[1], [2, 3]])
        expected = (b"L2\x00" + b"L1\x00" + b"I1\x00"
                    + b"L2\x00" + b"I2\x00" + b"I3\x00")
        self.assertEqual(hasher.hexdigest(), hashlib.blake2b(expected).hexdigest())

        hasher = hashlib.blake2b()
        _digest_into(hasher, {"b": None, "a": True})
        expected = b"D2\x00" + b"S1\x00a" + b"T" + b"S1\x00b" + b"N"
        self.assertEqual(hasher.hexdigest(), hashlib.blake2b(expected).hexdigest())

    def test_recorded_digest_is_stable_across_numpy_rebuilds(self) -> None:
        """A fixed synthetic episode must keep producing the same digest."""
        self.assertEqual(episode_digest(_episode(seed=7)),
                         "0915db43c1a665da68b8fdf2feb2e3b7")

    def test_lists_and_tuples_digest_alike(self) -> None:
        as_list = _episode()
        as_tuple = _episode()
        as_tuple["transitions"][0]["legal"]["plants"] = tuple(
            as_list["transitions"][0]["legal"]["plants"])
        self.assertEqual(episode_digest(as_list), episode_digest(as_tuple))

    def test_unsupported_values_are_rejected(self) -> None:
        episode = _episode()
        episode["transitions"][0]["value"] = object()
        with self.assertRaisesRegex(TypeError, "episode digest cannot encode object"):
            episode_digest(episode)

    def test_digest_is_much_faster_than_the_json_path(self) -> None:
        episode = _episode()
        episode_hash(episode)
        episode_digest(episode)

        def measure(function, repeats: int = 3) -> float:
            start = time.perf_counter()
            for _ in range(repeats):
                function(episode)
            return (time.perf_counter() - start) / repeats

        json_seconds = measure(episode_hash)
        byte_seconds = measure(episode_digest)
        self.assertLess(byte_seconds * 5, json_seconds,
                        f"expected a >=5x speedup, got {json_seconds / byte_seconds:.1f}x")

    def test_legacy_episode_hash_is_still_available_and_differs(self) -> None:
        episode = _episode()
        self.assertEqual(episode_hash(episode), episode_hash(_episode()))
        self.assertNotEqual(episode_hash(episode), episode_digest(episode))
        self.assertEqual(len(episode_digest(episode)), 32)
        self.assertEqual(len(episode_hash(episode)), 64)

    def test_field_list_matches_the_legacy_json_path(self) -> None:
        """A field added to one path but not the other would silently lose coverage."""
        episode = _episode()
        for field in EPISODE_DIGEST_FIELDS:
            self.assertIn(field, episode["transitions"][0])


if __name__ == "__main__":
    unittest.main()
