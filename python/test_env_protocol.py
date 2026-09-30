"""Tests for the environment wrapper's pure-Python validation and replay bookkeeping.

Nothing here launches ``pvz-portable``: every assertion targets a code path that runs
before the first byte reaches the child process, or that hashes files on disk. That is
deliberate -- these are exactly the paths that were silently broken by a missing import
while the process-level integration tests still passed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pvz_env
from pvz_common import (
    ENV_PROTOCOL_VERSION,
    REPLAY_FORMAT_VERSION,
    git_metadata,
)
from pvz_env import (
    LoadoutContext,
    PlayerProfileContext,
    PvZEnv,
    SeedCard,
    TaskSpec,
    training_task,
)


@contextmanager
def _resource_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "properties").mkdir()
        (root / "main.pak").write_bytes(b"main")
        (root / "properties" / "partner.xml").write_bytes(b"partner")
        yield root


class ConstructionTests(unittest.TestCase):
    def test_reset_events_cannot_leak_into_next_episode_income(self) -> None:
        env = PvZEnv(resource_dir="/nonexistent")
        initial = {"tick": 0, "wave_timer": 3000}
        env._annotate_observation(initial, {"sun_produced": 2**32 - 25}, reset_history=True)
        next_observation = {"tick": 150, "wave_timer": 2850}
        env._annotate_observation(next_observation, {"sun_produced": 0})
        self.assertEqual(next_observation["sun_income_rate"], 0.0)
        env._annotate_observation({"tick": 300, "wave_timer": 2700}, {"sun_produced": 25})
        self.assertEqual(sum(amount for _, amount in env._sun_production_history), 25)

    def test_construction_does_not_touch_the_resource_directory(self) -> None:
        """Hashing main.pak eagerly raised a bare FileNotFoundError before _start() could
        report the friendly "main.pak not found" message."""
        env = PvZEnv(resource_dir="/nonexistent/resource/directory")

        self.assertIsNone(env.episode)
        self.assertFalse(env._reset_done)
        self.assertIsNone(env._tick)

    def test_resource_hashes_are_computed_lazily_and_cached(self) -> None:
        with _resource_dir() as resources:
            env = PvZEnv(resource_dir=resources)

            first = env._resource_hashes()

            self.assertEqual(first, env._resource_hashes())
            self.assertTrue(all(len(digest) == 64 for digest in first))

    def test_a_missing_main_pak_is_reported_when_the_hashes_are_needed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = PvZEnv(resource_dir=directory)

            with self.assertRaises(FileNotFoundError):
                env._resource_hashes()

    def test_close_is_idempotent_on_an_environment_that_never_started(self) -> None:
        env = PvZEnv(resource_dir="/nonexistent")

        env.close()
        env.close()

        self.assertFalse(env._reset_done)


class ResetValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Every rejection below happens before the first command reaches the child.
        self.env = PvZEnv(resource_dir="/nonexistent")

    def test_level_and_seed_must_be_in_range(self) -> None:
        for level in (0, 51, "7", 7.0, True):
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, "level must be an integer"):
                    self.env.reset(level=level, seed=0)
        for seed in (-1, 2 ** 32, "1", 1.5, False):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed must be an unsigned"):
                    self.env.reset(level=1, seed=seed)

    def test_task_options_are_validated(self) -> None:
        cases = [
            (TaskSpec(level=1, seed=0, loadout_mode="random"), "only fixed loadouts"),
            (TaskSpec(level=1, seed=0, zombie_count_multiplier=0.5), "zombie_count_multiplier"),
            (TaskSpec(level=1, seed=0, zombie_count_multiplier=11.0), "zombie_count_multiplier"),
            (TaskSpec(level=1, seed=0, wave_cap=0), "wave_cap"),
            (TaskSpec(level=1, seed=0, wave_cap=51), "wave_cap"),
            (TaskSpec(level=1, seed=0, wave_cap=True), "wave_cap"),
            (TaskSpec(level=1, seed=0, preplanted=[(1, 0, 0)]), "preplanted must contain"),
            (TaskSpec(level=1, seed=0, preplanted=((49, 0, 0),)), "preplanted seed_type"),
            (TaskSpec(level=1, seed=0, preplanted=((1, 6, 0),)), "preplanted seed_type"),
            (TaskSpec(level=1, seed=0, preplanted=((1, 0, 9),)), "preplanted seed_type"),
            (TaskSpec(level=1, seed=0, playthrough=1), r"playthrough=2"),
            (TaskSpec(level=1, seed=0, forced_seeds=(0, 1, 2, 3)), "up to three cards"),
            (TaskSpec(level=1, seed=0, forced_seeds=(0, 0)), "must not contain duplicates"),
            (TaskSpec(level=1, seed=0, forced_seeds=(40,)), "base plant IDs"),
        ]
        for task, message in cases:
            with self.subTest(task=task):
                with self.assertRaisesRegex(ValueError, message):
                    self.env.reset(task=task)

    def test_task_options_are_encoded_in_the_reset_protocol(self) -> None:
        with _resource_dir() as resources:
            env = PvZEnv(resource_dir=resources)
            commands: list[str] = []
            observation = {
                "tick": 0, "wave": 0, "wave_count": 30, "wave_timer": 3000, "sun": 50,
                "plants": [], "zombies": [], "terminal": False, "result": 0,
            }

            def command(text: str) -> dict:
                commands.append(text)
                return {"ok": True, "observation": observation}

            env._command = command  # type: ignore[method-assign]
            env.reset(task=TaskSpec(level=7, seed=123, wave_cap=3, preplanted=((1, 1, 2), (0, 2, 4))))

            self.assertEqual(commands, [
                "RESET_V2 7 123 2 6 0 0 0 0 0 - - 0,1,2,3,4,5 1 3 1:1:2,0:2:4"
            ])
            self.assertEqual(env.episode["task"]["wave_cap"], 3)
            self.assertEqual(env.episode["task"]["preplanted"], [[1, 1, 2], [0, 2, 4]])

    def test_profile_options_are_validated(self) -> None:
        cases = [
            (PlayerProfileContext(seed_slot_count=5), "seed_slot_count"),
            (PlayerProfileContext(seed_slot_count=11), "seed_slot_count"),
            (PlayerProfileContext(rake_charges=-1), "rake_charges"),
            (PlayerProfileContext(imitater_owned=1), "ownership flags must be booleans"),
            (PlayerProfileContext(owned_upgrade_plants=(39,)), "upgrade seed IDs 40 through 47"),
            (PlayerProfileContext(owned_upgrade_plants=(40, 40)), "must not contain duplicates"),
        ]
        for profile, message in cases:
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(ValueError, message):
                    self.env.reset(task=TaskSpec(level=1, seed=0, profile=profile))

    def test_deck_entries_are_validated(self) -> None:
        cases = [
            ([], "deck must contain cards"),
            ([object()], "deck entries must be"),
            ([(0, 1, 2)], "deck entries must be"),
            ([0, 0], "must not contain duplicate card types"),
            ([49], "invalid seed type"),
            ([48], "invalid seed type"),
            ([SeedCard(48, 0)], "imitater card requires"),
        ]
        for deck, message in cases:
            with self.subTest(deck=deck):
                with self.assertRaisesRegex(ValueError, message):
                    self.env.reset(deck=deck)

    def test_deck_must_fit_the_profile_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit the profile's seed slots"):
            self.env.reset(deck=list(range(7)), task=TaskSpec(level=1, seed=0))

    def test_forced_seeds_must_be_present_in_the_deck(self) -> None:
        with self.assertRaisesRegex(ValueError, "deck must include every forced seed"):
            self.env.reset(deck=[0, 1], task=TaskSpec(level=1, seed=0, forced_seeds=(5,)))

    def test_an_imitater_card_is_accepted_when_the_profile_owns_one(self) -> None:
        """Deck validation must not reject a legal imitater just because it is exotic."""
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            # Validation passed, so the failure is now the missing child process.
            self.env.reset(deck=[SeedCard(48, 0), 1, 2],
                           task=TaskSpec(level=1, seed=0,
                                         profile=PlayerProfileContext(imitater_owned=True)))


class LifecycleGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PvZEnv(resource_dir="/nonexistent")

    def test_operations_before_reset_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            self.env.step({"type": "wait", "ticks": 60})
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            self.env.save_replay("/tmp/never-written.jsonl")
        with self.assertRaisesRegex(RuntimeError, "call reset"):
            with self.env.speculative():
                pass

    def test_snapshot_ids_must_be_positive_integers(self) -> None:
        for snapshot_id in (0, -1, "1", 1.0, True):
            with self.subTest(snapshot_id=snapshot_id):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.env.restore(snapshot_id)
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.env.release_snapshot(snapshot_id)

    def test_coordinates_must_be_integers(self) -> None:
        keys = ("packet", "col", "row")
        self.assertEqual(PvZEnv._coordinates({"packet": 1, "col": 2, "row": 3}, keys), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "must be integers"):
            PvZEnv._coordinates({"packet": 1.5, "col": 2, "row": 3}, keys)
        with self.assertRaisesRegex(ValueError, "must be integers"):
            PvZEnv._coordinates({"col": 2}, ("col", "row"))

    def test_replay_records_with_an_unknown_version_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported replay version"):
            self.env.replay_record({"format_version": REPLAY_FORMAT_VERSION - 1})
        with self.assertRaisesRegex(ValueError, "unsupported replay version"):
            self.env.replay_record({})


class BranchSnapshotTests(unittest.TestCase):
    """``BRANCH_SNAPSHOT_FAST`` is the command the search pays for every expansion.

    Its spec encoding lives in ``pvz_env.branch_action_token`` and nowhere else, so these
    tests pin the wire format that ``src/main.cpp::DecodeBranchAction`` parses.
    """

    def setUp(self) -> None:
        self.env = PvZEnv(resource_dir="/nonexistent")
        self.commands: list[str] = []

    def _stub(self, branches: list[dict]) -> None:
        def command(text: str) -> dict:
            self.commands.append(text)
            return {"ok": True, "branches": branches}

        self.env._command = command  # type: ignore[method-assign]

    def test_actions_are_encoded_as_colon_separated_tokens(self) -> None:
        self._stub([{"ok": True}, {"ok": True}, {"ok": True}])

        branches = self.env.branch_snapshot(7, [
            {"type": "plant", "packet": 2, "col": 3, "row": 4},
            {"type": "shovel", "col": 1, "row": 0},
            {"type": "wait", "ticks": 60},
        ])

        self.assertEqual(len(branches), 3)
        self.assertEqual(self.commands, ["BRANCH_SNAPSHOT_FAST 7 3 P:2:3:4 S:1:0 W:60"])

    def test_snapshot_id_must_be_a_positive_integer(self) -> None:
        self._stub([])
        for snapshot_id in (0, -1, "1", 1.0, True):
            with self.subTest(snapshot_id=snapshot_id):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.env.branch_snapshot(snapshot_id, [{"type": "wait", "ticks": 1}])
        self.assertEqual(self.commands, [])

    def test_an_empty_action_list_is_rejected_before_any_command(self) -> None:
        self._stub([])

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            self.env.branch_snapshot(1, [])

        self.assertEqual(self.commands, [])

    def test_more_than_the_batch_limit_is_rejected_before_any_command(self) -> None:
        self._stub([])

        with self.assertRaisesRegex(ValueError, "at most 128"):
            self.env.branch_snapshot(1, [{"type": "wait", "ticks": 1}] * (pvz_env.BRANCH_BATCH_LIMIT + 1))

        self.assertEqual(self.commands, [])

    def test_a_response_with_the_wrong_branch_count_is_rejected(self) -> None:
        """A truncated branch list must not be paired positionally with the actions."""
        self._stub([{"ok": True}])

        with self.assertRaisesRegex(ValueError, "could not branch"):
            self.env.branch_snapshot(1, [{"type": "wait", "ticks": 1}, {"type": "wait", "ticks": 1}])

    def test_unsupported_actions_are_rejected_by_the_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported branch action"):
            pvz_env.branch_action_token({"type": "sun"})
        with self.assertRaisesRegex(TypeError, "must be a dictionary"):
            pvz_env.branch_action_token("wait")


class SharedConstantTests(unittest.TestCase):
    def test_env_module_reexports_the_shared_protocol_versions(self) -> None:
        """pvz_env must not redefine the versions it validates against."""
        self.assertIs(pvz_env.ENV_PROTOCOL_VERSION, ENV_PROTOCOL_VERSION)
        self.assertIs(pvz_env.REPLAY_FORMAT_VERSION, REPLAY_FORMAT_VERSION)
        self.assertEqual(set(pvz_env.__all__), {
            "BRANCH_BATCH_LIMIT", "ENV_PROTOCOL_VERSION", "REPLAY_FORMAT_VERSION",
            "LoadoutContext", "PlayerProfileContext", "PvZEnv", "SeedCard",
            "SimulatorExited", "TaskSpec", "branch_action_token", "training_task",
        })

    def test_training_task_is_the_single_shared_task_factory(self) -> None:
        task = training_task(1234, 7, 2.5)

        self.assertEqual((task.level, task.seed, task.playthrough), (7, 1234, 2))
        self.assertEqual(task.zombie_count_multiplier, 2.5)
        self.assertEqual(task.profile, PlayerProfileContext())
        self.assertEqual(task.loadout_mode, "fixed")
        self.assertEqual(task.forced_seeds, ())

    def test_loadout_context_reads_the_observation(self) -> None:
        context = LoadoutContext.from_observation({"loadout_context": {
            "scene": 1, "seed_slot_count": 6, "free_slots": 4,
            "available_plants": [0, 1], "forced_seeds": [2], "zombie_roster": [0, 3],
        }})

        self.assertEqual(context.scene, 1)
        self.assertEqual(context.available_plants, (0, 1))
        self.assertEqual(context.forced_seeds, (2,))
        self.assertEqual(context.zombie_roster, (0, 3))


class ObservationDerivedInputTests(unittest.TestCase):
    @staticmethod
    def _observation(tick: int) -> dict:
        return {"tick": tick, "wave_timer": 2000}

    def test_sun_income_rate_uses_real_production_in_a_6000_tick_window(self) -> None:
        env = PvZEnv(resource_dir="/nonexistent")
        first = env._annotate_observation(self._observation(0), reset_history=True)
        second = env._annotate_observation(self._observation(3000), {"sun_produced": 30})
        third = env._annotate_observation(self._observation(8000), {"sun_produced": 12})
        fourth = env._annotate_observation(self._observation(10000))

        self.assertEqual(first["sun_income_rate"], 0.0)
        self.assertEqual(second["sun_income_rate"], 10.0)
        self.assertEqual(third["sun_income_rate"], 7.0)
        self.assertEqual(fourth["sun_income_rate"], 2.0)

    def test_wave_timer_must_be_present_in_real_observation(self) -> None:
        env = PvZEnv(resource_dir="/nonexistent")
        with self.assertRaisesRegex(RuntimeError, "missing the public wave_timer field"):
            env._annotate_observation({"tick": 10}, reset_history=True)

class ExperimentManifestTests(unittest.TestCase):
    """The manifest is written by ``save_replay`` and re-verified when a replay is loaded.

    Every one of these calls used to raise ``NameError`` because ``pvz_env`` had lost its
    ``hashlib`` import while the process-level tests kept passing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        revision, _ = git_metadata(Path(__file__).resolve().parent.parent)
        if revision is None:
            raise unittest.SkipTest("the workspace is not a git checkout")

    def test_manifest_is_written_verified_and_tamper_evident(self) -> None:
        with _resource_dir() as resources, tempfile.TemporaryDirectory() as directory:
            env = PvZEnv(resource_dir=resources)

            reference = env._write_manifest(Path(directory))

            self.assertEqual(reference["path"], "experiment_manifest.json")
            self.assertEqual(len(reference["sha256"]), 64)
            manifest_path = Path(directory) / reference["path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["working_tree_patch_sha256"]), 64)
            self.assertTrue((Path(directory) / manifest["working_tree_patch"]).is_file())

            env._check_manifest(reference, Path(directory))

            manifest_path.write_text('{"tampered":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                env._check_manifest(reference, Path(directory))

    def test_a_tampered_working_tree_patch_is_detected(self) -> None:
        with _resource_dir() as resources, tempfile.TemporaryDirectory() as directory:
            env = PvZEnv(resource_dir=resources)
            reference = env._write_manifest(Path(directory))

            (Path(directory) / "working_tree.patch").write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "working tree patch"):
                env._check_manifest(reference, Path(directory))

    def test_a_missing_manifest_reference_is_rejected(self) -> None:
        env = PvZEnv(resource_dir="/nonexistent")

        with self.assertRaisesRegex(ValueError, "no experiment manifest reference"):
            env._check_manifest({}, None)
        with self.assertRaisesRegex(ValueError, "no experiment manifest reference"):
            env._check_manifest({"path": "experiment_manifest.json", "sha256": "0" * 64}, None)


class EventKeyCanonicalisationTests(unittest.TestCase):
    """``events`` is JSON-decoded on every step, so its key strings must be shared.

    Measured on a real episode those keys cost 21.0 KiB -- 61% of the ``events`` field
    and 5.1% of the whole rollout payload, 41 MiB over a 2000-episode batch.  Each test
    below builds its keys with ``_uninterned`` so that a vacuous assertion -- one that
    would pass even if ``_canonical_events`` were deleted -- cannot slip through.
    """

    @staticmethod
    def _uninterned(name: str) -> str:
        """A string equal to *name* but a different object, the way JSON decoding makes it."""
        return "".join([name, ""])

    def test_the_helper_replaces_uninterned_keys_and_keeps_the_values(self) -> None:
        incoming = {self._uninterned(name): index
                    for index, name in enumerate(sorted(pvz_env.EVENT_KEYS))}
        for key in incoming:
            self.assertIsNot(key, pvz_env.EVENT_KEYS[key],
                             "test setup: the key was already the shared object")

        canonical = pvz_env._canonical_events(incoming)

        self.assertEqual(canonical, incoming)
        for key in canonical:
            self.assertIs(key, pvz_env.EVENT_KEYS[key])

    def test_an_unknown_key_is_passed_through_untouched(self) -> None:
        marker = self._uninterned("a-key-this-module-has-never-heard-of")
        canonical = pvz_env._canonical_events({self._uninterned("sun_spent"): 50, marker: 1})

        self.assertEqual(canonical, {"sun_spent": 50, marker: 1})
        known, unknown = list(canonical)
        self.assertIs(known, pvz_env.EVENT_KEYS["sun_spent"])
        self.assertIs(unknown, marker)

    def test_empty_and_non_dict_inputs_are_returned_unchanged(self) -> None:
        empty: dict[str, int] = {}
        self.assertIs(pvz_env._canonical_events(empty), empty)
        self.assertIsNone(pvz_env._canonical_events(None))
        not_a_dict = ["not", "a", "dict"]
        self.assertIs(pvz_env._canonical_events(not_a_dict), not_a_dict)

    def test_read_message_canonicalises_the_events_it_decodes(self) -> None:
        """The hook is in ``_read_message``, which is where every response is decoded."""
        payload = json.dumps({
            "protocol_version": ENV_PROTOCOL_VERSION,
            "ok": True,
            "events": {"sun_spent": 50, "level_won": False},
        })
        env = PvZEnv(resource_dir="/nonexistent")

        class _FakeProcess:
            stdout = [f"PVZENV {payload}\n"]

        env._process = _FakeProcess()  # type: ignore[assignment]
        response = env._read_message()

        self.assertEqual(response["events"], {"sun_spent": 50, "level_won": False})
        for key in response["events"]:
            self.assertIs(key, pvz_env.EVENT_KEYS[key],
                          f"key {key!r} survived decoding without canonicalisation")


if __name__ == "__main__":
    unittest.main()
