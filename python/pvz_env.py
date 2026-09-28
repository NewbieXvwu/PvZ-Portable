"""Thin Python control wrapper for the headless PvZ-Portable environment."""

from __future__ import annotations

import gzip
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterator, Sequence

from pvz_common import (
    ENV_PROTOCOL_VERSION,
    REPLAY_FORMAT_VERSION,
    canonical_digest,
    git_metadata,
    is_generated_artifact,
    sha256_bytes,
    sha256_file,
)

__all__ = [
    "BRANCH_BATCH_LIMIT",
    "ENV_PROTOCOL_VERSION",
    "REPLAY_FORMAT_VERSION",
    "LoadoutContext",
    "PlayerProfileContext",
    "PvZEnv",
    "SeedCard",
    "SimulatorExited",
    "TaskSpec",
    "branch_action_token",
    "training_task",
]

# ``BRANCH_SNAPSHOT_FAST`` refuses more than this many specs in one command
# (``src/main.cpp`` bounds ``branchCount`` at 128), so callers that want to expand a
# wider action set have to chunk.  The search itself never gets near this -- the root
# request saturates at 96 and internal nodes use ``candidate_limit`` -- but the audit
# tools deliberately enumerate every legal placement.
BRANCH_BATCH_LIMIT = 128


def branch_action_token(action: dict[str, Any]) -> str:
    """Encode one action as the compact spec ``BRANCH_SNAPSHOT_FAST`` expects.

    The protocol passes branch actions as a single space-delimited token per action,
    with ``:`` as the field separator, because the command line itself is
    whitespace-tokenised.  This is the only place that encoding is defined; the search
    teacher and the equivalence harness both go through it.
    """
    if not isinstance(action, dict):
        raise TypeError("action must be a dictionary")
    kind = action.get("type")
    if kind == "plant":
        return f"P:{int(action['packet'])}:{int(action['col'])}:{int(action['row'])}"
    if kind == "shovel":
        return f"S:{int(action['col'])}:{int(action['row'])}"
    if kind == "wait":
        return f"W:{int(action['ticks'])}"
    raise ValueError(f"unsupported branch action: {action}")


class SimulatorExited(RuntimeError):
    """The ``pvz-portable`` child process died without answering.

    A crash is a distinct failure from a rejected command or a protocol mismatch, and
    callers that are verifying behaviour (rather than driving training) need to tell them
    apart: a segfault in the simulator is a divergence, not a bug in their own action
    sequence.  Subclasses ``RuntimeError`` so existing handlers keep working.
    """

    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        super().__init__(f"PvZ-Portable exited unexpectedly with code {returncode}")


@dataclass(frozen=True)
class PlayerProfileContext:
    seed_slot_count: int = 6
    owned_upgrade_plants: tuple[int, ...] = ()
    imitater_owned: bool = False
    first_aid_owned: bool = False
    pool_cleaner_owned: bool = False
    roof_cleaner_owned: bool = False
    rake_charges: int = 0


@dataclass(frozen=True)
class TaskSpec:
    level: int = 1
    seed: int = 0
    playthrough: int = 2
    profile: PlayerProfileContext = field(default_factory=PlayerProfileContext)
    forced_seeds: tuple[int, ...] = ()
    loadout_mode: str = "fixed"
    zombie_count_multiplier: float = 1.0


def training_task(seed: int, level: int, zombie_count_multiplier: float = 1.0) -> TaskSpec:
    """The single ``TaskSpec`` shared by search collection, DAgger, PPO rollouts and benchmarks."""
    return TaskSpec(
        level=level,
        seed=seed,
        playthrough=2,
        profile=PlayerProfileContext(),
        zombie_count_multiplier=zombie_count_multiplier,
    )


@dataclass(frozen=True)
class LoadoutContext:
    scene: int
    seed_slot_count: int
    free_slots: int
    available_plants: tuple[int, ...]
    forced_seeds: tuple[int, ...]
    zombie_roster: tuple[int, ...]

    @classmethod
    def from_observation(cls, observation: dict[str, Any]) -> "LoadoutContext":
        context = observation["loadout_context"]
        return cls(
            scene=context["scene"],
            seed_slot_count=context["seed_slot_count"],
            free_slots=context["free_slots"],
            available_plants=tuple(context["available_plants"]),
            forced_seeds=tuple(context["forced_seeds"]),
            zombie_roster=tuple(context["zombie_roster"]),
        )


@dataclass(frozen=True)
class SeedCard:
    seed_type: int
    imitater_type: int | None = None


class PvZEnv:
    def __init__(
        self,
        resource_dir: str | os.PathLike[str],
        executable: str | os.PathLike[str] | None = None,
        save_dir: str | os.PathLike[str] | None = None,
        headless: bool = True,
        debug_replay: bool = False,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        binary_name = "pvz-portable.exe" if os.name == "nt" else "pvz-portable"
        self.executable = Path(executable) if executable else root / "build" / binary_name
        self.resource_dir = Path(resource_dir).expanduser().resolve()
        self.headless = headless
        self.debug_replay = debug_replay
        self._root = root
        self.save_dir = Path(save_dir).expanduser().resolve() if save_dir else None
        self._temporary_save: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._reset_done = False
        self._last_response: dict[str, Any] = {}
        self.episode: dict[str, Any] | None = None
        # Resource hashes and git metadata are computed on first use: hashing main.pak
        # eagerly here would raise a bare FileNotFoundError before ``_start`` can report
        # the friendly "main.pak not found in resource directory" message.
        self._resource_sha256: str | None = None
        self._properties_sha256: str | None = None
        self._source_metadata: tuple[str | None, bool | None] | None = None
        # Tick of the most recent observation, used to derive real tick deltas.
        self._tick: int | None = None

    def _resource_hashes(self) -> tuple[str, str]:
        if self._resource_sha256 is None or self._properties_sha256 is None:
            self._resource_sha256 = sha256_file(self.resource_dir / "main.pak")
            self._properties_sha256 = sha256_file(self.resource_dir / "properties" / "partner.xml")
        return self._resource_sha256, self._properties_sha256

    def _source_state(self) -> tuple[str | None, bool | None]:
        if self._source_metadata is None:
            self._source_metadata = git_metadata(self._root)
        return self._source_metadata

    def _adopt_tick(self, observation: dict[str, Any]) -> None:
        if "tick" in observation:
            self._tick = int(observation["tick"])

    def _start(self) -> None:
        if not self.executable.is_file():
            raise FileNotFoundError(f"PvZ-Portable executable not found: {self.executable}")
        if not (self.resource_dir / "main.pak").is_file():
            raise FileNotFoundError(f"main.pak not found in resource directory: {self.resource_dir}")
        if self.save_dir is None:
            self._temporary_save = tempfile.TemporaryDirectory(prefix="pvz-env-")
            save_dir = Path(self._temporary_save.name)
        else:
            save_dir = self.save_dir
            save_dir.mkdir(parents=True, exist_ok=True)

        self._process = subprocess.Popen(
            [
                str(self.executable),
                "-env" if self.headless else "-env-visible",
                "-resdir",
                str(self.resource_dir),
                "-savedir",
                str(save_dir),
            ],
            cwd=self.executable.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        ready = self._read_message()
        if not ready.get("ready") or ready.get("protocol_version") != ENV_PROTOCOL_VERSION:
            self._abort_process()
            raise RuntimeError(f"PvZ-Portable did not enter protocol-v{ENV_PROTOCOL_VERSION} environment mode: {ready}")

    def _abort_process(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            self._process.wait()
            self._process = None
        if self._temporary_save is not None:
            self._temporary_save.cleanup()
            self._temporary_save = None
        self._reset_done = False
        self._tick = None

    def _read_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("environment process is not running")
        for line in process.stdout:
            if line.startswith("PVZENV "):
                return json.loads(line[len("PVZENV ") :])
        raise SimulatorExited(process.poll())

    def _command(self, command: str) -> dict[str, Any]:
        if self._process is None:
            self._start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("environment process is not running")
        process.stdin.write(command + "\n")
        process.stdin.flush()
        response = self._read_message()
        if response.get("protocol_version") != ENV_PROTOCOL_VERSION:
            actual = response.get("protocol_version")
            self._abort_process()
            raise RuntimeError(
                f"environment protocol mismatch: expected {ENV_PROTOCOL_VERSION}, got {actual}"
            )
        self._last_response = response
        return response

    def reset(
        self,
        level: int = 1,
        seed: int = 0,
        deck: Sequence[int | SeedCard | tuple[int, int]] = (0, 1, 2, 3, 4, 5),
        task: TaskSpec | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if task is not None:
            level, seed = task.level, task.seed
        if type(level) is not int or not 1 <= level <= 50:
            raise ValueError("level must be an integer from 1 to 50")
        if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if task is None:
            task = TaskSpec(level=level, seed=seed)
        profile = task.profile
        if task.loadout_mode != "fixed":
            raise ValueError("only fixed loadouts are supported")
        if (type(task.zombie_count_multiplier) not in (int, float) or
                not 1.0 <= task.zombie_count_multiplier <= 10.0):
            raise ValueError("zombie_count_multiplier must be from 1 to 10")
        if type(task.playthrough) is not int or task.playthrough != 2:
            raise ValueError("only replay semantics (playthrough=2) are supported")
        if type(profile.seed_slot_count) is not int or not 6 <= profile.seed_slot_count <= 10:
            raise ValueError("seed_slot_count must be from 6 to 10")
        if type(profile.rake_charges) is not int or profile.rake_charges < 0:
            raise ValueError("rake_charges must be a non-negative integer")
        if any(type(value) is not bool for value in (
            profile.imitater_owned, profile.first_aid_owned, profile.pool_cleaner_owned, profile.roof_cleaner_owned
        )):
            raise ValueError("profile ownership flags must be booleans")
        if any(type(value) is not int or not 40 <= value <= 47 for value in profile.owned_upgrade_plants):
            raise ValueError("owned_upgrade_plants must contain upgrade seed IDs 40 through 47")
        if len(set(profile.owned_upgrade_plants)) != len(profile.owned_upgrade_plants):
            raise ValueError("owned_upgrade_plants must not contain duplicates")
        if len(task.forced_seeds) > 3:
            raise ValueError("forced_seeds support up to three cards")
        if any(type(value) is not int or not 0 <= value < 40 for value in task.forced_seeds):
            raise ValueError("forced_seeds must contain base plant IDs from 0 through 39")
        if len(set(task.forced_seeds)) != len(task.forced_seeds):
            raise ValueError("forced_seeds must not contain duplicates")

        cards: list[SeedCard] = []
        for card in deck:
            if type(card) is int:
                cards.append(SeedCard(card))
            elif isinstance(card, SeedCard):
                cards.append(card)
            elif isinstance(card, tuple) and len(card) == 2 and all(type(value) is int for value in card):
                cards.append(SeedCard(card[0], card[1]))
            else:
                raise ValueError("deck entries must be seed IDs, SeedCard values, or (imitater, target) pairs")
        if not cards or len(cards) > profile.seed_slot_count:
            raise ValueError("deck must contain cards and fit the profile's seed slots")
        if any(type(card.seed_type) is not int or card.seed_type < 0 or card.seed_type >= 49 or
               (card.imitater_type is not None and
                (type(card.imitater_type) is not int or not 0 <= card.imitater_type < 49 or card.imitater_type == 48)) or
               (card.seed_type == 48) != (card.imitater_type is not None) for card in cards):
            raise ValueError("deck contains an invalid seed type")
        if any(card.seed_type == 48 for card in cards) and not profile.imitater_owned:
            raise ValueError("an imitater card requires profile.imitater_owned")
        card_types = [card.seed_type for card in cards]
        if len(set(card_types)) != len(card_types):
            raise ValueError("deck must not contain duplicate card types")
        if any(seed_type not in card_types for seed_type in task.forced_seeds):
            raise ValueError("deck must include every forced seed")

        upgrades = ",".join(map(str, profile.owned_upgrade_plants)) or "-"
        forced = ",".join(map(str, task.forced_seeds)) or "-"
        deck_text = ",".join(
            str(card.seed_type) if card.imitater_type is None else f"{card.seed_type}:{card.imitater_type}"
            for card in cards
        )
        response = self._command(
            f"RESET_V2 {level} {seed} {task.playthrough} {profile.seed_slot_count} "
            f"{int(profile.imitater_owned)} {int(profile.first_aid_owned)} "
            f"{int(profile.pool_cleaner_owned)} {int(profile.roof_cleaner_owned)} {profile.rake_charges} "
            f"{upgrades} {forced} {deck_text} {task.zombie_count_multiplier:g}"
        )
        if not response.get("ok") or response.get("observation") is None:
            raise ValueError(f"PvZ-Portable rejected reset: {response}")
        self._reset_done = True
        resource_sha256, properties_sha256 = self._resource_hashes()
        source_revision, source_dirty = self._source_state()
        self._adopt_tick(response["observation"])
        self.episode = {
            "format_version": REPLAY_FORMAT_VERSION,
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "resource_sha256": resource_sha256,
            "properties_partner_sha256": properties_sha256,
            "level": level,
            "seed": seed,
            "deck": [asdict(card) for card in cards],
            "task": {
                "level": level,
                "seed": seed,
                "playthrough": task.playthrough,
                "profile": asdict(profile),
                "forced_seeds": list(task.forced_seeds),
                "loadout_mode": task.loadout_mode,
                "zombie_count_multiplier": task.zombie_count_multiplier,
            },
            "initial_state": self._state_record(response["observation"]),
            "operations": [],
            "final_state": self._state_record(response["observation"]),
        }
        return response["observation"], {"events": response.get("events", {})}

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._reset_done:
            raise RuntimeError("call reset() before step()")
        if not isinstance(action, dict):
            raise TypeError("action must be a dictionary")
        kind = action.get("type")
        if kind == "plant":
            packet, col, row = self._coordinates(action, ("packet", "col", "row"))
            command = f"PLANT {packet} {col} {row}"
        elif kind == "shovel":
            col, row = self._coordinates(action, ("col", "row"))
            command = f"SHOVEL {col} {row}"
        elif kind == "wait":
            ticks = action.get("ticks")
            if type(ticks) is not int or not 0 <= ticks <= 1_000_000:
                raise ValueError("wait ticks must be an integer from 0 to 1000000")
            command = f"WAIT {ticks}"
        else:
            raise ValueError("action type must be 'plant', 'shovel' or 'wait'")

        response = self._command(command)
        observation = response.get("observation")
        if observation is None:
            raise RuntimeError(f"environment returned no observation: {response}")
        # The real advance is the tick delta reported by the simulator, not the requested
        # wait duration: the environment may end the level or clamp a wait early.
        previous_tick = self._tick
        tick = int(observation["tick"])
        ticks_advanced = tick - previous_tick if previous_tick is not None else 0
        self._tick = tick
        info = {
            "ok": response.get("ok", False),
            "events": response.get("events", {}),
            "ticks_advanced": ticks_advanced,
        }
        self._record_operation({"kind": "action", "request": dict(action), "action": dict(action),
                                "ticks_advanced": ticks_advanced}, observation, info["events"])
        return observation, 0.0, bool(observation.get("terminal")), False, info

    def observe(self) -> dict[str, Any]:
        response = self._command("OBS")
        if response.get("observation") is None:
            raise RuntimeError(f"environment returned no observation: {response}")
        self._adopt_tick(response["observation"])
        return response["observation"]

    def privileged_state(self) -> dict[str, Any]:
        response = self._command("PRIV")
        if response.get("observation") is None:
            raise RuntimeError(f"environment returned no state: {response}")
        return response["observation"]

    def loadout_context(self) -> LoadoutContext:
        return LoadoutContext.from_observation(self.observe())

    def snapshot(self) -> int:
        response = self._command("SNAPSHOT")
        if not response.get("ok") or "snapshot_id" not in response:
            raise RuntimeError(f"environment could not save a snapshot: {response}")
        snapshot_id = int(response["snapshot_id"])
        self._record_operation({"kind": "snapshot", "id": snapshot_id}, response["observation"],
                               response.get("events", {}))
        return snapshot_id

    def restore(self, snapshot_id: int) -> dict[str, Any]:
        if type(snapshot_id) is not int or snapshot_id < 1:
            raise ValueError("snapshot_id must be a positive integer")
        response = self._command(f"RESTORE {snapshot_id}")
        if not response.get("ok") or response.get("observation") is None:
            raise ValueError(f"environment could not restore snapshot {snapshot_id}: {response}")
        observation = response["observation"]
        self._adopt_tick(observation)
        self._record_operation({"kind": "restore", "id": snapshot_id}, observation, response.get("events", {}))
        return observation

    def release_snapshot(self, snapshot_id: int) -> None:
        if type(snapshot_id) is not int or snapshot_id < 1:
            raise ValueError("snapshot_id must be a positive integer")
        response = self._command(f"DROP_SNAPSHOT {snapshot_id}")
        if not response.get("ok"):
            raise ValueError(f"environment could not release snapshot {snapshot_id}: {response}")

    def branch_snapshot(self, snapshot_id: int, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expand several actions from one parent snapshot in a single command.

        ``BRANCH_SNAPSHOT_FAST`` is *defined* as, per branch, "restore the parent,
        execute the action, save a child snapshot"; it exists so the search can pay one
        command round trip instead of three.  Each returned record therefore carries the
        branch's ``observation``, its ``snapshot_id``/``state_hash`` when the branch is
        non-terminal, and its ``events``.

        Two things the caller owns.  First, the child snapshots are registered with the
        simulator and must be released with :meth:`release_snapshot`; the parent stays
        valid.  Second, the simulator is left at the *last* branch's state, so the caller
        has to restore whatever state it wants next.  Like the search teacher, this
        bypasses the episode bookkeeping, so do not mix it with :meth:`save_replay`.
        """
        if type(snapshot_id) is not int or snapshot_id < 1:
            raise ValueError("snapshot_id must be a positive integer")
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty list")
        if len(actions) > BRANCH_BATCH_LIMIT:
            raise ValueError(
                f"BRANCH_SNAPSHOT_FAST accepts at most {BRANCH_BATCH_LIMIT} specs per command, "
                f"got {len(actions)}"
            )
        tokens = " ".join(branch_action_token(action) for action in actions)
        response = self._command(f"BRANCH_SNAPSHOT_FAST {snapshot_id} {len(actions)} {tokens}")
        branches = response.get("branches")
        if not response.get("ok") or not isinstance(branches, list) or len(branches) != len(actions):
            raise ValueError(f"environment could not branch snapshot {snapshot_id}: {response}")
        return branches

    @contextmanager
    def speculative(self) -> Iterator[int]:
        if not self._reset_done or self.episode is None:
            raise RuntimeError("call reset() before speculative()")
        operation_count = len(self.episode["operations"])
        final_state = self.episode["final_state"]
        snapshot_id = self.snapshot()
        try:
            yield snapshot_id
        finally:
            self.restore(snapshot_id)
            del self.episode["operations"][operation_count:]
            self.episode["final_state"] = final_state
            self.release_snapshot(snapshot_id)

    def save_replay(self, path: str | os.PathLike[str]) -> None:
        if self.episode is None:
            raise RuntimeError("call reset() before saving a replay")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._write_manifest(output.parent)
        header = {key: value for key, value in self.episode.items() if key not in ("operations", "final_state")}
        header["record_type"] = "header"
        header["manifest"] = manifest
        footer = {"record_type": "footer", "final_state": self.episode["final_state"]}
        if output.name.endswith(".gz"):
            with gzip.open(output, "wt", encoding="utf-8", compresslevel=6) as stream:
                for item in [header, *self.episode["operations"], footer]:
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            with output.open("w", encoding="utf-8") as stream:
                for item in [header, *self.episode["operations"], footer]:
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    def replay_record(self, record: dict[str, Any], manifest_directory: Path | None = None) -> dict[str, Any]:
        version = record.get("format_version")
        if version != REPLAY_FORMAT_VERSION:
            raise ValueError(f"unsupported replay version: {version}")
        self._check_manifest(record["manifest"], manifest_directory)
        task_data = dict(record["task"])
        task_data["profile"] = PlayerProfileContext(**task_data["profile"])
        task_data["forced_seeds"] = tuple(task_data["forced_seeds"])
        observation, _ = self.reset(deck=[SeedCard(**card) for card in record["deck"]], task=TaskSpec(**task_data))
        if self._state_record(observation) != record["initial_state"]:
            raise RuntimeError("replay diverged immediately after reset")
        snapshot_ids: dict[int, int] = {}
        for index, operation in enumerate(record["operations"]):
            kind = operation["kind"]
            if kind == "action":
                action = operation["action"]
                observation, _, _, _, info = self.step(action)
                if info["events"] != operation["events"]:
                    raise RuntimeError(f"replay events diverged at operation {index}")
                if info["ticks_advanced"] != operation["ticks_advanced"]:
                    raise RuntimeError(
                        f"replay tick count diverged at operation {index}: "
                        f"{info['ticks_advanced']} != {operation['ticks_advanced']}"
                    )
            elif kind == "snapshot":
                snapshot_ids[operation["id"]] = self.snapshot()
                observation = self.observe()
            elif kind == "restore":
                observation = self.restore(snapshot_ids[operation["id"]])
            else:
                raise ValueError(f"unknown replay operation: {kind}")
            actual = self._state_record(observation)
            expected = operation["state"]
            if actual != expected:
                raise RuntimeError(f"replay diverged at operation {index}")
            if operation.get("debug_state_sha256") and self._debug_state_sha256() != operation["debug_state_sha256"]:
                raise RuntimeError(f"full game state diverged at operation {index}")
        if self._state_record(observation) != record["final_state"]:
            raise RuntimeError("replay terminal state diverged")
        return observation

    def replay_file(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        source = Path(path)
        opener = gzip.open if source.name.endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8") as stream:
            first = stream.readline()
            if not first:
                raise ValueError("empty replay file")
            record = json.loads(first)
            if record.get("format_version") != REPLAY_FORMAT_VERSION:
                raise ValueError(f"unsupported replay version: {record.get('format_version')}")
            if record.get("record_type") != "header":
                raise ValueError("unsupported replay file format")
            record["operations"] = []
            footer_found = False
            for line in stream:
                item = json.loads(line)
                if item["record_type"] == "footer" and not footer_found:
                    record["final_state"] = item["final_state"]
                    footer_found = True
                elif item["record_type"] == "operation" and not footer_found:
                    record["operations"].append(item)
                else:
                    raise ValueError(f"invalid replay record: {item['record_type']}")
            if not footer_found:
                raise ValueError("replay file has no footer")
        return self.replay_record(record, source.parent)

    @staticmethod
    def _state_record(observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "tick": observation["tick"],
            "wave": observation["wave"],
            "wave_count": observation["wave_count"],
            "sun": observation["sun"],
            "plants": len(observation["plants"]),
            "zombies": len(observation["zombies"]),
            "terminal": observation["terminal"],
            "result": observation["result"],
        }

    def _record_operation(self, operation: dict[str, Any], observation: dict[str, Any],
                          events: dict[str, Any] | None = None) -> None:
        if self.episode is not None:
            operation["record_type"] = "operation"
            operation["events"] = events if events is not None else {}
            operation["tick"] = observation["tick"]
            operation["state"] = self._state_record(observation)
            if self.debug_replay:
                operation["debug_state_sha256"] = self._debug_state_sha256()
            self.episode["operations"].append(operation)
            self.episode["final_state"] = operation["state"]

    def _debug_state_sha256(self) -> str:
        return canonical_digest(self.privileged_state())

    def _manifest_data(self) -> tuple[dict[str, Any], bytes]:
        revision, _ = self._source_state()
        tracked = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--", ".", ":(exclude)artifacts/**"],
            cwd=self._root, check=True, capture_output=True
        ).stdout
        untracked_paths = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=self._root, check=True, capture_output=True,
        ).stdout.split(b"\0")
        untracked = []
        for raw_path in untracked_paths:
            if raw_path:
                path = os.fsdecode(raw_path)
                if is_generated_artifact(path):
                    continue
                diff = subprocess.run(
                    ["git", "diff", "--no-index", "--binary", "/dev/null", path],
                    cwd=self._root, capture_output=True,
                )
                untracked.append(diff.stdout)
        patch = tracked + b"".join(untracked)
        cache = self._root / "build" / "CMakeCache.txt"
        build_config: dict[str, str] = {}
        if cache.is_file():
            for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.split(":", 1)[0] in {
                        "BUILD_STATIC", "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
                        "CMAKE_GENERATOR", "DO_FIX_BUGS", "LOW_MEMORY", "PVZ_DEBUG",
                    }:
                        build_config[key.split(":", 1)[0]] = value
        executable_sha256 = sha256_file(self.executable) if self.executable.is_file() else None
        resource_sha256, properties_sha256 = self._resource_hashes()
        data = {
            "source_revision": revision,
            "source_dirty": bool(patch),
            "working_tree_patch_sha256": sha256_bytes(patch),
            "build_config": build_config,
            "resource_sha256": resource_sha256,
            "properties_partner_sha256": properties_sha256,
            "executable_sha256": executable_sha256,
        }
        return data, patch

    def _write_manifest(self, directory: Path) -> dict[str, str]:
        path = directory / "experiment_manifest.json"
        patch_path = directory / "working_tree.patch"
        current, patch = self._manifest_data()
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if any(manifest.get(key) != current.get(key) for key in current):
                raise RuntimeError(f"experiment manifest belongs to a different build: {path}")
            if not patch_path.is_file() or sha256_bytes(patch_path.read_bytes()) != manifest.get("working_tree_patch_sha256"):
                raise RuntimeError(f"experiment working tree patch is missing or changed: {patch_path}")
        else:
            patch_path.write_bytes(patch)
            manifest = {**current, "working_tree_patch": patch_path.name}
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        digest = canonical_digest(manifest)
        return {"path": path.name, "sha256": digest}

    def _check_manifest(self, reference: dict[str, str], directory: Path | None) -> None:
        if not reference or directory is None:
            raise ValueError(f"v{REPLAY_FORMAT_VERSION} replay has no experiment manifest reference")
        path = directory / reference["path"]
        if not path.is_file():
            raise FileNotFoundError(f"replay manifest not found: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        digest = canonical_digest(manifest)
        if digest != reference["sha256"]:
            raise ValueError(f"replay manifest digest mismatch: {path}")
        patch_path = path.parent / manifest.get("working_tree_patch", "")
        if not patch_path.is_file() or sha256_bytes(patch_path.read_bytes()) != manifest.get("working_tree_patch_sha256"):
            raise ValueError(f"replay working tree patch is missing or corrupted: {patch_path}")
        current, _ = self._manifest_data()
        mismatches = [key for key, value in current.items() if manifest.get(key) != value]
        if mismatches:
            raise RuntimeError("replay manifest mismatch: " + ", ".join(mismatches) +
                               "; replay with the matching Git revision")

    @staticmethod
    def _coordinates(action: dict[str, Any], keys: tuple[str, ...]) -> tuple[int, ...]:
        values = tuple(action.get(key) for key in keys)
        if any(type(value) is not int for value in values):
            raise ValueError(f"{', '.join(keys)} must be integers")
        return values

    def close(self) -> None:
        if self._process is not None:
            try:
                self._command("QUIT")
            except (BrokenPipeError, RuntimeError):
                pass
            self._process.wait()
            self._process = None
        if self._temporary_save is not None:
            self._temporary_save.cleanup()
            self._temporary_save = None
        self._reset_done = False
        self._tick = None

    def __enter__(self) -> "PvZEnv":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
