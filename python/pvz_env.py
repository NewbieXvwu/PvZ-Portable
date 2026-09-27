"""Thin Python control wrapper for the headless PvZ-Portable environment."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


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
    playthrough: int = 1
    profile: PlayerProfileContext = field(default_factory=PlayerProfileContext)
    forced_seeds: tuple[int, ...] = ()
    loadout_mode: str = "fixed"


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
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        binary_name = "pvz-portable.exe" if os.name == "nt" else "pvz-portable"
        self.executable = Path(executable) if executable else root / "build" / binary_name
        self.resource_dir = Path(resource_dir).expanduser().resolve()
        self.headless = headless
        self._resource_sha256 = self._hash_file(self.resource_dir / "main.pak")
        self._properties_sha256 = self._hash_file(self.resource_dir / "properties" / "partner.xml")
        self.save_dir = Path(save_dir).expanduser().resolve() if save_dir else None
        self._temporary_save: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._reset_done = False
        self.episode: dict[str, Any] | None = None
        self._source_revision, self._source_dirty = self._git_metadata(root)

    @staticmethod
    def _git_metadata(root: Path) -> tuple[str | None, bool | None]:
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
            ).stdout)
            return revision, dirty
        except (OSError, subprocess.CalledProcessError):
            return None, None

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
        if not ready.get("ready") or ready.get("protocol_version") != 1:
            raise RuntimeError(f"PvZ-Portable did not enter environment mode: {ready}")

    def _read_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("environment process is not running")
        for line in process.stdout:
            if line.startswith("PVZENV "):
                return json.loads(line[len("PVZENV ") :])
        raise RuntimeError(f"PvZ-Portable exited unexpectedly with code {process.poll()}")

    def _command(self, command: str) -> dict[str, Any]:
        if self._process is None:
            self._start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("environment process is not running")
        process.stdin.write(command + "\n")
        process.stdin.flush()
        return self._read_message()

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
        if type(task.playthrough) is not int or task.playthrough not in (1, 2):
            raise ValueError("playthrough must be 1 or 2")
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
        if len(task.forced_seeds) > 3 or (task.playthrough == 1 and task.forced_seeds):
            raise ValueError("forced_seeds are available only on playthrough 2, up to three cards")
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
            f"RESET_V1 {level} {seed} {task.playthrough} {profile.seed_slot_count} "
            f"{int(profile.imitater_owned)} {int(profile.first_aid_owned)} "
            f"{int(profile.pool_cleaner_owned)} {int(profile.roof_cleaner_owned)} {profile.rake_charges} "
            f"{upgrades} {forced} {deck_text}"
        )
        if not response.get("ok") or response.get("observation") is None:
            raise ValueError(f"PvZ-Portable rejected reset: {response}")
        self._reset_done = True
        self.episode = {
            "format_version": 2,
            "source_revision": self._source_revision,
            "source_dirty": self._source_dirty,
            "resource_sha256": self._resource_sha256,
            "properties_partner_sha256": self._properties_sha256,
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
            },
            "initial_state": self._state_record(response["observation"]),
            "actions": [],
            "operations": [],
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
        elif kind == "wait_decision":
            max_ticks = action.get("max_ticks", 1800)
            if type(max_ticks) is not int or not 1 <= max_ticks <= 1_000_000:
                raise ValueError("max_ticks must be an integer from 1 to 1000000")
            command = f"WAIT_DECISION {max_ticks}"
        else:
            raise ValueError("action type must be 'plant', 'shovel', 'wait' or 'wait_decision'")

        response = self._command(command)
        observation = response.get("observation")
        if observation is None:
            raise RuntimeError(f"environment returned no observation: {response}")
        info = {"ok": response.get("ok", False), "events": response.get("events", {})}
        if "ticks_advanced" in response:
            info["ticks_advanced"] = response["ticks_advanced"]
        self._record_operation({"kind": "step", "action": dict(action)}, observation)
        if self.episode is not None:
            self.episode["actions"].append(dict(action))
        return observation, 0.0, bool(observation.get("terminal")), False, info

    def observe(self) -> dict[str, Any]:
        response = self._command("OBS")
        if response.get("observation") is None:
            raise RuntimeError(f"environment returned no observation: {response}")
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
        self._record_operation({"kind": "snapshot", "id": snapshot_id}, response["observation"])
        return snapshot_id

    def restore(self, snapshot_id: int) -> dict[str, Any]:
        if type(snapshot_id) is not int or snapshot_id < 1:
            raise ValueError("snapshot_id must be a positive integer")
        response = self._command(f"RESTORE {snapshot_id}")
        if not response.get("ok") or response.get("observation") is None:
            raise ValueError(f"environment could not restore snapshot {snapshot_id}: {response}")
        observation = response["observation"]
        self._record_operation({"kind": "restore", "id": snapshot_id}, observation)
        return observation

    def save_replay(self, path: str | os.PathLike[str]) -> None:
        if self.episode is None:
            raise RuntimeError("call reset() before saving a replay")
        Path(path).write_text(json.dumps(self.episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def replay_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("format_version", 1) >= 2:
            task_data = record["task"]
            task_data["profile"] = PlayerProfileContext(**task_data["profile"])
            task_data["forced_seeds"] = tuple(task_data["forced_seeds"])
            observation, _ = self.reset(deck=[SeedCard(**card) for card in record["deck"]], task=TaskSpec(**task_data))
        else:
            observation, _ = self.reset(
                record["level"], record["seed"], record["deck"], task=TaskSpec(
                    level=record["level"], seed=record["seed"], playthrough=2
                )
            )
        expected = record.get("initial_state")
        if expected is not None and self._state_record(observation) != expected:
            raise RuntimeError("replay diverged immediately after reset")
        snapshot_ids: dict[int, int] = {}
        for index, operation in enumerate(record.get("operations", [])):
            kind = operation["kind"]
            if kind == "step":
                observation, _, _, _, _ = self.step(operation["action"])
            elif kind == "snapshot":
                snapshot_ids[operation["id"]] = self.snapshot()
                observation = self.observe()
            elif kind == "restore":
                observation = self.restore(snapshot_ids[operation["id"]])
            else:
                raise ValueError(f"unknown replay operation: {kind}")
            if self._state_record(observation) != operation["state"]:
                raise RuntimeError(f"replay diverged at operation {index}")
        return observation

    def replay_file(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.replay_record(record)

    @staticmethod
    def _state_record(observation: dict[str, Any]) -> dict[str, Any]:
        canonical = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {
            "tick": observation["tick"],
            "wave": observation["wave"],
            "sun": observation["sun"],
            "plants": len(observation["plants"]),
            "zombies": len(observation["zombies"]),
            "terminal": observation["terminal"],
            "observation_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }

    def _record_operation(self, operation: dict[str, Any], observation: dict[str, Any]) -> None:
        if self.episode is not None:
            operation["state"] = self._state_record(observation)
            self.episode["operations"].append(operation)

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

    def __enter__(self) -> "PvZEnv":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
