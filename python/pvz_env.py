"""Thin Python control wrapper for the headless PvZ-Portable environment."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence

from pvz_common import ENV_PROTOCOL_VERSION, OOM_SCORE_WORKER

__all__ = [
    "BRANCH_BATCH_LIMIT",
    "ENV_PROTOCOL_VERSION",
    "LoadoutContext",
    "PlayerProfileContext",
    "PvZEnv",
    "SeedCard",
    "SimulatorExited",
    "TaskSpec",
    "branch_action_token",
]

# ``BRANCH_SNAPSHOT_FAST`` refuses more than this many specs in one command
# (``src/main.cpp`` bounds ``branchCount`` at 128), so callers that want to expand a
# wider action set have to chunk.  The search itself never gets near this -- the root
# request saturates at 96 and internal nodes use ``candidate_limit`` -- but the audit
# tools deliberately enumerate every legal placement.
BRANCH_BATCH_LIMIT = 128

# ``events`` crosses the process boundary as JSON, so every single step builds a dict
# whose key strings are brand new objects.  Measured on a real episode, those eight
# keys cost 21.0 KiB per episode -- 61% of what the ``events`` field occupies and 5.1%
# of the whole rollout payload, 41 MiB over a 2000-episode batch.  Replacing them with
# one shared object per name leaves the dict equal to the one a reader would see.
EVENT_KEYS: dict[str, str] = {
    name: sys.intern(name) for name in (
        "zombies_killed",
        "plants_eaten",
        "plants_eaten_free",
        "sun_produced",
        "sun_spent",
        "mower_triggered",
        "waves_started",
        "level_won",
        "level_lost",
    )
}


def _canonical_events(events: Any) -> Any:
    """Rebuild *events* with shared key objects; unknown keys pass through unchanged."""
    if not isinstance(events, dict) or not events:
        return events
    return {EVENT_KEYS.get(key, key): value for key, value in events.items()}


def branch_action_token(action: dict[str, Any]) -> str:
    """Encode one action as the compact spec ``BRANCH_SNAPSHOT_FAST`` expects.

    The protocol passes branch actions as a single space-delimited token per action,
    with ``:`` as the field separator, because the command line itself is
    whitespace-tokenised.  This is the only place that encoding is defined; the search
    teacher and the equivalence harness both go through it.
    """
    if not isinstance(action, dict):
        raise TypeError("action must be a dictionary")
    if 'until' in action:
        raise ValueError('event waits require the native macro path; branch batching cannot drop until')
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
    wave_cap: int | None = None
    preplanted: tuple[tuple[int, int, int], ...] = ()
    # "original": conveyor levels keep the engine's own seed bank (10 empty packets, cards dealt
    # by Challenge::UpdateConveyorBelt) instead of writing the task deck into it.
    conveyor_start: str | None = None
    # Task family: "human" (pre-task66 cell rule), "human_v2" (task66 calibrated opacity rule),
    # "xray" (nothing hidden); None preserves legacy tasks and emits no trailing token.
    fog_observation: str | None = None


def fog_diagnostics(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """task66: per-zombie fog numbers from a privileged frame, keyed by zombie id.

    They live in their own fog_zombies array rather than as extra keys on the zombie objects,
    so a public frame is still exactly "the privileged zombies minus the hidden ones".  Absent
    on non-fog boards and on public frames.
    """
    return {entry['id']: entry for entry in state.get('fog_zombies') or []}




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




def _oom_score_engine() -> None:
    """preexec_fn for the simulator: under memory pressure the engine dies before its worker's driver.

    Runs between fork and exec, so it avoids Python's text-I/O machinery and writes the
    number with plain os.open/os.write (the parent may hold threads and their locks).
    """
    try:
        descriptor = os.open("/proc/self/oom_score_adj", os.O_WRONLY)
    except OSError:
        return
    try:
        os.write(descriptor, f"{OOM_SCORE_WORKER}\n".encode())
    except OSError:
        pass
    finally:
        os.close(descriptor)


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
        self.save_dir = Path(save_dir).expanduser().resolve() if save_dir else None
        self._temporary_save: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._reset_done = False
        self._last_response: dict[str, Any] = {}
        self._sparse_tracks_version = 0
        self._branch_observe_supported: bool | None = None
        self.episode: dict[str, Any] | None = None
        # Tick of the most recent observation, used to derive real tick deltas.
        self._tick: int | None = None
        self._sun_income_window_ticks = 6000
        self._sun_production_history: deque[tuple[int, int]] = deque()
        self._sun_history_start_tick: int | None = None
        self._sun_history_snapshots: dict[int, tuple[list[tuple[int, int]], int | None]] = {}
        self._snapshot_ticks: dict[int, int] = {}

    def _annotate_observation(self, observation: dict[str, Any],
                              events: dict[str, Any] | None = None,
                              reset_history: bool = False) -> dict[str, Any]:
        if "wave_timer" not in observation:
            raise RuntimeError("environment observation is missing the public wave_timer field")
        tick = int(observation["tick"])
        if reset_history:
            self._sun_production_history.clear()
            self._sun_history_start_tick = tick
            self._sun_history_snapshots.clear()
            self._snapshot_ticks.clear()
        if self._sun_history_start_tick is None:
            self._sun_history_start_tick = tick
        # A reset observation begins a new episode; events from the old board
        # cannot be part of its income history (also guards older binaries).
        if events is not None and not reset_history:
            produced = int(events.get("sun_produced", 0))
            if produced > 0:
                self._sun_production_history.append((tick, produced))
        cutoff = tick - self._sun_income_window_ticks
        while self._sun_production_history and self._sun_production_history[0][0] < cutoff:
            self._sun_production_history.popleft()
        elapsed = max(0, min(self._sun_income_window_ticks, tick - self._sun_history_start_tick))
        produced = sum(amount for sample_tick, amount in self._sun_production_history
                       if sample_tick >= cutoff)
        observation["sun_income_rate"] = produced * 1000.0 / elapsed if elapsed else 0.0
        return observation

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
            # one simulator per worker process: a threaded BLAS would oversubscribe the cores
            # PVZ_NO_POPUP: report fatal errors on stderr instead of a modal dialog that would hang the run
            env={**os.environ, "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"), "PVZ_NO_POPUP": "1"},
            # task34: never let a full WSL take the simulator's WSL access path down with it
            preexec_fn=_oom_score_engine if os.name == "posix" else None,
        )
        ready = self._read_message()
        self._rollout_version = ready.get("rollout_version", 0)
        self._sparse_tracks_version = ready.get("sparse_tracks_version", 0)
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
        self._sun_production_history.clear()
        self._sun_history_start_tick = None
        self._sun_history_snapshots.clear()

    def _read_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("environment process is not running")
        for line in process.stdout:
            if line.startswith("PVZENV "):
                response = json.loads(line[len("PVZENV ") :])
                # Canonicalise here so every consumer -- ``reset``, ``step`` and the
                # ``info`` dict that reaches the rollout buffer -- shares the same keys.
                if "events" in response:
                    response["events"] = _canonical_events(response["events"])
                return response
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
        if task.loadout_mode not in ("fixed", "user_pick"):
            raise ValueError("loadout_mode must be fixed or user_pick")
        user_pick = task.loadout_mode == 'user_pick'
        if user_pick and (self.headless or deck or task.conveyor_start or profile.imitater_owned or task.forced_seeds):
            raise ValueError('user_pick requires a visible window, empty deck, and no imitater/forced/conveyor cards')
        if (type(task.zombie_count_multiplier) not in (int, float) or
                not 0.1 <= task.zombie_count_multiplier <= 10.0):
            raise ValueError("zombie_count_multiplier must be from 0.1 to 10")
        if task.wave_cap is not None and (type(task.wave_cap) is not int or not 1 <= task.wave_cap <= 50):
            raise ValueError("wave_cap must be None or an integer from 1 to 50")
        if not isinstance(task.preplanted, tuple) or any(type(plant) is not tuple or len(plant) != 3
                                                        for plant in task.preplanted):
            raise ValueError("preplanted must contain (seed_type, row, col) tuples")
        if any(type(seed_type) is not int or not 0 <= seed_type <= 48 or
               type(row) is not int or not 0 <= row <= 5 or
               type(col) is not int or not 0 <= col <= 8
               for seed_type, row, col in task.preplanted):
            raise ValueError("preplanted seed_type, row and col must be in range")
        if task.conveyor_start not in (None, "original"):
            raise ValueError("conveyor_start must be None or 'original'")
        if task.fog_observation not in (None, "human", "human_v2", "xray", "storm"):
            raise ValueError("fog_observation must be None, 'human', 'human_v2', 'xray' or 'storm'")
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
        if (not cards and task.conveyor_start != "original" and not user_pick) or len(cards) > profile.seed_slot_count:
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
        preplanted = ",".join(f"{seed_type}:{row}:{col}" for seed_type, row, col in task.preplanted) or "-"
        deck_text = ",".join(
            str(card.seed_type) if card.imitater_type is None else f"{card.seed_type}:{card.imitater_type}"
            for card in cards
        ) or "-"   # empty deck (original conveyor start): "-" keeps the positional fields aligned
        command = (
            f"RESET_V2 {level} {seed} {task.playthrough} {profile.seed_slot_count} "
            f"{int(profile.imitater_owned)} {int(profile.first_aid_owned)} "
            f"{int(profile.pool_cleaner_owned)} {int(profile.roof_cleaner_owned)} {profile.rake_charges} "
            f"{upgrades} {forced} {deck_text} {task.zombie_count_multiplier:g} "
            f"{task.wave_cap or 0} {preplanted}"
        )
        # Only the original-conveyor tasks append this token: every earlier task's reset
        # message (and therefore its replay) stays byte-identical.
        if task.conveyor_start == "original":
            command += " conveyor_original"
        if task.fog_observation is not None:
            command += f" fog_{task.fog_observation}"
        if user_pick:
            command += ' user_pick'
        response = self._command(command)
        if not response.get("ok") or response.get("observation") is None:
            raise ValueError(f"PvZ-Portable rejected reset: {response}")
        if user_pick:
            cards = [SeedCard(kind) for kind in response['deck']]
        self._reset_done = True
        observation = self._annotate_observation(
            response["observation"], response.get("events", {}), reset_history=True)
        self._adopt_tick(observation)
        self.episode = {
            "level": level,
            "seed": seed,
            "deck": [asdict(card) for card in cards],
            "task": {
                "level": level,
                "seed": seed,
                "playthrough": task.playthrough,
                "profile": asdict(profile),
                "forced_seeds": list(task.forced_seeds),
                "loadout_mode": "fixed",
                "zombie_count_multiplier": task.zombie_count_multiplier,
                "wave_cap": task.wave_cap,
                "preplanted": [list(plant) for plant in task.preplanted],
                **({"conveyor_start": task.conveyor_start} if task.conveyor_start else {}),
                **({"fog_observation": task.fog_observation} if task.fog_observation else {}),
            },
            "initial_state": self._state_record(observation),
            "operations": [],
            "final_state": self._state_record(observation),
        }
        return observation, {"events": response.get("events", {}),
                             **({'deck': response['deck']} if user_pick else {})}

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._reset_done:
            raise RuntimeError("call reset() before step()")
        if not isinstance(action, dict):
            raise TypeError("action must be a dictionary")
        if 'until' in action:
            raise ValueError('event waits require EventWaitEnv and the versioned native capability')
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
        elif kind == "collect":
            x, y = self._coordinates(action, ("x", "y"))
            command = f"COLLECT {x} {y}"
        else:
            raise ValueError("action type must be 'plant', 'shovel', 'collect' or 'wait'")

        response = self._command(command)
        observation = response.get("observation")
        if observation is None:
            raise RuntimeError(f"environment returned no observation: {response}")
        events = response.get("events", {})
        self._annotate_observation(observation, events)
        # The real advance is the tick delta reported by the simulator, not the requested
        # wait duration: the environment may end the level or clamp a wait early.
        previous_tick = self._tick
        tick = int(observation["tick"])
        ticks_advanced = tick - previous_tick if previous_tick is not None else 0
        self._tick = tick
        info = {
            "ok": response.get("ok", False),
            "events": events,
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
        return self._annotate_observation(response["observation"], response.get("events", {}))

    def begin_demo(self) -> None:
        if not self._command("DEMO_BEGIN").get("ok"):
            raise RuntimeError("simulator does not support human demonstrations")

    def state_hash(self) -> str:
        response = self._command("STATE_HASH")
        if not response.get("ok") or not response.get("state_hash"):
            raise RuntimeError("simulator could not hash the current state")
        return response["state_hash"]

    def human_input(self) -> dict[str, Any]:
        response = self._command("HUMAN_POLL")
        if not response.get("ok"):
            raise RuntimeError("human input requires a visible demonstration session")
        return response

    def demo_key(self, key: str) -> None:
        """task80: push Esc/Space/window-close through the recorder's real SDL handler.

        Test seam for scripted regressions; rejected outside a visible demonstration session.
        """
        if key not in ("escape", "space", "close"):
            raise ValueError("key must be 'escape', 'space' or 'close'")
        response = self._command(f"DEMO_INPUT {key}")
        if not response.get("ok"):
            raise RuntimeError("key injection requires a visible demonstration session")

    def demo_click_packet(self, index: int) -> None:
        if type(index) is not int:
            raise TypeError("packet index must be an integer")
        response = self._command(f"DEMO_CLICK_PACKET {index}")
        if not response.get("ok"):
            raise RuntimeError("packet injection requires a visible demonstration session")

    def demo_click_cell(self, col: int, row: int) -> None:
        if type(col) is not int or type(row) is not int:
            raise TypeError("cell coordinates must be integers")
        response = self._command(f"DEMO_CLICK_CELL {col} {row}")
        if not response.get("ok"):
            raise RuntimeError("cell injection requires a visible demonstration session")

    def screenshot(self, path: str | Path) -> None:
        """Save the presented frame as a PNG (visible runs only, task66 calibration)."""
        response = self._command(f'SCREENSHOT "{Path(path).resolve()}"')
        if not response.get("ok"):
            raise RuntimeError(f"environment could not take a screenshot: {response}")

    def privileged_state(self) -> dict[str, Any]:
        response = self._command("PRIV")
        if response.get("observation") is None:
            raise RuntimeError(f"environment returned no state: {response}")
        return response["observation"]

    def critic_inputs(self, wave_index: int) -> dict[str, Any]:
        response = self._command(f"CRITIC_INPUTS {int(wave_index)}")
        if (not response.get("ok") or type(response.get("wave_timer")) is not int
                or not isinstance(response.get("wave_zombies"), list)):
            raise RuntimeError(f"environment returned invalid critic inputs: {response}")
        return response

    def snapshot_fast(self) -> int:
        """Search-only snapshot: preserve wrapper income/time without exporting the root again."""
        response = self._command("SNAPSHOT_FAST")
        if not response.get("ok") or "snapshot_id" not in response:
            raise RuntimeError(f"environment could not save a snapshot: {response}")
        snapshot_id = int(response["snapshot_id"])
        self._sun_history_snapshots[snapshot_id] = (list(self._sun_production_history), self._sun_history_start_tick)
        self._snapshot_ticks[snapshot_id] = self._tick
        return snapshot_id

    def restore_fast(self, snapshot_id: int) -> None:
        """Restore a search root and its wrapper state; no redundant observation or replay operation."""
        if snapshot_id not in self._snapshot_ticks:
            raise ValueError("restore_fast requires a live snapshot_fast root")
        response = self._command(f"RESTORE_FAST {snapshot_id}")
        if not response.get("ok"):
            raise ValueError(f"environment could not restore snapshot {snapshot_id}: {response}")
        history, start = self._sun_history_snapshots[snapshot_id]
        self._sun_production_history = deque(history)
        self._sun_history_start_tick = start
        self._tick = self._snapshot_ticks[snapshot_id]

    def snapshot(self) -> int:
        response = self._command("SNAPSHOT")
        if not response.get("ok") or "snapshot_id" not in response:
            raise RuntimeError(f"environment could not save a snapshot: {response}")
        snapshot_id = int(response["snapshot_id"])
        self._sun_history_snapshots[snapshot_id] = (
            list(self._sun_production_history), self._sun_history_start_tick)
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
        history = self._sun_history_snapshots.get(snapshot_id)
        if history is not None:
            self._sun_production_history = deque(history[0])
            self._sun_history_start_tick = history[1]
        self._adopt_tick(observation)
        self._annotate_observation(observation)
        self._record_operation({"kind": "restore", "id": snapshot_id}, observation, response.get("events", {}))
        return observation

    def release_snapshot(self, snapshot_id: int) -> None:
        if type(snapshot_id) is not int or snapshot_id < 1:
            raise ValueError("snapshot_id must be a positive integer")
        response = self._command(f"DROP_SNAPSHOT_FAST {snapshot_id}")
        if not response.get("ok"):
            raise ValueError(f"environment could not release snapshot {snapshot_id}: {response}")
        self._sun_history_snapshots.pop(snapshot_id, None)
        self._snapshot_ticks.pop(snapshot_id, None)

    def reseed(self, seed: int) -> None:
        """Draw a fresh, equally plausible future from the current state.

        A restored snapshot replays the true future exactly (same RNG, same wave table), so a
        search that rolls out from it would be clairvoyant. ``RESEED`` re-picks the zombies of
        every wave that has not started yet and reseeds in-play randomness; waves already sent
        and the wave count are unchanged. Call it right after :meth:`restore` in each rollout.
        """
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        response = self._command(f"RESEED {seed}")
        if not response.get("ok"):
            raise ValueError(f"environment could not reseed: {response}")

    def require_sparse_tracks(self) -> None:
        """Fail before collection starts when the requested CPU feature is absent."""
        if self._process is None:
            self._start()
        if self._sparse_tracks_version != 1:
            raise RuntimeError("CPU sparse tracks require a rebuilt simulator advertising sparse_tracks_version=1")

    def set_sim_skip(self, mask: int) -> None:
        """Approximate simulation (``SIM_SKIP``, bits in ``EnvSimSkip`` of src/PvzpLib/PvzpCommon.h).

        0 is the exact game. The setting is process-wide and survives RESET/RESTORE, so search
        code sets it around its rollouts and puts it back to 0 afterwards. Presentation skips are
        approximations to measure (scripts/sim_skip_agreement.py). Bit256 only changes traversal
        of otherwise inactive animation tracks and preserves the chosen presentation mode.
        """
        if type(mask) is not int or mask < 0 or mask & ~317:
            raise ValueError("SIM_SKIP accepts presentation bits 1, 4, 8, 16, 32 and sparse-track bit256")
        if mask & 256:
            self.require_sparse_tracks()
        if not self._command(f"SIM_SKIP {mask}").get("ok"):
            raise RuntimeError("simulator rejected SIM_SKIP (binary older than this wrapper?)")

    def set_token_observation(self, enabled: bool) -> None:
        """Opt in to task157 public obs-v3 arrays."""
        if not self._command(f"OBS_V3 {int(enabled)}").get("ok"):
            raise RuntimeError("simulator rejected OBS_V3 (requires task157 engine)")

    def set_lean_observation(self, lean: bool) -> None:
        """``OBS_LEAN``: compact observations (``"lean": 1``) until switched off again.

        Only for search rollouts: the lean form carries exactly what ``render_episode._frame``
        and the legal plant list need (positional arrays, no cells/projectiles/coins/shovels),
        so ``_frame`` of either form is identical but most other consumers would break.
        """
        if not self._command(f"OBS_LEAN {1 if lean else 0}").get("ok"):
            raise RuntimeError("simulator rejected OBS_LEAN (binary older than this wrapper?)")

    def load_value_net(self, path: str | os.PathLike[str], *, static: bool = False) -> None:
        filename = str(Path(path).resolve())
        if "\n" in filename or "\r" in filename:
            raise ValueError("value network path must not contain newlines")
        if not self._command(f"LOAD_VALUE_NET {int(static)} {json.dumps(filename)}").get("ok"):
            raise RuntimeError(f"simulator rejected value network: {filename}")

    def leaf_values(self, *, income: float | None = None, static: bool = False,
                    trace_path: str | os.PathLike[str] | None = None) -> list[float]:
        """Score the current Board without exporting an observation; optional feature trace for tests."""
        if income is None:
            income = self._annotate_observation({"tick": self._tick, "wave_timer": 0})["sun_income_rate"]
        if not math.isfinite(income):
            raise ValueError("income must be finite")
        suffix = f" {json.dumps(str(Path(trace_path).resolve()))}" if trace_path is not None else ""
        response = self._command(f"LEAF_VALUES {int(static)} BOARD {income:.17g}" + suffix)
        if not response.get("ok") or len(response.get("values", [])) != 1:
            raise RuntimeError(f"simulator rejected LEAF_VALUES: {response}")
        return response["values"]

    def leaf_feature_values(self, path: str | os.PathLike[str], count: int, *, static: bool = False) -> list[float]:
        """Score a float32 feature file produced by an external simulator or an oracle test."""
        if type(count) is not int or not 1 <= count <= 4096:
            raise ValueError("feature count must be from 1 to 4096")
        response = self._command(f"LEAF_VALUES {int(static)} FEATURES {count} {json.dumps(str(Path(path).resolve()))}")
        if not response.get("ok") or len(response.get("values", [])) != count:
            raise RuntimeError(f"simulator rejected LEAF_VALUES: {response}")
        return response["values"]

    def static_scores(self, snapshot_id: int, actions: list[dict[str, Any]], income: float) -> list[float | None]:
        """Restore, act, score each branch in the engine; rejected actions return None in place."""
        if type(snapshot_id) is not int or snapshot_id not in self._sun_history_snapshots:
            raise ValueError("snapshot_id must refer to a live snapshot")
        if not math.isfinite(income):
            raise ValueError("income must be finite")
        if not actions:
            return []
        if len(actions) > 1024:
            raise ValueError("STATIC_SCORES accepts at most 1024 actions")
        tokens = " ".join(branch_action_token(action) for action in actions)
        response = self._command(f"STATIC_SCORES {snapshot_id} {income:.17g} {len(actions)} {tokens}")
        if not response.get("ok") or len(response.get("values", [])) != len(actions):
            raise RuntimeError(f"simulator rejected STATIC_SCORES: {response}")
        return response["values"]

    def load_rollout_net(self, path: str | os.PathLike[str]) -> None:
        """Load weights exported by pvz_rollout_policy.export_student in this simulator process."""
        filename = str(Path(path).expanduser().resolve())
        if any(char in filename for char in ('\n', '\r')):
            raise ValueError("rollout network path must not contain newlines")
        with open(filename, "rb") as stream:
            version = stream.read(8)
            # LEGACY(pvzrn-versions): 002-005 are older exports; 006 is the current one.
            if version not in (b"PVZRN002", b"PVZRN003", b"PVZRN004", b"PVZRN005", b"PVZRN006"):
                raise ValueError("load_rollout_net requires an exported C++ network")
            observation = int.from_bytes(stream.read(16)[12:16], "little") if version in (b"PVZRN004", b"PVZRN005", b"PVZRN006") else 1
        if self._process is None:
            self._start()
        if version in (b"PVZRN004", b"PVZRN005", b"PVZRN006") and self._rollout_version < 2:
            raise RuntimeError("v2lite requires a rebuilt simulator with rollout_version >= 2")
        if observation == 3 and self._rollout_version < 3:
            raise RuntimeError("v2 requires a rebuilt simulator with rollout_version >= 3")
        if observation == 4 and self._rollout_version < 4:
            raise RuntimeError("v2fog requires a rebuilt simulator with rollout_version >= 4")
        if observation == 7 and self._rollout_version < 5:
            raise RuntimeError("v2bungee requires a rebuilt simulator with rollout_version >= 5")
        if observation == 8 and self._rollout_version < 6:
            raise RuntimeError("v2sleep requires a rebuilt simulator with rollout_version >= 6")
        quoted = '"' + filename.replace('\\', '\\\\').replace('"', '\\"') + '"'
        if not self._command(f"LOAD_ROLLOUT_NET {quoted}").get("ok"):
            raise RuntimeError(f"simulator rejected rollout network: {filename}")

    def deck_options(self) -> dict:
        """Read engine unlocks and raw per-card recommendation flags without advancing the game."""
        result = self._command('DECK_OPTIONS')
        if not result.get('ok'):
            raise RuntimeError('DECK_OPTIONS requires a reset board and an identity-capable simulator')
        return result

    def deck_warnings(self, deck) -> dict:
        """Whole-deck warnings from SeedChooserScreen::OnStartButton for an arbitrary deck.

        The engine runs its own CheckDeckWarnings on the reset board, so level/mode facts
        (pool, roof, fog, zombie roster, unlocks) come from the game, not from Python.
        """
        cards = [int(card) for card in deck]
        if not 0 <= len(cards) <= 10:
            raise ValueError('deck must have 0..10 cards')
        if any(card < 0 or card >= 64 for card in cards):
            raise ValueError('deck cards must be seed types in 0..63')
        result = self._command(f'DECK_WARNINGS {len(cards)}' + ''.join(f' {card}' for card in cards))
        if not result.get('ok'):
            raise RuntimeError('DECK_WARNINGS requires a reset board and a rebuilt simulator')
        return result

    def deck_pool(self) -> dict:
        """CrazyDavePickSeeds' candidate cards for the current level (engine query, no pick)."""
        result = self._command('DECK_POOL')
        if not result.get('ok'):
            raise RuntimeError('DECK_POOL requires a reset board and a rebuilt simulator')
        return result

    def _student_options(self, student_history, trace_path) -> str:
        suffix = ""
        if student_history is not None:
            plants = student_history.plants
            entries = " ".join(f"{int(row)} {int(col)} {int(kind)} {int(born)}"
                               for (row, col, kind), born in plants.items())
            wave = student_history.wave if student_history.wave is not None else -1
            suffix += f" V2H {int(wave)} {int(student_history.wave_tick)} {len(plants)} {entries}".rstrip()
        if trace_path is not None:
            filename = str(Path(trace_path).expanduser().resolve())
            if any(char in filename for char in ('\n', '\r')):
                raise ValueError("trace path must not contain newlines")
            suffix += ' TRACE "' + filename.replace('\\', '\\\\').replace('"', '\\"') + '"'
        if suffix and self._rollout_version < 2:
            raise RuntimeError("student history and tracing require rollout_version >= 2")
        return suffix

    def rollout_predict(self, previous_tick: int | None = None, previous_plant: bool = False, *,
                        student_history=None, trace_path=None) -> dict:
        """Diagnostic logits and greedy action on the current public board, without advancing it."""
        if not self._reset_done:
            raise RuntimeError("call reset() before rollout_predict()")
        income = self._annotate_observation({"tick": self._tick, "wave_timer": 0})["sun_income_rate"]
        response = self._command(f"ROLLOUT_PREDICT {previous_tick if previous_tick is not None else -1} "
                                 f"{int(previous_plant)} {income:.17g}"
                                 + self._student_options(student_history, trace_path))
        if not response.get("ok"):
            raise RuntimeError("simulator rejected ROLLOUT_PREDICT")
        if trace_path is not None:
            # Older engines reply before the local trace stream is closed. A read-only
            # round trip waits for that command scope to end before callers read the file.
            if not self._command("OBS").get("ok"):
                raise RuntimeError("simulator rejected rollout trace synchronization")
        return response

    def rollout(self, first_action: dict[str, Any] | None, until_tick: int, *,
                sim_skip: int = 0, max_actions: int = 1000000,
                snapshot_id: int | None = None, seed: int | None = None,
                student_history=None, trace_path=None, score_value: bool = False) -> dict[str, Any]:
        """Run the greedy student entirely in C++ and return its leaf (0 = whole game).

        None selects the first action with the student, with fresh previous-action context.
        Like branch_snapshot, this bypasses replay bookkeeping. Income samples retain every
        decision boundary so subsequent Python features and snapshot restores stay consistent.
        """
        if not self._reset_done:
            raise RuntimeError("call reset() before rollout()")
        if type(until_tick) is not int or not 0 <= until_tick <= 2 ** 31 - 1:
            raise ValueError("until_tick must be a non-negative signed 32-bit integer")
        if type(max_actions) is not int or not 1 <= max_actions <= 1000000:
            raise ValueError("max_actions must be from 1 to 1000000")
        if type(sim_skip) is not int or sim_skip < 0 or sim_skip & ~509:
            raise ValueError("rollout sim skip mask must combine the bits 1, 4, 8, 16, 32, 64, 128 and 256")
        if sim_skip & 256:
            self.require_sparse_tracks()
        event_wait = False
        if first_action is None:
            token = "A"
        else:
            action = dict(first_action)
            condition = action.pop("until", None)
            if condition is not None and (action.get("type") != "wait" or condition != "timeout"):
                raise ValueError("rollout first action only supports fixed or timeout waits")
            if action.get("type") not in ("plant", "shovel", "wait"):
                raise ValueError("rollout first action must be plant, shovel or wait")
            if action["type"] == "wait" and (type(action.get("ticks")) is not int or not 0 <= action["ticks"] <= 1000000):
                raise ValueError("wait ticks must be from 0 to 1000000")
            if action["type"] == "plant":
                self._coordinates(action, ("packet", "row", "col"))
            if action["type"] == "shovel":
                self._coordinates(action, ("col", "row"))
            event_wait = condition is not None
            token = branch_action_token(action)
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 32):
            raise ValueError("seed must be an unsigned 32-bit integer")
        if snapshot_id is not None and (type(snapshot_id) is not int or snapshot_id not in self._sun_history_snapshots):
            raise ValueError("snapshot_id must refer to a live snapshot of this environment")
        history, history_start = (list(self._sun_production_history), self._sun_history_start_tick)
        if snapshot_id is not None:
            saved = self._sun_history_snapshots[snapshot_id]
            history, history_start = saved[0], saved[1]
        samples = " ".join(f"{tick} {amount}" for tick, amount in history)
        command = "ROLLOUT_VALUE" if score_value else "ROLLOUT"
        response = self._command(f"{command} {snapshot_id or 0} {seed if seed is not None else -1} "
                                 f"{until_tick} {sim_skip} {max_actions} {int(event_wait)} {token} "
                                 f"{history_start} {len(history)} {samples}".rstrip()
                                 + self._student_options(student_history, trace_path))
        observation = response.get("observation")
        if not response.get("ok") or not isinstance(observation, dict):
            raise RuntimeError(f"simulator rejected ROLLOUT (binary older than this wrapper?): {response}")
        if response.get("estimated_result", -1) >= 0:
            observation["rollout_result"] = response["estimated_result"]
        if score_value:
            # Search immediately restores a root; no exported leaf or Python income history is needed.
            return observation
        self._sun_production_history = deque(history)
        self._sun_history_start_tick = history_start
        for tick, produced in response["substeps"]:
            if produced > 0:
                self._sun_production_history.append((tick, produced))
        self._annotate_observation(observation)
        self._adopt_tick(observation)
        return observation

    def branch_snapshot(self, snapshot_id: int, actions: list[dict[str, Any]], *,
                        retain_snapshots: bool = True) -> list[dict[str, Any]]:
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
        bypasses the episode bookkeeping.

        With retain_snapshots=False, return compact observations without saving children or hashing them.
        Older binaries use the original command and release its child snapshots immediately.
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
        if not retain_snapshots and self._branch_observe_supported is False:
            branches = self.branch_snapshot(snapshot_id, actions)
            for branch in branches:
                child = branch.pop("snapshot_id", None)
                branch.pop("state_hash", None)
                if child is not None:
                    self.release_snapshot(child)
            return branches
        command = "BRANCH_SNAPSHOT_FAST" if retain_snapshots else "BRANCH_OBSERVE"
        response = self._command(f"{command} {snapshot_id} {len(actions)} {tokens}")
        if not retain_snapshots and self._branch_observe_supported is None:
            self._branch_observe_supported = "branches" in response
            if not self._branch_observe_supported:
                return self.branch_snapshot(snapshot_id, actions, retain_snapshots=False)
        branches = response.get("branches")
        if not response.get("ok") or not isinstance(branches, list) or len(branches) != len(actions):
            raise ValueError(f"environment could not branch snapshot {snapshot_id}: {response}")
        base_history = list(self._sun_production_history)
        base_start = self._sun_history_start_tick
        try:
            for branch in branches:
                self._sun_production_history = deque(base_history)
                self._sun_history_start_tick = base_start
                if branch.get("observation") is not None:
                    self._annotate_observation(branch["observation"], branch.get("events", {}))
        finally:
            self._sun_production_history = deque(base_history)
            self._sun_history_start_tick = base_start
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
            self.episode["operations"].append(operation)
            self.episode["final_state"] = operation["state"]

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
        self._sun_production_history.clear()
        self._sun_history_start_tick = None
        self._sun_history_snapshots.clear()

    def __enter__(self) -> "PvZEnv":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
