"""Compare visible and headless environment ticks from identical resets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pvz_env import PlayerProfileContext, PvZEnv, TaskSpec


DECK = (0, 1, 2, 3, 4, 5)


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_pair(headless: PvZEnv, visible: PvZEnv, action: dict[str, Any]) -> dict[str, Any]:
    results = [env.step(action) for env in (headless, visible)]
    left, right = results
    if not left[4]["ok"] or not right[4]["ok"]:
        raise RuntimeError(f"action rejected: {action}")
    if left[0] != right[0] or left[4]["events"] != right[4]["events"]:
        raise RuntimeError(f"observation or events diverged after {action}")
    left_debug = headless.episode["operations"][-1].get("debug_state_sha256")
    right_debug = visible.episode["operations"][-1].get("debug_state_sha256")
    if left_debug != right_debug:
        raise RuntimeError(f"full game state diverged after {action}: {left_debug} != {right_debug}")
    return left[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[12, 18, 31, 50])
    parser.add_argument("--ticks", type=int, default=1500)
    args = parser.parse_args()
    if args.ticks < 1:
        parser.error("--ticks must be positive")

    for level in args.levels:
        task = TaskSpec(level=level, seed=1701, playthrough=2, profile=PlayerProfileContext())
        with PvZEnv(args.resource_dir, args.executable, headless=True, debug_replay=True) as headless, \
                PvZEnv(args.resource_dir, args.executable, headless=False, debug_replay=True) as visible:
            observations = [env.reset(deck=DECK, task=task)[0] for env in (headless, visible)]
            if observations[0] != observations[1]:
                raise RuntimeError(f"reset state diverged at level {level}")
            states = [env.privileged_state() for env in (headless, visible)]
            if digest(states[0]) != digest(states[1]):
                raise RuntimeError(f"reset full state diverged at level {level}")

            legal = next((item for item in observations[0]["legal_actions"]["plants"] if item["packet"] == 0), None)
            if legal is None:
                legal = next(iter(observations[0]["legal_actions"]["plants"]), None)
            cooldown_packet = None
            cooldown_ready_tick = 0
            if legal is not None:
                original_card_type = observations[0]["packets"][legal["packet"]]["type"]
                state = compare_pair(headless, visible, {"type": "plant", **legal})
                packet = state["packets"][legal["packet"]]
                if packet["type"] == original_card_type:
                    cooldown_packet = legal["packet"]
                    if packet["active"] or packet["refresh_time"] <= 0:
                        raise RuntimeError(f"planted card did not enter cooldown at level {level}")
                    cooldown_ready_tick = state["tick"] + packet["refresh_time"]
            elif level != 50:
                raise RuntimeError(f"no card can be planted to check cooldown at level {level}")
            for _ in range(20):
                compare_pair(headless, visible, {"type": "wait", "ticks": 1})
            snapshot_ids = [env.snapshot() for env in (headless, visible)]
            compare_pair(headless, visible, {"type": "wait", "ticks": 1})
            restored = [env.restore(snapshot_id) for env, snapshot_id in zip((headless, visible), snapshot_ids)]
            if restored[0] != restored[1] or digest(headless.privileged_state()) != digest(visible.privileged_state()):
                raise RuntimeError(f"snapshot restore diverged at level {level}")

            state = restored[0]
            for _ in range(args.ticks):
                state = compare_pair(headless, visible, {"type": "wait", "ticks": 1})
                if state["terminal"]:
                    break
            if cooldown_packet is not None and state["tick"] >= cooldown_ready_tick:
                packet = state["packets"][cooldown_packet]
                if not packet["active"] or packet["cooldown"] != 0:
                    raise RuntimeError(f"planted card did not refresh at level {level}: {packet}")
            print(json.dumps({"level": level, "tick": state["tick"], "fog": state["fog"],
                              "card_cooldown_checked": cooldown_packet is not None,
                              "night": state["night"], "visible_headless_equal": True}))


if __name__ == "__main__":
    main()
