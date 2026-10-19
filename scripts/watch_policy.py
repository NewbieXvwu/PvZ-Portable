"""Watch a student play the real game, optionally in a visible window.

This is a read-only viewer for humans. It never trains, never writes into
``artifacts/``, and never touches a running experiment. It exists because a
win rate alone cannot tell you *what the policy is doing* -- you have to look.
With ``--visible`` the simulator runs in ``-env-visible`` mode, so the normal game
window opens and animates while Python supplies the student's actions.

Usage::

    # the current best student on a 1.5x-zombie game of level 7, its own deck
    python scripts/watch_policy.py --visible --student artifacts/student/v31/model.pt --deck 0,1,2,3,4,5,7 \
        --multiplier 1.5 --seed 30000 --max-actions 3000 --hold-seconds 10

    # playback speed: PVZ_ENV_RENDER_EVERY logic ticks per drawn frame, PVZ_ENV_RENDER_MS pause per frame
    # (defaults 1 / 10 = normal speed); 5x:
    PVZ_ENV_RENDER_EVERY=5 python scripts/watch_policy.py --visible --student ...

    # replay one game of an evaluation (here v30 on 2.0x zombies, seed 30005) at 5x speed
    PVZ_ENV_RENDER_EVERY=5 python scripts/watch_policy.py --visible --student artifacts/student/v30/model.pt \
        --task-id day_l7_z20_eval --seed 30005 --max-actions 3000 --hold-seconds 10

Students are not in git; fetch one from HF (``transfer/v31_model.pt``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402

DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
DEFAULT_DECK = [0, 1, 2, 3, 4, 5]


def _student_actions(path: str):
    """The small no-memory student (python/pvz_rollout_policy.py, DESIGN §13), greedy."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from pvz_rollout_policy import act, load_student
    from render_episode import _frame
    net = load_student(path)
    memo = {"frame": None, "action": None}
    if net.obs != "v1":
        from pvz_student_observation import StudentHistory
        memo["history"] = StudentHistory(net.obs)

    def next_action(observation, _info=None):
        frame = _frame(observation)
        frame["action"] = memo["action"]
        with torch.no_grad():
            if "context" not in memo:      # the task is fixed for the whole game; read it from the first observation
                from pvz_value_net import observation_context
                memo["context"] = observation_context(observation)
            action = act(net, frame, memo["frame"], (observation.get("legal_actions") or {}).get("plants") or [],
                         context=memo["context"], history=memo.get("history"))
        memo["frame"], memo["action"] = frame, action
        return action
    return next_action


def run(student: str, level: int, seed: int, wave_cap: int | None, visible: bool, delay: float,
        max_actions: int, resource_dir: str, deck, quiet: bool, multiplier: float = 1.0,
        manifest: str | None = None, task_id: str | None = None) -> dict:
    next_action = _student_actions(student)
    env = EventWaitEnv(resource_dir=resource_dir, headless=not visible)
    from pvz_tasks import profile_for_deck
    if task_id:   # the exact game an evaluation played: same task, same seed (scripts/search_play.py)
        sys.path.insert(0, str(ROOT / "scripts"))
        from render_episode import load_training_task, training_task_kwargs
        kw = training_task_kwargs(load_training_task(Path(manifest), task_id), seed)
        deck, level = list(kw["deck"]), kw["level"]
        spec = TaskSpec(level=level, seed=kw["seed"], playthrough=2,
                        profile=profile_for_deck(tuple(deck), kw.get("first_aid_owned", False)),
                        **{k: v for k, v in kw["task_extra"].items() if v is not None})
    else:
        spec = TaskSpec(level=level, seed=seed, playthrough=2, profile=profile_for_deck(tuple(deck)),
                        zombie_count_multiplier=multiplier, wave_cap=wave_cap, preplanted=())
    observation, _ = env.reset(deck=deck, task=spec)

    counts = {"plant": 0, "shovel": 0, "wait": 0}
    zero_tick = 0
    info = None
    started = time.perf_counter()
    for step in range(max_actions):
        action = next_action(observation, info)
        counts[action["type"]] = counts.get(action["type"], 0) + 1
        if not quiet and (action["type"] != "wait" or step % 25 == 0):
            print(f"  step {step}: {json.dumps(action)} "
                  f"wave={observation['wave']} sun={observation['sun']}", flush=True)
        if delay:
            time.sleep(delay)
        observation, _, done, _, info = env.step(action)
        if info.get("ticks_advanced", 0) == 0:
            zero_tick += 1
        if done:
            break
    result = {
        "student": student, "level": level, "seed": seed, "wave_cap": wave_cap,
        "decisions": sum(counts.values()), "counts": counts,
        "zero_tick_waits": zero_tick, "terminal_wave": observation.get("wave"),
        "result": observation.get("result"), "won": observation.get("result") == 1,
        "ticks": observation.get("tick"),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--student", required=True, help="student model.pt (not in git; fetch from HF)")
    p.add_argument("--level", type=int, default=7)
    p.add_argument("--seed", type=int, default=1330000)
    p.add_argument("--multiplier", type=float, default=1.0, help="zombie-count multiplier (1.3 / 1.5 = the hard tasks)")
    p.add_argument("--wave-cap", type=int, default=None,
                   help="truncate to N waves; omit for the complete level")
    p.add_argument("--visible", action="store_true", help="open the real game window")
    p.add_argument("--delay", type=float, default=0.0, help="pause per action, seconds")
    p.add_argument("--max-actions", type=int, default=800)
    p.add_argument("--quiet", action="store_true", help="print only the final summary")
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    p.add_argument("--deck", default=",".join(str(c) for c in DEFAULT_DECK))
    p.add_argument("--task-id", help="replay a manifest task (e.g. day_l7_z20_eval) with --seed = the evaluation's seed")
    p.add_argument("--manifest", default=str(ROOT / "artifacts/stage4/day_l7_hard.json"))
    p.add_argument("--hold-seconds", type=float, default=0.0,
                   help="keep the window up after the episode ends")
    args = p.parse_args()

    deck = [int(c) for c in args.deck.split(",") if c.strip()]
    result = run(args.student, args.level, args.seed, args.wave_cap, args.visible, args.delay, args.max_actions,
                 args.resource_dir, deck, args.quiet, args.multiplier, args.manifest, args.task_id)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.hold_seconds:
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
