"""Drive the real game with a delivered checkpoint, optionally with a visible window.

This is a read-only viewer for humans. It never trains, never writes into
``artifacts/``, and never touches a running experiment. It exists because a
win rate alone cannot tell you *what the policy is doing* -- you have to look.

Two things it can show:

1. **The model playing.** Pass a checkpoint and it drives the simulator with the
   policy's own actions. With ``--visible`` the simulator runs in ``-env-visible``
   mode, so the normal game window opens and animates while Python supplies the
   actions over the existing protocol.
2. **The do-nothing baseline.** ``--policy donothing`` waits forever and never
   plants. On short tasks this baseline wins on its own (lawnmowers), which is
   exactly why a short-task win rate carries almost no policy information.
   Compare the two side by side before believing any short-task number.

Usage::

    # watch the delivered full-level7 checkpoint on a 5-wave task, window open
    python scripts/watch_policy.py --visible --wave-cap 5 --delay 0.15

    # same, but with the do-nothing baseline
    python scripts/watch_policy.py --visible --wave-cap 5 --policy donothing

    # headless, print the action mix only
    python scripts/watch_policy.py --wave-cap 5 --seed 1330000 --quiet

Checkpoints are not in git; fetch one first:

    python3 -c "from huggingface_hub import hf_hub_download; \\
        hf_hub_download('realnewbiexvwu/pvz-agent-artifacts', \\
        'mainline_full_level7_v1_seed0/runs/run_1/<name>.pt', repo_type='model', \\
        revision='evidence-mainline-full-level7-v1-seed0', local_dir='/tmp/pvz_infer')"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import policy_env  # noqa: E402
from pvz_agent_model import (  # noqa: E402
    GameplayModelV1, observation_tokens, policy_legal_summary, select_action,
)

DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
DEFAULT_DECK = [0, 1, 2, 3, 4, 5]


def _model_actions(model, cfg, deterministic: bool):
    """Yield one policy action per observation; state lives across calls.

    ``info`` from the previous ``env.step`` must be fed back: the wait head
    conditions on the previous wait's result and the ticks it actually
    advanced, and the replay path re-validates those two against each other.
    """
    hidden = prev = prev_wait = None
    elapsed, events = 0, {}

    def next_action(observation, info=None):
        nonlocal hidden, prev, prev_wait, elapsed, events
        if info is not None:
            prev_wait = info.get("wait_result")
            elapsed = int(info.get("ticks_advanced", 0) or 0)
            events = info.get("events", {}) or {}
        tensors, meta = observation_tokens(observation, cfg.get("input_flags", 0))
        legal = policy_legal_summary(observation, cfg)
        with torch.no_grad():
            out = model.step_tokens(
                tensors, meta, observation["wave"], hidden, prev, elapsed, events,
                prev_wait, planner_context=model.planner_context(observation))
        action, _, _ = select_action(model, out, legal, deterministic=deterministic)
        hidden, prev = out["hidden"], action
        return action

    return next_action


def _donothing_actions():
    def next_action(_observation, _info=None):
        return {"type": "wait", "ticks": 300, "until": "timeout"}
    return next_action


def _load_checkpoint(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = GameplayModelV1(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck.get("training_state", {})


def run(checkpoint: str | None, policy: str, level: int, seed: int, wave_cap: int | None,
        visible: bool, delay: float, max_actions: int, deterministic: bool,
        resource_dir: str, deck, quiet: bool) -> dict:
    if policy == "donothing":
        cfg = {"wait_mode": "events", "wait_mask": "progress_v1", "input_flags": 7}
        next_action = _donothing_actions()
        state = {}
    else:
        if not checkpoint:
            raise SystemExit("--checkpoint is required unless --policy donothing")
        model, cfg, state = _load_checkpoint(checkpoint)
        next_action = _model_actions(model, cfg, deterministic)

    env = policy_env(cfg, resource_dir=resource_dir, headless=not visible)
    spec = TaskSpec(level=level, seed=seed, playthrough=2,
                    zombie_count_multiplier=1.0, wave_cap=wave_cap, preplanted=())
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
        "policy": policy, "level": level, "seed": seed, "wave_cap": wave_cap,
        "decisions": sum(counts.values()), "counts": counts,
        "zero_tick_waits": zero_tick, "terminal_wave": observation.get("wave"),
        "result": observation.get("result"), "won": observation.get("result") == 1,
        "ticks": observation.get("tick"),
        "checkpoint_updates": state.get("updates"),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", help="local .pt path (not in git; fetch from HF)")
    p.add_argument("--policy", choices=("model", "donothing"), default="model")
    p.add_argument("--level", type=int, default=7)
    p.add_argument("--seed", type=int, default=1330000)
    p.add_argument("--wave-cap", type=int, default=None,
                   help="truncate to N waves; omit for the complete level")
    p.add_argument("--visible", action="store_true", help="open the real game window")
    p.add_argument("--delay", type=float, default=0.0, help="pause per action, seconds")
    p.add_argument("--max-actions", type=int, default=800)
    p.add_argument("--sampled", action="store_true", help="sample instead of greedy")
    p.add_argument("--quiet", action="store_true", help="print only the final summary")
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    p.add_argument("--deck", default=",".join(str(c) for c in DEFAULT_DECK))
    p.add_argument("--hold-seconds", type=float, default=0.0,
                   help="keep the window up after the episode ends")
    args = p.parse_args()

    deck = [int(c) for c in args.deck.split(",") if c.strip()]
    result = run(args.checkpoint, args.policy, args.level, args.seed, args.wave_cap,
                 args.visible, args.delay, args.max_actions, not args.sampled,
                 args.resource_dir, deck, args.quiet)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.hold_seconds:
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
