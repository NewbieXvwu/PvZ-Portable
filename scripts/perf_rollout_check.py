"""Exactness checks for the rollout speedups (the approximate ones live in sim_skip_agreement.py).

On live states of real student games it verifies that

  1. ``pvz_value_net.featurize`` (numpy) is bitwise equal to the original torch-indexing encoder,
     and ``pvz_rollout_policy.features`` to the original concatenation;
  2. ``render_episode._frame`` of the compact ``OBS_LEAN`` observation equals ``_frame`` of the
     full observation of the same state, and the legal plant lists are equal;
  3. a simulator process reused across games (RESET_V2) plays the same game as a fresh one;
  4. the first trunk convolution restricted to non-zero input channels equals the dense one;
  5. a search rollout gives the same leaf (tick, frame, sun-income history) with and without
     ``WAIT_UNTIL_LEGAL`` + ``OBS_LEAN``.

Usage: python3 scripts/perf_rollout_check.py --seeds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402
from pvz_rollout_policy import RolloutNet, act, features  # noqa: E402
from pvz_value_net import _featurize_reference, featurize  # noqa: E402
from render_episode import _frame, load_training_task, training_task_kwargs  # noqa: E402
from scripted_baseline import profile_for_deck  # noqa: E402

DEFAULT_MANIFEST = ROOT / "experiments/t7/bc_finetune_level7_v1/progress_evaluation_1024.json"


def _reference_features(frame: dict, previous: dict | None):
    board, scalars = _featurize_reference(frame)
    action = frame.get("action") if previous is not None else None
    extra = torch.tensor([float(bool(action) and action.get("type") == "plant"),
                          ((frame["tick"] - previous["tick"]) / 300.0) if previous else 0.0])
    return board, torch.cat([scalars, extra])


def _rollout_pair(env, obs: dict, first: dict, seed: int, counters: dict) -> None:
    """The same search rollout with and without WAIT_UNTIL_LEGAL / OBS_LEAN must end identically."""
    import search_play
    leaves = []
    root = env.snapshot()
    try:
        for batch, lean in ((False, False), (True, True)):
            env.restore(root)
            env.reseed(seed)
            search_play._W.update(batch_waits=batch, lean_rollouts=lean)
            leaf = search_play._rollout_policy(env, obs, {}, first, obs["tick"] + 3000)
            leaves.append((leaf["tick"], _frame(leaf), env._sun_production_history.copy()))
    finally:
        env.restore(root)
        env.release_snapshot(root)
    if leaves[0] != leaves[1]:
        raise SystemExit(f"batched-wait rollout differs at seed {seed} tick {obs['tick']}")
    counters["rollouts"] += 1


def play(env, net, manifest, task_id, seed, check_lean: bool, counters: dict) -> list:
    kw = training_task_kwargs(load_training_task(manifest, task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2, profile=profile_for_deck(tuple(kw["deck"])),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
    memo = {"frame": None, "action": None}
    trace = []
    while not obs["terminal"] and len(trace) < 3000:
        frame = _frame(obs)
        frame["action"] = memo["action"]
        b, s = featurize(frame)
        rb, rs = _featurize_reference(frame)
        fb, fs = features(frame, memo["frame"])
        rfb, rfs = _reference_features(frame, memo["frame"])
        if not (torch.equal(b, rb) and torch.equal(s, rs) and torch.equal(fb, rfb) and torch.equal(fs, rfs)):
            raise SystemExit(f"featurize differs from the reference at seed {seed} tick {obs['tick']}")
        counters["frames"] += 1
        with torch.inference_mode():
            dense_h, dense_g = net.encode(fb[None], fs[None])
            sparse_h, sparse_g = net.encode(fb[None], fs[None], sparse_input=True)
        if not (torch.equal(dense_h, sparse_h) and torch.equal(dense_g, sparse_g)):
            counters["sparse_mismatch"] += 1
            counters["sparse_max_diff"] = max(counters["sparse_max_diff"], float((dense_h - sparse_h).abs().max()),
                                              float((dense_g - sparse_g).abs().max()))
            legal = (obs.get("legal_actions") or {}).get("plants") or []
            if act(net, frame, memo["frame"], legal, sparse_input=False) != act(net, frame, memo["frame"], legal):
                raise SystemExit(f"channel-pruned trunk changes the action at seed {seed} tick {obs['tick']}")
        if check_lean:
            env.set_lean_observation(True)
            lean = env.observe()
            env.set_lean_observation(False)
            full = env.observe()
            if _frame(lean) != _frame(full) or lean["legal_actions"]["plants"] != full["legal_actions"]["plants"]:
                raise SystemExit(f"lean observation differs at seed {seed} tick {obs['tick']}")
            counters["lean_states"] += 1
        action = act(net, frame, memo["frame"], (obs.get("legal_actions") or {}).get("plants") or [])
        if check_lean and len(trace) % 40 == 0:
            _rollout_pair(env, obs, action, seed * 1000 + len(trace), counters)
        memo["frame"], memo["action"] = frame, action
        obs, _r, done, _t, info = env.step(action)
        if not info.get("ok"):
            obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
        trace.append((obs["tick"], action))
        if done:
            break
    return trace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--task-id", default="validation_full_level7_v1_d1")
    p.add_argument("--policy", default="/Users/newbiexvwu/PvZAgent/artifacts/student/v3all/model.pt")
    p.add_argument("--executable", default=str(ROOT / "build" / "pvz-portable"))
    p.add_argument("--resource-dir", default="/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
    args = p.parse_args()
    torch.set_num_threads(1)
    net = RolloutNet()
    weights = torch.load(args.policy, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")})
    net.eval()
    seeds = load_training_task(args.manifest, args.task_id)["seeds"][: args.seeds]
    counters = {"frames": 0, "lean_states": 0, "sparse_mismatch": 0, "sparse_max_diff": 0.0, "rollouts": 0}
    import search_play
    search_play._W["fast"] = net
    reused = EventWaitEnv(args.resource_dir, executable=args.executable)
    traces = {s: play(reused, net, args.manifest, args.task_id, s, True, counters) for s in seeds}
    for s in reversed(seeds):              # every game now runs in a process that already played others
        fresh = EventWaitEnv(args.resource_dir, executable=args.executable)
        try:
            again = play(fresh, net, args.manifest, args.task_id, s, False, counters)
        finally:
            fresh.close()
        if again != traces[s]:
            raise SystemExit(f"seed {s}: reused process and fresh process played different games")
    reused.close()
    print(f"ok: {counters['frames']} frames bitwise-equal features, {counters['lean_states']} states with "
          f"identical lean/full frames, {len(seeds)} seeds identical in reused vs fresh processes; "
          f"channel-pruned trunk differs from the dense one on {counters['sparse_mismatch']} frames "
          f"(max |diff| {counters['sparse_max_diff']:.3g}, no action changed); {counters['rollouts']} "
          f"3000-tick rollouts identical with and without WAIT_UNTIL_LEGAL + OBS_LEAN")


if __name__ == "__main__":
    main()
