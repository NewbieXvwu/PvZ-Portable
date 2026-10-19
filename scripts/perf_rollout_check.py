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
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402
from pvz_rollout_policy import RolloutNet, act, features, load_student, export_student  # noqa: E402
from pvz_value_net import _featurize_reference, featurize, observation_context  # noqa: E402
from render_episode import _frame, load_training_task, training_task_kwargs  # noqa: E402
from pvz_tasks import profile_for_deck  # noqa: E402

DEFAULT_MANIFEST = ROOT / "experiments/t7/bc_finetune_level7_v1/progress_evaluation_1024.json"


def check_cpp(args) -> None:
    import cProfile
    import pstats
    import search_play
    from collections import Counter
    net = load_student(args.policy)
    task_data = load_training_task(args.manifest, args.task_id)
    counters = {"frames": 0, "same_actions": 0, "max_logit_error": 0.0}
    with tempfile.TemporaryDirectory(prefix="pvz-rollout-check-") as directory, EventWaitEnv(
            args.resource_dir, executable=args.executable) as env:
        path = Path(directory) / "student.bin"
        export_student(net, path)
        env.load_rollout_net(path)
        search_play._W.update(fast=net, lean_rollouts=True, batch_waits=True, rollout_sim_skip=args.mask)
        if args.cpp_mode == "search":
            from pvz_value_net import load_value_net
            search_play._W["vnet"] = load_value_net(args.value)
        for seed in task_data["seeds"][: args.seeds]:
            kw = training_task_kwargs(task_data, seed)
            task = TaskSpec(level=kw["level"], seed=seed, playthrough=2, profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                            **{k: v for k, v in kw["task_extra"].items() if v is not None})
            obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
            task_context = observation_context(obs)
            search_play._W["context"] = task_context
            memo = {}
            if args.cpp_mode == "search":
                stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
                search_play._W["cpp_rollouts"] = args.cpp_rollouts
                profile = cProfile.Profile() if args.profile else None
                if profile: profile.enable()
                started = time.perf_counter()
            for decision in range(3000):
                if obs["terminal"]: break
                frame = _frame(obs)
                frame["action"] = memo.get("action")
                action = act(net, frame, memo.get("frame"), obs["legal_actions"]["plants"], context=task_context)
                if args.cpp_mode == "frames":
                    previous = memo.get("frame")
                    native = env.rollout_predict(previous["tick"] if previous else None,
                                                 memo.get("action", {}).get("type") == "plant")
                    board, scalars = features(frame, previous, task_context if net.context else None)
                    with torch.inference_mode():
                        hidden, context = net.encode(board[None], scalars[None], sparse_input=True)
                        errors = [(net.plant(context).flatten() - torch.tensor([native["plant"]])).abs().max(),
                                  (net.packet(context)[0] - torch.tensor(native["packet"])).abs().max()]
                        if native["cell_packet"] >= 0:
                            cells = net.cell_logits(hidden, context, torch.tensor([native["cell_packet"]]))[0]
                            errors.append((cells - torch.tensor(native["cell"])).abs().max())
                    counters["max_logit_error"] = max(counters["max_logit_error"], max(float(error) for error in errors))
                    counters["frames"] += 1
                    counters["same_actions"] += action == native["action"]
                elif args.cpp_mode == "throughput" and decision == 200:
                    root = env.snapshot()
                    times = {}
                    for cpp in (False, True):
                        search_play._W["cpp_rollouts"] = cpp
                        search_play._W["rollout_metrics"] = {}
                        started = time.perf_counter()
                        for repeat in range(args.repeats):
                            env.restore(root)
                            env.reseed(seed * 1000 + repeat)
                            search_play._rollout_policy(env, obs, action, obs["tick"] + 3000)
                        times["cpp" if cpp else "python"] = time.perf_counter() - started
                        if cpp: print(json.dumps(search_play._W["rollout_metrics"]), flush=True)
                    env.restore(root)
                    env.release_snapshot(root)
                    print(json.dumps({"seed": seed, "repeats": args.repeats, "seconds": times,
                                      "speedup": times["python"] / times["cpp"]}), flush=True)
                    break
                elif args.cpp_mode == "search":
                    action = search_play._search(env, obs, action, args, seed * 10000 + decision, stats)
                memo = {"frame": frame, "action": action}
                obs, *_rest, info = env.step(action)
                if not info.get("ok"): obs, *_rest, info = env.step({"type": "wait", "ticks": 60})
            if args.cpp_mode == "search":
                seconds = time.perf_counter() - started
                if profile:
                    profile.disable()
                    profile.dump_stats(str(args.profile))
                    pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(35)
                print(json.dumps({"environment_profile": env._command("ENV_PROFILE")}), flush=True)
                print(json.dumps({"seed": seed, "cpp": args.cpp_rollouts, "seconds": seconds, "won": obs["result"] == 1,
                                  "decisions": decision, "searched": stats["searched"], "overrides": stats["overrides"],
                                  "rollout_metrics": search_play._W.get("rollout_metrics")}), flush=True)
        if args.cpp_mode == "frames":
            counters["action_agreement"] = counters["same_actions"] / counters["frames"]
            print(json.dumps(counters), flush=True)
            if args.out: args.out.write_text(json.dumps(counters, indent=2) + "\n")
            if counters["frames"] < 2000 or counters["action_agreement"] < 1 or counters["max_logit_error"] > 1e-4:
                raise SystemExit("C++ frame equivalence threshold failed")


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
            leaf = search_play._rollout_policy(env, obs, first, obs["tick"] + 3000)
            leaves.append((leaf["tick"], _frame(leaf), env._sun_production_history.copy()))
    finally:
        env.restore(root)
        env.release_snapshot(root)
    if leaves[0] != leaves[1]:
        raise SystemExit(f"batched-wait rollout differs at seed {seed} tick {obs['tick']}")
    counters["rollouts"] += 1


def play(env, net, manifest, task_id, seed, check_lean: bool, counters: dict) -> list:
    kw = training_task_kwargs(load_training_task(manifest, task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2, profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
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
    p.add_argument("--cpp-mode", choices=("frames", "throughput", "search"))
    p.add_argument("--cpp-rollouts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value", default="/Users/newbiexvwu/PvZAgent/artifacts/value_net/v2/model.pt")
    p.add_argument("--mask", type=int, default=21)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--profile", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--per-card", type=int, default=1)
    p.add_argument("--min-gap", type=int, default=300)
    p.add_argument("--halving", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3000)
    p.set_defaults(margin=.02)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.cpp_mode:
        check_cpp(args)
        return
    net = load_student(args.policy)
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
