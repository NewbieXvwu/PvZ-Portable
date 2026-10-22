"""Per-decision time and byte breakdown of one collection game (task33).

Wraps the functions a collection worker actually calls, runs search_play.main() with the
arguments after "--", and writes one JSON line of totals a game to $HOTPATH_OUT.
No product code is changed: every wrapper is installed on the imported module object.

  HOTPATH_OUT=/tmp/hotpath.jsonl python3 scripts/hotpath_profile.py -- --seeds 1 --workers 1 ...
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import search_play as sp  # noqa: E402

OUT = Path(os.environ.get("HOTPATH_OUT", "/tmp/hotpath.jsonl"))
BYTES = os.environ.get("HOTPATH_BYTES", "1") != "0"
ACC: dict[str, float] = {}
COUNT: dict[str, int] = {}
DEPTH: dict[str, int] = {}


def timed(name):
    def wrap(function):
        @functools.wraps(function)
        def inner(*args, **kwargs):
            if DEPTH.get(name):
                return function(*args, **kwargs)
            DEPTH[name] = 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                DEPTH[name] = 0
                ACC[name] = ACC.get(name, 0.0) + time.perf_counter() - started
                COUNT[name] = COUNT.get(name, 0) + 1
        return inner
    return wrap


import pvz_value_net  # noqa: E402
import render_episode  # noqa: E402

for _name in ("_search", "_fast_action", "_continuations", "_leaf_values", "_verified_choice",
              "_rollout_policy", "_frame_json", "_static_frames", "_values", "_halving_choice"):
    if hasattr(sp, _name):
        setattr(sp, _name, timed(_name)(getattr(sp, _name)))

# Imported inside the functions, so the module attributes are what has to be wrapped.
pvz_value_net.featurize = timed("featurize")(pvz_value_net.featurize)
render_episode._frame = timed("_frame")(render_episode._frame)

# The archive write happens after the decision loop, so it never shows up in a per-decision timer.
# search_play imports it inside _play, so the module attribute is what has to be wrapped.
import pvz_frame_store  # noqa: E402

pvz_frame_store.write_frames_from_source = timed("write_frames")(pvz_frame_store.write_frames_from_source)

_play_original = sp._play


@functools.wraps(_play_original)
def play(job):
    from pvz_event_env import EventWaitEnv
    if not getattr(EventWaitEnv, "_hotpath_counted", False):
        original_read = EventWaitEnv._read_message
        original_command = EventWaitEnv._command

        def read(self):
            response = original_read(self)
            if BYTES:
                # Re-serialising is instrumentation cost, not product cost: it is the only way to
                # size a response without touching the reader. Turn it off (HOTPATH_BYTES=0) when
                # the number wanted is resident memory rather than bytes.
                ACC["resp_bytes"] = ACC.get("resp_bytes", 0) + len(json.dumps(response, separators=(",", ":")))
            ACC["responses"] = ACC.get("responses", 0) + 1
            # The engine times its own two halves of every ROLLOUT; summing them says whether a
            # lane's 24 ms is simulation or the C++ student's forward.
            for key in ("net_calls", "network_seconds", "simulation_seconds", "ticks"):
                if key in response:
                    ACC["engine_" + key] = ACC.get("engine_" + key, 0) + response[key]
            return response

        def command(self, text):
            ACC["req_bytes"] = ACC.get("req_bytes", 0) + len(text)
            return original_command(self, text)

        original_step = EventWaitEnv.step
        original_rollout = EventWaitEnv.rollout

        def step(self, action):
            started = time.perf_counter()
            try:
                return original_step(self, action)
            finally:
                ACC["step"] = ACC.get("step", 0.0) + time.perf_counter() - started
                COUNT["step"] = COUNT.get("step", 0) + 1

        def rollout(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original_rollout(self, *args, **kwargs)
            finally:
                ACC["rollout_call"] = ACC.get("rollout_call", 0.0) + time.perf_counter() - started
                COUNT["rollout_call"] = COUNT.get("rollout_call", 0) + 1

        EventWaitEnv._read_message = read
        EventWaitEnv._command = command
        EventWaitEnv.step = step
        EventWaitEnv.rollout = rollout
        EventWaitEnv._hotpath_counted = True
    ACC.clear()
    COUNT.clear()
    started = time.perf_counter()
    result = _play_original(job)
    total = time.perf_counter() - started
    import resource
    rss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # bytes on macOS, KB on Linux
    line = {"seed": job["seed"], "arm": job["arm"], "total_s": round(total, 3),
            "rss_self_mb": round(rss_self, 1),
            "seconds_reported": result.get("seconds"), "decisions": result.get("decisions"),
            "searched": result.get("searched"), "rollout_lanes": result.get("rollout_lanes"),
            "rollout_batches": result.get("rollout_batches"),
            **{f"{key}_s": round(value, 3) for key, value in ACC.items()},
            **{f"{key}_n": value for key, value in COUNT.items()}}
    with OUT.open("a") as stream:
        stream.write(json.dumps(line) + "\n")
    return result


sp._play = play
if os.environ.get("HOTPATH_REUSE"):
    # A/B of maxtasksperchild=1 (today) against a worker that keeps playing: same games either way,
    # RESET_V2 rebuilds the board from the seed, so only the per-game process cost differs.
    import multiprocessing as mp_module

    class _Context:
        def __init__(self, real):
            self._real = real

        def Pool(self, processes, initializer=None, initargs=(), maxtasksperchild=None):
            return self._real.Pool(processes, initializer=initializer, initargs=initargs,
                                   maxtasksperchild=None)

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _Module:
        def get_context(self, method):
            return _Context(mp_module.get_context(method))

        def __getattr__(self, name):
            return getattr(mp_module, name)

    sp.mp = _Module()

extra = sys.argv[1:]
if extra and extra[0] == "--":
    extra = extra[1:]
sys.argv = ["search_play.py"] + extra
sp.main()
