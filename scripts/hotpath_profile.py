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


for _name in ("_search", "_fast_action", "_continuations", "_leaf_values", "_verified_choice",
              "_rollout_policy", "_frame_json"):
    if hasattr(sp, _name):
        setattr(sp, _name, timed(_name)(getattr(sp, _name)))

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

        EventWaitEnv._read_message = read
        EventWaitEnv._command = command
        EventWaitEnv._hotpath_counted = True
    ACC.clear()
    COUNT.clear()
    started = time.perf_counter()
    result = _play_original(job)
    total = time.perf_counter() - started
    line = {"seed": job["seed"], "arm": job["arm"], "total_s": round(total, 3),
            "seconds_reported": result.get("seconds"), "decisions": result.get("decisions"),
            "searched": result.get("searched"), "rollout_lanes": result.get("rollout_lanes"),
            "rollout_batches": result.get("rollout_batches"),
            **{f"{key}_s": round(value, 3) for key, value in ACC.items()},
            **{f"{key}_n": value for key, value in COUNT.items()}}
    with OUT.open("a") as stream:
        stream.write(json.dumps(line) + "\n")
    return result


sp._play = play
extra = sys.argv[1:]
if extra and extra[0] == "--":
    extra = extra[1:]
sys.argv = ["search_play.py"] + extra
sp.main()
