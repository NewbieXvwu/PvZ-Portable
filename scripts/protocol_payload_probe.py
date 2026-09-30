"""Price the environment protocol: what crosses the pipe per decision, and what it costs.

The question this answers is not "is the simulator fast" but "how much of the
per-decision cost is *transport and decoding of data the trainer does not use*".
It instruments a real ``PvZEnv`` at the one place every response passes through
(:meth:`PvZEnv._read_message`) and separates:

  * C++ simulate + C++ JSON serialise + pipe write + pipe read  (the residual)
  * ``json.loads`` of the ``PVZENV {...}`` line
  * ``_canonical_events`` key sharing
  * the ``CRITIC_INPUTS`` round trip, which is a *second* synchronous request per
    decision

It also asserts the redundancy it suspects rather than assuming it: for every
decision it compares ``critic_inputs["wave_timer"]`` against the
``wave_timer`` that the immediately preceding ``OBS`` already carried.

Run with the mise interpreter (the managed one has no numpy/torch)::

    /Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3 \
        scripts/protocol_payload_probe.py --decisions 120
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_env import PvZEnv, _canonical_events  # noqa: E402
from pvz_agent_model import configure_torch_threads  # noqa: E402
from train_pvz_ppo import _task_spec  # noqa: E402
import train_pvz_ppo_task_family as trainer  # noqa: E402

LOCAL_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")


class InstrumentedEnv(PvZEnv):
    """A real env that records the size and decode cost of every response line."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_line_bytes: list[int] = []
        self.decode_seconds: list[float] = []
        self.events_seconds: list[float] = []
        self.observation_bytes: list[int] = []
        self.response_keys: list[frozenset[str]] = []

    def _read_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("environment process is not running")
        for line in process.stdout:
            if line.startswith("PVZENV "):
                payload = line[len("PVZENV "):]
                started = time.perf_counter()
                response = json.loads(payload)
                self.decode_seconds.append(time.perf_counter() - started)
                self.raw_line_bytes.append(len(line))
                self.response_keys.append(frozenset(response))
                if "observation" in response:
                    self.observation_bytes.append(len(json.dumps(response["observation"])))
                if "events" in response:
                    events_started = time.perf_counter()
                    response["events"] = _canonical_events(response["events"])
                    self.events_seconds.append(time.perf_counter() - events_started)
                return response
        raise RuntimeError("environment closed the pipe")


def _percentiles(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    return (f"p50={statistics.median(ordered) * 1e3:.3f} ms  "
            f"p90={ordered[int(len(ordered) * 0.9)] * 1e3:.3f} ms  "
            f"max={ordered[-1] * 1e3:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resource-dir", type=Path, default=LOCAL_RESOURCES)
    parser.add_argument("--decisions", type=int, default=120,
                        help="WAIT steps to sample; the default is ~3 episodes' worth")
    parser.add_argument("--wait-ticks", type=int, default=1,
                        help="ticks per WAIT; raise it to let the simulator do real work")
    parser.add_argument("--privileged", action="store_true",
                        help="also price the PRIV payload (57.9 KiB, unused by training)")
    parser.add_argument("--field-budget", type=int, default=0, metavar="N",
                        help="sample N observations and price every top-level field, "
                             "cross-referenced against the read sites in the repo")
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser()
    if not resource_dir.is_dir():
        raise SystemExit(f"resource dir not found: {resource_dir}")

    configure_torch_threads(1)
    train, _heldout = trainer._task_family()
    task = trainer._curriculum_tasks(train["tasks"], "cap1")[0]
    deck = task["deck"]
    spec = _task_spec(task, task["seeds"][0])

    with InstrumentedEnv(resource_dir) as env:
        observation, _ = env.reset(deck=deck, task=spec)
        step_seconds: list[float] = []
        critic_seconds: list[float] = []
        critic_timers: list[tuple[int, int]] = []
        for _ in range(args.decisions):
            wave = observation["wave"]
            public_wave_timer = observation["wave_timer"]
            critic_started = time.perf_counter()
            critic = env.critic_inputs(wave)
            critic_seconds.append(time.perf_counter() - critic_started)
            critic_timers.append((critic["wave_timer"], public_wave_timer))
            started = time.perf_counter()
            observation, _, done, _, info = env.step({"type": "wait", "ticks": args.wait_ticks})
            step_seconds.append(time.perf_counter() - started)
            if not info.get("ok"):
                raise SystemExit(f"WAIT was rejected: {info}")
            if done:
                observation, _ = env.reset(deck=deck, task=spec)

        privileged: list[float] = []
        if args.privileged:
            for _ in range(20):
                started = time.perf_counter()
                env.privileged_state()
                privileged.append(time.perf_counter() - started)

    # ``raw_line_bytes`` / ``decode_seconds`` hold one entry per PVZENV line seen,
    # which is reset + every CRITIC_INPUTS + every STEP, in order.  Separate them by
    # size: CRITIC_INPUTS lines are ~69 B, observations are ~12.8 KiB.
    critic_lines = [n for n in env.raw_line_bytes if n < 1024]
    obs_lines = [n for n in env.raw_line_bytes if n >= 1024]
    obs_decodes = [s for s, n in zip(env.decode_seconds, env.raw_line_bytes) if n >= 1024]
    critic_decodes = [s for s, n in zip(env.decode_seconds, env.raw_line_bytes) if n < 1024]

    total_step = sum(step_seconds)
    total_critic = sum(critic_seconds)
    total_decode = sum(obs_decodes) + sum(critic_decodes)
    total_events = sum(env.events_seconds)

    print(f"decisions sampled            {len(step_seconds)}")
    print(f"PVZENV lines seen            {len(env.raw_line_bytes)}"
          f"  ({len(obs_lines)} observation, {len(critic_lines)} critic)")
    print()
    print("--- per-decision bytes on the wire ---")
    print(f"  observation line   mean {statistics.mean(obs_lines):9.1f} B"
          f"   min {min(obs_lines)}   max {max(obs_lines)}")
    print(f"  observation JSON   mean {statistics.mean(env.observation_bytes):9.1f} B"
          f"   (the 'observation' object alone, re-serialised)")
    print(f"  critic line        mean {statistics.mean(critic_lines):9.1f} B")
    print(f"  bytes per decision total  {statistics.mean(obs_lines) + statistics.mean(critic_lines):9.1f} B")
    print()
    print("--- per-decision time ---")
    print(f"  env.step (whole)      {statistics.mean(step_seconds) * 1e3:8.3f} ms  {_percentiles(step_seconds)}")
    print(f"  env.critic_inputs     {statistics.mean(critic_seconds) * 1e3:8.3f} ms  {_percentiles(critic_seconds)}")
    print(f"  json.loads (obs)      {statistics.mean(obs_decodes) * 1e3:8.3f} ms  {_percentiles(obs_decodes)}")
    print(f"  json.loads (critic)   {statistics.mean(critic_decodes) * 1e3:8.3f} ms  {_percentiles(critic_decodes)}")
    print(f"  _canonical_events     {statistics.mean(env.events_seconds) * 1e3:8.3f} ms"
          if env.events_seconds else "  _canonical_events     n/a")
    print()
    print("--- shares of one decision (step + critic) ---")
    decision = total_step + total_critic
    print(f"  env.step total        {total_step * 1e3:9.1f} ms  {total_step / decision * 100:5.1f}%")
    print(f"    of which decode     {sum(obs_decodes) * 1e3:9.1f} ms"
          f"  {sum(obs_decodes) / decision * 100:5.1f}%  of the decision")
    print(f"  critic_inputs total   {total_critic * 1e3:9.1f} ms  {total_critic / decision * 100:5.1f}%")
    print(f"  decode total          {total_decode * 1e3:9.1f} ms  {total_decode / decision * 100:5.1f}%")
    print(f"  events key sharing    {total_events * 1e3:9.1f} ms  {total_events / decision * 100:5.1f}%")

    print()
    print("--- is the CRITIC_INPUTS round trip redundant? ---")
    same = sum(1 for a, b in critic_timers if a == b)
    print(f"  critic_inputs['wave_timer'] == observation['wave_timer']:"
          f"  {same}/{len(critic_timers)} decisions")
    print("  -> the public OBS already carries wave_timer (LawnApp.cpp:1524), so that")
    print("     half of the second round trip is re-fetching a value the trainer holds.")

    if privileged:
        priv_lines = [n for n in env.raw_line_bytes if n > 20_000]
        print()
        print("--- PRIV: the payload training never reads ---")
        print(f"  env.privileged_state()  {statistics.mean(privileged) * 1e3:8.3f} ms"
              f"  {_percentiles(privileged)}")
        if priv_lines:
            print(f"  PRIV line on the wire    {max(priv_lines):9.1f} B")
        print("  reachable only via debug_replay=True (verify_env_equivalence) and three")
        print("  benchmark scripts; the PPO trainer never sends PRIV.")

    if args.field_budget:
        _field_budget(resource_dir, args.field_budget)


def _field_budget(resource_dir: Path, samples: int) -> None:
    """Price every top-level observation field, and mark which ones nothing reads."""
    import re

    reads: set[str] = set()
    for path in sorted(ROOT.glob("python/*.py")) + sorted(ROOT.glob("scripts/*.py")):
        if path.name.startswith("test_") or path.name == "pvz_env.py":
            continue
        blob = path.read_text(encoding="utf-8", errors="replace")
        reads |= set(re.findall(r'observation(?:\[|\.get\()\s*["\']([a-z_]+)["\']', blob))

    train, _heldout = trainer._task_family()
    task = trainer._curriculum_tasks(train["tasks"], "cap1")[0]
    spec = _task_spec(task, task["seeds"][0])

    def encoded(value: Any) -> int:
        return len(json.dumps(value, separators=(",", ":")))

    seen: list[dict[str, Any]] = []
    with PvZEnv(resource_dir) as env:
        observation, _ = env.reset(deck=task["deck"], task=spec)
        for _ in range(samples):
            seen.append(observation)
            observation, _, done, _, info = env.step({"type": "wait", "ticks": 30})
            if done:
                observation, _ = env.reset(deck=task["deck"], task=spec)

    keys = list(seen[len(seen) // 2])
    totals = {key: sum(encoded(s[key]) for s in seen) for key in keys}
    count = len(seen)
    full = sum(encoded(s) for s in seen) / count
    print()
    print(f"--- per-field JSON budget ({count} observations, mean {full:,.0f} B) ---")
    print(f"  {'field':26s} {'bytes':>8s} {'share':>7s}  read by trainer?")
    for key in sorted(keys, key=lambda k: -totals[k]):
        mean = totals[key] / count
        mark = "yes" if key in reads else "*** NO READER ***"
        print(f"  {key:26s} {mean:8.1f} {mean / full * 100:6.1f}%  {mark}")
    unread = sum(totals[k] for k in keys if k not in reads) / count
    print(f"  {'UNREAD TOTAL':26s} {unread:8.1f} {unread / full * 100:6.1f}%")

    # The two fields that dominate are not large because they carry data the trainer
    # does not want; they are large because of how that data is spelled.
    with PvZEnv(resource_dir) as env:
        observation, _ = env.reset(deck=task["deck"], task=spec)
        for _ in range(60):
            observation, _, done, _, info = env.step({"type": "wait", "ticks": 30})
            if done:
                break
    legal = observation["legal_actions"]["plants"]
    if legal:
        rows = [[a["packet"], a["col"], a["row"]] for a in legal]
        columns = {key: [a[key] for a in legal] for key in ("packet", "col", "row")}
        print()
        print(f"  legal_actions.plants: {len(legal)} entries, {encoded(legal)} B")
        print(f"    as [[packet,col,row],...]           {encoded(rows):6d} B"
              f"  ({encoded(rows) / encoded(legal) * 100:5.1f}%)")
        print(f"    as {{'packet':[...],'col':[...]}}      {encoded(columns):6d} B"
              f"  ({encoded(columns) / encoded(legal) * 100:5.1f}%)")
        print(f"    information content (3 bytes/row)  {3 * len(legal):6d} B")
    cells = observation["cells"]
    empty = [c for c in cells if not c["plant_types"] and not c["grid_item_types"]]
    print(f"  cells: {len(cells)} entries, {encoded(cells)} B"
          f"  ({len(empty)} empty, costing {encoded(empty)} B)")
    if empty:
        print(f"    one empty cell: {json.dumps(empty[0], separators=(',', ':'))}")
        print(f"    -> {len(json.dumps(empty[0], separators=(',', ':')))} B for 6 small integers")


if __name__ == "__main__":
    main()
