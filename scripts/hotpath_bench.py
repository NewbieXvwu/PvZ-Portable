"""Hot-path measurements for the audit (task33). Read-only: plays games, writes nothing but stdout.

  python3 scripts/hotpath_bench.py protocol   # engine start, step and rollout round trips
  python3 scripts/hotpath_bench.py archive    # frame archive size / read cost
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_RESOURCE = Path(os.path.expanduser("~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))


def _rss(pid: int) -> int:
    out = os.popen(f"ps -o rss= -p {pid}").read().strip()
    return int(out or 0) * 1024


def load_exported_student(path):
    """The same load + C++ export search_play does per run (pvz_rollout_policy.load_student)."""
    from pvz_rollout_policy import export_student, load_student
    model = load_student(path)
    return model, export_student


def protocol(args) -> None:
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from pvz_env import TaskSpec

    level, deck = 7, (0, 1, 2, 3, 4, 5, 7)
    stats = {"request_bytes": 0, "response_bytes": 0, "commands": 0}
    original = EventWaitEnv._read_message

    def counted(self):
        before = stats["response_bytes"]
        response = original(self)
        return response

    EventWaitEnv._read_message = counted

    started = time.perf_counter()
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env._start()
    start_s = time.perf_counter() - started
    engine_pid = env._process.pid if env._process else None
    print(f"engine start: {start_s * 1000:.0f} ms, rss {_rss(engine_pid) / 1e6:.0f} MB, pid {engine_pid}")
    if args.rollout_policy:
        import tempfile
        keep = tempfile.TemporaryDirectory(prefix="pvz-bench-")
        exported = str(Path(keep.name) / "student.bin")
        started = time.perf_counter()
        model, export_student = load_exported_student(args.rollout_policy)
        export_student(model, exported)
        export_s = time.perf_counter() - started
        started = time.perf_counter()
        env.load_rollout_net(exported)
        print(f"student .pt -> .bin export {export_s * 1000:.0f} ms, load_rollout_net "
              f"{(time.perf_counter() - started) * 1000:.0f} ms")

    task = TaskSpec(level=level, seed=args.seed, playthrough=2, profile=profile_for_deck(deck, False))
    started = time.perf_counter()
    obs, _ = env.reset(deck=deck, task=task)
    print(f"reset: {(time.perf_counter() - started) * 1000:.0f} ms, rss {_rss(engine_pid) / 1e6:.0f} MB, "
          f"observation json {len(json.dumps(obs, separators=(',', ':'))) / 1024:.1f} KB")

    # step round trips: the student's action each decision
    from pvz_rollout_policy import RolloutNet
    times = []
    obs_sizes = []
    for _ in range(args.steps):
        started = time.perf_counter()
        obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
        times.append(time.perf_counter() - started)
        obs_sizes.append(len(json.dumps(obs, separators=(",", ":"))))
        if done:
            break
    print(f"step x{len(times)}: mean {statistics.mean(times) * 1000:.2f} ms "
          f"(median {statistics.median(times) * 1000:.2f}), response {statistics.mean(obs_sizes) / 1024:.1f} KB")

    # rollout round trips at several horizons: fixed cost vs per-tick cost
    env.reset(deck=deck, task=task)
    for horizon in args.horizons:
        samples = []
        sizes = []
        for _ in range(args.rollouts):
            started = time.perf_counter()
            leaf = env.rollout({"type": "wait", "ticks": 60}, obs["tick"] + horizon, sim_skip=args.sim_skip)
            samples.append(time.perf_counter() - started)
            sizes.append(len(json.dumps(leaf, separators=(",", ":"))))
        print(f"rollout horizon {horizon:5d} ticks: mean {statistics.mean(samples) * 1000:7.2f} ms, "
              f"min {min(samples) * 1000:7.2f}, leaf json {statistics.mean(sizes) / 1024:6.1f} KB, "
              f"rss {_rss(engine_pid) / 1e6:.0f} MB")
    env.close()


def archive(args) -> None:
    import numpy as np
    paths = sorted(Path(args.frames).glob("*/*/frames.npz"))
    if not paths:
        raise SystemExit(f"no frames.npz under {args.frames}")
    sample = paths[: args.files]
    started = time.perf_counter()
    rows = 0
    disk = 0
    raw = 0
    encoded = 0
    for path in sample:
        disk += path.stat().st_size
        with np.load(path, allow_pickle=False) as stored:
            rows += stored["boards"].shape[0]
            raw += stored["frames"].nbytes
            encoded += sum(stored[name].nbytes for name in stored.files if name not in ("frames", "encoder"))
    elapsed = time.perf_counter() - started
    total_disk = sum(path.stat().st_size for path in paths)
    print(f"{len(paths)} archives, {total_disk / 1e9:.2f} GB on disk, mean {total_disk / len(paths) / 1024:.0f} KB")
    print(f"read {len(sample)} files in {elapsed:.2f} s = {disk / 1e6 / elapsed:.0f} MB/s on disk, "
          f"{(raw + encoded) / 1e6 / elapsed:.0f} MB/s decompressed, {elapsed / len(sample) * 1000:.1f} ms a file")
    print(f"rows {rows}, {encoded / max(rows, 1):.0f} B encoded a row, raw frame json {raw / 1e6:.1f} MB "
          f"= {raw / (raw + encoded):.1%} of the decompressed payload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("protocol")
    p.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE)
    p.add_argument("--executable", default="/Users/newbiexvwu/PvZAgent/build/pvz-portable")
    p.add_argument("--seed", type=int, default=1740000)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--rollouts", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[250, 1000, 2500])
    p.add_argument("--sim-skip", type=int, default=61)
    p.add_argument("--rollout-policy", default="/Users/newbiexvwu/PvZAgent/artifacts/student/v32/model.pt")
    a = sub.add_parser("archive")
    a.add_argument("--frames", type=Path, required=True)
    a.add_argument("--files", type=int, default=600)
    args = parser.parse_args()
    {"protocol": protocol, "archive": archive}[args.command](args)


if __name__ == "__main__":
    main()
