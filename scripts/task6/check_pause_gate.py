#!/usr/bin/env python3
"""Touch the GPU pause file during a real game and verify an idle server at the boundary.

Runs four mixed collection games (one CPU worker and one GPU worker), writes frames
under --out, and never signals unrelated jobs. The output directory must be new.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from pvz_cuda_rollout import CudaRolloutClient


def descendants(pid):
    found, todo = [], [pid]
    while todo:
        current = todo.pop()
        found.append(current)
        try:
            todo.extend(map(int, Path(f"/proc/{current}/task/{current}/children").read_text().split()))
        except FileNotFoundError:
            pass
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--seed-offset", type=int, default=15000)
    p.add_argument("--hold-seconds", type=float, default=25)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pause = out / "training.pause"
    run = out / "mixed"
    cmd = [str(ROOT / "scripts/mixed_dagger_round.sh"), "--student", str(args.student.expanduser().resolve()),
           "--task", "day_l7_z17_train", "--seed-offset", str(args.seed_offset), "--games", "4",
           "--frames", str(out / "frames"), "--frames-label", "pausecheck", "--out", str(run),
           "--cpu-workers", "1", "--gpu-workers", "1"]
    env = {**os.environ, "PVZ_GPU_PAUSE_FILE": str(pause), "PVZ_CUDA_POLL_US": "100"}
    env.pop("PVZ_CUDA_DRAIN_WAITS", None)
    (out / "command.json").write_text(json.dumps({"command": cmd, "pause_file": str(pause)}, indent=2) + "\n")
    events, status_rows = [], []
    started = time.monotonic()
    def event(name, **values):
        row = {"event": name, "time": datetime.datetime.now().astimezone().isoformat(),
               "elapsed_s": time.monotonic() - started, **values}
        events.append(row)
        (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    client, paused_pids = None, set()
    touched, boundary, released, resumed = False, None, False, False
    initial_status = None
    peak_rss = peak_swap = 0
    minimum_available = 10**12
    with (out / "driver.txt").open("w") as log, (out / "resources.jsonl").open("w") as resources:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                assert time.monotonic() - started < 1800, "acceptance timed out"
                available = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines()
                                     if l.startswith("MemAvailable:")))
                minimum_available = min(minimum_available, available)
                owned = descendants(proc.pid)
                rss = swap = 0
                for pid in owned:
                    try:
                        text = Path(f"/proc/{pid}/status").read_text()
                        for label in ("VmRSS", "VmSwap"):
                            m = re.search(rf"^{label}:\s+(\d+)", text, re.M)
                            if m:
                                if label == "VmRSS": rss += int(m[1])
                                else: swap += int(m[1])
                    except FileNotFoundError:
                        pass
                peak_rss, peak_swap = max(peak_rss, rss), max(peak_swap, swap)
                if available < 2 * 1024 * 1024:
                    for pid in reversed(owned):
                        if pid not in paused_pids:
                            try:
                                os.kill(pid, signal.SIGSTOP)
                                paused_pids.add(pid)
                            except ProcessLookupError:
                                pass
                elif available >= 3 * 1024 * 1024 and paused_pids:
                    for pid in paused_pids:
                        try: os.kill(pid, signal.SIGCONT)
                        except ProcessLookupError: pass
                    paused_pids.clear()
                row = {"elapsed_s": time.monotonic() - started, "available_kib": available,
                       "rss_kib": rss, "swap_kib": swap, "memory_paused": bool(paused_pids)}
                resources.write(json.dumps(row) + "\n"); resources.flush()
                if paused_pids:
                    time.sleep(2)
                    continue
                gpu_log = run / "logs/gpu.log"
                text = gpu_log.read_text() if gpu_log.exists() else ""
                if client is None and not resumed:
                    socket = re.search(r"cuda rollout server: (.+), \d+ lanes", text)
                    if socket:
                        client = CudaRolloutClient(socket[1], timeout=600)
                        client.connect()
                if client is not None:
                    status = client.status()
                    status_rows.append({**row, **status})
                    if status["waves"] > 0 and not touched:
                        pause.touch()
                        touched = True
                        event("touch_during_game", status=status)
                    if touched and boundary is None and "GPU collection paused before seed" in text:
                        boundary = time.monotonic()
                        initial_status = status
                        event("game_boundary_paused", status=status)
                    if boundary is not None and not released:
                        assert status == initial_status, "collection continued while paused"
                        if time.monotonic() - boundary >= args.hold_seconds:
                            event("idle_verified", status=status, idle_seconds=time.monotonic() - boundary)
                            pause.unlink()
                            released = True
                            event("released")
                    elif released and status["waves"] > initial_status["waves"]:
                        resumed = True
                        event("kernels_resumed", status=status)
                        client.close()
                        client = None
                        # All remaining games now finish normally; no observer keeps the service alive.
                time.sleep(2)
            assert proc.returncode == 0, proc.returncode
            assert touched and released and resumed and boundary is not None
            games = [json.loads(l) for l in (run / "games.jsonl").read_text().splitlines()]
            assert len(games) == 4 and len({r["seed"] for r in games}) == 4
            assert len(list((out / "frames").glob("seed_*/pausecheck_fast_search/frames.npz"))) == 4
            event("complete", games=4, returncode=proc.returncode,
                  peak_rss_kib=peak_rss, peak_swap_kib=peak_swap, minimum_available_kib=minimum_available)
        finally:
            pause.unlink(missing_ok=True)
            if client is not None:
                client.close()
            if proc.poll() is None:
                for pid in reversed(descendants(proc.pid)):
                    try:
                        os.kill(pid, signal.SIGCONT)
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                proc.wait(timeout=30)
            (out / "server_status.json").write_text(json.dumps(status_rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
