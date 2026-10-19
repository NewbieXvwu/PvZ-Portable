#!/usr/bin/env python3
"""Measure 4+0, 2+2, 3+1, 4+0 mixed collection on the same seeds under observed load.

All output directories must be new. At most four games run concurrently. This
observes background jobs and never pauses them; only its own tree has a 2 GiB
available-memory pause guard. Shared-load results are local comparisons.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from check_pause_gate import descendants

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed-offset", type=int, default=15000)
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--background-pids", type=int, nargs="*", default=[])
    p.add_argument("--mixes", nargs="+", choices=("4:0", "2:2", "3:1"),
                   default=["4:0", "2:2", "3:1", "4:0"])
    p.add_argument("--cuda-binary", type=Path, default=ROOT / "build/rollout_cuda_small")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PVZ_CUDA_POLL_US": "100"}
    for name in ("PVZ_GPU_PAUSE_FILE", "PVZ_CUDA_DRAIN_WAITS"):
        env.pop(name, None)
    student = args.student.expanduser().resolve()
    provenance = {"time": datetime.datetime.now().astimezone().isoformat(),
                  "student": str(student), "student_sha256": hashlib.sha256(student.read_bytes()).hexdigest(),
                  "background_pids": args.background_pids, "games": args.games, "seed_offset": args.seed_offset,
                  "cuda_binary": str(args.cuda_binary.resolve()),
                  "cuda_binary_sha256": hashlib.sha256(args.cuda_binary.read_bytes()).hexdigest()}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results = []
    for index, mix in enumerate(args.mixes):
        cpu, gpu = map(int, mix.split(":"))
        name = f"mix{index}_cpu{cpu}_gpu{gpu}"
        directory = out / name
        cmd = [str(ROOT / "scripts/mixed_dagger_round.sh"), "--student", str(student),
               "--task", "day_l7_z17_train", "--seed-offset", str(args.seed_offset), "--games", str(args.games),
               "--frames", str(directory / "frames"), "--frames-label", "mixcompare", "--out", str(directory),
               "--cpu-workers", str(cpu), "--gpu-workers", str(gpu),
               "--cuda-binary", str(args.cuda_binary.resolve())]
        print(json.dumps({"event": "start", "name": name, "time": datetime.datetime.now().astimezone().isoformat(),
                          "command": cmd}), flush=True)
        directory.mkdir()
        minimum_available, peak_rss, peak_swap = 10**12, 0, 0
        background_continuous = True
        stopped = set()
        started, pauses, samples = time.monotonic(), [], []
        with (directory / "driver.txt").open("w") as log, (directory / "resources.jsonl").open("w") as resource_log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    available = int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines()
                                         if l.startswith("MemAvailable:")))
                    minimum_available = min(available, minimum_available)
                    pids = descendants(proc.pid)
                    rss = swap = 0
                    for pid in pids:
                        try:
                            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
                            rss += sum(int(l.split()[1]) for l in lines if l.startswith("VmRSS:"))
                            swap += sum(int(l.split()[1]) for l in lines if l.startswith("VmSwap:"))
                        except FileNotFoundError:
                            pass
                    peak_rss, peak_swap = max(peak_rss, rss), max(peak_swap, swap)
                    if available < 2 * 1024 * 1024 and not stopped:
                        for pid in reversed(pids):
                            try: os.kill(pid, signal.SIGSTOP); stopped.add(pid)
                            except ProcessLookupError: pass
                        pauses.append({"stop_s": time.monotonic() - started})
                    elif available >= 3 * 1024 * 1024 and stopped:
                        for pid in stopped:
                            try: os.kill(pid, signal.SIGCONT)
                            except ProcessLookupError: pass
                        stopped.clear()
                        pauses[-1]["resume_s"] = time.monotonic() - started
                    background_alive = [pid for pid in args.background_pids if Path(f"/proc/{pid}").exists()]
                    background_continuous &= background_alive == args.background_pids
                    row = {"elapsed_s": time.monotonic() - started, "available_kib": available,
                           "rss_kib": rss, "swap_kib": swap, "background_alive": background_alive,
                           "load": os.getloadavg(), "memory_paused": bool(stopped)}
                    if len(samples) % 5 == 0:
                        metrics = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                                  "--format=csv,noheader,nounits"],
                                                 capture_output=True, text=True, timeout=15)
                        row["gpu_util_memory_mib"] = metrics.stdout.strip()
                    samples.append(row)
                    resource_log.write(json.dumps(row) + "\n"); resource_log.flush()
                    time.sleep(2)
            finally:
                if proc.poll() is None:
                    for pid in reversed(descendants(proc.pid)):
                        try: os.kill(pid, signal.SIGCONT); os.kill(pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                    proc.wait(timeout=30)
        assert proc.returncode == 0, (name, proc.returncode)
        if gpu:
            assert "CUDA host wait: sleep 100 us" in (directory / ".parts/gpu/cuda_server.log").read_text(), \
                "binary did not actually enable sleep waiting"
        config = json.loads((directory / "config.json").read_text())
        games = [json.loads(l) for l in (directory / "games.jsonl").read_text().splitlines()]
        assert len(games) == args.games and len({r["seed"] for r in games}) == args.games
        row = {"name": name, "command": cmd, "time": datetime.datetime.now().astimezone().isoformat(),
               "cpu_workers": cpu, "gpu_workers": gpu, "games": args.games,
               "process_wall_s": time.monotonic() - started, "collection_wall_s": config["elapsed_wall_s"],
               "games_per_hour": args.games * 3600 / config["elapsed_wall_s"],
               "host_cpu_s_per_game": config["host_cpu_s_per_game"],
               "peak_rss_kib": peak_rss, "peak_swap_kib": peak_swap, "minimum_available_kib": minimum_available,
               "pauses": pauses, "background_continuous": background_continuous,
               "outcomes": [[r["seed"], r["won"], r["decisions"]] for r in sorted(games, key=lambda g: g["seed"])]}
        results.append(row)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
