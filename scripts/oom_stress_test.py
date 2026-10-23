#!/usr/bin/env python3
"""task34: controlled memory-pressure test for the OOM guards (Linux/WSL only).

Two fake collection workers raise their ``oom_score_adj`` to 1000 and allocate until
``MemAvailable`` falls below 5 % of ``MemTotal``.  While the system is under pressure the
script times ``tmux new-session -d`` and ``bash -c true`` (the task book's stand-in for a
new ``wsl.exe`` session, which cannot be entered from inside WSL) and then checks that the
two fake workers -- not init, sshd, tmux, earlyoom or the caller -- were the victims.

Refuses to start unless ``--i-know`` is given, no training is running, and at least
12 GiB is available.  Collection from other agents is recorded, not refused: their
workers carry score 0 while the fake workers carry 1000, so the guards must pick the
fake workers.  It only ever signals its own two children (SIGKILL as a last resort if
the guards did not act); init, sshd, tmux and earlyoom are observed, never touched.

Usage:
  python3 scripts/oom_stress_test.py            # prints the plan and refuses
  python3 scripts/oom_stress_test.py --i-know   # run (main-control approval required)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_common import OOM_SCORE_WORKER, set_oom_score_adj  # noqa: E402

GIB = 1024 ** 3
TRAINING = ("train_rollout_policy.py", "train_value_net.py", "train_value_branches.py")
COLLECTION = ("search_play.py", "pvz-portable")
MIN_START_BYTES = 12 * GIB
PRESSURE_FRACTION = 0.05
ALLOCATE_CHUNK = 64 * 1024 * 1024
WORKER_SLEEP = 0.05
OBSERVE_TIMEOUT = 120.0
DEATH_TIMEOUT = 45.0
LATENCY_LIMIT = 2.0


def meminfo() -> tuple[int, int]:
    """(MemTotal, MemAvailable) in bytes."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, rest = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable"):
            values[name] = int(rest.split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def command_matches(command: str, needles: tuple[str, ...]) -> bool:
    """An argv token's basename is exactly one of the needles.

    Plain substring matching would count earlyoom itself as collection because its
    ``--prefer '(pvz-portable|search_play|python3)'`` contains both names.
    """
    return any(Path(token).name in needles for token in command.split())


def matching_processes(needles: tuple[str, ...]) -> list[tuple[int, str]]:
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if command_matches(command, needles):
            found.append((int(entry.name), command.strip()))
    return found


def worker_main(index: int) -> int:
    if not set_oom_score_adj(OOM_SCORE_WORKER):
        print(f"worker {index}: cannot write oom_score_adj", flush=True)
        return 2
    print(f"worker {index} ready pid={os.getpid()} score={OOM_SCORE_WORKER}", flush=True)
    chunks = []
    while True:
        try:
            block = bytearray(ALLOCATE_CHUNK)
            block[::4096] = b"\x01" * (ALLOCATE_CHUNK // 4096)   # touch every page
        except MemoryError:
            print(f"worker {index}: MemoryError after {len(chunks)} chunks", flush=True)
            return 1
        chunks.append(block)
        time.sleep(WORKER_SLEEP)


def timed(command: list[str], timeout: float = 10.0) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return dict(seconds=time.monotonic() - started, returncode=result.returncode,
                    stderr=result.stderr.strip()[-200:])
    except subprocess.TimeoutExpired:
        return dict(seconds=time.monotonic() - started, returncode="timeout", stderr="")


def probe_latency() -> dict:
    name = f"oomstress_probe_{os.getpid()}"
    tmux = timed(["tmux", "new-session", "-d", "-s", name, "sleep", "0.2"])
    subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True)
    shell = timed(["bash", "-c", "true"])
    return {"tmux_new_session": tmux, "bash_true": shell}


def oom_score(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/oom_score_adj").read_text().strip()
    except OSError:
        return "gone"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--i-know", action="store_true",
                        help="confirm main-control approval: this deliberately drives the box into OOM")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return worker_main(args.worker)

    plan = dict(
        fake_workers=2, worker_score=OOM_SCORE_WORKER, chunk_mib=ALLOCATE_CHUNK // 1024 ** 2,
        stops_below=f"{PRESSURE_FRACTION:.0%} MemAvailable", latency_limit_s=LATENCY_LIMIT,
        refused_unless="--i-know",
    )
    if not args.i_know:
        print(json.dumps(dict(refused=True, reason="--i-know not given", plan=plan), indent=1))
        return 2
    if not Path("/proc/self/oom_score_adj").exists():
        print(json.dumps(dict(refused=True, reason="oom_score_adj is unavailable (not Linux)", plan=plan)))
        return 2

    training = matching_processes(TRAINING)
    other_collection = matching_processes(COLLECTION)
    mem_total, available = meminfo()
    if training:
        print(json.dumps(dict(refused=True, reason="training is running", training=training), indent=1))
        return 2
    if available < MIN_START_BYTES:
        print(json.dumps(dict(refused=True, reason=f"only {available / GIB:.1f} GiB available; "
                                                  f"need {MIN_START_BYTES / GIB:.0f} GiB", plan=plan), indent=1))
        return 2

    tmux_pids = [int(pid) for pid in subprocess.run(["pgrep", "-x", "tmux"], capture_output=True,
                                                     text=True).stdout.split()]
    sshd_pids = [int(pid) for pid in subprocess.run(["pgrep", "-x", "sshd"], capture_output=True,
                                                     text=True).stdout.split()]
    earlyoom = subprocess.run(["systemctl", "is-active", "earlyoom"], capture_output=True, text=True).stdout.strip()
    watched = {"init": [1], "tmux": tmux_pids, "sshd": sshd_pids}

    print(f"starting 2 fake workers with oom_score_adj={OOM_SCORE_WORKER}; "
          f"MemTotal {mem_total / GIB:.1f} GiB, MemAvailable {available / GIB:.1f} GiB, earlyoom={earlyoom}, "
          f"other collection processes left running: {len(other_collection)}", flush=True)
    workers = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
               for i in range(2)]
    for worker in workers:
        deadline = time.monotonic() + 10
        while oom_score(worker.pid) != str(OOM_SCORE_WORKER) and time.monotonic() < deadline:
            time.sleep(0.05)
        if oom_score(worker.pid) != str(OOM_SCORE_WORKER):
            print(f"worker {worker.pid} did not take its score: {oom_score(worker.pid)}", flush=True)
            for other in workers:
                other.kill()
            return 1

    threshold = int(mem_total * PRESSURE_FRACTION)
    minimum = available
    pressure_at = None
    latency = None
    deadline = time.monotonic() + OBSERVE_TIMEOUT
    while time.monotonic() < deadline:
        _total, now = meminfo()
        minimum = min(minimum, now)
        if now < threshold:
            pressure_at = now
            latency = probe_latency()
            break
        if all(worker.poll() is not None for worker in workers):
            break
        time.sleep(0.05)

    deadlines = time.monotonic() + DEATH_TIMEOUT
    while time.monotonic() < deadlines and any(worker.poll() is None for worker in workers):
        time.sleep(0.2)
    forced = []
    for worker in workers:
        if worker.poll() is None:
            worker.kill()
            forced.append(worker.pid)
    results = []
    for i, worker in enumerate(workers):
        output = worker.communicate()[0].strip()
        results.append(dict(index=i, pid=worker.pid, returncode=worker.returncode,
                            killed_by_signal=worker.returncode is not None and worker.returncode < 0,
                            output=output))

    alive = {name: [pid for pid in pids if Path(f"/proc/{pid}").exists()] for name, pids in watched.items()}
    late_available = meminfo()[1]
    # A kill can land between two 50 ms polls; the per-poll minimum still proves the box was
    # pushed below the threshold, while latency is only meaningful if the poll caught it live.
    pressure_observed = minimum < threshold
    passed = (
        pressure_observed
        and latency is not None
        and latency["tmux_new_session"]["seconds"] < LATENCY_LIMIT
        and latency["bash_true"]["seconds"] < LATENCY_LIMIT
        and all(result["killed_by_signal"] for result in results)
        and not forced
        and all(alive[name] == pids for name, pids in watched.items())
    )
    print(json.dumps(dict(
        passed=passed, pressure_observed=pressure_observed, mem_total_gib=round(mem_total / GIB, 1),
        mem_available_start_gib=round(available / GIB, 1),
        minimum_available_gib=round(minimum / GIB, 2),
        minimum_available_percent=round(100 * minimum / mem_total, 2),
        pressure_at_gib=None if pressure_at is None else round(pressure_at / GIB, 2),
        latency=latency, workers=results, forced_kills=forced,
        watched=watched, watched_alive=alive, earlyoom=earlyoom,
        other_collection=[dict(pid=pid, command=command) for pid, command in other_collection],
        available_after_gib=round(late_available / GIB, 1),
    ), indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
