"""Record host/GPU load during a CUDA evaluation, without reading process arguments or environment.

The PID/state files locate jobs; live /proc identities determine whether the recorder keeps waiting.
Linux/WSL only. Outputs are local experiment evidence, not repository delivery artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import subprocess
import time


def processes() -> dict[int, dict]:
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
            status = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)
            result[int(path.name)] = {"ppid": int(stat[1]), "start": stat[19], "state": stat[0],
                "rss_kib": int(status.get("VmRSS", "0 kB").split()[0]),
                "swap_kib": int(status.get("VmSwap", "0 kB").split()[0]),
                "name": status["Name"].strip()}
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def gpu_query(fields: str, applications: bool = False) -> str:
    flag = "--query-compute-apps=" if applications else "--query-gpu="
    try:
        return subprocess.check_output(["nvidia-smi", flag + fields, "--format=csv,noheader,nounits"],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until-pid", nargs="+", type=int, required=True)
    parser.add_argument("--state", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--samples", type=int, default=0, help="0: continue while a located job/controller is live")
    args = parser.parse_args()
    if args.interval <= 0 or args.samples < 0:
        parser.error("interval must be positive and samples nonnegative")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    identities, samples = {}, 0
    with gzip.open(args.out, "at", encoding="utf-8") as output:
        while True:
            table = processes()
            roots, stages = set(args.until_pid), []
            for path in args.state:
                try:
                    state = json.loads(path.read_text())
                    roots.update(int(state[key]) for key in ("pid", "controller_pid") if key in state)
                    stages.append({"source": str(path), **state})
                except (OSError, json.JSONDecodeError):
                    continue
            for pid in roots:
                if pid in table and pid not in identities:
                    identities[pid] = table[pid]["start"]
            live = {pid for pid in roots if pid in table and table[pid]["start"] == identities.get(pid)
                    and table[pid]["state"] != "Z"}
            owned = set(live)
            while True:
                children = {pid for pid, data in table.items() if data["ppid"] in owned}
                if children <= owned:
                    break
                owned.update(children)
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            row = {"time": datetime.now().astimezone().isoformat(), "monotonic": time.monotonic(),
                "live_roots": sorted(live), "stages": stages,
                "host_loadavg": Path("/proc/loadavg").read_text().split()[:3],
                "host_available_kib": int(mem["MemAvailable"].split()[0]),
                "host_swap_used_kib": int(mem["SwapTotal"].split()[0]) - int(mem["SwapFree"].split()[0]),
                "host_cpu_ticks": {parts[0]: list(map(int, parts[1:]))
                    for line in Path("/proc/stat").read_text().splitlines()
                    if (parts := line.split()) and parts[0].startswith("cpu")},
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "evaluation_rss_kib": sum(table[pid]["rss_kib"] for pid in owned),
                "evaluation_swap_kib": sum(table[pid]["swap_kib"] for pid in owned),
                "evaluation_processes": {str(pid): table[pid] for pid in sorted(owned)},
                "gpu_util_memory_clock_power": gpu_query("utilization.gpu,memory.used,clocks.current.sm,power.draw"),
                "gpu_compute_processes": gpu_query("pid,process_name,used_gpu_memory", applications=True)}
            output.write(json.dumps(row) + "\n")
            output.flush()
            samples += 1
            if not live or (args.samples and samples >= args.samples):
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
