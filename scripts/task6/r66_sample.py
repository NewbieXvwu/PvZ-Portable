#!/usr/bin/env python3
"""Sample training/collection CPU, GPU use and memory for the contention test."""
import json, os, subprocess, sys, time
from pathlib import Path

TRAIN = int(sys.argv[1]); OUT = Path(sys.argv[2]); SECONDS = float(sys.argv[3])
FIELDS = ("pid", "comm", "state", "ppid")

def proc_cpu(pid):
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None

def tree_cpu(pid):
    total, stack = 0.0, [pid]
    while stack:
        p = stack.pop()
        c = proc_cpu(p)
        if c is not None: total += c
        try:
            for line in Path(f"/proc/{p}/task/{p}/children").read_text().split():
                stack.append(int(line))
        except Exception:
            pass
    return total

def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        util, used = out.split(",")
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return {"util": int(util), "used_mib": int(used),
                "apps": [a.split(", ")[0] for a in apps if a]}
    except Exception as e:
        return {"error": str(e)}

start = time.time()
with OUT.open("a") as fh:
    while time.time() - start < SECONDS:
        row = {"t": round(time.time(), 3), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "train_cpu_s": tree_cpu(TRAIN), "load": os.getloadavg()[0],
               "avail_mib": int([l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable")][0].split()[1]) // 1024,
               "gpu": gpu()}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        time.sleep(5)
