"""Linux host CPU accounting for collection workers and their simulator processes."""
from pathlib import Path
import os
import resource
import subprocess
import time


def usage(who=resource.RUSAGE_SELF):
    value = resource.getrusage(who)
    return {"user_s": value.ru_utime, "system_s": value.ru_stime}


def process_usage(pid):
    # comm may contain spaces or parentheses; the remaining fields begin at state (field 3).
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    clock = os.sysconf("SC_CLK_TCK")
    return {"user_s": int(fields[11]) / clock, "system_s": int(fields[12]) / clock}


def difference(after, before):
    result = {key: after[key] - before[key] for key in ("user_s", "system_s")}
    result["total_s"] = sum(result.values())
    return result


def wait_with_cpu_usage(process, timeout=30):
    """Reap this direct child once, retaining wait4's own and reaped-descendant CPU time."""
    if process.returncode is not None:
        raise RuntimeError("child already reaped; exact CPU accounting is unavailable")
    until = time.monotonic() + timeout
    while True:
        pid, status, value = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return {"user_s": value.ru_utime, "system_s": value.ru_stime,
                    "total_s": value.ru_utime + value.ru_stime,
                    "returncode": process.returncode}
        if time.monotonic() >= until:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(0.01)
