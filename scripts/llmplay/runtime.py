"""平台无关的本地路径、环境指纹和离线自检；不访问网络。"""
from __future__ import annotations

import json
import os
import platform
from functools import lru_cache
from pathlib import Path

DEFAULT_SANDBOX_OUT = Path("/mnt/data/out")


def output_path(root):
    override = os.environ.get("LLMPLAY_OUT")
    if override:
        return Path(override).expanduser().resolve()
    candidate = DEFAULT_SANDBOX_OUT
    if candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK):
        return candidate.resolve()
    return (Path(root).parent / "out").resolve()


@lru_cache(maxsize=1)
def environment_fingerprint():
    model = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        available = len(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        available = os.cpu_count()
    return dict(cpu_model=model, cpu_count=os.cpu_count(),
                available_cpu_count=available,
                python_version=platform.python_version(), machine=platform.machine(), system=platform.system())


def paths():
    from .game import ROOT, _default_executable, _default_resource_dir
    from . import budget
    from .resources import app_data
    from .delivery import bundle_dir
    executable, resources = _default_executable(), _default_resource_dir()
    data = dict(package_dir=str(ROOT), resource_dir=str(resources), out_dir=str(output_path(ROOT)),
                bundle_dir=str(bundle_dir()),
                budget_file=str(budget.session().path),
                app_data_dir=str(app_data(executable, resources, budget.session())) if executable.is_file() else None)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data
