#!/usr/bin/env python3
"""task105 流水线的看守进程：崩溃后自动续跑，直到 pipeline_status 记为 complete。

用法（台式机，ml venv）：
  python3 scripts/task105_supervise.py <zip> [--max-restarts 50] [--sleep 60]

流水线本身可断点续跑（games.jsonl + 结果 json + 导入回执），所以这里只需要在
子进程非零退出后等一小会儿再拉起；已有流水线在跑时只等待。
不部署、不提交，只管把 adaptive → register → 结果整理跑完；不加复验尾链。
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task105'
STATUS = OUT / 'pipeline_status.json'
LOCK = OUT / 'pipeline.lock'


def pipeline_running() -> bool:
    """用 flock 探测流水线是否还活着；活着就等它退出。"""
    with LOCK.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def state() -> str | None:
    try:
        value = json.loads(STATUS.read_text()).get('state')
        if value == 'complete':
            # A legacy pipeline could exit zero despite failed individual scans.
            summary = json.loads((OUT / 'delivery_summary.json').read_text())
            if not summary.get('complete'):
                return 'incomplete'
        return value
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--max-restarts', type=int, default=50)
    parser.add_argument('--sleep', type=int, default=60)
    parser.add_argument('--skip-import', action='store_true',
                        help='拉起流水线时复用合并后的 scan_jobs.json，不重新导入')
    args = parser.parse_args()
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log = OUT / 'supervise.out'
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'supervise.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    restarts = 0
    while True:
        if pipeline_running():
            time.sleep(args.sleep)
            continue
        if state() == 'complete':
            print('pipeline complete', flush=True)
            return
        if restarts >= args.max_restarts:
            print(f'giving up after {restarts} restarts', flush=True)
            raise SystemExit(2)
        with log.open('a') as stream:
            stream.write(f'--- restart {restarts} at {time.strftime("%F %T")} ---\n')
            stream.flush()
            command = [sys.executable, str(ROOT / 'scripts/task105_pipeline.py'), str(args.input)]
            if args.skip_import:
                command.append('--skip-import')
            code = subprocess.run(command, cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        if code == 0 and state() == 'complete':
            print('pipeline complete', flush=True)
            return
        restarts += 1
        time.sleep(args.sleep)


if __name__ == '__main__':
    main()
