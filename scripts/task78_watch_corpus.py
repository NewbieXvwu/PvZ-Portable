"""Resume the preregistered formal path automatically once the v5 corpus appears.

The corpus is the only missing input for task78 items 1 and 3. This watcher polls the
acceptance check at a low rate and, the moment it passes, runs the resumable formal
runner exactly once. It starts no game and no GPU work until then, and it never
bypasses the check: a partial or mis-hashed corpus keeps it waiting.

Usage:
    python scripts/task78_watch_corpus.py --interval 600 [--source-root DIR]
    python scripts/task78_watch_corpus.py --once      # single evaluation, for tests/ops
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/newbiexvwu/.venvs/ml/bin/python3"
SCRIPTS = ROOT / "scripts"
LOG = ROOT / "artifacts/task78/corpus_watch.log"
STOP = ROOT / "artifacts/task78/corpus_watch.stop"


DESCRIPTOR = ROOT / "artifacts/task78/original_v5/corpus.json"


def corpus_ready(source_root=None):
    """True only when the descriptor passes AND the shards reproduce v5's row counts.

    With --source-root, a corpus that has been dropped somewhere but has no descriptor
    yet is hashed into one first; the row check still has to pass before anything runs.
    """
    check = str(SCRIPTS / "task78_corpus_check.py")
    if source_root and not DESCRIPTOR.exists():
        build = subprocess.run([PY, check, "--write-descriptor", "--source-root", str(source_root)],
                               capture_output=True, text=True)
        if build.returncode != 0:
            return False, "descriptor build failed rc=%d (files incomplete or ambiguous)" % build.returncode
    result = subprocess.run([PY, check, "--verify-rows"], capture_output=True, text=True)
    if result.returncode == 0:
        return True, "descriptor accepted; v5 fit/validation row counts reproduced"
    paths = subprocess.run([PY, check, "--paths"], capture_output=True, text=True)
    if paths.returncode == 0:
        return False, "PATHS_READY descriptor missing or rows unverified"
    return False, "files and descriptor both missing"


def run_formal():
    with LOG.open("a") as log:
        log.write("FORMAL_START %s\n" % time.strftime("%Y-%m-%dT%H:%M:%S"))
        log.flush()
        result = subprocess.run([PY, str(SCRIPTS / "task78_formal_run.py")],
                                stdout=log, stderr=subprocess.STDOUT)
        log.write("FORMAL_DONE rc=%d %s\n" % (result.returncode, time.strftime("%Y-%m-%dT%H:%M:%S")))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--source-root", type=Path,
                        help="where a dropped corpus should be hashed into a descriptor from")
    args = parser.parse_args()
    while True:
        if STOP.exists():
            print("STOP file present; leaving the watch loop", flush=True)
            return 0
        ready, detail = corpus_ready(args.source_root)
        if ready:
            print("CORPUS_READY; starting the formal path", flush=True)
            return run_formal()
        print("CORPUS_MISSING %s %s" % (time.strftime("%Y-%m-%dT%H:%M:%S"), detail), flush=True)
        if args.once:
            print(detail)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
