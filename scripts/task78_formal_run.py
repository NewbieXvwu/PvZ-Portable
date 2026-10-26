"""One resumable command for the preregistered formal path (task78 item 3).

Steps: corpus acceptance -> train the formal candidate on the original corpus ->
freeze the formal panel -> v5 arm -> v6 arm -> report. Each step is a separate
process; the panel tools are idempotent (they dedupe by seed and write once), so
re-running after an interruption resumes instead of duplicating work.

The corpus step is the gate: without a restored original corpus this exits before
starting any game or any GPU work, and says which files are missing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/newbiexvwu/.venvs/ml/bin/python3"
FORMAL = ROOT / "artifacts/task78/formal_v6"
LOG = ROOT / "artifacts/task78/formal_run.log"


def plan():
    scripts = ROOT / "scripts"
    return [
        ("corpus", [PY, str(scripts / "task78_corpus_check.py")]),
        ("train", [PY, str(scripts / "task78_train.py"), "--corpus", "original",
                   "--out", str(FORMAL), "--device", "cuda"]),
        ("freeze", [PY, str(scripts / "task78_formal_ab.py"), "freeze"]),
        ("v5", [PY, str(scripts / "task78_formal_ab.py"), "v5"]),
        ("v6", [PY, str(scripts / "task78_formal_ab.py"), "v6"]),
        ("report", [PY, str(scripts / "task78_formal_ab.py"), "report"]),
    ]


def trained():
    return (FORMAL / "model.pt").is_file()


def run_step(name, command, env=None):
    with LOG.open("a") as log:
        log.write("START %s %s\n" % (name, " ".join(command)))
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write("DONE %s rc=%d\n" % (name, result.returncode))
        log.flush()
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from", dest="start", default="corpus")
    args = parser.parse_args()
    steps = plan()
    names = [name for name, _ in steps]
    if args.start not in names:
        raise SystemExit("unknown step %s; choose one of %s" % (args.start, names))
    if args.dry_run:
        print(json.dumps(dict(steps=[dict(name=n, command=c) for n, c in steps],
                              trained=trained()), indent=1))
        return 0
    for name, command in steps[names.index(args.start):]:
        if name == "train" and trained():
            print("SKIP train (formal candidate already exists)")
            continue
        code = run_step(name, command)
        if code != 0:
            print("STOP %s rc=%d; see %s" % (name, code, LOG))
            return code
        print("OK %s" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
