"""Mechanical acceptance check for a restored v5 corpus (task78 item 1).

The preregistered merge needs the exact 442,601-row corpus plus a descriptor. This
check never writes to the corpus: it verifies the descriptor against the frozen
requirements and reports every missing file, hash mismatch and split disagreement, so
that "the corpus is back" can be decided mechanically instead of by eye.

Usage:
    python scripts/task78_corpus_check.py             # descriptor in this worktree
    python scripts/task78_corpus_check.py --paths     # only the original absolute paths
    python scripts/task78_corpus_check.py --descriptor /path/to/corpus.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "MEMORY/docs/task78-v5-corpus-requirements.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requirements():
    return json.loads(REQUIREMENTS.read_text())


def check_paths(spec):
    rows = []
    missing = 0
    for path in spec["original_branch_paths"]:
        present = Path(path).is_file()
        missing += 0 if present else 1
        rows.append(dict(kind="branch", path=path, exists=present))
    directory = Path(spec["original_decision_directory"])
    shards = 0
    if directory.is_dir():
        shards = sum(1 for item in directory.iterdir() if item.suffix in (".npz", ".pt"))
    rows.append(dict(kind="decision_directory", path=str(directory), exists=directory.is_dir(),
                     shards=shards, expected=spec["decision_shards"]))
    audit = Path(spec["original_external_audit"])
    rows.append(dict(kind="external_audit", path=str(audit), exists=audit.is_file()))
    ok = missing == 0 and shards == spec["decision_shards"] and audit.is_file()
    return dict(ok=ok, missing_branches=missing, rows=rows)


def verify_file(row, label, problems):
    source = Path(str(row.get("path", "")))
    if not source.is_file():
        problems.append("missing %s %s" % (label, source))
        return
    if row.get("sha256") and sha256(source) != row["sha256"]:
        problems.append("sha256 mismatch %s %s" % (label, source))


def check_descriptor(spec, path):
    """Validate a descriptor against the schema frozen in the requirements file."""
    if not path.is_file():
        return dict(ok=False, descriptor=str(path), exists=False, problems=["descriptor missing"])
    document = json.loads(path.read_text())
    problems = []
    if document.get("baseline_sha256") != spec["baseline_sha256"]:
        problems.append("baseline_sha256 differs from the adopted v5 checkpoint")
    branches = document.get("branches") or []
    if len(branches) != len(spec["original_branch_paths"]):
        problems.append("branch count %d != %d" % (len(branches), len(spec["original_branch_paths"])))
    for row in branches:
        verify_file(row, "branch file", problems)
    decision = Path(str(document.get("decision_source_path", "")))
    if not decision.is_dir():
        problems.append("decision_source_path missing %s" % decision)
    shards = document.get("decision_shards") or []
    if len(shards) != spec["decision_shards"]:
        problems.append("decision shard count %d != %d" % (len(shards), spec["decision_shards"]))
    for row in shards:
        verify_file(row, "decision shard", problems)
    audit = document.get("audit_manifest") or {}
    if not audit:
        problems.append("audit_manifest missing")
    else:
        verify_file(audit, "audit manifest", problems)
    files = document.get("episode_files") or []
    names = {Path(str(row.get("path", ""))).name for row in files}
    for name in spec["required_episode_cache"]:
        if name not in names:
            problems.append("episode cache lacks %s" % name)
    for row in files:
        verify_file(row, "episode file", problems)
    if "splits" in document and document["splits"] != spec["required_splits"]:
        problems.append("declared splits differ from the frozen split")
    return dict(ok=not problems, descriptor=str(path), exists=True,
                branches=len(branches), decision_shards=len(shards),
                episode_files=len(files), problems=problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path,
                        default=ROOT / "artifacts/task78/original_v5/corpus.json")
    parser.add_argument("--paths", action="store_true",
                        help="only check the original absolute paths, no descriptor")
    args = parser.parse_args()
    spec = load_requirements()
    report = dict(requirements=str(REQUIREMENTS))
    report.update(check_paths(spec) if args.paths else check_descriptor(spec, args.descriptor))
    report["ready"] = bool(report.get("ok"))
    print(json.dumps(report, indent=1))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
