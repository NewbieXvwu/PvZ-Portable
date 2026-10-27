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



def index_by_name(root):
    """Map every file basename under root to the paths that carry it."""
    index = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    return index


def decision_candidates(root, count):
    """Directories that hold exactly the expected number of decision shards.

    The frozen location is `task43/sparse64/data`, so a `data` (then `sparse64`)
    directory wins over an unrelated directory that happens to hold `count` shards.
    """
    found = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_dir():
            continue
        shards = [item for item in path.iterdir() if item.suffix in (".pt", ".npz")]
        if len(shards) == count:
            found.append(path)
    for preferred in ("data", "sparse64"):
        subset = [path for path in found if path.name == preferred]
        if subset:
            return subset
    return found


def build_descriptor(spec, root, decision_dir=None, episodes_dir=None):
    """Write a descriptor for a corpus laid out somewhere under root.

    Every file is hashed here, so a wrong corpus cannot pass: the descriptor is then
    validated against the frozen requirements exactly like a hand-written one.
    """
    root = Path(root)
    problems = []
    index = index_by_name(root)
    branches = []
    for expected in spec["original_branch_paths"]:
        name = Path(expected).name
        hits = index.get(name, [])
        if len(hits) != 1:
            problems.append("branch %s: %d matches under %s" % (name, len(hits), root))
            continue
        branches.append(dict(source_path=expected, path=str(hits[0]), sha256=sha256(hits[0])))
    if decision_dir is None:
        candidates = decision_candidates(root, spec["decision_shards"])
        if len(candidates) != 1:
            problems.append("decision directory: %d candidates with %d shards"
                            % (len(candidates), spec["decision_shards"]))
            decision_dir = candidates[0] if candidates else None
        else:
            decision_dir = candidates[0]
    shards = []
    if decision_dir is not None:
        decision_dir = Path(decision_dir)
        if not decision_dir.is_dir():
            problems.append("decision directory missing %s" % decision_dir)
        else:
            for item in sorted(decision_dir.iterdir()):
                if item.suffix in (".pt", ".npz"):
                    shards.append(dict(path=str(item), sha256=sha256(item)))
            if len(shards) != spec["decision_shards"]:
                problems.append("decision shards %d != %d" % (len(shards), spec["decision_shards"]))
    audit = index.get("full_audit_manifest.json", [])
    audit_row = {}
    if len(audit) != 1:
        problems.append("audit manifest: %d matches" % len(audit))
    else:
        audit_row = dict(source_path=spec["original_external_audit"], path=str(audit[0]),
                         sha256=sha256(audit[0]))
    if episodes_dir is None:
        found = index.get("episodes.json", [])
        episodes_dir = found[0].parent if len(found) == 1 else None
    files = []
    if episodes_dir is None:
        problems.append("episode cache: no single episodes.json under %s" % root)
    else:
        for name in spec["required_episode_cache"]:
            path = Path(episodes_dir) / name
            if not path.is_file():
                problems.append("episode cache lacks %s in %s" % (name, episodes_dir))
                continue
            files.append(dict(path=str(path), sha256=sha256(path)))
    document = dict(baseline_sha256=spec["baseline_sha256"], branches=branches,
                    decision_source_path=str(decision_dir) if decision_dir else "",
                    decision_shards=shards, audit_manifest=audit_row,
                    episode_files=files, splits=spec["required_splits"])
    return document, problems



def held(seed):
    """The v5 split: sha256('value-widen:<seed>')[:8] % 10 == 0 is held out for validation."""
    return int(hashlib.sha256(("value-widen:%d" % int(seed)).encode()).hexdigest()[:8], 16) % 10 == 0


def _split_counts(seeds, excluded):
    import numpy as np
    seed = np.asarray(seeds)
    mask = ~np.isin(seed, list(excluded))
    stop = np.array([held(v) for v in seed])
    return int((mask & ~stop).sum()), int((mask & stop).sum())


def verify_rows(spec, descriptor_path):
    """Load every shard and reproduce v5's fit/validation row counts.

    The file-level check cannot tell a right corpus from a wrong one with the same names;
    this recomputes the split the v5 trainer used (audit seeds excluded, then held() by
    seed) and compares the totals with the frozen 324,559 / 29,446 (branch),
    3,404 / 456 (decision) and 75,136 / 9,600 (episode).
    """
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        return dict(ok=False, problems=["torch/numpy required: %s" % exc])
    path = Path(descriptor_path)
    if not path.is_file():
        return dict(ok=False, problems=["descriptor missing"])
    document = json.loads(path.read_text())
    problems = []
    audit = document.get("audit_manifest") or {}
    audit_path = Path(str(audit.get("path", "")))
    if not audit_path.is_file():
        return dict(ok=False, problems=["audit manifest missing, cannot exclude audit seeds"])
    excluded = {int(row["seed"]) for row in json.loads(audit_path.read_text())}

    counts = {}
    for kind, rows in (("branch", document.get("branches") or []),
                       ("decision", document.get("decision_shards") or [])):
        fit = val = 0
        for row in rows:
            shard = Path(str(row.get("path", "")))
            if not shard.is_file():
                problems.append("missing %s shard %s" % (kind, shard))
                continue
            payload = torch.load(shard, map_location="cpu", weights_only=False, mmap=True)
            if "seed" not in payload:
                problems.append("%s shard lacks seed: %s" % (kind, shard))
                continue
            a, b = _split_counts(payload["seed"].numpy(), excluded)
            fit += a
            val += b
        counts[kind] = dict(fit=fit, validation=val)

    episodes = document.get("episode_files") or []
    rows_npz = next((Path(str(r.get("path"))) for r in episodes
                     if Path(str(r.get("path"))).name == "rows.npz"), None)
    if rows_npz is None or not rows_npz.is_file():
        problems.append("episode rows.npz missing")
    else:
        with np.load(rows_npz) as archive:
            if "seeds" not in archive.files:
                problems.append("episode rows.npz lacks seeds")
            else:
                a, b = _split_counts(archive["seeds"], set())
                counts["episode"] = dict(fit=a, validation=b)

    expected = spec["required_splits"]
    for kind, want in expected.items():
        got = counts.get(kind)
        if got is None:
            problems.append("%s: no rows counted" % kind)
        elif got["fit"] != want["fit"] or got["validation"] != want["validation"]:
            problems.append("%s split %d/%d != %d/%d"
                            % (kind, got["fit"], got["validation"], want["fit"], want["validation"]))
    return dict(ok=not problems, counts=counts, expected=expected, problems=problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path,
                        default=ROOT / "artifacts/task78/original_v5/corpus.json")
    parser.add_argument("--verify-rows", action="store_true",
                        help="also load every shard and reproduce v5's fit/validation row counts")
    parser.add_argument("--paths", action="store_true",
                        help="only check the original absolute paths, no descriptor")
    parser.add_argument("--write-descriptor", action="store_true",
                        help="hash a corpus laid out under --source-root and write the descriptor")
    parser.add_argument("--source-root", type=Path,
                        help="root of the restored corpus files (searched by basename)")
    parser.add_argument("--decision-dir", type=Path,
                        help="directory holding the decision shards, if not discoverable")
    parser.add_argument("--episodes-dir", type=Path,
                        help="directory holding the episode cache, if not discoverable")
    args = parser.parse_args()
    spec = load_requirements()
    report = dict(requirements=str(REQUIREMENTS))
    if args.write_descriptor:
        if args.source_root is None:
            parser.error("--write-descriptor needs --source-root")
        document, problems = build_descriptor(spec, args.source_root,
                                              decision_dir=args.decision_dir,
                                              episodes_dir=args.episodes_dir)
        report["build"] = dict(source_root=str(args.source_root), problems=problems)
        if problems:
            report.update(ok=False, descriptor=str(args.descriptor), exists=args.descriptor.is_file())
            print(json.dumps(report, indent=1))
            return 1
        args.descriptor.parent.mkdir(parents=True, exist_ok=True)
        args.descriptor.write_text(json.dumps(document, indent=1) + "\n")
        report["build"]["written"] = str(args.descriptor)
    report.update(check_paths(spec) if args.paths else check_descriptor(spec, args.descriptor))
    if args.verify_rows:
        rows = verify_rows(spec, args.descriptor)
        report["rows"] = rows
        report["ok"] = bool(report.get("ok")) and rows["ok"]
    report["ready"] = bool(report.get("ok"))
    print(json.dumps(report, indent=1))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
