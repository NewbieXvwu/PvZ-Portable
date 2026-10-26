"""The corpus acceptance check must fail closed on every mismatch."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import task78_corpus_check as check


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make(tmp_path, names):
    files = []
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        files.append(path)
    return files


def spec(tmp_path, branches):
    return dict(baseline_sha256="baseline",
                original_branch_paths=[str(p) for p in branches],
                decision_shards=2,
                required_episode_cache=["episodes.json", "boards.npy"],
                required_splits=dict(branch=dict(fit=1, validation=1)))


def descriptor(tmp_path, branches, shards, episode, **overrides):
    decision = tmp_path / "decision"
    decision.mkdir(exist_ok=True)
    document = dict(baseline_sha256="baseline",
                    branches=[dict(source_path=str(p), path=str(p), sha256=digest(p))
                              for p in branches],
                    decision_source_path=str(decision),
                    decision_shards=[dict(path=str(p), sha256=digest(p)) for p in shards],
                    audit_manifest=dict(path=str(episode[0]), sha256=digest(episode[0])),
                    episode_files=[dict(path=str(p), sha256=digest(p)) for p in episode])
    document.update(overrides)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(document))
    return path


def complete(tmp_path):
    branches = make(tmp_path, ["a.pt", "b.pt"])
    shards = make(tmp_path, ["s0.npz", "s1.npz"])
    episode = make(tmp_path, ["episodes.json", "boards.npy"])
    return branches, shards, episode, descriptor(tmp_path, branches, shards, episode)


def test_missing_descriptor_is_not_ready(tmp_path):
    report = check.check_descriptor(spec(tmp_path, []), tmp_path / "absent.json")
    assert report["ok"] is False and report["problems"] == ["descriptor missing"]


def test_complete_descriptor_is_ready(tmp_path):
    branches, shards, episode, path = complete(tmp_path)
    report = check.check_descriptor(spec(tmp_path, branches), path)
    assert report["ok"] is True, report["problems"]
    assert report["branches"] == 2 and report["decision_shards"] == 2


def test_hash_mismatch_and_missing_file_fail(tmp_path):
    branches, shards, episode, path = complete(tmp_path)
    branches[0].write_bytes(b"y")
    report = check.check_descriptor(spec(tmp_path, branches), path)
    assert any("sha256 mismatch" in item for item in report["problems"])
    branches[0].unlink()
    report = check.check_descriptor(spec(tmp_path, branches), path)
    assert any("missing branch file" in item for item in report["problems"])


def test_counts_baseline_and_episode_names_are_enforced(tmp_path):
    branches, shards, episode, path = complete(tmp_path)
    assert any("baseline_sha256" in item for item in check.check_descriptor(
        spec(tmp_path, branches), descriptor(tmp_path, branches, shards, episode,
                                             baseline_sha256="other"))["problems"])
    assert any("branch count" in item for item in check.check_descriptor(
        spec(tmp_path, branches), descriptor(tmp_path, branches[:1], shards, episode))["problems"])
    assert any("decision shard count" in item for item in check.check_descriptor(
        spec(tmp_path, branches), descriptor(tmp_path, branches, shards[:1], episode))["problems"])
    thin = make(tmp_path / "thin", ["episodes.json"])
    report = check.check_descriptor(spec(tmp_path, branches),
                                    descriptor(tmp_path, branches, shards, thin))
    assert any("episode cache lacks boards.npy" in item for item in report["problems"])


def test_paths_mode_reports_missing_original_locations(tmp_path):
    document = spec(tmp_path, [tmp_path / "absent.pt"])
    document["original_decision_directory"] = str(tmp_path / "nodir")
    document["original_external_audit"] = str(tmp_path / "absent.json")
    report = check.check_paths(document)
    assert report["ok"] is False and report["missing_branches"] == 1
    assert report["rows"][-1]["kind"] == "external_audit" and report["rows"][-1]["exists"] is False
