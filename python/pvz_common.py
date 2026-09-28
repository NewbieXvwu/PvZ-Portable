"""Dependency-free constants and helpers shared by every layer.

This module deliberately imports nothing from the rest of the project: it is the
bottom of the dependency graph, so ``pvz_env`` (which must not import the training
modules) and every training entry point can share one definition of the protocol
version, the observation/task schema versions, the training seed, and file hashing.

Keeping the versions here is what stops the observation/task version from being a
literal ``2`` copy-pasted across four files.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ENV_PROTOCOL_VERSION = 3
REPLAY_FORMAT_VERSION = 5
OBSERVATION_VERSION = 2
TASK_VERSION = 2

# Seeds the training entry points must pass to ``random.seed`` / ``torch.manual_seed``.
TRAINING_SEED = 17

# Search, imitation and PPO all emit values in this closed interval.
VALUE_RANGE = (-1.0, 1.0)

# Generated files that must not make a recorded source tree look dirty.
_GENERATED_PREFIXES = ("artifacts/",)
_GENERATED_NAMES = {"experiment_manifest.json", "working_tree.patch"}
_GENERATED_SUFFIXES = (".jsonl", ".jsonl.gz")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Hex sha256 of an in-memory buffer.

    Every module that hashes a buffer goes through here, so no call site needs to
    import ``hashlib`` itself (and none can forget to).
    """
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: Any) -> str:
    """Hex sha256 of a canonical JSON encoding of *value*.

    Canonical means sorted keys, compact separators and UTF-8: two structurally
    equal values always produce the same digest, and no call site re-derives the
    encoding by hand.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def is_generated_artifact(path: str) -> bool:
    """True for replay/manifest/patch outputs that are not part of the source revision."""
    return (
        path.startswith(_GENERATED_PREFIXES)
        or Path(path).name in _GENERATED_NAMES
        or path.endswith(_GENERATED_SUFFIXES)
    )


def git_metadata(root: Path) -> tuple[str | None, bool | None]:
    """Return ``(HEAD revision, dirty)`` for *root*.

    The dirty flag ignores generated artifacts so that every caller records the same
    value for the same working tree.
    """
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout
        dirty = any(
            not is_generated_artifact(line[3:].strip().split(" -> ")[-1])
            for line in status.splitlines()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None
