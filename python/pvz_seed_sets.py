"""Schema check for the frozen development/final-test seed set files.

The files themselves are still read by ``check_task_manifests`` when it records the
held-out manifest, which takes the fields it needs directly.  This module is the one
place that validates their schema, and ``test_training_semantics`` pins that check.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_seed_set(path: Path, level: int = 7, expected_role: str | None = None) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("level") != level or data.get("playthrough") != 2:
        raise ValueError("frozen seed file must use schema 1 for Adventure-II and playthrough 2")
    if expected_role is not None and data.get("role") != expected_role:
        raise ValueError(f"expected frozen seed role {expected_role!r}, got {data.get('role')!r}")
    first = int(data["first_seed"])
    count = int(data["count"])
    seeds = list(range(first, first + count))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("frozen seed file must define unique non-negative seeds")
    return seeds
