"""Where the training packs live and what makes two of them the same (task33).

A pack holds the exact rows one training run needs, so its identity is the whole option
tuple that changes those rows. Both the packer and the trainer derive the same path from
the same tuple, which is why neither has to be told where a pack is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


PACK_VERSION = 1


def archive_fingerprint(data: Path, labels) -> str:
    """What the pack was built from: every archive's path and size, so a new or changed game invalidates it.

    mtime is deliberately not part of it: a copy that keeps the bytes (HF transfer, scp) keeps the size,
    and a training run must not care when the file was written.
    """
    from pvz_frame_store import frame_paths
    data = Path(data)
    lines = sorted(f"{path.relative_to(data)}:{path.stat().st_size}"
                   for path in frame_paths(data) if path.parent.name in labels)
    return hashlib.sha1("|".join(lines).encode()).hexdigest()[:16]


def pack_options(labels, wait_keep: float = 1.0, obs: str = "v1", cells: bool = False,
                 aux_horizon: int = 0, packet_head: str = "slot", with_context: bool = False,
                 label_weight: dict[str, float] | None = None, sources: str | None = None) -> dict:
    """Everything that changes the rows or their values: a pack is valid for exactly one tuple."""
    return {"version": PACK_VERSION, "sources": sources, "labels": sorted(labels), "wait_keep": round(float(wait_keep), 6),
            "obs": obs, "cells": bool(cells), "aux_horizon": int(aux_horizon), "packet_head": packet_head,
            "with_context": bool(with_context),
            "label_weight": {k: round(float(v), 6) for k, v in sorted((label_weight or {}).items())}}


def pack_path(data: Path, options: dict) -> Path:
    key = hashlib.sha1(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
    return Path(data) / "packs" / f"{key}.npz"


