"""Where the training packs live and what makes two of them the same (task33).

A corpus index pins immutable per-label segments. Its identity is the whole row option
tuple; a segment excludes unrelated labels and weights. Both packer and trainer derive
the same index path, so neither needs an explicit pack path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


PACK_VERSION = 2  # label-local wait sampling; immutable segments + small corpus index


def archive_fingerprint(data: Path, labels, llm_demo_import=None, llm_demo_results=(1,)) -> str:
    """What the pack was built from: every archive's path and size, so a new or changed game invalidates it.

    mtime is deliberately not part of it: a copy that keeps the bytes (HF transfer, scp) keeps the size,
    and a training run must not care when the file was written.
    """
    data = Path(data)
    return fingerprint_paths(data, training_paths(data, labels, llm_demo_import, llm_demo_results))


def training_paths(data, labels, llm_demo_import=None, llm_demo_results=(1,)):
    from pvz_frame_store import frame_paths
    paths = [path for path in frame_paths(data) if path.parent.name in labels]
    if llm_demo_import is not None:
        from pvz_llm_demo_rows import demo_paths
        paths.extend(demo_paths(Path(data), Path(llm_demo_import), llm_demo_results))
    return sorted(paths, key=lambda path: str(path.parent))


def fingerprint_paths(data: Path, paths) -> str:
    lines = []
    for path in paths:
        line = f"{path.relative_to(data)}:{path.stat().st_size}"
        selection = training_rows(path)
        if selection is not None:
            line += ':training_rows=' + json.dumps(selection, separators=(',', ':'))
        lines.append(line)
    lines.sort()
    return hashlib.sha1("|".join(lines).encode()).hexdigest()[:16]


def training_rows(frame_path):
    """Optional exact supervision rows, preserving all real prefix frames for history.

    Ordinary archives retain their original fingerprint and sampling contract. Explicit
    rows are retained exactly, including wait corrections, with no wait downsampling.
    """
    path = Path(frame_path).parent / 'meta.json'
    if not path.exists():
        return None
    meta = json.loads(path.read_text())
    rows = meta.get('training_rows')
    if rows is None:
        return None
    if (not isinstance(rows, list) or any(type(i) is not int or i < 1 for i in rows)
            or rows != sorted(set(rows))):
        raise ValueError(f'{path}: training_rows must be sorted unique positive decision indices')
    return rows


def segment_options(options: dict, label: str, sources: str) -> dict:
    """Unrelated labels/weights never invalidate an immutable label segment."""
    base = {key: value for key, value in options.items()
            if key not in ('llm_demo_import', 'llm_demo_results')}
    return {**base, 'labels': [label], 'sources': sources,
            'label_weight': {label: options['label_weight'][label]}
            if label in options['label_weight'] else {}}


def segment_path(data: Path, options: dict) -> Path:
    return Path(data) / 'packs' / 'segments' / pack_path(data, options).name


def pack_options(labels, wait_keep: float = 1.0, obs: str = "v1", cells: bool = False,
                 aux_horizon: int = 0, packet_head: str = "slot", with_context: bool = False,
                 label_weight: dict[str, float] | None = None, sources: str | None = None,
                 llm_demo_import=None, llm_demo_results=(1,)) -> dict:
    """Everything that changes the rows or their values: a pack is valid for exactly one tuple."""
    options = {"version": PACK_VERSION, "sources": sources, "labels": sorted(labels), "wait_keep": round(float(wait_keep), 6),
            "obs": obs, "cells": bool(cells), "aux_horizon": int(aux_horizon), "packet_head": packet_head,
            "with_context": bool(with_context),
            "label_weight": {k: round(float(v), 6) for k, v in sorted((label_weight or {}).items())}}
    if llm_demo_import is not None:
        options.update(llm_demo_import=str(llm_demo_import), llm_demo_results=sorted(llm_demo_results))
    return options


def pack_path(data: Path, options: dict) -> Path:
    key = hashlib.sha1(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
    return Path(data) / "packs" / f"{key}.npz"




def pack_flags(options: dict) -> list[str]:
    """Packer argv for a row contract; also used in missing-pack diagnostics."""
    flags = ['--wait-keep', str(options['wait_keep']), '--obs', options['obs'],
             '--packet-head', options['packet_head'], '--aux-horizon', str(options['aux_horizon'])]
    if options['cells']:
        flags.append('--cells')
    if options['with_context']:
        flags.append('--context')
    if options['label_weight']:
        flags += ['--label-weight', ','.join(f'{k}={v}' for k,v in options['label_weight'].items())]
    if options.get('llm_demo_import'):
        flags += ['--llm-demo-import', options['llm_demo_import'], '--llm-demo-results',
                  ','.join(map(str, options['llm_demo_results']))]
    return flags
