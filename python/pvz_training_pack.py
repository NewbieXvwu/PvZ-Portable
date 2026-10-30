"""Where the training packs live and what makes two of them the same (task33).

A corpus index pins immutable per-label segments. Its identity is the whole row option
tuple; a segment excludes unrelated labels and weights. Both packer and trainer derive
the same index path, so neither needs an explicit pack path.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


PACK_VERSION = 2  # label-local wait sampling; immutable segments + small corpus index


def training_frame_paths(data: Path, labels, token_input='compact'):
    from pvz_frame_store import frame_paths
    return [path for path in frame_paths(Path(data)) if path.parent.name in labels
            and (token_input != 'obs_v3' or (path.parent / 'tokens_v3.npz').is_file())]


def archive_fingerprint(data: Path, labels, token_input='compact') -> str:
    """What the pack was built from: every archive's path and size, so a new or changed game invalidates it.

    mtime is deliberately not part of it: a copy that keeps the bytes (HF transfer, scp) keeps the size,
    and a training run must not care when the file was written.
    """
    data = Path(data)
    return fingerprint_paths(data, training_frame_paths(data, labels, token_input), token_input)


def fingerprint_paths(data: Path, paths, token_input='compact') -> str:
    lines = []
    for path in paths:
        line = f"{path.relative_to(data)}:{path.stat().st_size}"
        selection = training_rows(path)
        if selection is not None:
            line += ':training_rows=' + json.dumps(selection, separators=(',', ':'))
        if token_input == 'obs_v3':
            import zipfile
            attachment = path.parent / 'tokens_v3.npz'
            if attachment.is_file():
                # Read ZIP metadata only; adding/changing an attachment invalidates rich packs.
                with zipfile.ZipFile(attachment) as archive:
                    members = [(m.filename, m.CRC, m.file_size) for m in archive.infolist()]
                line += ':obs_v3=' + json.dumps(members, separators=(',', ':'))
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
    result = {**options, 'labels': [label], 'sources': sources,
            'label_weight': {label: options['label_weight'][label]}
            if label in options['label_weight'] else {}}
    result.pop('label_keep', None)
    keep = label_keep_options([label], options.get('label_keep'))
    if keep:
        result['label_keep'] = keep
    return result


def label_keep_options(labels, rules):
    """Regexes match from the start of a label; overlapping rules use the lowest rate."""
    if not isinstance(rules, dict) and rules is not None:
        raise ValueError('label_keep: expected object')
    compiled = []
    for pattern, rate in (rules or {}).items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError('label_keep: expected nonempty regex')
        try:
            regex = re.compile(pattern)
        except re.error as error:
            raise ValueError(f'label_keep: invalid regex {pattern!r}') from error
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f'label_keep.{pattern}: expected fraction in 0..1')
        compiled.append((regex, rate))
    return {re.escape(label) + '$': rate for label in sorted(labels)
            if (rate := min((rate for regex, rate in compiled if regex.match(label)), default=1.)) < 1}


def keep_archive(data, path, rules):
    rate = min((rate for pattern, rate in (rules or {}).items() if re.match(pattern, path.parent.name)),
               default=1.)
    # Stable across corpus growth, rounds, copies and archive compression formats.
    key = path.parent.relative_to(data).as_posix().encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') < rate * 2**64


def segment_path(data: Path, options: dict) -> Path:
    return Path(data) / 'packs' / 'segments' / pack_path(data, options).name


def pack_options(labels, wait_keep: float = 1.0, obs: str = "v1", cells: bool = False,
                 aux_horizon: int = 0, packet_head: str = "slot", with_context: bool = False,
                 label_weight: dict[str, float] | None = None, sources: str | None = None,
                 label_keep: dict[str, float] | None = None, token_input: str = "compact") -> dict:
    """Everything that changes the rows or their values: a pack is valid for exactly one tuple."""
    keep = label_keep_options(labels, label_keep)
    return {**({"token_input": token_input} if token_input != "compact" else {}), "version": PACK_VERSION, "sources": sources, "labels": sorted(labels), "wait_keep": round(float(wait_keep), 6),
            **({'token_storage': 'sparse-block32-v1'} if obs == 'v3' else {}),
            "obs": obs, "cells": bool(cells), "aux_horizon": int(aux_horizon), "packet_head": packet_head,
            "with_context": bool(with_context), **({'label_keep': keep} if keep else {}),
            "label_weight": {k: round(float(v), 6) for k, v in sorted((label_weight or {}).items())}}


def pack_path(data: Path, options: dict) -> Path:
    key = hashlib.sha1(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
    return Path(data) / "packs" / f"{key}.npz"




def pack_flags(options: dict) -> list[str]:
    """Packer argv for a row contract; also used in missing-pack diagnostics."""
    flags = ['--wait-keep', str(options['wait_keep']), '--obs', options['obs'],
             '--packet-head', options['packet_head'], '--aux-horizon', str(options['aux_horizon'])]
    if options.get('token_input', 'compact') != 'compact':
        flags += ['--token-input', options['token_input']]
    if options['cells']:
        flags.append('--cells')
    if options['with_context']:
        flags.append('--context')
    if options['label_weight']:
        flags += ['--label-weight', ','.join(f'{k}={v}' for k,v in options['label_weight'].items())]
    if options.get('label_keep'):
        flags += ['--label-keep', json.dumps(options['label_keep'])]
    return flags
