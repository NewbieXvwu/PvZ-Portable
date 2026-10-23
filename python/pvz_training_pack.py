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


CHUNK_BYTES = 1 << 26


def mapped_dir(pack: Path) -> Path:
    """The mmap-friendly unpacking of a pack: one .npy per member, next to the archive."""
    return Path(str(pack) + ".d")


def _read_header(source):
    """Shape, order and dtype of one .npy stream, leaving the stream right before the payload."""
    import numpy as np
    version = np.lib.format.read_magic(source)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(source)
    return np.lib.format.read_array_header_2_0(source)


def _payload_sha(source) -> str:
    """SHA256 of the payload after the .npy header, streamed (a 2.4 GB member must not be read in one go)."""
    _read_header(source)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def unpack(pack: Path, dest: Path | None = None, check: bool = True) -> dict:
    """Decompress every zip member into its own .npy and verify payload identity.

    np.load(..., mmap_mode='r') cannot map zip members: reading them puts ~9 GB of arrays in anonymous
    memory, which defeats the 10 GiB MemAvailable floor. The .npy pages written here are file-backed and
    reclaimable, so the trainer keeps its footprint evictable.
    """
    import zipfile

    import numpy as np

    pack = Path(pack)
    dest = Path(dest) if dest is not None else mapped_dir(pack)
    dest.mkdir(parents=True, exist_ok=True)
    digests = {}
    with zipfile.ZipFile(pack) as archive:
        for name in archive.namelist():
            with archive.open(name) as source:
                shape, fortran, dtype = _read_header(source)
                if dtype.kind == "O":
                    raise ValueError(f"{name}: object members cannot be unpacked")
                count = int(np.prod(shape)) if shape else 1
                target = dest / name
                array = np.lib.format.open_memmap(target, mode="w+", dtype=dtype, shape=shape,
                                                  fortran_order=bool(fortran))
                if count * dtype.itemsize <= CHUNK_BYTES:
                    payload = source.read(count * dtype.itemsize)
                    if len(payload) != count * dtype.itemsize:
                        raise EOFError(f"{name}: truncated member")
                    array[...] = np.frombuffer(payload, dtype=dtype).reshape(shape, order="F" if fortran else "C")
                else:
                    flat = array.reshape(-1, order="F" if fortran else "C")
                    done, step = 0, CHUNK_BYTES // dtype.itemsize
                    while done < count:
                        width = min(step, count - done)
                        payload = source.read(width * dtype.itemsize)
                        if len(payload) != width * dtype.itemsize:
                            raise EOFError(f"{name}: truncated member")
                        flat[done:done + width] = np.frombuffer(payload, dtype=dtype)
                        done += width
                array.flush()
                del array
            with open(target, "rb") as source:
                payload = _payload_sha(source)
            if check:
                with archive.open(name) as source:
                    expected = _payload_sha(source)
                if expected != payload:
                    raise ValueError(f"{name}: unpacked payload differs from the pack")
            digests[name] = payload
    (dest / "manifest.json").write_text(json.dumps({"pack": pack.name, "members": digests}, indent=1) + "\n")
    return digests



