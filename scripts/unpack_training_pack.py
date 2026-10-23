"""Unpack a training pack into one .npy per member, so the trainer can map it (task36).

A pack is a zip, and np.load(..., mmap_mode='r') ignores the mapping for zip members: the trainer used to
pull the whole corpus into anonymous memory (8.8 GB), which the memory guard could pause but never get back
under the 10 GiB MemAvailable floor. The .npy files written here are mapped file pages, so they stay evictable.

Every member's payload is hashed against the pack before the manifest is written; a pack whose directory is
missing or incomplete is simply read the old way.

  python3 scripts/unpack_training_pack.py artifacts/task27/training_frames/packs/<key>.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_training_pack import mapped_dir, unpack  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pack", type=Path)
    p.add_argument("--dest", type=Path, default=None, help="default: <pack>.d")
    p.add_argument("--no-check", action="store_true", help="skip the payload hash comparison")
    args = p.parse_args()
    pack = args.pack.resolve()
    started = time.perf_counter()
    digests = unpack(pack, args.dest, check=not args.no_check)
    print(json.dumps({"pack": str(pack), "dest": str(args.dest or mapped_dir(pack)),
                      "members": len(digests), "bytes": sum(path.stat().st_size
                                                             for path in (args.dest or mapped_dir(pack)).glob("*.npy")),
                      "s": round(time.perf_counter() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
