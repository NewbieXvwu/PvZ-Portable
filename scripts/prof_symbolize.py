#!/usr/bin/env python3
"""Flat profile from scripts/prof_sample.c samples.

This machine has no perf and no valgrind, so the simulator is sampled by a SIGPROF
handler (scripts/prof_sample.c) that records the interrupted program counter.  This
script maps those PCs to functions with addr2line and prints the shares.

    cc -shared -fPIC -O2 -o /tmp/prof_sample.so scripts/prof_sample.c -ldl
    PVZ_PROF_OUT=/tmp/pcs.bin LD_PRELOAD=/tmp/prof_sample.so python3 scripts/bench_engine.py bench ...
    python3 scripts/prof_symbolize.py --samples /tmp/pcs.bin --binary build/pvz-portable --top 15
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 8}Q", data))


def symbolize(binary: Path, offsets: list[int]) -> dict[int, str]:
    unique = sorted(set(offsets))
    payload = "".join(f"0x{offset:x}\n" for offset in unique)
    result = subprocess.run(["addr2line", "-f", "-C", "-e", str(binary)], input=payload,
                            capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    names = {}
    for index, offset in enumerate(unique):
        name = lines[2 * index] if 2 * index < len(lines) else "??"
        names[offset] = name.split("(")[0].strip() or "??"
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("/tmp/pvz_pcs.bin"))
    parser.add_argument("--binary", type=Path, default=Path("build/pvz-portable"))
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    offsets = load(args.samples)
    if not offsets:
        raise SystemExit(f"no samples in {args.samples}")
    names = symbolize(args.binary, offsets)
    counts = Counter(names[offset] for offset in offsets)
    total = sum(counts.values())
    rows = [{"function": name, "samples": count, "share": round(count / total, 4)}
            for name, count in counts.most_common(args.top)]
    print(f"{total} samples from {args.samples}")
    for row in rows:
        print(f"  {row['share'] * 100:5.1f}%  {row['function']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"samples": total, "binary": str(args.binary), "top": rows},
                                       indent=1) + "\n")


if __name__ == "__main__":
    main()
