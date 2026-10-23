#!/usr/bin/env python3
"""Flat profile from scripts/prof_sample.c samples.

This machine has no perf and no valgrind, so the simulator is sampled by a SIGPROF
handler (scripts/prof_sample.c) that records the interrupted program counter.  The
handler resolves each sample to (module, offset) at exit, so samples that land in
libc are reported as such instead of being lost as "??"; the executable's offsets go
through addr2line.

    cc -shared -fPIC -O2 -o /tmp/prof_sample.so scripts/prof_sample.c -ldl
    PVZ_PROF_OUT=/tmp/pcs.bin LD_PRELOAD=/tmp/prof_sample.so \
        python3 scripts/bench_engine.py bench ...
    python3 scripts/prof_symbolize.py --samples /tmp/pcs.bin --binary build/pvz-portable
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path


def load(path: Path) -> tuple[list[str], list[int], list[int]]:
    data = path.read_bytes()
    head, _, rest = data.partition(b"\n")
    magic, version, count = head.decode().split()
    if magic != "PVZPROF" or version != "1":
        raise SystemExit(f"unexpected sample file header: {head!r}")
    names = []
    for _ in range(int(count)):
        line, _, rest = rest.partition(b"\n")
        names.append(line.decode())
    total = len(rest) // 12
    offsets = list(struct.unpack(f"<{total}Q", rest[: total * 8]))
    modules = list(struct.unpack(f"<{total}I", rest[total * 8:]))
    return names, offsets, modules


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


def debug_binary(binary: Path, debug_root: Path | None) -> Path:
    """Use matching build-id debug data without installing it system-wide."""
    if debug_root is not None:
        result = subprocess.run(['readelf', '-n', str(binary)], capture_output=True, text=True, check=True)
        match = re.search(r'Build ID: ([0-9a-f]+)', result.stdout)
        if match:
            build_id = match[1]
            debug = debug_root / '.build-id' / build_id[:2] / (build_id[2:] + '.debug')
            if debug.is_file():
                return debug
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=Path("/tmp/pvz_pcs.bin"))
    parser.add_argument("--binary", type=Path, default=Path("build/pvz-portable"))
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--debug-root", type=Path, help="unpacked usr/lib/debug, matched by ELF build ID")
    args = parser.parse_args()

    names, offsets, modules = load(args.samples)
    if not offsets:
        raise SystemExit(f"no samples in {args.samples}")
    # The executable is whichever module the binary path names; everything else is reported
    # by library name so libc (malloc/memcpy) is visible instead of hiding in "??".
    target = str(args.binary.resolve())
    executable = next((index for index, name in enumerate(names) if Path(name).resolve() == Path(target)),
                      None)
    if executable is None:
        # A saved binary may have a new filename; its captured module still has
        # the original executable name. --binary supplies that exact saved build.
        candidates = [index for index, name in enumerate(names) if Path(name).name == "pvz-portable"]
        if len(candidates) == 1:
            executable = candidates[0]
    by_module: dict[int, list[int]] = {}
    for offset, module in zip(offsets, modules):
        by_module.setdefault(module, []).append(offset)
    labels: dict[tuple[int, int], str] = {}
    for module, values in by_module.items():
        if module == 0xFFFFFFFF:
            for value in set(values):
                labels[(module, value)] = "??"
        elif module == executable:
            resolved = symbolize(args.binary, values)
            for value, name in resolved.items():
                labels[(module, value)] = name
        else:
            library = Path(names[module])
            # Shared libraries often retain dynamic function symbols even without
            # debug line tables (e.g. sgemm_kernel_HASWELL). Resolve those too:
            # merging all BLAS work into one module hid the largest CPU hotspot.
            resolved = symbolize(debug_binary(library, args.debug_root), values) if library.is_file() else {}
            for value in set(values):
                name = resolved.get(value, "??")
                labels[(module, value)] = f"[{library.name}]" + (f" {name}" if name != "??" else "")
    counts = Counter(labels[(module, offset)] for offset, module in zip(offsets, modules))
    total = sum(counts.values())
    rows = [{"function": name, "samples": count, "share": round(count / total, 4)}
            for name, count in counts.most_common(args.top)]
    print(f"{total} samples from {args.samples}")
    for row in rows:
        print(f"  {row['share'] * 100:5.1f}%  {row['function']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"samples": total, "binary": str(args.binary),
                                        "debug_root": str(args.debug_root) if args.debug_root else None,
                                        "modules": names, "top": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
