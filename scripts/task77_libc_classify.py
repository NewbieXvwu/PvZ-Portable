#!/usr/bin/env python3
"""Classify the libc samples that addr2line cannot name (task77 report section 4).

This box has no perf and no libc debug symbols (apt-get download is blocked by the
proxy), so a sampled PC inside libc is classified from the instructions at the address:
glibc's string routines are the only hot code built around rep stos / rep movs, and the
AVX variants around vmovdqu/vmovdqa.  The window heuristic is deliberately crude — it
answers "is this memset, a copy, or something else", not which caller paid for it.

    python3 scripts/task77_libc_classify.py --out artifacts/task77/libc_classification.json

The sample set is the same twenty games as the hotspot table: profile_complete for
day/night, the three good pool samples plus the resampled one, fog and roof as profiled.
"""
from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from prof_symbolize import load

# The pool sample whose parallel arrays disagree (report section 4); kept, never read.
INVALID = {'pcs.522801.bin'}
FOLDERS = [('day', 'profile_complete/day'), ('night', 'profile_complete/night'),
           ('pool', 'profile/pool'), ('pool', 'profile_recovery/pool'),
           ('fog', 'profile/fog'), ('roof', 'profile/roof')]
WINDOW = 12


def classify_window(text: str) -> str:
    if 'rep stos' in text:
        return 'memset (rep stos)'
    if 'rep movs' in text:
        return 'memcpy/memmove (rep movs)'
    if 'vmovdqu' in text or 'vmovdqa' in text:
        return 'vector copy/set (avx)'
    return 'other libc'


def instructions(library: Path, low: int, high: int) -> tuple[list[int], list[str]]:
    output = subprocess.run(['objdump', '-d', '--no-show-raw-insn',
                             f'--start-address={max(0, low - 32)}', f'--stop-address={high + 64}',
                             str(library)], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in output.splitlines():
        head, sep, rest = line.partition(':')
        if not sep or not head.strip():
            continue
        try:
            rows.append((int(head.strip(), 16), rest.strip()))
        except ValueError:
            continue
    rows.sort()
    return [row[0] for row in rows], [row[1] for row in rows]


def classify(library: Path, offsets: list[int]) -> dict[int, str]:
    if not offsets:
        return {}
    keys, texts = instructions(library, min(offsets), max(offsets))
    labels = {}
    for offset in offsets:
        index = bisect.bisect_right(keys, offset) - 1
        labels[offset] = classify_window(' '.join(texts[max(0, index): index + WINDOW]))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task77/libc_classification.json')
    args = parser.parse_args()
    total, libc, counts = 0, 0, Counter()
    cells = Counter()
    for cell, relative in FOLDERS:
        for path in sorted((ROOT / 'artifacts/task77' / relative).glob('pcs.*.bin')):
            if path.name in INVALID:
                continue
            names, offsets, modules = load(path)
            total += len(offsets)
            cells[cell] += len(offsets)
            grouped = {}
            for offset, module in zip(offsets, modules):
                grouped.setdefault(module, []).append(offset)
            for module, values in grouped.items():
                library = Path(names[module])
                if 'libc' not in library.name:
                    continue
                labels = classify(library, values)
                for value in values:
                    counts[labels.get(value, 'other libc')] += 1
                libc += len(values)
    if not total or not libc:
        raise SystemExit('no samples to classify')
    print(f'{total} samples, {libc} in libc ({libc / total:.2%})')
    for name, value in counts.most_common():
        print(f'  {name}: {value} ({value / libc:.1%} of libc, {value / total:.2%} of all)')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'total_samples': total, 'cells': dict(cells),
                                    'libc_samples': libc, 'classes': dict(counts),
                                    'invalid_excluded': sorted(INVALID)}, indent=1) + '\n')


if __name__ == '__main__':
    main()
