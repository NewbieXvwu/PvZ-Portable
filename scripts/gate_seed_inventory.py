#!/usr/bin/env python3
"""task88 step 1: inventory every registered seed segment and compute free gaps.

The gate pool question is "which seeds has anything ever claimed?", so this script
collects *declared* segments mechanically instead of trusting prose:

  design      ranges written in each worktree DESIGN.md, parsed with a regex
              (dash ranges and [a, b) pool notation), kept when both ends look
              like seeds
  configs     every JSON with seed-bearing keys (seeds, seed_block, seed_window,
              future_seed_start, base/stride/per_task, tasks as {name: [seeds]})
  manifests   per-task [min(seeds), max(seeds)+1) spans from every JSON under the
              worktrees, the main checkout and ~/pvz-logs
  ledger      the out-of-git gate ledger (pool + allocated blocks)

Each source contributes half-open intervals; they are merged and the complement
inside [0, 2**32) is reported as free gaps. --verify-lo/--verify-hi re-scans every
JSON/JSONL file on the machine for a literal integer inside a window, which is the
check that matters for a candidate pool: a declared-segment audit can miss a seed
that only survives in a recorded game.

Nothing here is hand-copied: rerun the script and the table regenerates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pvz_promotion import START_SEED_LEDGER_DEFAULT, START_SEED_POOL  # noqa: E402

MAIN_CHECKOUT = Path.home() / "PvZ-search"
LOG_ROOT = Path.home() / "pvz-logs"
SEED_LIMIT = 1 << 32

RANGE = re.compile(r"(?<![\d,.])(\d{1,3}(?:,\d{3})+|\d{4,9})\s*[\u2013\-~]\s*(\d{1,3}(?:,\d{3})+|\d{4,9})(?![\d,])")
POOL = re.compile(r"\[\s*(\d{1,3}(?:,\d{3})*|\d+)\s*,\s*(\d{1,3}(?:,\d{3})*|\d+)\s*\)")
PREFILTER = (b'"seeds"', b'"seed_block"', b'"seed_window"', b'"future_seed_start"', b'"base"')
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.mypy_cache', '.pytest_cache'}
SKIP_PARTS = ('/artifacts/student/frames', '/artifacts/human_demo/raw', '/artifacts/research')
SKIP_NAMES = ('gate-seed-width-check.json', 'gate-seed-dryrun.json', 'gate-seed-inventory.json')
MAX_JSON_BYTES = 32 << 20


def num(text):
    return int(text.replace(',', ''))


def worktree_roots():
    """Every local worktree plus the main checkout plus the log directory."""
    roots = []
    try:
        listing = subprocess.run(['git', 'worktree', 'list', '--porcelain'], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
        roots += [line.split('worktree ', 1)[1] for line in listing.splitlines()
                  if line.startswith('worktree ')]
    except (OSError, subprocess.CalledProcessError):
        roots.append(str(ROOT))
    for extra in (MAIN_CHECKOUT, LOG_ROOT):
        if extra.exists() and str(extra) not in roots:
            roots.append(str(extra))
    return roots


def design_segments(path):
    """Seed-looking ranges in DESIGN prose, e.g. 43,000,000-43,008,191 or [31024, 65536)."""
    try:
        text = Path(path).read_text(errors='replace')
    except OSError:
        return []
    segments = []
    for regex in (RANGE, POOL):
        for match in regex.finditer(text):
            low, high = num(match.group(1)), num(match.group(2))
            if 30000 <= low < high <= SEED_LIMIT and high - low <= 50_000_000:
                segments.append(dict(name=match.group(0), low=low, high=high))
    return segments


def _int_list(value):
    return (isinstance(value, list) and value
            and all(type(item) is int and not isinstance(item, bool) for item in value))


def json_segments(document, label):
    """Declared seed intervals inside one JSON document."""
    found = []
    stack = [(label, document)]
    while stack:
        name, item = stack.pop()
        if isinstance(item, dict):
            seeds = item.get('seeds')
            if _int_list(seeds) and all(0 <= seed < SEED_LIMIT for seed in seeds):
                found.append(dict(name=name, low=min(seeds), high=max(seeds) + 1, kind='task seeds'))
            block = item.get('seed_block') or item.get('seed_window')
            if (isinstance(block, list) and len(block) == 2
                    and all(type(x) is int for x in block) and 0 <= block[0] < block[1] <= SEED_LIMIT):
                found.append(dict(name=name, low=block[0], high=block[1], kind='declared block'))
            start, count = item.get('future_seed_start'), item.get('future_seed_count')
            if type(start) is int and type(count) is int and 0 <= start < SEED_LIMIT and count > 0:
                found.append(dict(name=name + '/future', low=start, high=min(start + count, SEED_LIMIT),
                                  kind='future seeds'))
            base, stride, per_task = item.get('base'), item.get('stride'), item.get('per_task')
            if type(base) is int and type(stride) is int and type(per_task) is int and base >= 30000:
                tasks = item.get('tasks') if isinstance(item.get('tasks'), dict) else {}
                width = max(len(tasks), 1) * stride
                found.append(dict(name=name + '/block', low=base, high=min(base + width, SEED_LIMIT),
                                  kind='seed block'))
            tasks = item.get('tasks')
            if isinstance(tasks, dict):
                for key, value in tasks.items():
                    if _int_list(value) and all(0 <= seed < SEED_LIMIT for seed in value):
                        found.append(dict(name=key, low=min(value), high=max(value) + 1, kind='task seeds'))
            for key, value in item.items():
                if key != 'tasks':
                    stack.append((f'{name}.{key}' if isinstance(key, str) else name, value))
        elif isinstance(item, list):
            for index, value in enumerate(item):
                if isinstance(value, (dict, list)):
                    stack.append((f'{name}[{index}]', value))
    return found


def scan_json(roots):
    """Walk every JSON file that carries seed-bearing keys."""
    files = parsed = 0
    segments = []
    for root in roots:
        for folder, subfolders, names in os.walk(root):
            subfolders[:] = [name for name in subfolders if name not in SKIP_DIRS]
            if any(part in folder for part in SKIP_PARTS):
                subfolders[:] = []
                continue
            for name in names:
                if not name.endswith('.json') or name in SKIP_NAMES:
                    continue
                path = Path(folder) / name
                try:
                    if path.stat().st_size > MAX_JSON_BYTES:
                        continue
                    raw = path.read_bytes()
                except OSError:
                    continue
                files += 1
                if not any(key in raw for key in PREFILTER):
                    continue
                try:
                    document = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                parsed += 1
                rel = str(path.relative_to(root)) if str(path).startswith(root) else str(path)
                for segment in json_segments(document, rel):
                    segment['source'] = str(path)
                    segments.append(segment)
    return dict(files_seen=files, files_parsed=parsed, segments=segments)


def merge(intervals):
    ordered = sorted((low, high) for low, high in intervals)
    merged = []
    for low, high in ordered:
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def free_gaps(occupied, limit, minimum):
    gaps, cursor = [], 0
    for low, high in occupied:
        if low - cursor >= minimum:
            gaps.append(dict(low=cursor, high=low, size=low - cursor))
        cursor = max(cursor, high)
    if limit - cursor >= minimum:
        gaps.append(dict(low=cursor, high=limit, size=limit - cursor))
    return gaps


def verify_window(roots, low, high):
    """Every literal integer inside [low, high) in every JSON/JSONL file on the machine."""
    assert len(str(low)) == len(str(high)), 'window must not straddle a digit width'
    width = len(str(low))
    # The run must sit outside any identifier/decimal fraction: preceded by a
    # non-word char (': ', ', ', '['), followed by neither a digit nor a dot.
    # Otherwise every 9-digit float mantissa (0.100845682) and every hex state
    # hash with a digit run becomes a false hit (measured 1688 -> real hits).
    pattern = re.compile((r'(?<![\d.\w])\d{%d}(?![\d.])' % width).encode())
    hits, files = [], 0
    for root in roots:
        for folder, subfolders, names in os.walk(root):
            subfolders[:] = [name for name in subfolders if name not in SKIP_DIRS]
            if any(part in folder for part in SKIP_PARTS):
                subfolders[:] = []
                continue
            for name in names:
                if not name.endswith(('.json', '.jsonl')) or name in SKIP_NAMES:
                    continue
                path = Path(folder) / name
                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                files += 1
                for match in pattern.finditer(raw):
                    value = int(match.group(0))
                    if low <= value < high:
                        hits.append(dict(path=str(path), value=value))
    return dict(files=files, hits=hits)


def compact(result, widest=40):
    """The git-committed form: merged intervals plus a few named examples.

    The full per-declaration table is a scan artifact (thousands of task rows, a
    megabyte of JSON); what a reviewer needs in the repository is the merged
    occupancy, the per-source counts and the widest named segments. A document
    that is already compact (the committed summary) is returned unchanged, so
    --from-inventory works on it instead of raising KeyError on 'sources'.
    """
    if 'sources' not in result:
        return result
    by_kind = {}
    for segment in result['sources']:
        entry = by_kind.setdefault(segment['kind'], dict(declarations=0, low=segment['low'],
                                                        high=segment['high']))
        entry['declarations'] += 1
        entry['low'] = min(entry['low'], segment['low'])
        entry['high'] = max(entry['high'], segment['high'])
    widest_segments = sorted(result['sources'], key=lambda s: s['high'] - s['low'], reverse=True)[:widest]
    return dict(
        schema=result['schema'],
        roots=result['roots'],
        scan=result['scan'],
        occupied=result['occupied'],
        occupied_seeds=result['occupied_seeds'],
        free_gaps_ge_1m=result['free_gaps_ge_1m'],
        sources_by_kind=by_kind,
        widest_segments=[dict(name=s['name'], kind=s['kind'], low=s['low'], high=s['high'],
                              source=Path(s['source']).name) for s in widest_segments],
        **{key: result[key] for key in ('candidate', 'verification') if key in result},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=None,
                        help='full scan result (default: a temp file when --summary-out is set)')
    parser.add_argument('--summary-out', type=Path,
                        default=ROOT / 'MEMORY/docs/gate-seed-inventory.json',
                        help='compact, git-committed summary')
    parser.add_argument('--from-inventory', type=Path, default=None,
                        help='re-summarise a previous full result without rescanning')
    parser.add_argument('--roots', type=Path, nargs='*', default=None,
                        help='roots to scan (default: every local worktree, main checkout, ~/pvz-logs)')
    parser.add_argument('--quick', action='store_true', help='design ranges + ledger + configs only')
    parser.add_argument('--verify-lo', type=int, default=None)
    parser.add_argument('--verify-hi', type=int, default=None)
    parser.add_argument('--candidate', type=int, nargs=2, default=None,
                        help='LO HI candidate pool to report against the free gaps')
    parser.add_argument('--ignore-start-pool', action='store_true',
                        help='drop the task99 start-cell pool and its ledger from occupancy; use it '
                             'to audit a candidate that is itself a registered pool')
    args = parser.parse_args()
    if args.from_inventory is not None:
        result = json.loads(args.from_inventory.read_text())
        if args.candidate:
            low, high = args.candidate
            overlap = [[max(low, a), min(high, b)] for a, b in result['occupied'] if a < high and low < b]
            result['candidate'] = dict(low=low, high=high, size=high - low,
                                       disjoint=not overlap, overlaps=overlap,
                                       self_registered=[low, high] == list(START_SEED_POOL))
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(compact(result), indent=1) + '\n')
        print(json.dumps(result.get('candidate', {}), indent=1))
        return 0

    roots = [str(path) for path in (args.roots or worktree_roots())]
    design = []
    for root in roots:
        path = Path(root) / 'DESIGN.md'
        for segment in design_segments(path):
            design.append(dict(segment, source=str(path), kind='design range'))

    ledger_path = Path.home() / 'pvz-logs/gate_seed_ledger.json'
    ledger = []
    if ledger_path.exists():
        document = json.loads(ledger_path.read_text())
        pool = document.get('pool')
        if isinstance(pool, list) and len(pool) == 2:
            ledger.append(dict(name='gate ledger pool', low=pool[0], high=pool[1],
                               source=str(ledger_path), kind='gate ledger'))
        for index, block in enumerate(document.get('allocated', [])):
            ledger.append(dict(name=f'gate ledger block {index}', low=block[0], high=block[1],
                               source=str(ledger_path), kind='gate ledger'))

    # task99: the start-cell renewal pool is registered here as a constant (not only as a
    # ledger) so a scan run before the first renewal still reports it as occupied.
    start_segments = [dict(name='start cell renewal pool', low=START_SEED_POOL[0],
                           high=START_SEED_POOL[1], kind='start cell pool',
                           source=str(ROOT / 'python/pvz_promotion.py'))]
    start_ledger_path = Path(START_SEED_LEDGER_DEFAULT)
    if start_ledger_path.exists():
        document = json.loads(start_ledger_path.read_text())
        for index, block in enumerate(document.get('allocated') or []):
            start_segments.append(dict(name=f'start cell ledger block {index}', low=block[0],
                                       high=block[1], kind='start cell ledger',
                                       source=str(start_ledger_path)))

    scanned = dict(files_seen=0, files_parsed=0, segments=[])
    if not args.quick:
        scanned = scan_json(roots)
    segments = design + ledger + ([] if args.ignore_start_pool else start_segments) + scanned['segments']
    occupied = merge((segment['low'], segment['high']) for segment in segments)
    gaps = free_gaps(occupied, SEED_LIMIT, 1_000_000)

    result = dict(
        schema='gate_seed_inventory_v1',
        roots=roots,
        scan=dict(files_seen=scanned['files_seen'], files_parsed=scanned['files_parsed'],
                  segments=len(scanned['segments']), quick=bool(args.quick)),
        sources=sorted(segments, key=lambda s: (s['low'], s['high'])),
        occupied=occupied,
        free_gaps_ge_1m=gaps,
        occupied_seeds=sum(high - low for low, high in occupied),
    )
    if args.candidate:
        low, high = args.candidate
        overlap = [[max(low, a), min(high, b)] for a, b in occupied if a < high and low < b]
        result['candidate'] = dict(low=low, high=high, size=high - low,
                                   disjoint=not overlap, overlaps=overlap,
                                   self_registered=[low, high] == list(START_SEED_POOL))
    if args.verify_lo is not None:
        result['verification'] = dict(low=args.verify_lo, high=args.verify_hi,
                                      **verify_window(roots, args.verify_lo, args.verify_hi))
        result['verification']['hit_count'] = len(result['verification']['hits'])
        result['verification']['hits'] = result['verification']['hits'][:50]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(compact(result), indent=1) + "\n")
    summary = {key: result[key] for key in ('scan', 'occupied_seeds')}
    summary['occupied_intervals'] = len(occupied)
    summary['free_gaps_ge_1m'] = gaps[:10]
    if 'candidate' in result:
        summary['candidate'] = result['candidate']
    if 'verification' in result:
        summary['verification'] = {key: result['verification'][key]
                                   for key in ('low', 'high', 'files', 'hit_count')}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
