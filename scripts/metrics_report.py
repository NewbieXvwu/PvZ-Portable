#!/usr/bin/env python3
"""Summarize pvz runtime metrics, grouping daily segments by process identity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pvz_metrics import DEFAULT_DIRECTORY, summarize


def report(paths):
    processes = defaultdict(list)
    malformed = 0
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get('event') == 'sample':
                    processes[(row['name'], row['pid'])].append(row)
            except (ValueError, KeyError, TypeError):
                malformed += 1  # includes a partial last line from a live writer
    return dict(processes=[dict(name=name, pid=pid, **summarize(rows))
                           for (name, pid), rows in sorted(processes.items())],
                malformed_lines=malformed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path, help='JSONL files or metrics directories')
    args = parser.parse_args()
    paths = []
    for path in args.paths or [DEFAULT_DIRECTORY]:
        if not path.is_absolute():
            parser.error('all paths must be absolute')
        paths.extend(sorted(path.glob('*.jsonl')) if path.is_dir() else [path])
    print(json.dumps(report(sorted(set(paths))), indent=2))


if __name__ == '__main__':
    main()
