#!/usr/bin/env python3
"""Run the existing A4 workload and apply task95's >=0.98 throughput requirement."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def evaluate(report: dict) -> dict:
    games = report['games']
    if (Counter(row['cell'] for row in games) != {'day': 8, 'pool': 8}
            or len({(row['cell'], row['seed']) for row in games}) != 16):
        raise ValueError('requires the complete 16-pair A4 day/pool panel')
    ratios = []
    for row in games:
        old, new = row['old_seconds'], row['new_seconds']
        if not all(math.isfinite(value) and value > 0 for value in (old, new)):
            raise ValueError('worker durations must be finite and positive')
        ratios.append(old / new)
    mismatches = [row for row in games if row['old_decisions'] != row['new_decisions']]
    median = statistics.median(ratios)
    return {**report, 'legacy_a4_time_rule_pass': report.get('pass'),
            'throughput_ratio_definition': 'old_worker_seconds / new_worker_seconds',
            'paired_throughput_ratios': ratios, 'median_throughput_ratio': median,
            'minimum_throughput_ratio': 0.98, 'decision_mismatch': mismatches,
            'pass': median >= 0.98 and not mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--new', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = args.out.with_name(args.out.stem + '_a4_raw.json')
    subprocess.run([sys.executable, str(ROOT / 'scripts/task70_throughput_abba.py'),
                    '--old', str(args.old.resolve(strict=True)),
                    '--new', str(args.new.resolve(strict=True)), '--out', str(raw)], check=True)
    result = evaluate(json.loads(raw.read_text()))
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
                     ('median_throughput_ratio', 'minimum_throughput_ratio', 'pass')}), flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
