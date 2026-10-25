#!/usr/bin/env python3
"""Summarize probe traces, with first divergent operation and exact coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path


def compare(left, right):
    first_hash = first_observation = first_events = None
    count = 0
    with left.open() as a, right.open() as b:
        for index, (x, y) in enumerate(zip_longest(a, b)):
            if x is None or y is None:
                raise ValueError(f'trace lengths differ at {index}: {left}, {right}')
            x, y = json.loads(x), json.loads(y)
            if x['operation'] != y['operation']:
                raise ValueError('trace operations differ')
            if first_hash is None and x['hash'] != y['hash']:
                first_hash = x['operation']
            if first_observation is None and x['observation'] != y['observation']:
                first_observation = x['operation']
            if first_events is None and x.get('events') != y.get('events'):
                first_events = x['operation']
            count += 1
    return dict(states_compared=count, first_hash_diff=first_hash,
                first_observation_diff=first_observation, first_events_diff=first_events)


def summarize(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    summary['reset_hashes'] = sorted({r['initial_hash'] for r in summary['runs']})
    summary['variable_segments'] = sorted({key for run in summary['runs']
        for key, value in summary['runs'][0]['segments'].items() if run['segments'][key] != value})
    summary['comparisons_to_first'] = [compare(directory / 'trace_0.jsonl', directory / f'trace_{i}.jsonl')
                                       for i in range(1, len(summary['runs']))]
    summary['trace_sha256'] = [hashlib.sha256((directory / f'trace_{i}.jsonl').read_bytes()).hexdigest()
                               for i in range(len(summary['runs']))]
    return summary


def compare_decisions(full_trace, decision_trace):
    """Check every sampled boundary against the uncoalesced, per-operation reference."""
    selected = iter(map(json.loads, decision_trace.read_text().splitlines()))
    target = next(selected, None)
    full_events, decision_events = {}, {}
    count = 0
    first_observation = first_events = None
    with full_trace.open() as stream:
        for line in stream:
            record = json.loads(line)
            for name, value in record.get('events', {}).items():
                full_events[name] = full_events.get(name, 0) + value
            if target is None:
                raise ValueError('decision trace ended before full trace')
            if record['operation'] == target['operation']:
                for name, value in target.get('events', {}).items():
                    decision_events[name] = decision_events.get(name, 0) + value
                if first_observation is None and record['observation'] != target['observation']:
                    first_observation = record['operation']
                if first_events is None and full_events != decision_events:
                    first_events = record['operation']
                count += 1
                target = next(selected, None)
    if target is not None:
        raise ValueError('full trace ended before decision trace')
    return dict(states_compared=count, first_observation_diff=first_observation,
                first_cumulative_events_diff=first_events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--full-reference', type=Path,
                        help='compare first trace in each directory to this uncoalesced trace')
    args = parser.parse_args()
    report = {str(directory): summarize(directory) for directory in args.directories}
    if args.full_reference:
        for directory in args.directories:
            report[str(directory)]['uncoalesced_reference'] = compare_decisions(
                args.full_reference, directory / 'trace_0.jsonl')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    for directory, result in report.items():
        print(directory, len(result['runs']), 'runs;', len(result['reset_hashes']), 'reset hashes;',
              result['comparisons_to_first'])


if __name__ == '__main__':
    main()
