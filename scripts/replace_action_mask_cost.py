"""Measure the observation/legal-mask cost added by the replace action.

The replace mask is produced by LawnApp::EnvironmentReplacements, which saves a
full snapshot, shovels, tests every packet and restores the snapshot for each
occupied cell; EnvironmentRollout::Predict calls it on every rollout prediction.
This benchmark isolates the cost of one observe() at two board densities so the
A/B wall-clock numbers can be attributed instead of guessed.  It launches one
game process at a time and never touches the frozen A/B outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from pvz_env import PvZEnv, TaskSpec, profile_for_deck

RESOURCE = Path('/home/newbiexvwu/.cache/pvz-research-resources')
DECK = (8, 1, 0, 4, 16, 30)
LEVEL = 11
SEED = 6901


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preplanted(count):
    cells = []
    for row in range(4):
        for col in range(8):
            if len(cells) < count:
                cells.append((8, row, col))
    if len(cells) != count:
        raise ValueError(f'cannot preplant {count} cells without touching graves')
    return tuple(cells)


def measure(executable, count, reps, warm):
    env = PvZEnv(RESOURCE, executable=executable)
    try:
        env.reset(deck=DECK, task=TaskSpec(level=LEVEL, seed=SEED,
                                           profile=profile_for_deck(DECK),
                                           preplanted=preplanted(count)))
        for _ in range(warm):
            env.observe()
        samples = []
        observation = None
        for _ in range(reps):
            start = time.perf_counter()
            observation = env.observe()
            samples.append(time.perf_counter() - start)
        return dict(median_ms=round(statistics.median(samples) * 1000, 3),
                    min_ms=round(min(samples) * 1000, 3),
                    max_ms=round(max(samples) * 1000, 3),
                    legal_cells=len(observation['legal_actions'].get('plants', [])))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--warm', type=int, default=5)
    parser.add_argument('--cells', type=int, nargs='+', default=[8, 32])
    args = parser.parse_args()
    engines = dict(treatment=(args.root / 'build/pvz-portable').resolve(),
                   baseline=(args.root / 'artifacts/task69/baseline-engine').resolve())
    result = dict(purpose='isolate the per-observe cost of the replace legal mask',
                  task=dict(level=LEVEL, seed=SEED, deck=list(DECK)),
                  resource=str(RESOURCE), reps=args.reps, warmup=args.warm,
                  engines={name: dict(path=str(path), sha256=sha(path))
                           for name, path in engines.items()},
                  measurements={})
    for count in args.cells:
        result['measurements'][str(count)] = {
            name: measure(path, count, args.reps, args.warm)
            for name, path in engines.items()}
    for count in args.cells:
        row = result['measurements'][str(count)]
        row['ratio_treatment_over_baseline'] = round(
            row['treatment']['median_ms'] / row['baseline']['median_ms'], 2)
    counts = sorted(args.cells)
    if len(counts) >= 2:
        lo, hi = str(counts[0]), str(counts[-1])
        span = counts[-1] - counts[0]
        result['marginal_ms_per_occupied_cell'] = {
            name: round((result['measurements'][hi][name]['median_ms'] -
                         result['measurements'][lo][name]['median_ms']) / span, 4)
            for name in engines}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['measurements'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
