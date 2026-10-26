"""Reanalyse task92's existing games; never launch training or game processes."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from select_student import score_rows, sha

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('a', 'c', 'd2')
HELDOUT = (124, 135, 169)  # task84, also preregistered in task92's task document


def expected_gain(scores, quality, k):
    """Enumerate uniform empirical candidate draws; first training slot wins ties.

    Each row of scores is one panel resample. Repeated model identities share
    scores, since the same policy on the same panel is deterministic.
    """
    scores = np.atleast_2d(scores)
    quality = np.asarray(quality)
    gain = np.zeros(len(scores))
    optimal = np.zeros(len(scores))
    combinations = list(itertools.product(range(len(quality)), repeat=k))
    for candidates in combinations:
        candidates = np.asarray(candidates)
        chosen = candidates[np.argmax(scores[:, candidates], axis=1)]
        gain += quality[chosen] - quality[candidates].mean()
        optimal += quality[chosen] == quality[candidates].max()
    return gain / len(combinations), optimal / len(combinations)


def paired_analysis(outcomes, quality=HELDOUT, draws=20000, seed=20261026):
    """Bootstrap within each task, using the SAME seed indices for all models."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((draws, len(quality)), dtype=np.int64)
    totals = np.zeros(len(quality), dtype=np.int64)
    pair_variances = np.zeros((len(quality), len(quality)))
    for task in outcomes:
        rows = np.asarray(task, dtype=np.int64)
        totals += rows.sum(axis=0)
        samples += rows[rng.integers(0, len(rows), size=(draws, len(rows)))].sum(axis=1)
        for i, j in itertools.combinations(range(len(quality)), 2):
            pair_variances[i, j] += len(rows) * np.var(rows[:, i] - rows[:, j], ddof=1)
    winner = np.argmax(samples, axis=1)
    result = dict(draws=draws, random_seed=seed, totals=totals.tolist(),
                  bootstrap_winner_probability=(np.bincount(winner, minlength=len(quality)) / draws).tolist(),
                  paired_difference_sd={f'{NAMES[i]}-{NAMES[j]}': float(np.sqrt(pair_variances[i, j]))
                                        for i, j in itertools.combinations(range(len(quality)), 2)},
                  empirical_candidate_draws={})
    for k in (2, 3):
        gain, optimal = expected_gain(samples, quality, k)
        observed, _ = expected_gain(totals, quality, k)
        result['empirical_candidate_draws'][str(k)] = dict(
            observed_panel_gain=float(observed[0]), bootstrap_mean_gain=float(gain.mean()),
            bootstrap_gain_interval95=np.quantile(gain, [.025, .975]).tolist(),
            bootstrap_optimal_probability=float(optimal.mean()))
    return result


def analyse(offline, manifest, draws=20000, seed=20261026):
    receipt = offline / 'selection.json'
    selection = json.loads(receipt.read_text())
    panel = json.loads(manifest.read_text())
    paths = sorted(selection['scores'])
    if tuple(Path(path).stem for path in paths) != NAMES:
        raise ValueError('expected the frozen A/C/D2 candidate order')
    sources = {str(receipt): sha(receipt), str(manifest): sha(manifest)}
    totals = dict.fromkeys(NAMES, 0)
    outcomes = []
    for task in panel['tasks']:
        columns = []
        expected = task['seeds']
        for index, (name, model) in enumerate(zip(NAMES, paths)):
            folder = offline / 'selection' / f'c{index}_{task["task_id"]}'
            games, config = folder / 'games.jsonl', folder / 'config.json'
            settings = json.loads(config.read_text())
            if (settings['rollout_policy'] != model or settings['task_id'] != task['task_id']
                    or settings['arms'] != 'fast' or int(settings['seed_offset']) != 0
                    or int(settings['seeds']) != len(expected)):
                raise ValueError(f'wrong candidate/task slice: {config}')
            wins = score_rows(games, expected)
            if wins != selection['scores'][model]['wins'][task['task_id']]:
                raise ValueError(f'receipt score disagrees with games: {games}')
            by_seed = {row['seed']: int(row['won']) for row in
                       (json.loads(line) for line in games.read_text().splitlines() if line.strip())}
            columns.append([by_seed[value] for value in expected])
            totals[name] += wins
            sources[str(games)] = sha(games)
            sources[str(config)] = sha(config)
        outcomes.append(np.asarray(columns).T)
    if any(totals[name] != selection['scores'][path]['total'] for name, path in zip(NAMES, paths)):
        raise ValueError('receipt totals disagree with games')
    if selection['best'] != paths[int(np.argmax(list(totals.values())))]:
        raise ValueError('receipt did not choose maximum total wins')
    return dict(schema_version=1, candidates=list(NAMES), heldout_wins=list(HELDOUT),
                heldout_source='MEMORY/docs/task92-multiseed-select.md: task84 evidence table',
                scope='Conditional illustration from three heterogeneous fixed policies, not a production gain forecast. '
                      'Heldout counts are fixed; bootstrap intervals cover panel resampling only. '
                      'No new training or games; original panel and selection rule unchanged.',
                sources_sha256=sources, **paired_analysis(outcomes, draws=draws, seed=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', type=Path, default=ROOT / 'artifacts/task92/offline')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/student_selection.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = analyse(args.offline, args.manifest)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'sources_sha256'}, indent=2))


if __name__ == '__main__':
    main()
