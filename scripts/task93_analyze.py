"""Audit task93 deck coverage or compare complete, paired student evaluations."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pair_compare import mcnemar_exact
from pvz_promotion import summarize, _wilson
from pvz_tasks import deck_for_seed


def load_games(path, seeds, arm, decks, *, legacy_decks=None, require_complete=True):
    """Strict comparisons by default; explicit queue probes may inspect valid partial rows."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['arm'] != arm:
            raise ValueError(f'{path}: unexpected arm {row["arm"]}')
        seed = row['seed']
        if seed in rows:
            raise ValueError(f'{path}: duplicate seed {seed}')
        if row.get('future_seed') is not None or row.get('terminal') is not True:
            raise ValueError(f'{path}: not a terminal cold-start game: {seed}')
        if type(row.get('won')) is not bool:
            raise ValueError(f'{path}: invalid outcome: {seed}')
        deck = row.get('deck')
        if deck is None and legacy_decks is not None:
            deck = legacy_decks.get(seed)
        if seed not in decks or deck != list(decks[seed]):
            raise ValueError(f'{path}: unexpected seed/deck: {seed}')
        rows[seed] = row
    if require_complete and set(rows) != set(seeds):
        raise ValueError(f'{path}: incomplete seed set; missing={sorted(set(seeds)-set(rows))}')
    return rows


def spread(wins, total):
    rates = [100 * value / total for value in wins]
    return dict(draws=len(wins), wins=wins, mean_pct=statistics.mean(rates),
                sample_std_pp=statistics.stdev(rates) if len(rates) > 1 else None,
                range_pp=max(rates)-min(rates))


def comparison(candidate, baseline, seeds):
    report = summarize([dict(candidate=candidate[s]['won'], incumbent=baseline[s]['won'])
                        for s in seeds])
    report['mcnemar_exact_two_sided'] = mcnemar_exact(report['gain'], report['harm'])
    return report


def coverage(manifest, cold):
    report = {}
    for task in manifest['tasks']:
        seeds = task['splits']['cold']
        decks = {s: deck_for_seed(task, s) for s in seeds}
        path = cold / task['task_id'] / 'games.jsonl'
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(raw) != 2 * len(seeds) or {r['arm'] for r in raw} != {'fast', 'fast_search'}:
            raise ValueError(f'{path}: expected both complete cold-start arms')
        # Use the same strict checks without creating intermediate files.
        arms = {}
        for arm in ('fast', 'fast_search'):
            rows = [r for r in raw if r['arm'] == arm]
            if (len(rows) != len(seeds) or {r['seed'] for r in rows} != set(seeds)
                    or any(r.get('terminal') is not True or r.get('future_seed') is not None
                           or type(r.get('won')) is not bool
                           or r.get('deck') != list(decks[r['seed']]) for r in rows)):
                raise ValueError(f'{path}: incomplete or invalid {arm} arm')
            wins = sum(r['won'] for r in rows)
            arms[arm] = dict(games=len(rows), wins=wins, wilson95=_wilson(wins, len(rows)))
        counts = Counter(card for deck in decks.values() for card in deck)
        report[task['task_id']] = dict(arms=arms,
            plant_counts={str(card): counts[card] for card in range(48)},
            length_counts=dict(sorted(Counter(map(len, decks.values())).items())))
    return report


def evaluation_config(path, task):
    """Older task84 rows lack a deck field; require their recorded task contract."""
    config_path = Path(path).with_name('config.json')
    config = json.loads(config_path.read_text())
    source = Path(config['manifest'])
    recorded = next(t for t in json.loads(source.read_text())['tasks']
                    if t['task_id'] == config['task_id'])
    if (recorded != task or config['arms'] != 'fast'
            or float(config['dagger_beta']) != 0 or config.get('demo_archive', 'None') != 'None'):
        raise ValueError(f'{config_path}: evaluation contract differs')
    return ({s: list(deck_for_seed(recorded, s)) for s in recorded['seeds']},
            dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 executable=config['executable']))


def named_runs(values, seeds, decks, task):
    runs, sources = {}, {}
    for value in values:
        name, path = value.split('=', 1)
        if not name or name in runs:
            raise ValueError(f'duplicate/empty run name: {name}')
        legacy_decks, contract = evaluation_config(path, task)
        runs[name] = load_games(path, seeds, 'fast', decks, legacy_decks=legacy_decks)
        sources[name] = dict(path=str(Path(path).resolve()),
                             sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), **contract)
    return runs, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    cold = commands.add_parser('coverage')
    cold.add_argument('--cold', type=Path, required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--task-id', required=True)
    compare.add_argument('--candidate', action='append', required=True, metavar='NAME=GAMES_JSONL')
    compare.add_argument('--baseline', action='append', required=True, metavar='NAME=GAMES_JSONL')
    compare.add_argument('--incumbent', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.command == 'coverage':
        result = coverage(manifest, args.cold)
    else:
        task = next(t for t in manifest['tasks'] if t['task_id'] == args.task_id)
        seeds = task['seeds']
        if len(seeds) != 192 or len(set(seeds)) != 192:
            raise ValueError('task93 comparison requires exactly 192 distinct evaluation seeds')
        decks = {s: deck_for_seed(task, s) for s in seeds}
        candidates, candidate_sources = named_runs(args.candidate, seeds, decks, task)
        baselines, baseline_sources = named_runs(args.baseline, seeds, decks, task)
        legacy_decks, incumbent_contract = evaluation_config(args.incumbent, task)
        incumbent = load_games(args.incumbent, seeds, 'fast', decks, legacy_decks=legacy_decks)
        if len(candidates) < 2 or len(baselines) < 2:
            raise ValueError('task93 requires at least two candidate and baseline training draws')
        result = dict(task_id=args.task_id, pairs=len(seeds),
            candidates=candidate_sources, baselines=baseline_sources,
            paired={name: {base: comparison(run, old, seeds) for base, old in baselines.items()}
                    for name, run in candidates.items()},
            incumbent_source=dict(path=str(args.incumbent.resolve()),
                sha256=hashlib.sha256(args.incumbent.read_bytes()).hexdigest(), **incumbent_contract),
            incumbent_paired={name: comparison(run, incumbent, seeds) for name, run in candidates.items()},
            candidate_spread=spread([sum(r['won'] for r in run.values()) for run in candidates.values()], len(seeds)),
            baseline_spread=spread([sum(r['won'] for r in run.values()) for run in baselines.values()], len(seeds)),
            limitation='Two training draws provide descriptive spread, not proof that tail risk is eliminated.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
