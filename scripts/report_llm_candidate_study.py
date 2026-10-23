"""Render the task52 frozen-study summary as markdown tables; never starts a game.

Reads only summary.json plus the independent audit produced by the finalizer, so the
human-readable report cannot drift from the audited numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('day', 'roof', 'fog')
ARMS = ('B_low', 'B_novel')


def read(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=3):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def pct(value):
    return '—' if value is None else f'{100*value:.1f}%'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return '\n'.join(lines)


def render(summary, audit):
    if summary['status'] != 'complete' or audit['status'] != 'complete':
        raise SystemExit('refusing to render an incomplete study')
    if summary['contract'] != audit['contract'] or audit['games_verified'] != 144:
        raise SystemExit('summary and audit disagree')
    out = [f"contract: `{summary['contract']}`", '',
           f"audit: {audit['games_verified']} games, {audit['llm_events_verified']} LLM events, "
           f"{audit['evidence_files']} evidence files, bundle `{audit['evidence_bundle_sha256']}`", '']
    paired, activity, leaf, tokens, fog = [], [], [], [], []
    for family in FAMILIES:
        arms = summary['families'][family]['comparisons']
        for arm in ARMS:
            row = arms[arm]
            independent = audit['paired_comparison'][family][arm]
            for key in ('before_wins', 'after_wins', 'net_wins', 'improved', 'regressed',
                        'wave_deltas', 'mean_wave_delta'):
                assert row[key] == independent[key], (family, arm, key)
            paired.append([family, arm, row['before_wins'], row['after_wins'], row['net_wins'],
                           row['improved'], row['regressed'], fmt(row['paired_binomial_p']),
                           fmt(row['mean_wave_delta'], 2),
                           f"{row['wave_delta_range'][0]:+d}..{row['wave_delta_range'][1]:+d}",
                           ','.join(f'{d:+d}' for d in row['wave_deltas'])])
            counts = row['counts']
            activity.append([family, arm, row['queries'], fmt(row['queries_per_game'], 2),
                             row['cache_hits'], row['actual_api_attempts'],
                             counts.get('returned', 0), counts.get('valid', 0),
                             counts.get('illegal', 0), counts.get('malformed', 0),
                             counts.get('excess', 0), counts.get('duplicates', 0),
                             pct(row['illegal_rate']), row['selected_queries'],
                             row['selected_novel_queries'], row['novel_proposals'],
                             pct(row['selected_novel_proposal_rate']),
                             pct(row['selected_novel_query_rate']),
                             row['search_points']])
            leaf.append([family, arm, len(row['selected_leaf_advantages']),
                         ','.join(fmt(v, 4) for v in row['selected_leaf_advantages']) or '—',
                         row['terminal_diagnostic_pairs'], row['terminal_diagnostic_complete'],
                         row['terminal_diagnostic_net_wins'],
                         ','.join(f'{d:+d}' for d in row['terminal_diagnostic_wave_deltas']) or '—'])
            usage = row['logical_tokens']
            tokens.append([family, arm, usage.get('promptTokenCount', 0),
                           usage.get('candidatesTokenCount', 0),
                           usage.get('thoughtsTokenCount', 0),
                           usage.get('totalTokenCount', 0)])
            if family == 'fog':
                valid = counts.get('valid', 0)
                proportion = lambda n: pct(n / valid if valid else None)
                fog.append([arm, valid,
                            pct(row['legal_clear_fog_proportion']),
                            proportion(row['legal_proposal_plant_types'].get('22', 0)),
                            proportion(row['legal_proposal_plant_types'].get('25', 0)),
                            proportion(row['legal_proposal_plant_types'].get('27', 0)),
                            proportion(row['legal_proposal_plant_types'].get('31', 0)),
                            proportion(row['legal_proposal_plant_types'].get('26', 0)),
                            proportion(row['legal_proposal_plant_types'].get('23', 0)),
                            pct(row['raw_resolvable_clear_fog_proportion'])])
    out += ['## Paired result (A reused for both arms)', '',
            table(['site', 'arm', 'A wins', 'B wins', 'net', 'improved', 'regressed', 'exact p',
                   'mean wave Δ', 'range', 'per-seed Δ'], paired), '',
            '## Proposal activity', '',
            table(['site', 'arm', 'queries', '/game', 'cache hits', 'API attempts', 'returned',
                   'valid', 'illegal', 'malformed', 'excess', 'duplicates', 'illegal rate',
                   'selected any', 'selected novel', 'novel proposals', 'novel selected/proposal',
                   'novel selected/query', 'search points'], activity), '',
            '## Leaf advantage and one-step terminal diagnostic', '',
            table(['site', 'arm', 'novel selections', 'leaf advantage', 'terminal pairs',
                   'completed', 'net wins', 'wave Δ'], leaf), '',
            '## Logical tokens (events replayed from cache counted separately from API attempts)', '',
            table(['site', 'arm', 'prompt', 'candidate', 'thought', 'total'], tokens), '',
            '## Fog proposal composition (legal proposals only, f8 deck)', '',
            table(['arm', 'valid', 'clear-fog share 22/25/27', '22 Torchwood', '25 Plantern',
                   '27 Blover', '31 Magnet-shroom', '26 Cactus', '23 Tall-nut', 'raw clear-fog share'],
                  fog), '']
    ledger = summary['api_ledger_including_retries_and_interrupted_games']
    usage = ledger['tokens']
    out += ['## API ledger (all attempts, including retries and interrupted games)', '',
            f"attempts {ledger['attempts']}, failed {ledger['failed_attempts']}, "
            f"prompt {usage.get('promptTokenCount', 0)}, candidate {usage.get('candidatesTokenCount', 0)}, "
            f"thought {usage.get('thoughtsTokenCount', 0)}, total {usage.get('totalTokenCount', 0)}", '',
            'monetary cost: unknown (gateway tariff not provided)', '']
    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task52/study-v1')
    args = p.parse_args()
    print(render(read(args.out / 'summary.json'), read(args.out / 'independent-audit.json')))


if __name__ == '__main__':
    main()
