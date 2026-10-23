"""Render the task52 frozen-study summary as markdown tables; never starts a game.

Reads only summary.json, the independent audit and the audited traces, so the
human-readable report and the KB-scale conclusion layer cannot drift from evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('day', 'roof', 'fog')
ARMS = ('B_low', 'B_novel')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def dose(out, contract):
    """Index-aligned trace comparison: how often did the arm actually change a decision?"""
    rows, pairs = [], 0
    for family_index, family in enumerate(FAMILIES):
        for arm in ARMS:
            for offset in range(16):
                seed = 52000000 + 1000*family_index + offset
                pair = []
                for name in ('A', arm):
                    folder = out/'games'/family/name/str(seed)
                    result = read(folder/'result.json')
                    assert result['contract'] == contract
                    assert sha(folder/'trace.json') == result['trace_sha256']
                    trace = read(folder/'trace.json')
                    assert len(trace) == result['decisions']
                    pair.append((result, trace))
                (a_result, a_trace), (b_result, b_trace) = pair
                pairs += 1
                common = min(len(a_trace), len(b_trace))
                differ = [i for i in range(common) if a_trace[i]['chosen'] != b_trace[i]['chosen']]
                if not differ and len(a_trace) == len(b_trace):
                    continue
                index = differ[0] if differ else None
                rows.append(dict(family=family, arm=arm, seed=seed, first_diff_index=index,
                                 first_diff_tick=a_trace[index]['tick'] if index is not None else None,
                                 differing_decisions=len(differ), decisions_a=len(a_trace),
                                 decisions_b=len(b_trace), final_tick_a=a_result['final_tick'],
                                 final_tick_b=b_result['final_tick'], final_wave_a=a_result['final_wave'],
                                 final_wave_b=b_result['final_wave'], won_a=a_result['won'], won_b=b_result['won']))
    return dict(pairs=pairs, diverged=len(rows), rows=rows)


def render(summary, audit, study):
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
    rows = study['rows']
    detail = (table(['site', 'arm', 'seed', 'first tick', 'differing decisions', 'decisions A/B',
                     'final tick A/B', 'final wave A/B', 'won A/B'],
                    [[r['family'], r['arm'], r['seed'], r['first_diff_tick'] if r['first_diff_tick'] is not None else '—',
                      r['differing_decisions'], f"{r['decisions_a']}/{r['decisions_b']}",
                      f"{r['final_tick_a']}/{r['final_tick_b']}", f"{r['final_wave_a']}/{r['final_wave_b']}",
                      f"{int(r['won_a'])}/{int(r['won_b'])}"] for r in rows])
             if rows else '(no paired game changed a decision)')
    out += ['## Intervention dose (index-aligned audited traces)', '',
            f"{study['diverged']}/{study['pairs']} paired games changed at least one decision.", '',
            detail, '']
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
    p.add_argument('--json', type=Path)
    args = p.parse_args()
    out = args.out
    summary, audit, plan = read(out/'summary.json'), read(out/'independent-audit.json'), read(out/'plan.json')
    study = dose(out, summary['contract'])
    print(render(summary, audit, study))
    if args.json:
        payload = dict(contract=summary['contract'], study=str(out),
                       audit={k: audit[k] for k in ('status', 'games_verified', 'llm_events_verified',
                                                    'evidence_files', 'evidence_bundle_sha256', 'auditor_sha256')},
                       frozen_files=plan['files'], preregistration_sha256=plan['preregistration_sha256'],
                       families={family: summary['families'][family]['comparisons'] for family in FAMILIES},
                       api_ledger=summary['api_ledger_including_retries_and_interrupted_games'],
                       intervention_dose=study,
                       monetary_cost_reason='gateway tariff not provided')
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
