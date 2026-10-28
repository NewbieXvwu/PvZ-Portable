"""Diagnose where an unused card is dropped by the teacher's search.

Each run directory comes from scripts/roof_teacher_diagnostic.py: it records
every search decision with the legal static-scored cells, the candidate set, the
rollout values, and the executed action. This script walks that funnel per card:

    search decisions -> card has a legal cell -> card is a candidate
                     -> candidate wins the rollout -> candidate is executed

A card that never reaches "legal cell" is blocked before the candidate set is
built. On this engine a plant action is only legal when the player can pay for
it, so a card whose cost is above the sun the policy ever holds is invisible to
the search no matter what the value net thinks of it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import pvz_constants  # noqa: E402


def _costs() -> dict[int, int]:
    return {row['type']: row['cost'] for row in pvz_constants.plant_defs()}


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    def q(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1) + 0.5))], 6)
    return dict(median=round(statistics.median(ordered), 6), p10=q(0.1), p90=q(0.9),
                min=round(ordered[0], 6), max=round(ordered[-1], 6), n=len(ordered))


def _load(pattern: str) -> list[dict]:
    runs = []
    for directory in sorted(glob.glob(pattern)):
        path = Path(directory)
        search = path / 'search.jsonl'
        result_path = path / 'result.json'
        if not search.exists() or not result_path.exists():
            continue
        rows = [json.loads(line) for line in search.read_text().splitlines() if line.strip()]
        result = json.loads(result_path.read_text())
        if 'deck' not in result:
            initial = path / 'initial.json'
            if initial.exists():
                packets = json.loads(initial.read_text())['observation'].get('packets') or []
                result['deck'] = [int(p['type']) for p in packets if int(p.get('type', -1)) >= 0]
        if 'level' not in result:
            meta = next(iter(sorted(path.glob('frames/*/*/meta.json'))), None)
            if meta is not None:
                block = json.loads(meta.read_text())
                result['level'] = (block.get('task') or {}).get('level') or (block.get('context') or {}).get('level')
            else:
                result['level'] = int(path.name.split('-')[0].lstrip('l'))
        runs.append(dict(path=path, rows=rows, result=result))
    return runs


def deck_table(runs: list[dict], costs: dict[int, int]) -> list[dict]:
    """Per-card funnel for every card that appears in a deck: is the gate affordability?"""
    table: dict[int, dict] = {}
    for run in runs:
        deck = run['result'].get('deck') or []
        rows = run['rows']
        for packet, card in enumerate(deck):
            row = table.setdefault(card, dict(card=card, cost=costs.get(card), packet=packet,
                                              decisions=0, affordable=0, legal=0, candidate=0,
                                              chosen=0, planted=0))
            row['decisions'] += len(rows)
            row['affordable'] += sum(1 for r in rows if r['sun'] >= (costs.get(card) or 0))
            for r in rows:
                if packet in {s['action']['packet'] for s in r.get('static_scores') or []}:
                    row['legal'] += 1
                if packet in [c.get('packet') for c in r.get('candidates') or []]:
                    row['candidate'] += 1
                if (r.get('chosen') or {}).get('packet') == packet:
                    row['chosen'] += 1
            row['planted'] += int((run['result'].get('planted_types') or {}).get(str(card), 0))
    return sorted(table.values(), key=lambda row: (row['cost'] or 0, row['card']))


def analyse(runs: list[dict], cards: list[int], threshold: float, costs: dict[int, int]) -> dict:
    report = dict(runs=len(runs), threshold=threshold, cards={}, deck=deck_table(runs, costs))
    for card in cards:
        per_level: dict[int, dict] = {}
        games = []
        cells = Counter()
        gaps, ranks, static_ranks = [], [], []
        for run in runs:
            deck = run['result'].get('deck') or []
            if card not in deck:
                continue
            packet = deck.index(card)
            level = int(run['result'].get('level') or 0)
            rows = run['rows']
            legal = candidate = best = chosen_card = 0
            suns = []
            for row in rows:
                suns.append(row['sun'])
                legal_packets = {s['action']['packet'] for s in row.get('static_scores') or []}
                candidate_packets = [c.get('packet') for c in row.get('candidates') or []]
                if packet not in legal_packets:
                    continue
                legal += 1
                if packet not in candidate_packets:
                    continue
                candidate += 1
                index = candidate_packets.index(packet)
                action = row['candidates'][index]
                cells[(action.get('col'), action.get('row'))] += 1
                values = [float(v) if isinstance(v, (int, float)) else -math.inf
                          for v in row.get('values_after') or []]
                finite = _finite(row.get('values_after') or [])
                if finite and math.isfinite(values[index]):
                    gaps.append(max(finite) - values[index])
                    ranks.append(sorted(finite, reverse=True).index(values[index]) + 1)
                bests = {}
                for score in row.get('static_scores') or []:
                    p = score['action']['packet']
                    bests[p] = max(bests.get(p, -math.inf), score['value'])
                if packet in bests:
                    order = sorted(bests.items(), key=lambda item: -item[1])
                    static_ranks.append([p for p, _ in order].index(packet) + 1)
                if row.get('best') == index:
                    best += 1
                if (row.get('chosen') or {}).get('packet') == packet:
                    chosen_card += 1
            planted = int((run['result'].get('planted_types') or {}).get(str(card), 0))
            game = dict(level=level, seed=run['result'].get('seed'), won=bool(run['result'].get('won')),
                        decisions=len(rows), legal=legal, candidate=candidate, best=best,
                        chosen=chosen_card, planted=planted,
                        max_sun=max(suns) if suns else None,
                        affordable=sum(1 for s in suns if s >= costs.get(card, 0)))
            games.append(game)
            bucket = per_level.setdefault(level, dict(games=0, decisions=0, legal=0, candidate=0,
                                                     best=0, chosen=0, planted=0, wins=0,
                                                     affordable=0, max_sun=0))
            bucket['games'] += 1
            bucket['wins'] += int(game['won'])
            bucket['decisions'] += len(rows)
            bucket['legal'] += legal
            bucket['candidate'] += candidate
            bucket['best'] += best
            bucket['chosen'] += chosen_card
            bucket['planted'] += planted
            bucket['affordable'] += game['affordable']
            bucket['max_sun'] = max(bucket['max_sun'], game['max_sun'] or 0)
        total = dict(games=len(games), decisions=sum(g['decisions'] for g in games),
                     legal=sum(g['legal'] for g in games), candidate=sum(g['candidate'] for g in games),
                     best=sum(g['best'] for g in games), chosen=sum(g['chosen'] for g in games),
                     planted=sum(g['planted'] for g in games),
                     affordable=sum(g['affordable'] for g in games),
                     games_with_legal=sum(1 for g in games if g['legal']),
                     games_with_candidate=sum(1 for g in games if g['candidate']),
                     games_with_affordable=sum(1 for g in games if g['affordable']))
        if total['games_with_legal'] == 0:
            verdict = 'never_legal'
        elif total['candidate'] == 0:
            verdict = 'legal_but_never_candidate'
        elif total['best'] == 0:
            verdict = 'candidate_never_wins'
        elif total['chosen'] == 0:
            verdict = 'wins_but_never_executed'
        else:
            verdict = 'reachable'
        report['cards'][str(card)] = dict(
            cost=costs.get(card), verdict=verdict, total=total, by_level=per_level, games=games,
            candidate_gaps=_quantiles(gaps), candidate_value_rank=_quantiles([float(r) for r in ranks]),
            static_rank=_quantiles([float(r) for r in static_ranks]),
            cells=[dict(col=c, row=r, n=n) for (c, r), n in cells.most_common()])
    return report


def markdown(report: dict) -> str:
    lines = ['# 老师搜索盲点漏斗', '',
             f"- 运行数：{report['runs']}，门槛 {report['threshold']}", '']
    if report.get('deck'):
        lines.append('## 卡组逐卡漏斗（所有局合计）')
        lines.append('')
        lines.append('| 卡 | 花费 | 阳光够的决策 | 有合法格 | 进候选 | 执行 | 种下 |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for row in report['deck']:
            lines.append(f"| {row['card']} | {row['cost']} | {row['affordable']} | {row['legal']} | "
                         f"{row['candidate']} | {row['chosen']} | {row['planted']} |")
        lines.append('')
    names = {26: '仙人掌', 39: '西瓜投手', 44: '冰西瓜', 43: '香蒲'}
    for card, block in report['cards'].items():
        total = block['total']
        lines.append(f"## {card} {names.get(int(card), '')}（花费 {block['cost']}）—— {block['verdict']}")
        lines.append('')
        lines.append('| 层 | 次数 | 占搜索决策 | 有该卡的局 |')
        lines.append('|---|---:|---:|---:|')
        rows = [('搜索决策', total['decisions'], 1.0, total['games']),
                ('有合法格', total['legal'], total['legal'] / max(1, total['decisions']), total['games_with_legal']),
                ('进候选', total['candidate'], total['candidate'] / max(1, total['decisions']), total['games_with_candidate']),
                ('候选第一', total['best'], total['best'] / max(1, total['decisions']), ''),
                ('实际执行', total['chosen'], total['chosen'] / max(1, total['decisions']), ''),
                ('种下株数', total['planted'], '', '')]
        for name, count, share, games in rows:
            lines.append(f"| {name} | {count} | {share:.1%} | {games} |" if isinstance(share, float) else f"| {name} | {count} | | {games} |")
        lines.append('')
        lines.append(f"- 决策时阳光 ≥ 花费的次数：{total['affordable']}（{total['affordable'] / max(1, total['decisions']):.1%}），"
                     f"有该次数的局 {total['games_with_affordable']}/{total['games']}")
        if block['candidate_gaps']:
            gap = block['candidate_gaps']
            lines.append(f"- 进候选时与最佳候选的续走差：中位 {gap['median']}，p90 {gap['p90']}，最大 {gap['max']}（门槛 {report['threshold']}）")
        if block['static_rank']:
            lines.append(f"- 静态网在卡间排名：中位 {block['static_rank']['median']}（{block['static_rank']['n']} 个决策）")
        if block['cells']:
            cells = ', '.join(f"({c['col']},{c['row']})×{c['n']}" for c in block['cells'][:6])
            lines.append(f"- 静态网选的格子：{cells}")
        lines.append('')
        lines.append('| 关 | 局 | 胜 | 搜索决策 | 合法 | 候选 | 执行 | 种下 | 最大阳光 | 可负担决策 |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for level, row in sorted(block['by_level'].items()):
            lines.append(f"| {level} | {row['games']} | {row['wins']} | {row['decisions']} | {row['legal']} | "
                         f"{row['candidate']} | {row['chosen']} | {row['planted']} | {row['max_sun']} | {row['affordable']} |")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', action='append', required=True, help='glob of diagnostic run directories')
    parser.add_argument('--card', default='26', help='comma separated SeedType ids')
    parser.add_argument('--threshold', type=float, default=0.02)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--md', type=Path)
    options = parser.parse_args()
    runs = []
    for pattern in options.runs:
        runs.extend(_load(pattern))
    cards = [int(part) for part in options.card.split(',')]
    report = analyse(runs, cards, options.threshold, _costs())
    text = markdown(report)
    if options.out:
        options.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    if options.md:
        options.md.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
