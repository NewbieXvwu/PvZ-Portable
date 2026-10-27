"""汇总 task129 卡组胜局、输局证据及单卡/组件关联。"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from deck_rules import DAY_MUSHROOMS, DEPENDENCIES
from pvz_constants import PLANT_NAME, plant_defs
from pvz_frame_store import read_frames
from render_episode import _mower_fired_waves
from task129_deck_sweep import MANIFEST, OUT


def rate(rows):
    games = sum(row['games'] for row in rows)
    return sum(row['won'] for row in rows) / games if games else None


def association(rows, selected):
    present = [row for row in rows if selected(set(row['deck']))]
    absent = [row for row in rows if not selected(set(row['deck']))]
    within_level = {}
    for level in sorted({row['level'] for row in rows}):
        with_rate = rate([row for row in present if row['level'] == level])
        without_rate = rate([row for row in absent if row['level'] == level])
        if with_rate is not None and without_rate is not None:
            within_level[level] = 100 * (with_rate - without_rate)
    return dict(with_decks=len(present), without_decks=len(absent),
                with_win_rate=rate(present), without_win_rate=rate(absent),
                within_level_delta_pp=within_level,
                mean_within_level_delta_pp=(sum(within_level.values()) / len(within_level)
                                            if within_level else None))


def loss_evidence(task, row):
    frames = read_frames(OUT / 'frames' / f'seed_{row["seed"]}' /
                         f'{task["task_id"]}_fast_search' / 'frames.npz')
    fired = _mower_fired_waves({'frames': frames})
    crossed = []
    for zombie in row['final_zombies']:
        kind = zombie['type']
        edge = (-150 if kind in (3, 23, 32) else -175 if kind in (7, 12, 22)
                else -130 if kind in (8, 9, 11) else -100)
        if int(zombie['x']) <= edge:
            crossed.append(zombie)
    return dict(seed=row['seed'], final_wave=row['final_wave'], terminal=row['terminal'],
                first_mower_wave=min((wave for wave, _lane in fired), default=None),
                crossed_zombies=crossed)


def main():
    manifest = json.loads(MANIFEST.read_text())
    costs = {row['type']: row['cost'] for row in plant_defs()}
    summaries = []
    for task in manifest['tasks']:
        path = OUT / task['task_id'] / 'games.jsonl'
        games = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        planted = Counter()
        for game in games:
            planted.update({int(card): count for card, count in game['planted_types'].items()})
        summary = dict(task_id=task['task_id'], level=task['level'], deck=task['deck'],
                       card_names=[PLANT_NAME[card] for card in task['deck']],
                       mean_card_cost=sum(costs[card] for card in task['deck']) / len(task['deck']),
                       games=len(games), won=sum(row['won'] for row in games),
                       seconds=sum(row['seconds'] for row in games), planted_types=dict(planted))
        if len(games) == 8 and summary['won'] <= 2:
            losses = [loss_evidence(task, row) for row in games if not row['won']]
            summary['losses'] = losses
            summary['crossed_types'] = dict(Counter(zombie['type'] for loss in losses
                                                    for zombie in loss['crossed_zombies']))
            summary['first_mower_waves'] = dict(Counter(str(loss['first_mower_wave']) for loss in losses))
        summaries.append(summary)
    completed = [row for row in summaries if row['games'] == 8]
    components = {f'{PLANT_NAME[card]}+底座或配套':
                  lambda cards, card=card, choices=choices: card in cards and bool(cards & choices)
                  for card, choices in DEPENDENCIES.items()}
    components['咖啡豆+蘑菇'] = lambda cards: 35 in cards and bool(cards & DAY_MUSHROOMS)
    components['投手+坚果或南瓜'] = lambda cards: bool(cards & {32, 34, 39, 44}) and bool(cards & {3, 23, 30})
    components['寒冰菇+咖啡豆'] = lambda cards: {14, 35} <= cards
    components['大喷菇+咖啡豆'] = lambda cards: {10, 35} <= cards
    components['忧郁菇+大蒜'] = lambda cards: {42, 36} <= cards
    report = dict(status='complete' if len(completed) == 54 else 'partial',
                  games=sum(row['games'] for row in summaries), won=sum(row['won'] for row in summaries),
                  completed_decks=len(completed), tasks=summaries,
                  by_level={level: dict(games=sum(row['games'] for row in summaries if row['level'] == level),
                                        won=sum(row['won'] for row in summaries if row['level'] == level))
                            for level in sorted({row['level'] for row in summaries})},
                  cards={card: dict(name=PLANT_NAME[card], **association(completed, lambda cards, card=card: card in cards))
                         for card in sorted({card for row in completed for card in row['deck']})},
                  components={name: association(completed, selected) for name, selected in components.items()},
                  evidence_note='破防按Zombie::CheckForBoardEdge的类型阈值筛选终局越界者，多只时不指定唯一触发者。首次用车波次取决策帧首次观测。关联为描述性，按关差值仅使用同关同时有/无该卡或组件的卡组。')
    target = ROOT / 'MEMORY/docs/task129-results.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'games', 'won', 'completed_decks', 'by_level')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
