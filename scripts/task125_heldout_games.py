"""task125: read the round-21 gate's per-game records; no game process, no replay.

Answers "where does the candidate diverge on the two heldout tasks" from
round_000021/evaluations/<hash>/{candidate,incumbent}/<task>/games.jsonl.
"""
import collections
import json
import sys
from pathlib import Path

BASE = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')
NAMES = {0: 'Peashooter', 1: 'Sunflower', 2: 'CherryBomb', 3: 'Wallnut', 4: 'PotatoMine',
         5: 'SnowPea', 7: 'Repeater', 16: 'Lilypad', 27: 'Blover', 32: 'CabbagePult',
         33: 'FlowerPot', 38: 'Marigold', 43: 'Cattail'}
DECKS = {'day_l7_eval_d7p_holdout1': [0, 1, 2, 27, 5, 32, 3],
         'day_l9_eval_d7p_holdout2': [7, 2, 5, 38, 4, 1, 3]}
TASKS = ('day_l7_eval_d7', 'day_l10_orig_eval', 'day_l7_eval_d7p_holdout1', 'day_l9_eval_d7p_holdout2')


def load(root: Path, side: str, task: str) -> dict:
    games = {}
    with open(root / side / task / 'games.jsonl') as f:
        for line in f:
            row = json.loads(line)
            games[row['seed']] = row
    return games


def main() -> None:
    round_dir = BASE / 'round_000021'
    root = Path(json.loads((round_dir / 'complete.json').read_text())['evaluation_directory'])
    receipt = {'evaluation_directory': str(root), 'tasks': {}}
    for task in TASKS:
        if not (root / 'candidate' / task / 'games.jsonl').is_file():
            continue
        cand, inc = load(root, 'candidate', task), load(root, 'incumbent', task)
        seeds = sorted(set(cand) & set(inc))
        n = len(seeds)
        harm = [s for s in seeds if not cand[s]['won'] and inc[s]['won']]
        entry = {'seeds': n,
                 'candidate_won': sum(cand[s]['won'] for s in seeds),
                 'incumbent_won': sum(inc[s]['won'] for s in seeds),
                 'harm': len(harm),
                 'gain': sum(1 for s in seeds if cand[s]['won'] and not inc[s]['won'])}
        if task in DECKS:
            deck = DECKS[task]
            # slot -> SeedType is only trustworthy if every game's slot/type counts pair up
            mismatches = sum(1 for g in (cand, inc) for r in g.values()
                             for slot, count in r['planted_slots'].items()
                             if str(deck[int(slot)]) not in r['planted_types']
                             or r['planted_types'][str(deck[int(slot)])] != count)
            entry['deck'] = deck
            entry['slot_type_mismatches'] = mismatches
        for label, games in (('candidate', cand), ('incumbent', inc)):
            mix = collections.Counter()
            for s in seeds:
                mix.update({int(k): v for k, v in games[s]['planted_types'].items()})
            entry[label] = {
                'plants_per_game': round(sum(mix.values()) / n, 2),
                'mowers_per_game': round(sum(games[s]['mowers_used'] for s in seeds) / n, 2),
                'mean_final_wave': round(sum(games[s]['final_wave'] for s in seeds) / n, 2),
                'mix_per_game': {NAMES.get(k, str(k)): round(v / n, 2) for k, v in mix.most_common(8)},
            }
        if harm:
            entry['harm_only'] = {}
            for label, games in (('candidate', cand), ('incumbent', inc)):
                mix = collections.Counter()
                for s in harm:
                    mix.update({int(k): v for k, v in games[s]['planted_types'].items()})
                entry['harm_only'][label] = {
                    'plants_per_game': round(sum(mix.values()) / len(harm), 2),
                    'mowers_per_game': round(sum(games[s]['mowers_used'] for s in harm) / len(harm), 2),
                    'mean_final_wave': round(sum(games[s]['final_wave'] for s in harm) / len(harm), 2),
                    'mix_per_game': {NAMES.get(k, str(k)): round(v / len(harm), 2)
                                     for k, v in mix.most_common(8)},
                }
        receipt['tasks'][task] = entry
    out = Path(__file__).resolve().parents[1] / 'artifacts/task125/heldout_games.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
