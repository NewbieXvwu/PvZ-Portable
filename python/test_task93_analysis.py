"""Integrity checks for task93's paired evidence, independent of model behavior."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task93_analyze import comparison, load_games, spread


def test_missing_duplicate_wrong_arm_deck_and_nonterminal_are_rejected(tmp_path):
    valid = [dict(seed=s, arm='fast', terminal=True, won=bool(s % 2), deck=[0, 1])
             for s in (1, 2)]
    path = tmp_path / 'games.jsonl'
    decks = {s: [0, 1] for s in (1, 2)}
    corruptions = [valid[:1], valid + valid[:1]]
    for field, value in [('arm', 'fast_search'), ('deck', [1, 0]), ('terminal', False),
                         ('future_seed', 123), ('won', 1)]:
        corruptions.append([{**valid[0], field: value}, valid[1]])
    for rows in corruptions:
        path.write_text('\n'.join(map(json.dumps, rows)))
        with pytest.raises(ValueError):
            load_games(path, (1, 2), 'fast', decks)
    path.write_text('\n'.join(map(json.dumps, valid)))
    assert set(load_games(path, (1, 2), 'fast', decks)) == {1, 2}


def test_spread_uses_percentage_points_not_wins():
    result = spread([124, 169], 192)
    assert result['mean_pct'] == pytest.approx(76.3020833333)
    assert result['range_pp'] == pytest.approx(23.4375)
    assert result['sample_std_pp'] == pytest.approx(16.5728151841)


def test_paired_direction_and_exact_p():
    candidate = {s: {'won': s < 8} for s in range(8)}
    baseline = {s: {'won': s < 2} for s in range(8)}
    report = comparison(candidate, baseline, range(8))
    assert (report['gain'], report['harm']) == (6, 0)
    assert report['mcnemar_exact_two_sided'] == pytest.approx(0.03125)
    assert report['delta_rate'] == 0.75
