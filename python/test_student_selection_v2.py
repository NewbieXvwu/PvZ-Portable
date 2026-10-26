"""task92b: incumbent-only screening, ties, and paired resampling."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task92b_selection import eligible_windows, kendall_tau, summarize
from pvz_tasks import deck_for_seed, random_deck_covered, random_deck_draws
from gate_seed_inventory import scan_json, scan_roots_once, verify_window


def test_kendall_handles_ties_and_reversed_order():
    assert kendall_tau([4, 3, 2, 1], [1, 2, 3, 4]) == -1
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1
    assert kendall_tau([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert kendall_tau([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(np.sqrt(5 / 6))


def test_seed_audit_deduplicates_nested_roots_without_losing_records(tmp_path):
    child = tmp_path / 'worktrees/child'
    child.mkdir(parents=True)
    (child / 'manifest.json').write_text('{"tasks":[{"seeds":[126000001]}]}')
    roots = [str(child), str(tmp_path), str(child)]
    assert scan_roots_once(roots) == [str(tmp_path)]
    assert scan_json(roots)['files_parsed'] == 1
    verification = verify_window(roots, 126000000, 126020000)
    assert verification['files'] == 1
    assert [hit['value'] for hit in verification['hits']] == [126000001]


def test_bootstrap_keeps_policy_outcomes_paired_and_stable_ties():
    # Paired all-model ties cannot introduce any apparent winner advantage.
    report = summarize([np.array([[0, 0, 0, 0], [1, 1, 1, 1]])], draws=100, seed=7)
    assert report['selected'] == 'a'
    assert report['bootstrap_winner_probability'] == dict(a=1, c=0, d2=0, r11=0)
    assert not report['acceptance_passed']
    report = summarize([np.tile([0, 0, 1, 0], (10, 1))], draws=100, seed=7)
    assert report['bootstrap_heldout_best_probability'] == 1
    assert report['acceptance_passed']


def test_random_family_requires_exact_engine_validated_receipt():
    task = dict(level=7, deck=[], deck_random=dict(
        engine_options=dict(level=7, playthrough=2, available_plants=list(range(40))),
        excluded_decks=[], validated_decks={}))
    attempt, deck = next(random_deck_draws(task, 126016000))
    task['deck_random']['validated_decks']['126016000'] = dict(attempt=attempt, deck=list(deck))
    assert deck_for_seed(task, 126016000) == deck
    with pytest.raises(ValueError, match='no engine-validation receipt'):
        deck_for_seed(task, 126016001)
    task['deck_random']['validated_decks']['126016000']['deck'] = list(reversed(deck))
    with pytest.raises(ValueError, match='engine-validated receipt'):
        deck_for_seed(task, 126016000)
    pool = dict(task, level=21)
    assert not random_deck_covered(pool, [0, 1, 2, 3, 4, 5])


def test_window_screening_uses_r11_only_excludes_mixed_and_checks_suffix(tmp_path):
    r11 = tmp_path / 'r11.pt'
    r11.write_bytes(b'r11')
    groups = [dict(base='t', multiplier=1., beta=0., student_round=5, label='old', games=2),
              dict(base='t', multiplier=1., beta=.5, student_round=11, label='mixed', games=2),
              dict(base='t', multiplier=1., beta=0., student_round=11, label='new', games=2,
                   task_id='t', manifest='/source.json')]
    folder = tmp_path / 'new_2'
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps(dict(rollout_policy=str(r11), dagger_beta='0', task_id='t')))
    (folder / 'games.jsonl').write_text('\n'.join(json.dumps(dict(won=bool(w), terminal=True)) for w in [1, 0]))
    state = dict(student=str(r11), student_round=11, log=[dict(groups=groups)],
                 cells=dict(t=dict(windows={'1.0': dict(recent=[1, 1, 1, 0])})))
    cells, _ = eligible_windows(state, runs=tmp_path, r11=r11)
    assert len(cells) == 1
    assert cells[0]['rate'] == .5 and cells[0]['games'] == 2
    state['cells']['t']['windows']['1.0']['recent'][-1] = 1
    with pytest.raises(ValueError, match='window suffix'):
        eligible_windows(state, runs=tmp_path, r11=r11)
