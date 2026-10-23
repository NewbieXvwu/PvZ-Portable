"""Quality promotion gates: paired inference, complete coverage and zone-specific regressions."""
import copy
import math

import pytest

from pvz_promotion import PANEL_TASKS, decide, freeze_panel, paired_interval, summarize


def manifest():
    tasks = [dict(task_id=name, level=1, seeds=list(range(30000, 30128)), deck=[0, 1, 2, 3, 4, 5],
                  zombie_count_multiplier=1., wave_cap=None) for _, name in PANEL_TASKS]
    return freeze_panel(dict(panel_version='v2', groups={name: zone for zone, name in PANEL_TASKS}, tasks=tasks))


def outcomes(panel, candidate, incumbent):
    return [dict(task_id=task['task_id'], seed=seed, zone=panel['groups'][task['task_id']],
                 candidate=candidate, incumbent=incumbent) for task in panel['tasks'] for seed in task['seeds']]


def test_newcombe_published_reference_and_edge_uncertainty():
    # Official statsmodels method-10 example [[10,5],[2,20]], independently published limits.
    low, high = paired_interval(10, 5, 2, 20)
    assert low == pytest.approx(-.0643926050711, abs=1e-12)
    assert high == pytest.approx(.2211160178958, abs=1e-12)
    low, high = paired_interval(64, 0, 0, 0)
    assert low < -.03 and high > 0  # no fictitious zero-width CI for all-win A/A
    with pytest.raises(ValueError):
        paired_interval(0, 0, 0, 0)


def test_real_improvement_passes_and_reversed_regression_fails():
    panel = manifest()
    positive = decide(outcomes(panel, True, False), panel)
    assert positive['approved'] and positive['overall']['ci95'][0] > -.03
    negative = decide(outcomes(panel, False, True), panel)
    assert not negative['approved']
    assert all('significant_zone_regression:' + zone in negative['reasons'] for zone in positive['zones'])


def test_zone_veto_blocks_overall_improvement():
    panel = manifest()
    pairs = outcomes(panel, True, False)
    for row in pairs:
        if row['zone'] == 'night':
            row.update(candidate=False, incumbent=True)
    result = decide(pairs, panel)
    assert result['overall']['ci95'][0] > -.03
    assert not result['approved'] and result['reasons'] == ['significant_zone_regression:night']


def test_panel_is_unchanged_except_seed_subset_and_outcomes_are_complete():
    panel = manifest()
    assert len({seed for task in panel['tasks'] for seed in task['seeds']}) == 64
    pairs = outcomes(panel, True, False)
    for broken in (pairs[:-1], pairs + [pairs[0]], pairs[:-1] + [pairs[0]]):
        with pytest.raises(ValueError):
            decide(broken, panel)
    wrong = copy.deepcopy(pairs)
    wrong[0]['zone'] = 'pool'
    with pytest.raises(ValueError):
        decide(wrong, panel)
    assert summarize([dict(candidate=True, incumbent=False)])['mcnemar_worse_p'] == 1
