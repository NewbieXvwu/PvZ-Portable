"""Quality promotion gate (v5): paired inference, complete coverage, zone vetoes."""
import copy
import json
from pathlib import Path

import pytest

from pvz_promotion import (GATE_VERSION, PANEL_TASKS_V5, SEED_POOLS_V2, V5_MONITOR_TASKS,
                           decide, decide_scored, freeze_panel, paired_interval, policy, summarize)

ROOT = Path(__file__).resolve().parents[1]
ADVENTURE_ZONES = ('day', 'fog', 'night', 'pool', 'roof')


def v5_source():
    return json.loads((ROOT / 'configs/gate_v5_panel.json').read_text())


def v5_panel(start=100_000_000):
    return freeze_panel(v5_source(), version=GATE_VERSION, seed_block=(start, start + 1920))


def outcomes(panel, candidate, incumbent):
    return [dict(task_id=task['task_id'], seed=seed, zone=panel['groups'][task['task_id']],
                 candidate=candidate, incumbent=incumbent,
                 candidate_wave=20 if candidate else 10, incumbent_wave=20 if incumbent else 10)
            for task in panel['tasks'] for seed in task['seeds']]


def paired_tables(panel, tables):
    """Rewrite a both-win panel so each zone shows (both_win, gain, harm, both_lose) counts."""
    pairs = outcomes(panel, True, True)
    values = [(True, True), (True, False), (False, True), (False, False)]
    for zone, counts in tables.items():
        rows = [row for row in pairs if row['zone'] == zone]
        assert len(rows) == sum(counts)
        offset = 0
        for count, (candidate, incumbent) in zip(counts, values):
            for row in rows[offset:offset + count]:
                row.update(candidate=candidate, incumbent=incumbent)
            offset += count
    return pairs


def zone_sizes(panel):
    sizes = {}
    for task in panel['tasks']:
        if task['task_id'] not in V5_MONITOR_TASKS:
            sizes[panel['groups'][task['task_id']]] = sizes.get(panel['groups'][task['task_id']], 0) + len(task['seeds'])
    return sizes


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
    panel = v5_panel()
    positive = decide(outcomes(panel, True, False), panel)
    assert positive['approved'] and positive['overall']['ci95'][0] > -.03
    negative = decide(outcomes(panel, False, True), panel)
    assert not negative['approved']
    assert all('significant_zone_regression:' + zone in negative['reasons']
               for zone in set(positive['zones']) - {'roof'})
    assert 'significant_roof_wave_regression:roof' in negative['reasons']


def test_zone_veto_blocks_overall_improvement():
    panel = v5_panel()
    pairs = outcomes(panel, True, False)
    for row in pairs:
        if row['zone'] == 'night':
            row.update(candidate=False, incumbent=True)
    result = decide(pairs, panel)
    assert result['overall']['ci95'][0] > -.03
    assert not result['approved'] and result['reasons'] == ['significant_zone_regression:night']


def test_panel_covers_every_registered_pair_exactly_once():
    panel = v5_panel()
    assert len({seed for task in panel['tasks'] for seed in task['seeds']}) == 1920
    pairs = outcomes(panel, True, False)
    for broken in (pairs[:-1], pairs + [pairs[0]], pairs[:-1] + [pairs[0]]):
        with pytest.raises(ValueError):
            decide(broken, panel)
    wrong = copy.deepcopy(pairs)
    wrong[0]['zone'] = 'heldout'
    with pytest.raises(ValueError):
        decide(wrong, panel)
    assert summarize([dict(candidate=True, incumbent=False)])['mcnemar_worse_p'] == 1


def test_single_strong_zone_regression_still_vetoes_under_holm():
    panel = v5_panel()
    # Eight harms, zero gains in day: p = 1/256 < .05/5. Overall stays noninferior.
    day = [row for row in outcomes(panel, True, True) if row['zone'] == 'day']
    for row in day[:8]:
        row.update(candidate=False)
    result = decide(day + [row for row in outcomes(panel, True, True) if row['zone'] != 'day'], panel)
    assert result['overall']['ci95'][0] > -.03
    assert result['zones']['day']['mcnemar_worse_p'] == 1/256
    assert result['reasons'] == ['significant_zone_regression:day']


def test_holm_stepdown_stops_after_the_first_nonrejection():
    panel = v5_panel()
    sizes = zone_sizes(panel)
    harms = dict(zip(['day', 'night', 'pool', 'fog'], (8, 7, 5, 4)))
    tables = {zone: (sizes[zone] - count, 0, count, 0) for zone, count in harms.items()}
    result = decide(paired_tables(panel, tables), panel)
    assert result['reasons'] == ['significant_zone_regression:day',
                                 'significant_zone_regression:night']
    assert result['zones']['fog']['holm_adjusted_p'] == pytest.approx(2 * 1/32)
    assert 'holm_adjusted_p' not in result['zones']['roof']
    # An overall regression stays an independent reason on the same step-down rules.
    assert 'overall_ci_lower_not_above_minus_3pp' in decide(outcomes(panel, False, True), panel)['reasons']


def test_ordered_gate_pools_never_overlap_and_fit_many_rounds():
    assert len(SEED_POOLS_V2) >= 2
    ordered = sorted(SEED_POOLS_V2)
    assert ordered == list(SEED_POOLS_V2)  # consumption order is the registered order
    for (low, high), (next_low, _) in zip(ordered, ordered[1:]):
        assert low < high <= next_low  # disjoint, never touching
    assert SEED_POOLS_V2[-1][1] - SEED_POOLS_V2[-1][0] >= 200 * 1920


def test_policy_and_committed_source_panel_are_the_same_adventure_set():
    assert GATE_VERSION == 6
    v5 = policy()
    assert v5['pair_count'] == 1920 and v5['episode_count'] == 3840
    assert v5['seeds_per_task'] == 45 and v5['monitor_seeds_per_task'] == 96
    assert len(v5['tasks']) == 40 and sum(task['seeds'] for task in v5['tasks']) == 1920
    assert [task['task_id'] for task in v5['tasks'] if task['monitor']] == list(V5_MONITOR_TASKS)
    assert 'monitor only' in v5['heldout'] and 'never enters the overall rate' in v5['heldout']
    document = v5_source()
    assert [(task['task_id'], document['groups'][task['task_id']]) for task in document['tasks']] == \
        [(name, zone) for zone, name, _ in PANEL_TASKS_V5]
    for version in (1, 2, 3, 4, 5):
        with pytest.raises(ValueError):
            policy(version)


def test_panel_partitions_one_block_and_scores_only_adventure_pairs():
    panel = v5_panel()
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    assert panel['panel_version'] == 'v6-task168'
    assert len(seeds) == len(set(seeds)) == 1920
    assert min(seeds) == 100_000_000 and max(seeds) == 100_001_919
    assert sorted(set(panel['groups'].values())) == sorted(ADVENTURE_ZONES + ('heldout',))
    counts = [len(task['seeds']) for task in panel['tasks'] if task['task_id'] not in V5_MONITOR_TASKS]
    assert sorted(counts) == [45] * 20 + [46] * 18
    result = decide(outcomes(panel, True, False), panel)
    assert result['version'] == 6 and result['approved']
    assert result['overall']['pairs'] == 1728                # heldout never enters the overall rate
    assert sorted(result['zones']) == list(ADVENTURE_ZONES)
    assert sorted(result['monitor']) == list(V5_MONITOR_TASKS)
    assert all(cell['pairs'] == 96 for cell in result['monitor'].values())


def test_second_pool_window_freezes_and_blocks_outside_the_pools_are_rejected():
    start = SEED_POOLS_V2[1][0]
    panel = v5_panel(start)
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    assert panel['seed_block'] == [start, start + 1920]
    assert min(seeds) == start and max(seeds) == start + 1919
    for block in ((SEED_POOLS_V2[0][1] - 1, SEED_POOLS_V2[0][1] - 1 + 1920),
                  (SEED_POOLS_V2[0][1], SEED_POOLS_V2[0][1] + 1920),
                  (2_000_000_000, 2_000_001_920)):
        with pytest.raises(ValueError):
            freeze_panel(v5_source(), version=GATE_VERSION, seed_block=block)


@pytest.mark.parametrize('zone', ['pool', 'roof'])
def test_heldout_collapse_never_vetoes_while_a_zone_collapse_still_does(zone):
    panel = v5_panel()
    pairs = outcomes(panel, True, True)
    for row in pairs:
        if row['task_id'] in V5_MONITOR_TASKS:
            row.update(candidate=False, incumbent=True)
    result = decide(pairs, panel)
    assert result['approved'] and result['reasons'] == []
    assert all(cell['harm'] == 96 for cell in result['monitor'].values())
    assert result['overall']['harm'] == 0 and result['overall']['pairs'] == 1728
    for row in pairs:
        if row['zone'] == zone:
            row.update(candidate=False, incumbent=True)
            row.update(candidate_wave=10, incumbent_wave=20)
    reasons = decide(pairs, panel)['reasons']
    assert ('significant_roof_wave_regression:roof' if zone == 'roof'
            else 'significant_zone_regression:' + zone) in reasons
    assert not any('heldout' in reason for reason in reasons)


def test_freeze_pins_the_block_size_source_version_and_difficulty():
    document = v5_source()
    with pytest.raises(ValueError):
        freeze_panel(document, version=GATE_VERSION)                                   # block required
    with pytest.raises(ValueError, match='wrong size'):
        freeze_panel(document, version=GATE_VERSION, seed_block=(100_000_000, 100_001_919))
    with pytest.raises(ValueError):
        freeze_panel(document, version=GATE_VERSION, seed_block=(2_000_000_000, 2_000_001_920))
    with pytest.raises(ValueError):
        freeze_panel(v5_source(), version=4, seed_block=(100_000_000, 100_001_920))
    swapped = copy.deepcopy(document)
    swapped['groups']['task101_l7_day_d7'] = 'night'
    with pytest.raises(ValueError, match='task101_l7_day_d7'):
        freeze_panel(swapped, version=GATE_VERSION, seed_block=(100_000_000, 100_001_920))
    easier = copy.deepcopy(document)
    easier['tasks'][0]['zombie_count_multiplier'] = 0.5
    with pytest.raises(ValueError, match='original-difficulty'):
        freeze_panel(easier, version=GATE_VERSION, seed_block=(100_000_000, 100_001_920))


def test_decide_scored_is_the_shared_rule_for_scored_pairs():
    rows = [dict(zone='day', candidate=True, incumbent=True)] * 400
    rows += [dict(zone='night', candidate=False, incumbent=True)] * 8
    result = decide_scored(rows + [dict(zone='night', candidate=True, incumbent=True)] * 400)
    assert result['version'] == 6 and not result['approved']
    assert result['reasons'] == ['significant_zone_regression:night']
