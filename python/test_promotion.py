"""Quality promotion gates: paired inference, complete coverage and zone-specific regressions."""
import copy

import pytest

from pvz_promotion import PANEL_TASKS, decide, freeze_panel, paired_interval, policy, summarize


def manifest():
    tasks = [dict(task_id=name, level=1, seeds=list(range(30000, 30128)), deck=[0, 1, 2, 3, 4, 5],
                  zombie_count_multiplier=1., wave_cap=None) for _, name in PANEL_TASKS]
    return freeze_panel(dict(panel_version='v2', groups={name: zone for zone, name in PANEL_TASKS}, tasks=tasks), version=1)


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

# ---- task73b: gate v2 (fresh seeds per round, 1536 pairs, ledger pool) ----

V2_BLOCK_SIZE = 192 * 8


def source_document():
    return dict(panel_version='v2', groups={name: zone for zone, name in PANEL_TASKS},
                tasks=[dict(task_id=name, seeds=list(range(30000, 30128)), wave_cap=None,
                            zombie_count_multiplier=1, level=1, deck=[0, 1, 2, 3, 4, 5])
                       for _, name in PANEL_TASKS])


def v2_panel(start=31024):
    return freeze_panel(source_document(), version=2, seed_block=(start, start + V2_BLOCK_SIZE))


def test_v2_policy_records_fresh_seed_design_and_v1_stays_byte_compatible():
    v2 = policy(2)
    assert v2['version'] == 2 and v2['pair_count'] == 1536 and v2['seeds_per_task'] == 192
    assert v2['episode_count'] == 3072 and v2['seed_pool'] == [31024, 65536]
    assert 'never' in v2['seed_reuse'] and 'cached' in v2['incumbent_rerun']
    assert len(v2['tasks']) == 8
    # The v1 policy dict is embedded in every task49A certificate contract; it must
    # not gain, lose or reorder keys, or old certificates would stop verifying.
    v1 = policy(1)
    assert list(v1) == ['version', 'alpha', 'noninferiority_margin', 'overall_interval',
                        'zone_veto', 'pair_count', 'episode_count', 'seeds_per_task', 'tasks']
    assert v1['pair_count'] == 64 and v1['version'] == 1 and v1['seeds_per_task'] == 8
    with pytest.raises(ValueError):
        policy(3)


def test_v2_panel_partitions_block_and_decide_covers_1536_pairs():
    panel = v2_panel()
    assert panel['panel_version'] == 'v2-task73b' and panel['seed_block'] == [31024, 31024 + V2_BLOCK_SIZE]
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    assert len(seeds) == 1536 and len(set(seeds)) == 1536
    assert min(seeds) == 31024 and max(seeds) == 31024 + V2_BLOCK_SIZE - 1
    assert all(not (30000 <= seed < 31024) for seed in seeds)  # never dips into consumed windows
    positive = decide(outcomes(panel, True, False), panel)
    assert positive['version'] == 2 and positive['approved'] and positive['overall']['pairs'] == 1536
    negative = decide(outcomes(panel, False, True), panel)
    assert not negative['approved']
    assert all('significant_zone_regression:' + zone in negative['reasons'] for zone in positive['zones'])
    for broken in (outcomes(panel, True, False)[:-1], outcomes(panel, True, False) + [outcomes(panel, True, False)[0]]):
        with pytest.raises(ValueError):
            decide(broken, panel)


def test_v2_seed_block_is_validated_against_pool_and_size():
    with pytest.raises(ValueError):
        freeze_panel(source_document(), version=2)
    with pytest.raises(ValueError):
        freeze_panel(source_document(), version=2, seed_block=(31024, 31024 + V2_BLOCK_SIZE - 1))
    with pytest.raises(ValueError):
        freeze_panel(source_document(), version=2, seed_block=(30000, 30000 + V2_BLOCK_SIZE))
    with pytest.raises(ValueError):
        freeze_panel(source_document(), version=2, seed_block=(65536 - V2_BLOCK_SIZE + 1, 65536 + 1))
    with pytest.raises(ValueError):
        freeze_panel(source_document(), version=1, seed_block=(31024, 31024 + V2_BLOCK_SIZE))
    with pytest.raises(ValueError):
        freeze_panel(source_document(), version=3, seed_block=(31024, 31024 + V2_BLOCK_SIZE))

