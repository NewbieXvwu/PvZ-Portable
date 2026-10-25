"""Quality promotion gates: paired inference, complete coverage and zone-specific regressions."""
import copy

import pytest

from pvz_promotion import (PANEL_TASKS, PANEL_TASKS_V3, SEED_POOLS_V2, SEED_POOL_V2, decide,
                           freeze_panel, paired_interval, policy, summarize)


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
V3_BLOCK_SIZE = 192 * 10


def source_document():
    return dict(panel_version='v2', groups={name: zone for zone, name in PANEL_TASKS},
                tasks=[dict(task_id=name, seeds=list(range(30000, 30128)), wave_cap=None,
                            zombie_count_multiplier=1, level=1, deck=[0, 1, 2, 3, 4, 5])
                       for _, name in PANEL_TASKS])


def v3_source_document():
    """The registered unified v2 schema plus the practiced fog/roof cells (task73c)."""
    document = source_document()
    for zone, name in PANEL_TASKS_V3:
        if (zone, name) not in PANEL_TASKS:
            document['groups'][name] = zone
            document['tasks'].append(dict(task_id=name, seeds=list(range(43000000, 43000128)),
                                          wave_cap=None, zombie_count_multiplier=1, level=31,
                                          deck=[0, 1, 2, 3, 4, 5]))
    return document


def v2_panel(start=31024):
    return freeze_panel(source_document(), version=2, seed_block=(start, start + V2_BLOCK_SIZE))


def v3_panel(start=31024):
    return freeze_panel(v3_source_document(), version=3, seed_block=(start, start + V3_BLOCK_SIZE))


# ---- task88: ordered gate pool list (second window after the first is spent) ----

def test_gate_pools_are_ordered_disjoint_and_keep_the_legacy_first_window():
    assert SEED_POOL_V2 == (31024, 65536)  # v2/v3 were registered against this window
    assert SEED_POOLS_V2[0] == SEED_POOL_V2
    assert len(SEED_POOLS_V2) >= 2
    ordered = sorted(SEED_POOLS_V2)
    assert ordered == list(SEED_POOLS_V2)  # consumption order is the registered order
    for (low, high), (next_low, _) in zip(ordered, ordered[1:]):
        assert low < high <= next_low  # disjoint, never touching
    # The new window must fit >= 200 v3 rounds (1920 pairs each) with room to spare.
    assert SEED_POOLS_V2[-1][1] - SEED_POOLS_V2[-1][0] >= 200 * V3_BLOCK_SIZE


def test_frozen_panel_for_an_issued_block_stays_byte_identical():
    """Lock the v2/v3 panels of an already-issued block: task88 must not move a seed.

    Both digests were taken from the code as of commit 04b0e75b, before the pool
    list existed. A change here means an old certificate no longer replays.
    """
    import hashlib
    import json
    for panel, digest in ((v2_panel(), '3b6e2e62031345294a534f8dc07bd448e23274b867f40859cf3eff67335a361d'),
                          (v3_panel(), 'f29f3c828117447b9c6fceb596dbadb39175f425ab20b6c9f0e2984be6435c99')):
        assert hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest() == digest


def test_second_window_freezes_and_pools_outside_the_list_are_rejected():
    start = SEED_POOLS_V2[1][0]
    panel = v3_panel(start)
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    assert panel['seed_block'] == [start, start + V3_BLOCK_SIZE]
    assert min(seeds) == start and max(seeds) == start + V3_BLOCK_SIZE - 1
    assert len(set(seeds)) == V3_BLOCK_SIZE
    # A block that straddles the end of a pool, or sits between pools, is refused.
    for block in ((SEED_POOLS_V2[0][1] - 1, SEED_POOLS_V2[0][1] - 1 + V3_BLOCK_SIZE),
                  (SEED_POOLS_V2[0][1], SEED_POOLS_V2[0][1] + V3_BLOCK_SIZE),
                  (2_000_000_000, 2_000_000_000 + V3_BLOCK_SIZE)):
        with pytest.raises(ValueError):
            freeze_panel(v3_source_document(), version=3, seed_block=block)


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
        policy(4)


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
        freeze_panel(source_document(), version=4, seed_block=(31024, 31024 + V2_BLOCK_SIZE))


# ---- task73c: gate v3 (fog + roof zones, replaced holdout, 1920 pairs) ----

V3_ZONES = ('day', 'fog', 'heldout', 'night', 'pool', 'roof')


def test_v3_policy_covers_practiced_zones_and_keeps_older_policies_frozen():
    v3 = policy(3)
    assert v3['version'] == 3 and v3['pair_count'] == 1920 and v3['seeds_per_task'] == 192
    assert v3['episode_count'] == 3840 and v3['seed_pool'] == [31024, 65536]
    zones = [task['zone'] for task in v3['tasks']]
    assert zones.count('fog') == 1 and zones.count('roof') == 1
    names = [task['task_id'] for task in v3['tasks']]
    assert 'day_l9_eval_d7p_holdout1' not in names          # floor task is gone
    assert 'day_l9_eval_d7p_holdout2' in names              # measured in-band-adjacent replacement
    assert 'fog_user_l31_snow' in names and 'roof_l41_train_r7' in names
    # v1/v2 policy dicts are embedded in old certificate contracts: byte-for-byte stable.
    assert policy(2)['pair_count'] == 1536 and len(policy(2)['tasks']) == 8
    assert policy(1)['pair_count'] == 64


def test_v3_panel_partitions_block_across_six_zones():
    panel = v3_panel()
    assert panel['panel_version'] == 'v3-task73c' and panel['seed_block'] == [31024, 31024 + V3_BLOCK_SIZE]
    assert sorted(set(panel['groups'].values())) == sorted(V3_ZONES)
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    assert len(seeds) == 1920 and len(set(seeds)) == 1920
    assert min(seeds) == 31024 and max(seeds) == 31024 + V3_BLOCK_SIZE - 1
    per_zone = {}
    for task in panel['tasks']:
        per_zone[panel['groups'][task['task_id']]] = per_zone.get(panel['groups'][task['task_id']], 0) + len(task['seeds'])
    assert per_zone == {'day': 384, 'fog': 192, 'heldout': 384, 'night': 384, 'pool': 384, 'roof': 192}
    positive = decide(outcomes(panel, True, False), panel)
    assert positive['version'] == 3 and positive['approved'] and positive['overall']['pairs'] == 1920
    assert sorted(positive['zones']) == sorted(V3_ZONES)
    negative = decide(outcomes(panel, False, True), panel)
    assert not negative['approved']
    assert all('significant_zone_regression:' + zone in negative['reasons'] for zone in V3_ZONES)


def test_v3_freeze_validates_source_panel_groups_and_block():
    document = v3_source_document()
    with pytest.raises(ValueError):
        freeze_panel(document, version=3)                                   # block required
    with pytest.raises(ValueError):
        freeze_panel(document, version=3, seed_block=(31024, 31024 + V3_BLOCK_SIZE - 1))
    with pytest.raises(ValueError):
        freeze_panel(document, version=3, seed_block=(30000, 30000 + V3_BLOCK_SIZE))
    with pytest.raises(ValueError):
        freeze_panel(document, version=3, seed_block=(65536 - V3_BLOCK_SIZE + 1, 65536 + 1))
    # A v3 task whose group was not registered under the v3 zone name is rejected.
    swapped = copy.deepcopy(document)
    swapped['groups']['fog_user_l31_snow'] = 'day'
    with pytest.raises(ValueError, match='fog_user_l31_snow'):
        freeze_panel(swapped, version=3, seed_block=(31024, 31024 + V3_BLOCK_SIZE))
    # Non-original difficulty stays rejected in v3 (the frozen fog/roof cells are 1.0).
    easier = copy.deepcopy(document)
    easier['tasks'][-1]['zombie_count_multiplier'] = 0.5
    with pytest.raises(ValueError, match='original-difficulty'):
        freeze_panel(easier, version=3, seed_block=(31024, 31024 + V3_BLOCK_SIZE))
    # The v2 spec must not accept a v3 task: panel membership is version-pinned.
    with pytest.raises(ValueError):
        freeze_panel(document, version=2, seed_block=(31024, 31024 + V3_BLOCK_SIZE))


def test_v3_decision_uses_zone_veto_on_new_zones_too():
    panel = v3_panel()
    pairs = outcomes(panel, True, False)
    for row in pairs:
        if row['zone'] in ('fog', 'roof'):
            row.update(candidate=False, incumbent=True)
    result = decide(pairs, panel)
    assert not result['approved']
    assert sorted(result['reasons']) == ['significant_zone_regression:fog',
                                         'significant_zone_regression:roof']
