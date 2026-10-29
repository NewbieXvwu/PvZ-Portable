"""task168: exact signed ranks and paired wave recording, no games."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_promotion import decide, decide_scored, freeze_panel, policy, summarize, summarize_waves, wilcoxon_worse
from promotion_gate import _paired_rows


@pytest.mark.parametrize('recorded_waves', [False, True])
def test_offline_roof_replay_without_contract_hashes(tmp_path, recorded_waves):
    from promotion_gate import contract
    from replay_roof_wave_gate import replay

    folder = tmp_path / 'round_1/evaluations/model'
    folder.mkdir(parents=True)
    pair = dict(task_id='roof', seed=1, zone='roof', candidate=True, incumbent=False)
    if recorded_waves:
        pair.update(candidate_wave=12, incumbent_wave=10)
    else:
        for side, won, wave in [('candidate', True, 12), ('incumbent', False, 10)]:
            episode = folder / side / 'roof'
            episode.mkdir(parents=True)
            (episode / 'games.jsonl').write_text(json.dumps(dict(seed=1, won=won, final_wave=wave)) + '\n')
    frozen = contract('candidate.pt', 'incumbent.pt', 'panel', 'engine', 'value', 'resources')
    (folder / 'decision.json').write_text(json.dumps(dict(contract=frozen, paired=[pair],
                                                        approved=False, reasons=['old decision'])))
    result = replay(tmp_path)
    assert result['rounds'][0]['round'] == 1
    assert result['rounds'][0]['roof_pairs']['roof'] == [[1, 12, 10, True, False]]
    assert result['rounds'][0]['zones']['roof']['waves']['mean_delta_wave'] == 2


@pytest.mark.parametrize('differences,statistic,p', [
    ([0, 0], 0, 1), ([-1, -1], 0, .25), ([1, 1], 3, 1),
    ([-1, 0, 1], 1.5, .75), ([-1, 2, -3], 2, .375),
    # Wilcoxon's published corn example, sign reversed for the lower-tail alternative.
    ([-6, -8, -14, -16, -23, -24, -28, -29, -41, 48, -49, -56, -60, 67, -75],
     24, .0206298828125),
])
def test_exact_wilcoxon_roof_ties_zeros_and_direction(differences, statistic, p):
    result = wilcoxon_worse(differences)
    assert result['positive_rank_sum'] == statistic
    assert result['wilcoxon_worse_p'] == p
    assert result['nonzero_pairs'] == sum(d != 0 for d in differences)


def test_roof_wave_recording():
    task = dict(task_id='roof', seeds=[2, 1])
    outcomes = dict(candidate={1: dict(won=False, final_wave=5), 2: dict(won=False, final_wave=8)},
                    incumbent={1: dict(won=False, final_wave=6), 2: dict(won=False, final_wave=7)})
    recorded = _paired_rows(task, 'roof', outcomes)
    assert [(row['seed'], row['candidate_wave'], row['incumbent_wave']) for row in recorded] == [(2, 8, 7), (1, 5, 6)]
    assert summarize_waves(recorded)['wilcoxon_worse_p'] == .75
    assert summarize_waves(recorded)['mean_delta_wave'] == 0


def v6_pairs():
    source = json.loads((ROOT / 'configs/gate_v5_panel.json').read_text())
    panel = freeze_panel(source, version=6, seed_block=(100_000_000, 100_001_920))
    pairs = [dict(task_id=task['task_id'], seed=seed, zone=panel['groups'][task['task_id']],
                  candidate=True, incumbent=True, candidate_wave=12, incumbent_wave=12)
             for task in panel['tasks'] for seed in task['seeds']]
    return panel, pairs


def test_v6_roof_endpoint_uses_90_pairs_and_preserves_overall_and_monitoring():
    panel, pairs = v6_pairs()
    assert panel['panel_version'] == 'v6-task168'
    assert policy()['pair_count'] == 1920
    for row in pairs:
        if row['zone'] in ('roof', 'heldout'):
            row.update(candidate=False, incumbent=False)
        if row['zone'] == 'roof':
            row.update(candidate_wave=10, incumbent_wave=20)
    result = decide(pairs, panel)
    assert not result['approved']
    assert result['reasons'] == ['significant_roof_wave_regression:roof']
    assert result['overall'] == summarize([row for row in pairs if row['zone'] != 'heldout'])
    assert all(cell['harm'] == 0 for cell in result['monitor'].values())
    assert result['zones']['roof']['waves']['pairs'] == 90
    assert result['zones']['roof']['waves']['wilcoxon_worse_p'] == 2**-90
    assert 'holm_adjusted_p' not in result['zones']['roof']
    for row in pairs:
        if row['zone'] == 'roof':
            row.update(candidate_wave=20, incumbent_wave=10)
    assert decide(pairs, panel)['approved']


def test_v6_four_binary_zones_keep_mcnemar_and_use_four_way_holm():
    panel, pairs = v6_pairs()
    for row in pairs[:7]:
        row['candidate'] = False
    scored = [row for row in pairs if row['zone'] != 'heldout']
    new = decide(pairs, panel)
    assert summarize(scored) == new['overall']
    for zone in ('day', 'night', 'pool', 'fog'):
        assert {key: value for key, value in new['zones'][zone].items() if key != 'holm_adjusted_p'} == \
            summarize([row for row in scored if row['zone'] == zone])
    assert new['zones']['day']['holm_adjusted_p'] == 4 / 128
    assert 'significant_zone_regression:day' in new['reasons']
    pairs[-1]['candidate'] = False  # A heldout loss still cannot veto.
    assert decide(pairs, panel)['reasons'] == new['reasons']


@pytest.mark.parametrize('round_number,approved', [(26, False), (27, True), (28, False), (29, False), (36, True)])
def test_v6_preregistered_historical_roof_replay(round_number, approved):
    evidence = json.loads((ROOT / 'MEMORY/docs/task168-roof-wave-replay.json').read_text())
    record = next(row for row in evidence['rounds'] if row['round'] == round_number)
    pairs = []
    for zone, counts in record['zones'].items():
        if zone == 'roof':
            continue
        for name, candidate, incumbent in [('both_win', True, True), ('gain', True, False),
                                            ('harm', False, True), ('both_lose', False, False)]:
            pairs.extend(dict(zone=zone, candidate=candidate, incumbent=incumbent) for _ in range(counts[name]))
    for name, rows in record['roof_pairs'].items():
        pairs.extend(dict(task_id=name, seed=seed, zone='roof', candidate=candidate, incumbent=incumbent,
                          candidate_wave=cwave, incumbent_wave=iwave)
                     for seed, cwave, iwave, candidate, incumbent in rows)
    result = decide_scored(pairs)
    assert result['approved'] is approved
    assert result['reasons'] == record['new_reasons']
    if round_number in (27, 29):
        expected = (5.53, 3.11) if round_number == 27 else (-2.09, -4.64)
        tasks = result['zones']['roof']['wave_tasks']
        assert [tasks[name]['mean_delta_wave'] for name in sorted(tasks)] == pytest.approx(expected, abs=.005)
        assert result['zones']['roof']['waves']['veto'] is (round_number == 29)
    if round_number == 29:
        assert result['zones']['roof']['wave_tasks']['task101_l41_roof_demo']['wilcoxon_worse_p'] > .01
