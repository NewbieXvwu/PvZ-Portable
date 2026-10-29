import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import value_retrain as value


def comparisons():
    games, roots = [], []
    for zone in ('day', 'pool'):
        games.extend(dict(zone=zone, seed=seed, incumbent=0.3, candidate=0.2) for seed in range(8))
        for seed in range(4):
            roots.append(dict(zone=zone, seed=seed, candidates=['a', 'b', 'c', 'd'], student='a',
                              true=[0., 0., 1., 1.], true_wave=[1, 1, 2, 2],
                              values=dict(incumbent=[.8, .7, .2, .1], candidate=[.1, .2, .7, .8])))
    return games, roots


def test_value_loop_decision_improves_both_and_protects_each_terrain():
    games, roots = comparisons()
    assert value.decide(games, roots, {'day', 'pool'})['approved']
    for game in games:
        if game['zone'] == 'pool':
            game['candidate'] = .34
        else:
            game['candidate'] = .01
    result = value.decide(games, roots, {'day', 'pool'})
    assert not result['approved']
    assert 'pool: Brier worsened' in result['reasons']


def test_value_loop_rejects_equal_ranking_and_missing_terrain():
    games, roots = comparisons()
    for root in roots:
        root['values']['candidate'] = root['values']['incumbent']
    assert not value.decide(games, roots, {'day', 'pool'})['approved']
    assert not value.decide(games, roots, {'day', 'pool', 'roof'})['approved']


def test_value_loop_bounded_rows_and_warm_start(tmp_path, monkeypatch):
    import torch
    import train_value_net as training
    from pvz_value_net import ValueNet, load_value_net
    from pvz_frame_store import write_frames
    from test_value_net_rows import frames
    monkeypatch.setattr(training, 'CONTEXT', False)
    monkeypatch.setattr(training, 'ENCODER_VERSION', 1)
    torch.set_num_threads(1)
    records = []
    for seed in range(12):
        archive = tmp_path / f'seed_{seed}' / 'c1_fast_search'
        archive.mkdir(parents=True)
        write_frames(archive / 'frames.jsonl', frames(12))
        meta = archive / 'meta.json'
        meta.write_text(json.dumps(dict(task=dict(seed=seed), outcome=dict(won=seed % 2 == 0))))
        records.append(dict(path=str(meta), seed=seed, zone='day'))
    initial = ValueNet(width=4)
    checkpoint = dict(state_dict=initial.state_dict(), width=4, context=False, trained_on=[99])
    path = tmp_path / 'initial.pt'
    torch.save(checkpoint, path)
    request = dict(incumbent=str(path), data=str(tmp_path))
    out = tmp_path / 'out'
    out.mkdir()
    result = value.train_candidate(records, request, checkpoint, out)
    assert result['rows'] == 48
    assert load_value_net(out / 'model.pt').conv[0].out_channels == 4
    assert set(torch.load(out / 'model.pt', weights_only=False)['trained_on']) == set(range(12)) | {99}
    games, tensors = training.load_games(tmp_path, meta_paths=[Path(records[0]['path'])], frames_per_game=2)
    assert len(games) == 1 and len(tensors['labels']) == 2


def test_value_loop_demo_identity_preserves_future_and_original_start(tmp_path):
    archive = tmp_path / 'seed_5_f101' / 'c1_fast_search'
    archive.mkdir(parents=True)
    (archive / 'meta.json').write_text(json.dumps(dict(task=dict(seed=5), outcome=dict(won=True),
        demo_archive='/original', demo_decision=18)))
    request = dict(data=str(tmp_path), groups=[dict(label='c1', base='roof',
        definition=dict(level=42, deck=[0, 1, 2, 3, 32, 33], task_id='roof'))])
    record, = value.catalog(request)
    assert record['seed'] == 101 and record['base_seed'] == 5
    assert record['kwargs']['seed'] == 5
    assert record['kwargs']['demo_archive'] == '/original'
    assert record['kwargs']['demo_decision'] == 18


def test_value_loop_ranking_replays_demo_and_reseeds(monkeypatch):
    import human_demo
    import value_ranking_check as ranking
    from types import SimpleNamespace
    calls = []
    env = SimpleNamespace(reseed=lambda seed: calls.append(('reseed', seed)),
                          load_rollout_net=lambda path: None)
    def replay(env, archive, **kwargs):
        calls.append((str(archive), kwargs['decision']))
        return dict(terminal=True), None, []
    monkeypatch.setattr(human_demo, 'replay', replay)
    monkeypatch.setattr(ranking, '_W', dict(args=SimpleNamespace(dump_shards=None), env=env,
        student=SimpleNamespace(obs='v1'), weights='weights'))
    roots, leaves = ranking._game((101, dict(seed=5, level=42, deck=[0, 1, 2, 3, 32, 33],
        task_extra={}, demo_archive='/original', demo_decision=18)))
    assert roots == leaves == []
    assert calls == [('/original', 18), ('reseed', 101)]


def test_value_loop_monitor_warns_only_after_three_rounds(tmp_path):
    import curriculum_queue as queue
    from types import SimpleNamespace
    args = SimpleNamespace(state=tmp_path / 'state.json', log=tmp_path / 'log')
    rows = [dict(won=False, value_monitor=dict(value='v5', selected=dict(
        all=dict(count=2, total=1.8, squared=1.62), executed=dict(count=1, total=.9, squared=.81))))
        for _ in range(8)]
    summary = queue.value_monitor_summary(rows)
    assert abs(summary['v5:executed']['prediction_sum'] - 7.2) < 1e-12
    state = dict(log=[])
    for round_index in range(1, 4):
        entry = dict(round=round_index, groups=[dict(value_monitor=summary)])
        queue.record_value_monitor(args, state, entry)
        assert bool(entry['value_warnings']) == (round_index == 3)
        state['log'].append(entry)
    assert (tmp_path / 'state.pipeline/round_000003/value_monitor.json').is_file()


def test_value_loop_monitor_records_anchored_choice_not_raw_winner(monkeypatch):
    import search_play as search
    from types import SimpleNamespace
    wait = dict(type='wait', ticks=60, until='timeout')
    plant = dict(type='plant', packet=0, row=0, col=0)
    args = SimpleNamespace(value_monitor=True, verify_samples=0, samples=3)
    stats = {}
    monkeypatch.setattr(search, '_save_plan', lambda *args: None)
    monkeypatch.setattr(search, '_probe_decision', lambda *args: None)
    chosen = search._verified_choice(None, {}, 0, [wait, plant], [.7, .71], [3, 3], 1, 0,
                                     wait, args, 0, stats)
    assert chosen == wait and stats['last_leaf_value'] == .7
