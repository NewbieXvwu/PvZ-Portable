import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import value_retrain as value


def comparisons():
    games, roots = [], []
    for zone in ('day', 'pool'):
        games.extend(dict(zone=zone, incumbent=0.3, candidate=0.2) for _ in range(8))
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
