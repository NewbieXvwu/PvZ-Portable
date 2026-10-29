"""task167 小型私有权重与共包夹具，不运行生产训练或对局。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_recipe import load_recipe, training_row_options, validate, write_recipe
from pvz_training_pack import keep_archive
from pvz_training_rows import SharedPackAdjustment


def archive(data, seed, label, level=7, deck=None):
    path = data / f'seed_{seed}' / label / 'frames.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    task = dict(level=level)
    if deck is not None:
        task['deck'] = deck
    (path.parent / 'meta.json').write_text(json.dumps(dict(task=task)))
    return path


def test_disabled_is_bitwise_identity_without_metadata(tmp_path):
    weights = torch.tensor([0., -0., 1.5, 12.25])
    original = weights.numpy().tobytes()
    before = torch.random.get_rng_state()
    adjust = SharedPackAdjustment(tmp_path)
    adjust.observe(tmp_path / 'seed_1/z_old/frames.npz', np.arange(4))
    adjust.apply_keep(weights, torch.tensor([False, False, True, True]))
    adjust.apply_deck_cap(weights, torch.zeros(4, dtype=torch.bool), None)
    assert weights.numpy().tobytes() == original
    assert torch.equal(torch.random.get_rng_state(), before)


def test_whole_archive_keep_is_stable_across_rounds_and_roots(tmp_path):
    results = []
    for root, span in [('round_1', 3), ('round_9_copy', 5)]:
        data = tmp_path / root
        adjust = SharedPackAdjustment(data, .25)
        expected = []
        for seed in range(64):
            label = 'z17dag35b_fast_search' if seed % 2 else 'd7p1m_fast_search'
            path = archive(data, seed, label)
            adjust.observe(path, np.arange(span))
            expected.append(keep_archive(data, path, {'.*': .25}))
        # z-like labels from another level and fresh course L7 both stay intact.
        for seed, label, level in [(70, 'z_other', 8), (71, 'c55_l7', 7)]:
            adjust.observe(archive(data, seed, label, level), np.arange(span))
        weights = torch.ones(66 * span)
        validation = torch.zeros(len(weights), dtype=torch.bool)
        adjust.apply_keep(weights, validation)
        rows = weights.reshape(66, span)
        assert all(torch.equal(row, torch.full((span,), float(keep))) for row, keep in zip(rows[:64], expected))
        assert rows[64:].eq(1).all() and 0 < sum(expected) < 64
        results.append(rows[:, 0].tolist())
    assert results[0] == results[1]
    # A validation archive remains an untouched whole archive.
    validation = torch.ones(len(weights), dtype=torch.bool)
    original = torch.arange(len(weights), dtype=torch.float32)
    actual = original.clone()
    adjust.apply_keep(actual, validation)
    assert torch.equal(actual, original)


def test_deck_cap_uses_actual_cards_and_final_weight_mass(tmp_path):
    adjust = SharedPackAdjustment(tmp_path, deck_weight_cap=.4)
    specs = [(7, None), (7, [1, 0]), (7, [0, 1, 1]), (8, [0, 1]), (30, [])]
    for i, (level, deck) in enumerate(specs):
        adjust.observe(archive(tmp_path, i, 'teacher', level, deck), np.arange(3))
    # First row of each archive is validation. Archive 0's packed deck agrees with
    # archive 1's differently ordered explicit deck; duplicate plants remain distinct.
    kinds = np.tile([0, 1, -1, -1, -1, -1, -1, -1, -1, -1], (15, 1)).astype(np.int8)
    packets = SimpleNamespace(columns=[kinds])
    validation = torch.tensor([True, False, False] * 5)
    weights = torch.tensor([999., 200., 200., 888., 100., 100., 777., 10., 10.,
                            666., 10., 10., 555., 10., 10.])
    original = weights.clone()
    adjust.apply_deck_cap(weights, validation, packets)
    assert torch.equal(weights[validation], original[validation])
    assert (weights <= original).all()
    mass = [float(weights[[1, 2, 4, 5]].sum()), float(weights[[7, 8]].sum()),
            float(weights[[10, 11]].sum()), float(weights[[13, 14]].sum())]
    assert max(mass) / sum(mass) <= .4 + 1e-7
    assert torch.equal(weights[[7, 8, 10, 11, 13, 14]], original[[7, 8, 10, 11, 13, 14]])
    assert len(adjust.report['decks']['signatures']) == 4
    assert any(record['deck'] == [] for record in adjust.report['decks']['signatures'])
    repeated = original.clone()
    adjust.apply_deck_cap(repeated, validation, packets)
    assert torch.equal(weights, repeated)


def test_recipes_share_a_pack_and_pipeline_freezes_each_b_variant(tmp_path):
    import curriculum_queue as queue
    a = load_recipe(ROOT / 'recipes/dw4i2_v2timer.json')
    for filename, field, value in [('dw4i2_v2timer_l7keep25.json', 'history_l7_keep', .25),
                                   ('dw4i2_v2timer_deckcap05.json', 'deck_weight_cap', .05)]:
        path = ROOT / 'recipes' / filename
        b = load_recipe(path)
        assert b[field] == value and {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)} == {field}
        assert training_row_options(a) == training_row_options(b)
        pipeline = queue.TrainingPipeline.__new__(queue.TrainingPipeline)
        pipeline.args = SimpleNamespace(recipe=ROOT / 'recipes/dw4i2_v2timer.json',
            alternate_recipe=path, candidates_per_recipe=2, state=tmp_path / 'state.json', log=tmp_path / 'log')
        pipeline.directory = tmp_path / filename
        pipeline.build_workers = 1
        pipeline.state = dict(round=52, log=[], training_jobs=[])
        pipeline._plan_build = Mock()
        pipeline._snapshot_selection = Mock()
        pipeline._seeded = Mock()
        pipeline.executor = Mock()
        pipeline.futures = []
        pipeline.enqueue()
        job = pipeline.state['training_jobs'][0]
        assert job['training_seeds'] == [1, 2, 3, 4]
        assert [r['recipe_family'] for r in job['candidate_recipes']] == ['A', 'A', 'B', 'B']
        frozen = load_recipe(Path(job['candidate_recipes'][2]['recipe']))
        assert frozen[field] == value and pipeline._plan_build.call_count == 1
    for field, value in [('history_l7_keep', -1), ('deck_weight_cap', 0), ('deck_weight_cap', 1.1)]:
        with pytest.raises(ValueError):
            validate(dict(a, **{field: value}))


def test_cli_scales_private_weights_without_repacking_or_training(tmp_path, monkeypatch):
    import train_rollout_policy as trainer
    from test_incremental_pack import fixture, options
    from pack_training_rows import pack
    data = tmp_path / 'data'
    fixture(data, ['fresh', 'z_old'])
    for path in data.glob('*/*/meta.json'):
        meta = json.loads(path.read_text())
        meta['context']['level'] = 7
        path.write_text(json.dumps(meta))
    index = pack(data, options(data, ['fresh', 'z_old']), tmp_path / 'cache')
    original = trainer.read_pack(index, data)[5].clone()
    common = dict(schema_version=1, kind='training', data=str(data), labels=['fresh', 'z_old'],
                  device='cpu', threads=1, width=4, wait_keep=1., validation=dict(seed=0, fraction=.34),
                  batch_size=2, validation_batch_size=2, packet_balance=.5,
                  seed=167, out=str(tmp_path / 'student'))
    write_recipe(tmp_path / 'a.json', common)
    write_recipe(tmp_path / 'b.json', dict(common, history_l7_keep=0.))
    monkeypatch.setattr(trainer, 'Metrics', Mock())
    monkeypatch.setattr(trainer, 'set_oom_score_adj', lambda *_: None)
    loaded = {}
    old_load = trainer.load
    def load(*args, **kwargs):
        result = old_load(*args, **kwargs)
        loaded['weight'] = result[5]
        loaded['seed'] = np.asarray(result[6])
        return result
    monkeypatch.setattr(trainer, 'load', load)
    class StopBeforeUpdate(Exception):
        pass
    losses = []
    def stop(step, loss):
        assert step == 1 and np.isfinite(loss)
        losses.append(loss)
        raise StopBeforeUpdate
    def run(recipe, *extra):
        monkeypatch.setattr(sys, 'argv', ['train', '--recipe', str(recipe), *extra])
        with pytest.raises(StopBeforeUpdate):
            trainer.main(step_observer=stop)
        return loaded['weight'].clone()
    reference = run(tmp_path / 'a.json')
    write_recipe(tmp_path / 'disabled.json', dict(common, history_l7_keep=1., deck_weight_cap=1.))
    assert torch.equal(run(tmp_path / 'disabled.json'), reference) and losses[0] == losses[1]
    actual = run(tmp_path / 'b.json', '--pack-recipe', str(tmp_path / 'a.json'))
    note = json.loads((tmp_path / 'student/shared_pack_weights.json').read_text())
    assert note['post_pack']['history_l7']['dropped_training_rows'] > 0
    assert note['effective_training_rows'] < note['training_rows_before']
    assert note['planned_optimizer_steps'] == 12 * -(-note['training_rows_before'] // 2)
    assert actual.eq(0).any()
    assert torch.equal(actual[actual > 0], reference[actual > 0])
    assert torch.equal(trainer.read_pack(index, data)[5], original)
