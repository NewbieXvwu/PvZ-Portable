"""task131: a wider input preserves the parent's function before training."""
import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_rollout_policy import RolloutNet, warm_start_student


@pytest.mark.parametrize('obs', ['v1', 'v2', 'v2boss', 'v2bungee', 'v2sleep'])
@pytest.mark.parametrize('context,head', [(False, 'identity'), (True, 'slot')])
def test_widened_parent_preserves_all_outputs(obs, context, head):
    torch.set_num_threads(1)
    torch.manual_seed(131)
    old = RolloutNet(8, 'meanmax', context, obs, row_pool=True, aux=True, packet_head=head).eval()
    new = RolloutNet(8, 'meanmax', context, 'v2sleep', row_pool=True, aux=True, packet_head=head).eval()
    saved = copy.deepcopy(old.state_dict())
    warm_start_student(new, old)
    for name, weight in saved.items():
        actual = new.state_dict()[name]
        if name in ('trunk.0.weight', 'scalar.0.weight'):
            torch.testing.assert_close(actual[:, :weight.shape[1]], weight, rtol=0, atol=0)
            assert not actual[:, weight.shape[1]:].any()
        else:
            torch.testing.assert_close(actual, weight, rtol=0, atol=0)
    board = torch.randn(2, old.trunk[0].in_channels, 6, 10)
    scalars = torch.randn(2, old.scalar[0].in_features)
    wider_board = torch.zeros(2, new.trunk[0].in_channels, 6, 10)
    wider_scalars = torch.zeros(2, new.scalar[0].in_features)
    wider_board[:, :board.shape[1]] = board
    wider_scalars[:, :scalars.shape[1]] = scalars
    info = torch.zeros(2, 10, 4)
    info[:, :, 0] = torch.arange(10)
    packet = torch.tensor([1, 3])
    with torch.inference_mode():
        expected = old(board, scalars, packet, auxiliary=True, packet_info=info)
        actual = new(wider_board, wider_scalars, packet, auxiliary=True, packet_info=info)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-6)


def test_widening_rejects_semantically_different_inputs():
    with pytest.raises(ValueError, match='input upgrade'):
        warm_start_student(RolloutNet(8, obs='v2sleep'), RolloutNet(8, obs='v2fog'))
    with pytest.raises(ValueError, match='context'):
        warm_start_student(RolloutNet(8, context=True, obs='v2sleep'), RolloutNet(8, obs='v2'))


@pytest.mark.parametrize('new_fields', [False, True])
def test_sleep_cache_reuses_only_missing_field_v2_rows(tmp_path, monkeypatch, new_fields):
    import numpy as np
    import pvz_frame_store as store
    from test_bungee_inputs import frame

    first = frame()
    first.pop('bungee')
    if new_fields:
        first['sleep'] = [[0, 1, 1, 0]]
    second = copy.deepcopy(first)
    second.update(tick=160, action=dict(type='wait', ticks=60))
    path = tmp_path / 'seed_30000/example/frames.npz'
    store.write_frames(path, [first, second])
    old = store.cache_episode(path, tmp_path / 'cache', 'v2')
    source = store.frame_bytes(path)
    expected = store.encode_student_frames([first, second], source, 'v2sleep')
    original = store.encode_student_frames
    calls = []
    def encode(*args):
        calls.append(args[-1])
        return original(*args)
    monkeypatch.setattr(store, 'encode_student_frames', encode)
    upgraded = store.cache_episode(path, tmp_path / 'cache', 'v2sleep')
    assert calls == (['v2sleep'] if new_fields else [])
    with store.read_episode_cache(upgraded) as actual:
        for name, array in expected.items():
            np.testing.assert_array_equal(actual[name], array)
    with np.load(upgraded, allow_pickle=False) as actual:
        assert ('zero_pad_from' in actual.files) is not new_fields
    assert old.exists() and old != upgraded
    assert store.cache_episode(path, tmp_path / 'cache', 'v2sleep') == upgraded


def test_curriculum_recipe_trains_one_step_from_old_parent(tmp_path, monkeypatch):
    import train_rollout_policy as trainer
    from pvz_frame_store import write_frames, cache_episode
    from pvz_recipe import write_recipe
    from pack_training_rows import packed_load
    from test_bungee_inputs import frame

    old = RolloutNet(8, 'meanmax', False, 'v2', row_pool=True, packet_head='identity')
    parent = tmp_path / 'parent.pt'
    torch.save(dict(state_dict=old.state_dict(), pool='meanmax', obs='v2', row_pool=True,
                    packet_head='identity'), parent)
    for seed in (30000, 30001):
        first = frame()
        first.pop('bungee')
        second = copy.deepcopy(first)
        second.update(tick=160, action=dict(type='wait', ticks=60))
        path = tmp_path / f'data/seed_{seed}/example/frames.npz'
        write_frames(path, [first, second])
        cache_episode(path, tmp_path / 'cache', 'v2')
    packed_load(tmp_path / 'data', {'example'}, obs='v2sleep', packet_head='identity',
                cache_dir=tmp_path / 'cache')
    recipe = tmp_path / 'recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(tmp_path / 'data'),
        labels=['example'], out=str(tmp_path / 'out'), init=str(parent), obs='v2sleep',
        width=8, pool='meanmax', row_pool=True, packet_head='identity', device='cpu',
        epochs=1, max_steps=1, validation=dict(seed=0, fraction=.5)))
    monkeypatch.setattr(sys, 'argv', ['train_rollout_policy.py', '--recipe', str(recipe)])
    trainer.main()
    checkpoint = torch.load(tmp_path / 'out/model.pt', weights_only=False)
    assert checkpoint['obs'] == 'v2sleep'
    assert checkpoint['state_dict']['trunk.0.weight'].shape[1] == 113


def test_sleep_enqueue_trains_both_recipes_from_scratch(tmp_path, monkeypatch):
    """task143: production trains every round from scratch; the queue must not
    couple a v2sleep round to whichever checkpoint happens to be incumbent."""
    import curriculum_queue as queue
    from pvz_recipe import load_recipe, write_recipe
    from test_curriculum_pipeline import setup

    args, state = setup(tmp_path)
    recipe = dict(load_recipe(args.recipe), obs='v2sleep', init=None)
    write_recipe(args.recipe, recipe)
    args.alternate_recipe = tmp_path / 'b.json'
    write_recipe(args.alternate_recipe, recipe)
    args.candidates_per_recipe = 2
    args.selection_seed_ledger = tmp_path / 'selection_ledger.json'
    monkeypatch.setattr(queue.TrainingPipeline, '_run', lambda *args: {})
    monkeypatch.setattr(queue.TrainingPipeline, '_plan_build', lambda *args: None)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        pipeline.enqueue()
        pipeline.futures[0][1].result(timeout=5)
        job = state['training_jobs'][0]
        assert len(job['candidate_recipes']) == 4
        assert all(load_recipe(Path(candidate['recipe']))['init'] is None
                   for candidate in job['candidate_recipes'])
    finally:
        pipeline.executor.shutdown(wait=True)
