import copy
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from pvz_entity import ENTITY_FIELDS, ENTITY_TOKENS, entity_features
from pvz_rollout_policy import RolloutNet, export_student, features, identity_scalars, load_student, packet_features, warm_start_student
from pack_training_rows import packed_load, build_from_archives, assert_payload_equal
from pvz_recipe import write_recipe


def frame():
    return dict(plants=[[2, 3, 1, 300, 300]], zombies=[[1, 0, 420.125, .25, 200, True, False, 0, 0, 9]],
                packets=[[slot, kind, True, 0, 100, 25] for slot, kind in enumerate(range(6))],
                mowers=[], lanes=list(range(5)), tick=0, wave=0, sun=1000, scene={'wave_count': 10})


def inputs(record):
    board, scalars = features(record, None)
    return board[None], torch.cat((identity_scalars(record, scalars), entity_features(record).flatten()))[None]


def test_entity_permutation_precision_and_empty_inputs():
    torch.set_num_threads(1)
    torch.manual_seed(153)
    net = RolloutNet(8, packet_head='identity', arch='entity_attn').eval()
    original = frame()
    reordered = copy.deepcopy(original)
    reordered['packets'] = [[slot, *packet[1:]] for slot, packet in enumerate(reversed(original['packets']))]
    first = net(*inputs(original), torch.tensor([0]), auxiliary=False, packet_info=packet_features(original)[None])
    second = net(*inputs(reordered), torch.tensor([5]), packet_info=packet_features(reordered)[None])
    torch.testing.assert_close(first['plant'], second['plant'], rtol=0, atol=0)
    torch.testing.assert_close(first['cell'], second['cell'], rtol=0, atol=0)
    torch.testing.assert_close(first['packet'][0, :6].flip(0), second['packet'][0, :6], rtol=0, atol=0)
    shifted = copy.deepcopy(original)
    shifted['zombies'][0][2] += 1
    changed = net(*inputs(shifted), packet_info=packet_features(shifted)[None])
    assert not torch.equal(first['plant'], changed['plant'])
    empty = frame()
    empty.update(plants=[], zombies=[], packets=[])
    context, cards = net.entity(entity_features(empty)[None])
    assert torch.isfinite(context).all() and torch.isfinite(cards).all()
    assert not cards.abs().sum()


def test_cards_do_not_enter_the_board_context():
    torch.set_num_threads(1)
    torch.manual_seed(153)
    net = RolloutNet(8, packet_head='identity', arch='entity_attn').eval()
    first, second = frame(), frame()
    second['packets'] = [[slot, 20 + slot, True, 0, 100, 25] for slot in range(6)]
    a, a_cards = net.entity(entity_features(first)[None])
    b, b_cards = net.entity(entity_features(second)[None])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a_cards, b_cards)


def test_deck_dropout_hides_one_or_two_cards_per_sample_in_training():
    torch.set_num_threads(1)
    net = RolloutNet(8, packet_head='identity', arch='entity_attn').train()
    net.entity.deck_dropout = 1.0
    entities = entity_features(frame())[None].repeat(8, 1, 1)
    card = entities[:, :, 0] == 1
    valid = (entities[:, :, 0] > 0) & (entities[:, :, 1] >= 0)
    torch.manual_seed(7)
    hidden = net.entity.hidden_cards(valid, card)
    count = hidden.sum(1)
    assert hidden[:, 10:].sum() == 0 and (count >= 1).all() and (count <= 2).all()
    torch.manual_seed(7)
    cards = net.entity(entities)[1]
    assert (cards[hidden[:, :10]].abs() == 0).all()
    net.eval()
    assert net.entity.hidden_cards(valid, card).sum() == 0


def test_entity_warm_start_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    initial = RolloutNet(8, packet_head='identity').eval()
    net = RolloutNet(8, packet_head='identity', arch='entity_attn').eval()
    warm_start_student(net, initial)
    record = frame()
    board, scalars = features(record, None)
    old = initial(board[None], identity_scalars(record, scalars)[None], packet_info=packet_features(record)[None])
    new = net(*inputs(record), packet_info=packet_features(record)[None])
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    path = tmp_path / 'model.pt'
    torch.save(dict(state_dict=net.state_dict(), packet_head='identity', arch='entity_attn'), path)
    restored = load_student(path)
    import pytest
    with pytest.raises(ValueError, match='native export'):
        export_student(restored, tmp_path / 'model.bin')
    loss = restored(*inputs(record), packet_info=packet_features(record)[None])['plant'].sum()
    loss.backward()
    assert restored.entity.output.weight.grad.abs().sum() > 0


def test_entity_pack_and_training(tmp_path):
    from test_identity_training import IdentityTrainingTest
    torch.set_num_threads(1)
    data = tmp_path / 'data'
    IdentityTrainingTest().fixture(data)
    loaded = packed_load(data, {'example'}, packet_head='identity', arch='entity_attn',
                         cells=True, aux_horizon=2, cache_dir=tmp_path / 'cache', rebuild=True)
    fresh = build_from_archives(data, {'example'}, packet_head='identity', arch='entity_attn',
                                cells=True, aux_horizon=2)[0]
    assert_payload_equal(fresh, loaded)
    assert loaded[1].shape[1] == 115 + ENTITY_TOKENS * ENTITY_FIELDS
    recipe = dict(schema_version=1, kind='training', data=str(data), labels=['example'], out=str(tmp_path / 'model'),
                  packet_head='identity', arch='entity_attn', width=8, seed=1, epochs=1,
                  cell_soft_weight=1, aux_weight=.1, aux_horizon=2, batch_size=32)
    write_recipe(tmp_path / 'recipe.json', recipe)
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/train_rollout_policy.py'),
                             '--recipe', str(tmp_path / 'recipe.json')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert load_student(tmp_path / 'model/model.pt').arch == 'entity_attn'
    from pvz_training_pack import archive_fingerprint, pack_options, pack_path
    from pack_training_rows import prune
    entity_path = pack_path(data, pack_options({'example'}, packet_head='identity', arch='entity_attn',
                                              cells=True, aux_horizon=2, sources=archive_fingerprint(data, {'example'})))
    packed_load(data, {'example'}, packet_head='identity', cells=True, aux_horizon=2)
    base_path = pack_path(data, pack_options({'example'}, packet_head='identity', cells=True, aux_horizon=2,
                                            sources=archive_fingerprint(data, {'example'})))
    prune(data, {'example'}, base_path)
    assert entity_path.is_file()


def test_entity_alternate_recipe_keeps_architecture(tmp_path, monkeypatch):
    import curriculum_queue as queue
    from pvz_recipe import load_recipe
    from test_curriculum_pipeline import setup
    args, state = setup(tmp_path)
    recipe = dict(load_recipe(args.recipe), packet_head='identity')
    write_recipe(args.recipe, recipe)
    args.alternate_recipe = tmp_path / 'entity.json'
    write_recipe(args.alternate_recipe, dict(recipe, arch='entity_attn'))
    args.candidates_per_recipe = 1
    args.selection_seed_ledger = tmp_path / 'selection_ledger.json'
    monkeypatch.setattr(queue.TrainingPipeline, '_run', lambda *args: {})
    monkeypatch.setattr(queue.TrainingPipeline, '_plan_build', lambda *args: None)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        pipeline.enqueue()
        pipeline.futures[0][1].result(timeout=5)
        candidates = state['training_jobs'][0]['candidate_recipes']
        assert [load_recipe(candidate['recipe']).get('arch', 'grid') for candidate in candidates] == ['grid', 'entity_attn']
    finally:
        pipeline.executor.shutdown(wait=True)
