"""task192: small CPU cases for the opt-in attribute path and frozen study queue."""
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_rollout_policy import RolloutNet, load_student, export_student
from pvz_student_observation import OBS_EXTRA
from pvz_value_net import BOARD_CHANNELS, SCALARS


@pytest.fixture(scope='module')
def legacy():
    # Compare to the actual pinned research code, not a second hand-written implementation.
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        '5ef7ad93af44145668b27cc9ba6cb4bf3b5bab12:python/pvz_rollout_policy.py'], text=True)
    module = types.ModuleType('task192_legacy')
    exec(compile(source, '<pinned task162 rollout>', 'exec'), module.__dict__)
    return module


@pytest.mark.parametrize('head', ['slot', 'identity'])
@pytest.mark.parametrize('row_pool', [False, True])
def test_card_attributes_disabled_is_bitwise_pinned_network(legacy, head, row_pool):
    torch.set_num_threads(1)
    options = dict(width=8, pool='meanmax', obs='v2timer', row_pool=row_pool, packet_head=head)
    torch.manual_seed(1); control = legacy.RolloutNet(**options).eval()
    torch.manual_seed(1); actual = RolloutNet(**options, card_attributes=False).eval()
    assert control.state_dict().keys() == actual.state_dict().keys()
    for key, value in control.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    assert not hasattr(actual, 'plant_attributes')
    board = torch.randn(3, BOARD_CHANNELS + OBS_EXTRA['v2timer'][0], 6, 10)
    scalars = torch.randn(3, SCALARS + 2 + OBS_EXTRA['v2timer'][1])
    info = torch.rand(3, 10, 4); info[:, :, 0] = torch.arange(10)
    info[:, -1, 0] = -1
    packets = torch.tensor([0, 2, 4])
    for sparse in (False, True):
        old_h, old_g = control.encode(board, scalars, sparse_input=sparse, packet_info=info)
        h, g = actual.encode(board, scalars, sparse_input=sparse, packet_info=info)
        assert torch.equal(h, old_h) and torch.equal(g, old_g)
        assert torch.equal(actual.packet_logits(g, info), control.packet_logits(old_g, info))
        assert torch.equal(actual.cell_logits(h, g, packets, info), control.cell_logits(old_h, old_g, packets, info))


def test_card_attributes_engine_facts_include_upgrade_bases():
    from pvz_card_attributes import card_attribute_matrix
    from pvz_constants import plant_defs
    matrix, names = card_attribute_matrix()
    assert matrix.shape == (54, 65) and matrix.nbytes == 14040
    assert card_attribute_matrix()[0] is matrix
    definitions = plant_defs()
    for definition in definitions:
        kind = definition['type']
        for column, key in enumerate(('cost', 'recharge', 'launch')):
            expected = definition[key] / max(d[key] for d in definitions)
            assert matrix[kind, column] == pytest.approx(expected)
    assert matrix[8, names.index('IsNocturnal')] == 1
    assert matrix[0, names.index('IsNocturnal')] == 0
    assert matrix[43, names.index('IsAquatic')] == 1
    assert matrix[35, names.index('IsFlying')] == 1  # exact engine label
    assert matrix[46, names.index('IsSpiky')] == 1
    assert matrix[43, names.index('upgrade_base/16')] == 1
    assert matrix[47, names.index('upgrade_base/34')] == 1
    assert matrix[40, names.index('upgrade_base/7')] == 1
    assert matrix[7, names.index('subclass/SUBCLASS_SHOOTER')] == 1


def test_card_attributes_board_projection_slot_symmetry_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    net = RolloutNet(width=8, obs='v2timer', pool='meanmax', row_pool=True,
                     packet_head='identity', card_attributes=True).eval()
    board = torch.zeros(2, BOARD_CHANNELS + OBS_EXTRA['v2timer'][0], 6, 10)
    board[:, 7, 2, 3] = .5
    scalars = torch.zeros(2, SCALARS + 2 + OBS_EXTRA['v2timer'][1])
    info = torch.zeros(2, 10, 4)
    info[:, :, 0] = torch.arange(10)
    info[:, 9, 0] = -1
    info[:, :, 1:] = torch.rand(2, 10, 3)
    captured = []
    handle = net.trunk[0].register_forward_pre_hook(lambda _layer, args: captured.append(args[0].detach()))
    result = net(board, scalars, torch.tensor([1, 1]), packet_info=info)
    handle.remove()
    assert torch.equal(captured[0][:, board.shape[1]:, 2, 3], net.plant_attributes[7].expand(2, -1) * .5)
    assert torch.equal(captured[0][:, :board.shape[1]], board)
    order = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    moved = net(board, scalars, torch.tensor([0, 0]), packet_info=info[:, order])
    torch.testing.assert_close(result['plant'], moved['plant'], rtol=0, atol=0)
    torch.testing.assert_close(result['cell'], moved['cell'], rtol=0, atol=0)
    torch.testing.assert_close(result['packet'][:, order], moved['packet'], rtol=0, atol=0)
    sum(value[:, :1].square().sum() if value.ndim > 1 else value.square().sum()
        for value in result.values()).backward()
    assert net.trunk[0].weight.grad[:, board.shape[1]:].abs().sum() > 0
    checkpoint = dict(state_dict=net.state_dict(), width=8, obs='v2timer', pool='meanmax', row_pool=True,
                      packet_head='identity', card_attributes=True)
    path = tmp_path / 'model.pt'; torch.save(checkpoint, path)
    restored = load_student(path)
    actual = restored(board, scalars, torch.tensor([1, 1]), packet_info=info)
    for key in result:
        assert torch.equal(result[key], actual[key])
    with pytest.raises(NotImplementedError):
        export_student(restored, tmp_path / 'unsupported.bin')
    assert not (tmp_path / 'unsupported.bin').exists()


def test_card_attributes_train_and_matched_preview_keeps_control_results(tmp_path, monkeypatch):
    import train_rollout_policy as trainer
    import eval_window_policy as evaluator
    from test_identity_training import IdentityTrainingTest
    from pack_training_rows import packed_load
    from pvz_recipe import write_recipe
    IdentityTrainingTest().fixture(tmp_path / 'data')
    packed_load(tmp_path / 'data', {'example'}, obs='v2', packet_head='identity', cells=True, rebuild=True)
    pack = next((tmp_path / 'data/packs').glob('*.npz'))
    out, control = tmp_path / 'arm', tmp_path / 'control'
    path = tmp_path / 'recipe.json'
    write_recipe(path, dict(schema_version=1, kind='training', data=str(tmp_path / 'data'), labels=['example'],
        out=str(out), width=8, obs='v2', pool='meanmax', row_pool=True, seed=1, epochs=1, max_steps=1,
        batch_size=16, validation_batch_size=16, packet_head='identity', cell_soft_weight=3, card_attributes=True))
    monkeypatch.setattr(sys, 'argv', ['train', '--recipe', str(path)])
    trainer.main()
    checkpoint = torch.load(out / 'model.pt', weights_only=False)
    assert checkpoint['card_attributes'] is True and checkpoint['optimizer_steps'] == 1
    assert load_student(out / 'model.pt').card_attributes
    control.mkdir()
    base = RolloutNet(width=8, obs='v2', pool='meanmax', row_pool=True, packet_head='identity')
    torch.save(dict(state_dict=base.state_dict(), width=8, obs='v2', pool='meanmax', row_pool=True,
                    packet_head='identity'), control / 'model.pt')
    original = '{"existing_control_result": true}\n'
    (control / 'heldout.json').write_text(original)
    monkeypatch.setattr(sys, 'argv', ['eval', '--recipe', str(path), '--pack', str(pack),
        '--merge-dir', str(tmp_path / 'merge'), '--out', str(out), '--batch-size', '2',
        '--preview-rows', '2', '--preview-only', '--peer-out', str(control), '--peers-inline'])
    evaluator.main()
    result = json.loads((out / 'heldout.json').read_text())
    assert not result['evaluation_complete']
    assert result['evaluated_rows'] == result['preview_rows'] < result['heldout_rows']
    compared = result['comparisons'][str(control)]
    assert compared['all']['rows'] == result['evaluated_rows']
    for group in ('unseen', 'day', 'night', 'pool', 'fog', 'roof'):
        assert compared[group]['control']['rows'] == result['groups'][group]['rows']
    assert compared['roof']['delta']['joint_acc'] is None
    assert (control / 'heldout.json').read_text() == original


def test_card_attributes_queue_inherits_fixed_rows_budget_and_order(tmp_path):
    from pvz_recipe import write_recipe, load_recipe
    from queue_decision_study import enqueue_card_attributes
    queue = tmp_path / 'queue'; queue.mkdir()
    recipe = tmp_path / 'base.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data='/frozen', labels=['r42'],
        out='/old', epochs=1, seed=1, width=64, packet_head='identity'))
    old = dict(id='r42_cnn64_s1', recipe=str(recipe), out='/old', priority=10, enabled=True,
        revision='old', code_root='/old-code', arch='cnn', seed=1, host_gib=5, gpu_gib=4,
        depends_on=['/split/ready.json', '/budget/ready.json'],
        command=['python', '/old-code/scripts/train_rollout_policy.py', '--recipe', str(recipe),
                 '--pack', '/frozen/pack.npz', '--deck-holdout', '/same-mask', '--training-budget', '/same-budget',
                 '--budget-arm', 'cnn64'],
        evaluate=['python', '/old-code/scripts/eval_window_policy.py', '--recipe', str(recipe),
                  '--out', '/old', '--preview-rows', '32768'])
    source = queue / 'r42_cnn64_s1.json'; source.write_text(json.dumps(old))
    enqueue_card_attributes(tmp_path, recipe, 'pinned', Path('/new-code'), sys.executable)
    job = json.loads((queue / 'cnn64_cardattr_s1.json').read_text())
    assert json.loads(source.read_text()) == old
    assert job['after_jobs'] == ['r42_cnn64_s1'] and job['priority'] == 11
    assert job['depends_on'] == old['depends_on']
    assert job['command'][job['command'].index('--budget-arm') + 1] == 'cnn64'
    assert '--peers-inline' in job['evaluate'] and '--preview-only' in job['evaluate']
    variant = load_recipe(Path(job['recipe']), 'training')
    base = load_recipe(recipe, 'training')
    assert {key for key in base if base[key] != variant[key]} == {'card_attributes', 'out', 'provenance'}
