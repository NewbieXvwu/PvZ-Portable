import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
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
    assert torch.isfinite(net.entity(entity_features(empty)[None])).all()


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
    export_student(restored, tmp_path / 'model.bin')
    assert (tmp_path / 'model.bin').read_bytes()[:8] == b'PVZRN007'
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


def test_entity_cpp_public_features_and_logits(tmp_path):
    import os
    import pytest
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_student_observation import StudentHistory
    from pvz_tasks import deck_for_level, profile_for_deck
    from render_episode import _frame
    from pvz_rollout_policy import act
    root = Path(__file__).resolve().parents[1]
    binary = Path(os.environ.get('PVZ_EXECUTABLE', root / 'build/pvz-portable'))
    resource = Path(os.environ.get('PVZ_RESOURCE_DIR', '/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'))
    if not binary.is_file() or not (resource / 'main.pak').is_file():
        pytest.skip('requires local engine and resources')
    torch.set_num_threads(1)
    torch.manual_seed(153)
    maximum = 0
    with EventWaitEnv(resource, executable=binary) as env:
        for width, row_pool, level in ((16, False, 7), (64, True, 5), (64, True, 21), (16, True, 36), (16, True, 45), (16, True, 50)):
            net = RolloutNet(width, 'meanmax', obs='v2timer', row_pool=row_pool,
                             packet_head='identity', arch='entity_attn').eval()
            with torch.no_grad():
                net.plant.bias.fill_(1)
                net.plant.weight.zero_()
            export_student(net, tmp_path / 'model.bin')
            env.load_rollout_net(tmp_path / 'model.bin')
            deck = deck_for_level(level)
            live, _ = env.reset(deck=deck, task=TaskSpec(level=level, seed=30000, profile=profile_for_deck(deck)))
            while live['tick'] < 5000 and not live.get('terminal'):
                legal = live['legal_actions']['plants']
                live, *_ = env.step({'type': 'plant', **legal[0]} if legal else {'type': 'wait', 'ticks': 60})
            history = StudentHistory('v2timer')
            previous = previous_action = None
            for _ in range(3):
                record = _frame(live)
                record['action'] = previous_action
                trace = tmp_path / 'trace.jsonl'
                native = env.rollout_predict(previous['tick'] if previous else None,
                                             bool(previous_action and previous_action['type'] == 'plant'),
                                             student_history=history, trace_path=trace)
                diagnostic = json.loads(trace.read_text())
                entities = entity_features(record)
                np.testing.assert_array_equal(np.asarray(diagnostic['scalars'][-ENTITY_TOKENS * ENTITY_FIELDS:], np.float32), entities.numpy().ravel())
                board, scalars = history.features(record, previous)
                scalars = torch.cat((identity_scalars(record, scalars), entities.flatten()))
                info = packet_features(record)[None]
                with torch.inference_mode():
                    hidden, context = net.encode(board[None], scalars[None], packet_info=info)
                    pairs = [(net.plant(context).flatten(), torch.tensor([native['plant']], dtype=torch.float32)),
                             (net.packet_logits(context, info)[0], torch.tensor(native['packet'], dtype=torch.float32))]
                    if native['cell_packet'] >= 0:
                        pairs.append((net.cell_logits(hidden, context, torch.tensor([native['cell_packet']]), info)[0], torch.tensor(native['cell'], dtype=torch.float32)))
                    for python, cpp in pairs:
                        maximum = max(maximum, float((python - cpp).abs().max()))
                        torch.testing.assert_close(python, cpp, rtol=0, atol=1e-5)
                chosen = act(net, record, previous, live['legal_actions']['plants'], history=history)
                assert chosen == native['action']
                previous, previous_action = record, chosen
                live, *_ = env.step(chosen)
    print({'entity_cpp_max_abs_error': maximum, 'decisions': 18})
