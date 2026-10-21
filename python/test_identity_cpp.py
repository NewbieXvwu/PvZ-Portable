"""PVZRN006 CPU parity on live boards; tests keep exactly one simulator alive."""
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import RolloutNet, act, export_student, features, packet_features, identity_scalars
from pvz_student_observation import StudentHistory
from pvz_tasks import profile_for_deck
from pvz_value_net import observation_context
from render_episode import _frame

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', str(ROOT / 'build/pvz-portable')))


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'needs local CPU build and resources')
class IdentityCppTest(unittest.TestCase):
    def test_random_identity_heads_on_real_positions(self):
        torch.set_num_threads(1)
        torch.manual_seed(19006)
        maximum = 0.
        decisions = 0
        with tempfile.TemporaryDirectory() as folder, EventWaitEnv(RESOURCE, executable=BINARY) as env:
            path = Path(folder) / 'identity.bin'
            for obs, pool, ctx, row_pool in itertools.product(('v1', 'v2lite', 'v2'), ('mean', 'meanmax'), (False, True), (False, True)):
                net = RolloutNet(64 if row_pool else 13, pool, ctx, obs, row_pool, packet_head='identity',
                                  plant_embedding_dim=None if row_pool else 7).eval()
                export_student(net, path)
                env.load_rollout_net(path)
                deck = (7, 2, 0, 5, 1, 4, 3)
                live, _ = env.reset(deck=deck, task=TaskSpec(level=9, seed=30001, profile=profile_for_deck(deck)))
                # Real occupied positions rather than only an empty board, independent of random policy.
                while live['tick'] < 6000:
                    legal = live['legal_actions']['plants']
                    action = {'type': 'plant', **legal[0]} if legal else {'type': 'wait', 'ticks': 60}
                    live, *_ = env.step(action)
                history = StudentHistory(obs) if obs != 'v1' else None
                task_context = observation_context(live) if ctx else None
                previous = previous_action = None
                for _ in range(8):
                    frame = _frame(live)
                    frame['action'] = previous_action
                    native = env.rollout_predict(previous['tick'] if previous else None,
                                                 bool(previous_action and previous_action['type'] == 'plant'),
                                                 student_history=history)
                    b, s = history.features(frame, previous, task_context) if history else features(frame, previous, task_context)
                    s = identity_scalars(frame, s)
                    info = packet_features(frame)[None]
                    with torch.inference_mode():
                        h, g = net.encode(b[None], s[None], packet_info=info)
                        pairs = [(net.plant(g).flatten(), torch.tensor([native['plant']], dtype=torch.float32)),
                                 (net.packet_logits(g, info)[0], torch.tensor(native['packet'], dtype=torch.float32))]
                        if native['cell_packet'] >= 0:
                            pairs.append((net.cell_logits(h, g, torch.tensor([native['cell_packet']]), info)[0],
                                          torch.tensor(native['cell'], dtype=torch.float32)))
                        for python, cpp in pairs:
                            maximum = max(maximum, float((python - cpp).abs().max()))
                            torch.testing.assert_close(python, cpp, rtol=0, atol=1e-5)
                            self.assertEqual(int(python.argmax()), int(cpp.argmax()))
                    chosen = act(net, frame, previous, live['legal_actions']['plants'], context=task_context, history=history)
                    self.assertEqual(chosen, native['action'])
                    previous, previous_action = frame, chosen
                    live, *_ = env.step(chosen)
                    decisions += 1
        print(json.dumps({'identity_cpu_max_abs_error': maximum, 'configurations': 24,
                          'real_decisions': decisions, 'all_argmax_and_actions_equal': True}), flush=True)

    def test_native_identity_ties_are_independent_of_slot_order(self):
        torch.set_num_threads(1)
        net = RolloutNet(8, packet_head='identity').eval()
        with torch.no_grad():
            net.plant.weight.zero_()
            net.plant.bias.fill_(1)
            for parameter in net.packet.parameters():
                parameter.zero_()
        selected = []
        with tempfile.TemporaryDirectory() as folder, EventWaitEnv(RESOURCE, executable=BINARY) as env:
            path = Path(folder) / 'ties.bin'
            export_student(net, path)
            env.load_rollout_net(path)
            for deck in ((0, 1, 2, 3, 4, 5, 7), (7, 5, 4, 3, 2, 1, 0), (4, 7, 1, 0, 5, 2, 3)):
                live, _ = env.reset(deck=deck, task=TaskSpec(level=9, seed=30000, profile=profile_for_deck(deck)))
                native = env.rollout_predict()['action']
                self.assertEqual(native, act(net, _frame(live), None, live['legal_actions']['plants']))
                self.assertEqual(native['type'], 'plant')
                selected.append(deck[native['packet']])
        self.assertEqual(len(set(selected)), 1)

    def test_live_conveyor_free_cards_and_empty_slots_match_python(self):
        torch.set_num_threads(1)
        torch.manual_seed(19010)
        net = RolloutNet(64, 'meanmax', obs='v2', row_pool=True, packet_head='identity').eval()
        with torch.no_grad():
            net.plant.weight.zero_()
            net.plant.bias.fill_(1)
        empty = free = 0
        maximum = 0.
        deck = (0, 1, 2, 3, 4, 5, 7)
        with tempfile.TemporaryDirectory() as folder, EventWaitEnv(RESOURCE, executable=BINARY) as env:
            path = Path(folder) / 'conveyor.bin'
            export_student(net, path)
            env.load_rollout_net(path)
            live, _ = env.reset(deck=deck, task=TaskSpec(level=10, seed=30000, profile=profile_for_deck(deck)))
            history = StudentHistory('v2')
            previous = action = None
            for _ in range(80):
                frame = _frame(live)
                frame['action'] = action
                empty += any(p[1] < 0 for p in frame['packets'])
                legal = live['legal_actions']['plants']
                free += any(frame['packets'][p['packet']][5] > frame['sun'] for p in legal)
                native = env.rollout_predict(previous['tick'] if previous else None,
                                             bool(action and action['type'] == 'plant'), student_history=history)
                b, s = history.features(frame, previous)
                s = identity_scalars(frame, s)
                info = packet_features(frame)[None]
                with torch.inference_mode():
                    h, g = net.encode(b[None], s[None], packet_info=info)
                    pairs = [(net.plant(g).flatten(), torch.tensor([native['plant']], dtype=torch.float32)),
                             (net.packet_logits(g, info)[0], torch.tensor(native['packet'], dtype=torch.float32))]
                    if native['cell_packet'] >= 0:
                        pairs.append((net.cell_logits(h, g, torch.tensor([native['cell_packet']]), info)[0],
                                      torch.tensor(native['cell'], dtype=torch.float32)))
                    for python, cpp in pairs:
                        maximum = max(maximum, float((python - cpp).abs().max()))
                        torch.testing.assert_close(python, cpp, rtol=0, atol=1e-5)
                        self.assertEqual(int(python.argmax()), int(cpp.argmax()))
                chosen = act(net, frame, previous, legal, history=history)
                self.assertEqual(chosen, native['action'])
                previous, action = frame, chosen
                live, *_ = env.step(chosen)
        self.assertGreater(empty, 0)
        self.assertGreater(free, 0)
        print(json.dumps(dict(conveyor_decisions=80, empty_slot_positions=empty,
                              free_card_positions=free, maximum_error=maximum)), flush=True)


if __name__ == '__main__':
    unittest.main()
