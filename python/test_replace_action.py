"""Native atomic replacement, terrain/layers and unchanged old-action integration tests."""
import os
from pathlib import Path
import tempfile
import unittest

from pvz_env import PvZEnv, TaskSpec, SeedCard, profile_for_deck

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/home/newbiexvwu/.cache/pvz-research-resources'))
BASELINE = ROOT / 'artifacts/task69/baseline-engine'


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'needs native engine/resources')
class ReplaceActionTests(unittest.TestCase):
    def setUp(self):
        self.env = PvZEnv(RESOURCE, executable=BINARY)
        self.addCleanup(self.env.close)

    def reset(self, level=11, plants=((8, 0, 0),), deck=(8, 1, 0, 4, 16, 30)):
        return self.env.reset(deck=deck, task=TaskSpec(level=level, seed=6901,
            profile=profile_for_deck(deck), preplanted=plants))[0]

    def action(self, packet=1, col=0, row=0):
        return dict(type='replace', packet=packet, col=col, row=row)

    def assert_rejected_unchanged(self, action):
        before = self.env.observe()
        state = self.env.state_hash()
        after, _, _, _, info = self.env.step(action)
        self.assertFalse(info['ok'])
        self.assertEqual(info['ticks_advanced'], 0)
        self.assertEqual(state, self.env.state_hash())
        self.assertEqual(before, after)
        self.assertTrue(all(not v for v in info['events'].values()))

    def test_atomic_success_and_failed_budget_cooldown_identity(self):
        obs = self.reset()
        self.assertIn(self.action(), obs['legal_actions']['plants'])
        self.assert_rejected_unchanged(self.action(packet=2))  # pea costs 100 > 50
        self.assert_rejected_unchanged(self.action(packet=0))  # same effective puff
        self.assert_rejected_unchanged(self.action(col=1))     # empty cell
        self.assert_rejected_unchanged(self.action(col=-1))
        self.assert_rejected_unchanged(self.action(packet=99))
        after, _, _, _, info = self.env.step(self.action())
        self.assertTrue(info['ok'])
        self.assertEqual((after['tick'], after['sun']), (obs['tick'], 0))
        self.assertEqual(info['events']['sun_spent'], 50)
        self.assertEqual([p['type'] for p in after['plants']], [1])
        # Sunflower cooldown remains authoritative, including when another target exists.
        self.env.step(dict(type='plant', packet=0, col=1, row=0))
        self.assert_rejected_unchanged(self.action(col=1))

    def test_imitater_uses_effective_native_identity(self):
        self.reset(deck=(8, SeedCard(48, 8), 1, 0, 16, 30))
        self.assert_rejected_unchanged(self.action())

    def compare_with_native_shovel_then_plant(self, level, plants, deck, packet, row=0, col=0):
        before = self.reset(level, plants, deck)
        root = self.env.snapshot_fast()
        original = self.env.state_hash()
        self.env.legal_replacements()
        self.assertEqual(self.env.state_hash(), original)
        after, _, _, _, info = self.env.step(self.action(packet, col, row))
        self.assertTrue(info['ok'])
        outcome = self.env.state_hash()
        self.env.restore_fast(root)
        self.assertTrue(self.env.step(dict(type='shovel', col=col, row=row))[-1]['ok'])
        sequential = self.env.step(dict(type='plant', packet=packet, col=col, row=row))
        self.assertTrue(sequential[-1]['ok'])
        self.assertEqual(outcome, self.env.state_hash())
        self.assertEqual(after, sequential[0])
        self.assertEqual(after['tick'], before['tick'])
        self.env.release_snapshot(root)
        return after

    def test_pool_base_and_pumpkin_native_target(self):
        deck = (8, 1, 16, 30, 43, 0)
        after = self.compare_with_native_shovel_then_plant(31,
            ((16, 2, 0), (0, 2, 0), (30, 2, 0)), deck, 0, row=2)
        self.assertEqual(sorted(p['type'] for p in after['plants']), [8, 16, 30])
        # Native cattail shovel creates a lily when protected by pumpkin.
        after = self.compare_with_native_shovel_then_plant(31,
            ((16, 2, 0), (43, 2, 0), (30, 2, 0)), deck, 0, row=2)
        self.assertIn(16, [p['type'] for p in after['plants']])
        self.reset(31, ((16, 2, 0),), deck)
        self.assert_rejected_unchanged(self.action(packet=0, row=2))  # removal loses water base

    def test_roof_base_and_bare_pot_rollback(self):
        deck = (8, 1, 33, 0, 30, 4)
        after = self.compare_with_native_shovel_then_plant(41, ((0, 0, 0),), deck, 1)
        self.assertIn(33, [p['type'] for p in after['plants']])
        self.reset(41, (), deck)
        self.assert_rejected_unchanged(self.action(packet=0))  # cannot remove sole pot and plant puff

    def test_grave_cells_never_become_replacement_targets(self):
        obs = self.reset(plants=())
        graves = [g for g in obs['grid_items'] if g.get('type') == 1]
        self.assertTrue(graves, 'fixture must contain a native grave')
        for g in graves:
            self.assert_rejected_unchanged(self.action(packet=0, col=g['col'], row=g['row']))

    def test_snapshot_branch_replay_and_determinism(self):
        self.reset()
        root = self.env.snapshot_fast()
        action = self.action()
        branch = self.env.branch_snapshot(root, [action])[0]
        self.assertTrue(branch['ok'])
        self.env.restore_fast(root)
        direct = self.env.step(action)[0]
        self.assertEqual(branch['observation']['plants'], direct['plants'])
        first = self.env.state_hash()
        self.env.restore(branch['snapshot_id'])
        self.assertEqual(first, self.env.state_hash())
        self.env.restore_fast(root)
        self.env.step(action)
        self.assertEqual(first, self.env.state_hash())
        self.env.release_snapshot(branch['snapshot_id'])
        self.env.release_snapshot(root)
        self.reset(level=7)
        self.env.step(action)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'replace-replay.json.gz'
            self.env.save_replay(path)
            replayed = PvZEnv(RESOURCE, executable=BINARY)
            try:
                import gzip, json
                with gzip.open(path, 'rt') as stream:
                    records = [json.loads(line) for line in stream]
                record = {**records[0], **records[-1], 'operations': records[1:-1]}
                replayed.replay_record(record, path.parent)
                self.assertEqual(self.env.state_hash(), replayed.state_hash())
            finally:
                replayed.close()

    def test_static_scores_match_actual_replacement_and_rejected_null(self):
        import sys
        import torch
        sys.path.insert(0, str(ROOT / 'scripts'))
        from pvz_value_net import ValueNet
        from export_value_net import export_value_net
        torch.set_num_threads(1)
        torch.manual_seed(69)
        self.reset()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'static.bin'
            export_value_net(ValueNet(8).eval(), path)
            self.env.load_value_net(path, static=True)
            root = self.env.snapshot_fast()
            scores = self.env.static_scores(root, [self.action(), self.action(packet=2)], 0)
            self.assertIsNone(scores[1])
            self.env.restore_fast(root)
            self.env.step(self.action())
            self.assertAlmostEqual(scores[0], self.env.leaf_values(income=0, static=True)[0], places=6)
            self.env.release_snapshot(root)

    def test_cpp_and_python_student_choose_same_replacement_cell(self):
        import sys
        import torch
        sys.path.insert(0, str(ROOT / 'scripts'))
        from pvz_rollout_policy import RolloutNet, export_student, act
        from render_episode import _frame
        torch.set_num_threads(1)
        net = RolloutNet(width=8).eval()
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net.plant.bias.fill_(2)
            net.packet.bias[1] = 2
        obs = self.reset()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'student.bin'
            export_student(net, path)
            self.env.load_rollout_net(path)
            native = self.env.rollout_predict()
            action = act(net, _frame(obs), None, obs['legal_actions']['plants'])
            self.assertEqual(action, self.action())  # tie picks first row/column, now occupied
            self.assertEqual(native['action'], action)
            root = self.env.snapshot_fast()
            self.env.rollout(action, obs['tick'] + 60, snapshot_id=root)
            self.env.release_snapshot(root)
            # Slot zero has only replacement cells; slot one still has ordinary water
            # cells. The native packet tie must not depend on mask insertion order.
            obs = self.reset(31, tuple((8, row, col) for row in (0, 1, 4, 5)
                                      for col in range(9)), (1, 16, 8, 0, 30, 4))
            with torch.no_grad():
                net.packet.bias.zero_()
            export_student(net, path)
            self.env.load_rollout_net(path)
            native = self.env.rollout_predict()
            action = act(net, _frame(obs), None, obs['legal_actions']['plants'])
            self.assertEqual(action, self.action(packet=0))
            self.assertEqual(action, native['action'])

    @unittest.skipUnless(BASELINE.is_file(), 'requires frozen task69 baseline')
    def test_old_actions_and_full_states_bitwise_equal_to_baseline(self):
        baseline = PvZEnv(RESOURCE, executable=BASELINE)
        try:
            frames = []
            for level in (7, 11, 31, 41):
                deck = (8, 1, 0, 4, 33 if level == 41 else 16, 30)
                old, _ = baseline.reset(deck=deck, task=TaskSpec(level=level, seed=6901,
                    profile=profile_for_deck(deck), preplanted=((8, 0, 0),)))
                new = self.reset(level=level, deck=deck)
                actions = [dict(type='plant', packet=1, col=1, row=0),
                           dict(type='shovel', col=0, row=0),
                           dict(type='wait', ticks=300), dict(type='plant', packet=0, col=0, row=0),
                           dict(type='collect', x=0, y=0), dict(type='wait', ticks=1200)]
                for action in [None] + actions:
                    if action:
                        old, *old_info = baseline.step(action)
                        new, *new_info = self.env.step(action)
                        self.assertEqual(old_info, new_info)
                    # Only the action mask is extended. Every old observation field is unchanged.
                    new['legal_actions']['plants'] = [a for a in new['legal_actions']['plants']
                                                       if a.get('type') != 'replace']
                    self.assertEqual(old, new)
                    self.assertEqual(baseline.state_hash(), self.env.state_hash())
                    import sys
                    sys.path.insert(0, str(ROOT / 'scripts'))
                    from render_episode import _frame
                    frames.append({**_frame(new), 'action': action})
            # The compatibility fingerprint must mean the old encoded arrays really match.
            import importlib.util, json
            import numpy as np
            from unittest.mock import patch
            import pvz_frame_store
            def load_old(name):
                path = ROOT / 'artifacts/task69/baseline-source/python' / f'{name}.py'
                spec = importlib.util.spec_from_file_location('task69_old_' + name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            old_policy, old_store = load_old('pvz_rollout_policy'), load_old('pvz_frame_store')
            source = ''.join(json.dumps(f, separators=(',', ':')) + '\n' for f in frames).encode()
            current = pvz_frame_store.encode_frames(frames, source)
            with patch('pvz_rollout_policy.features', old_policy.features):
                expected = old_store.encode_frames(frames, source)
            for key in current:
                np.testing.assert_array_equal(current[key], expected[key], err_msg=key)
        finally:
            baseline.close()


class ReplacementTeacherTests(unittest.TestCase):
    def test_study_observer_records_layers_sun_and_never_calls_the_environment(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from replace_action_measure import Measurements
        observer = Measurements()
        before = dict(tick=0, sun=50, plants=[dict(type=8, row=0, col=0),
                                             dict(type=30, row=0, col=0)])
        after = dict(tick=0, sun=0, plants=[dict(type=1, row=0, col=0),
                                           dict(type=30, row=0, col=0)])
        observer.before(object(), before, {}, 0)  # any attempted env call would fail
        observer.after(after, dict(type='replace', packet=1, row=0, col=0), {}, False,
                       dict(events=dict(sun_spent=50, sun_produced=0)))
        result = observer.result()
        self.assertEqual(result['replace_events'][0]['removed'], [8])
        self.assertEqual(result['replace_events'][0]['added'], [1])
        self.assertEqual((result['sun_spent'], result['sun_remaining']), (50, 0))
        self.assertEqual(result['replace_attempts'], 1)
        self.assertEqual(before['plants'][0]['type'], 8)

    def test_each_card_has_independent_best_plant_and_replace_candidate(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'scripts'))
        import search_play
        legal = [dict(packet=0, row=0, col=col, **({'type': 'replace'} if col > 1 else {}))
                 for col in range(4)]
        captured = []
        args = SimpleNamespace(min_gap=0, per_card=1, halving=True, student_candidates=0)
        env = SimpleNamespace(snapshot_fast=lambda: 1, restore_fast=lambda _root: None,
                              release_snapshot=lambda _root: None)
        def scores(_env, _root, actions, _income):
            return [(a['col'], a) for a in actions]
        with patch.dict(search_play._W, {}, clear=True), \
             patch.object(search_play, '_static_scores', side_effect=scores), \
             patch.object(search_play, '_halving_choice',
                          side_effect=lambda _env, _obs, _root, candidates, *args:
                              captured.extend(candidates) or search_play.NOOP):
            search_play._search(env, dict(tick=0, legal_actions=dict(plants=legal)),
                                search_play.NOOP, args, 69, dict(searched=0))
        self.assertEqual({(a['type'], a['col']) for a in captured if a['type'] != 'wait'},
                         {('plant', 1), ('replace', 3)})

    def test_verification_attempts_and_revocations_use_same_margin_for_both_types(self):
        import sys
        from collections import Counter
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'scripts'))
        import search_play
        args = SimpleNamespace(samples=3, horizon=2500, verify_samples=9)
        for kind in ('plant', 'replace'):
            stats = dict(overrides=0, override_kinds=Counter())
            challenger = dict(type=kind, packet=0, row=0, col=0)
            futures = [dict(rollout_result=.8)] * 9 + [dict(rollout_result=.1)] * 9
            with patch.object(search_play, '_rollout_policy', side_effect=futures):
                chosen = search_play._verified_choice(None, dict(tick=0), 1,
                    [search_play.NOOP, challenger], [.4, .6], [3, 3], 1, 0,
                    search_play.NOOP, args, 69, stats)
            self.assertEqual(chosen, search_play.NOOP)
            self.assertEqual(stats['verification_by_type'], {kind: dict(attempts=1, revoked=1)})


if __name__ == '__main__':
    unittest.main()
