"""Synthetic policy/collector contracts; these do not certify native physics or CUDA."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from pvz_agent_model import (GameplayModelV1, WAIT_TICKS, configure_torch_threads,
                             hard_behavior_cloning_loss, legal_summary,
                             model_architecture_version, observation_tokens, pack_tokens,
                             replay_log_probs, select_action, unpack_tokens)
from pvz_common import ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION
from pvz_env import PvZEnv
from pvz_event_env import EventWaitEnv, policy_env
from pvz_research import capture_rng, load_config, restore_rng
from pvz_seed_jobs import atomic_numpy, read_numpy
from pvz_wait_events import CONDITIONS, summarize_wait_records
from test_observation_context import SMALL, source
from train_pvz_ppo import add_advantages, collect_task_episode, episode_digest, episode_hash, train_update
import t4_capability_profile as profile


def wait_result(action, actual, *, reason=None):
    reason = reason or ('max_ticks' if action['until'] == 'timeout' else 'condition')
    return dict(version=1, condition=action['until'], requested_ticks=action['ticks'],
                actual_ticks=actual, logic_steps=action['ticks'] if reason == 'max_ticks' else actual,
                stalled_clock_steps=action['ticks']-actual if reason == 'max_ticks' else 0,
                initial_condition_satisfied=action['until'] == 'left_zone_occupied' and actual == 0,
                reason=reason, triggered=[reason])


def fixture(model):
    """Mixed actions, six conditions, zero/short actual waits and unequal sequences."""
    episodes = []
    for seed, length in enumerate((11, 8)):
        transitions, hidden, previous, previous_result, delta = [], None, None, None, 0
        obs = source()
        for index in range(length):
            obs['tick'] += delta
            tensors, metadata = observation_tokens(obs, 7)
            legal = legal_summary(obs['legal_actions'])
            if index % 4 == 1:
                action = {'type': 'plant', **obs['legal_actions']['plants'][index % 3]}
            elif index % 4 == 3:
                col, row = obs['legal_actions']['shovels'][0]
                action = dict(type='shovel', row=row, col=col)
            else:
                condition = CONDITIONS[(index // 2 + seed) % len(CONDITIONS)]
                action = dict(type='wait', ticks=WAIT_TICKS[index % 3])
                if model.config.get('wait_mode') == 'events':
                    action['until'] = condition
            with torch.no_grad():
                output = model.step_tokens(tensors, metadata, obs['wave'], hidden, previous,
                                           delta, {}, previous_result)
                _, log_prob, _ = select_action(model, output, legal, action=action)
                value = model.privileged_value_from_extra(output, [0.] * 16)
            duration = (0 if action['type'] != 'wait' or action.get('until') == 'left_zone_occupied'
                        else min(action['ticks'], (1, 37, 149)[index % 3]))
            result = wait_result(action, duration) if 'until' in action else None
            transitions.append(dict(tokens=pack_tokens(tensors, metadata), wave=obs['wave'],
                previous_action=previous, previous_wait_result=previous_result,
                elapsed_since_previous_observation=delta, events={}, legal=legal,
                action=action, wait_result=result, log_prob=log_prob.item(), value=value.item(),
                critic_extra=[0.] * 16, action_duration_ticks=duration, discount=.99**(duration/300),
                reward=-1. if index == length-1 else 0.))
            hidden, previous, previous_result, delta = output['hidden'], action, result, duration
        episodes.append(dict(seed=seed, task_seed=seed+100, task_id='synthetic', result=2,
                             transitions=transitions))
    add_advantages(episodes, .95)
    return episodes


def update(model, episodes, optimizer):
    return train_update(model, episodes, optimizer, torch.device('cpu'), 2, 3,
                        .2, .5, .01, minibatch_chunks=2, attention_backend='dense')


class SyntheticEnv(PvZEnv):
    """Deterministic fixture; never starts the simulator and has no physical claims."""
    def __init__(self, terminal_after=3, bad_result=False):
        super().__init__('/tmp/not-native-resources')
        self.terminal_after, self.bad_result = terminal_after, bad_result

    def reset(self, **kwargs):
        self.obs, self.calls = source(), 0
        self.obs['sun'] = 50
        self.obs['legal_actions'] = dict(plants=[], shovels=[], wait=True)
        return copy.deepcopy(self.obs), {}

    def critic_inputs(self, wave):
        return dict(wave_zombies=[0, 1])

    def step(self, action):
        self.calls += 1
        duration = (0, 37, 3)[(self.calls-1) % 3]
        self.obs['tick'] += duration
        self.obs.update(terminal=self.calls >= self.terminal_after,
                        result=1 if self.calls >= self.terminal_after else 0)
        info = dict(ok=True, ticks_advanced=duration, events={})
        if 'until' in action:
            info['wait_result'] = wait_result(action, duration,
                                             reason='terminal' if self.obs['terminal'] else 'condition')
            if self.bad_result and self.obs['terminal']:
                info['wait_result']['actual_ticks'] += 1
        return copy.deepcopy(self.obs), 0., self.obs['terminal'], False, info


class SyntheticEventEnv(SyntheticEnv, EventWaitEnv):
    pass


TASK = dict(task_id='synthetic', level=1, playthrough=2, zombie_count_multiplier=1.,
            wave_cap=1, preplanted=[], deck=[0, 1, 2, 3, 4, 5], sun_start=50)


class EventWaitPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.model = GameplayModelV1({**SMALL, 'wait_mode': 'events'}).eval()
        self.obs = source()
        self.output = self.model.step(self.obs)

    def test_paired_parameters_and_legacy_rng_streams_match_for_three_seeds(self):
        for seed in (0, 1, 2):
            states, draws = [], []
            for mode in (None, 'fixed', 'events'):
                torch.manual_seed(seed)
                model = GameplayModelV1({**SMALL, **({'wait_mode': mode} if mode else {})})
                states.append(model.state_dict())
                draws.append(torch.rand(5))
            self.assertEqual(set(states[1]), set(states[2]))
            for name in states[1]:
                torch.testing.assert_close(states[1][name], states[2][name], rtol=0, atol=0)
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
            torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
            torch.testing.assert_close(draws[1], draws[2], rtol=0, atol=0)

    def test_architecture_tag_is_configuration_specific(self):
        self.assertEqual(model_architecture_version(SMALL), 7)
        for mode in ('fixed', 'events'):
            self.assertEqual(model_architecture_version({**SMALL, 'wait_mode': mode}), 8)
        for mode in ('event', None, True):
            with self.assertRaisesRegex(ValueError, 'wait_mode'):
                GameplayModelV1({**SMALL, 'wait_mode': mode})

    def test_wait_probability_and_entropy_include_the_six_way_condition(self):
        output = dict(self.output, wait_condition_logits=torch.tensor([0., 1., 2., 3., 4., 5.]))
        action = dict(type='wait', ticks=150, until='sun_increased')
        _, lp, entropy = select_action(self.model, output, self.obs, action=action)
        distributions = [torch.distributions.Categorical(logits=output[key])
                         for key in ('type_logits', 'wait_logits', 'wait_condition_logits')]
        expected = sum(d.log_prob(torch.tensor(i)) for d, i in zip(distributions, (2, 1, 3)))
        torch.testing.assert_close(lp, expected)
        torch.testing.assert_close(entropy, sum(d.entropy() for d in distributions))
        _, other_lp, _ = select_action(self.model, output, self.obs,
                                       action={**action, 'until': 'timeout'})
        self.assertAlmostEqual(float(lp-other_lp), 3., places=6)
        torch.testing.assert_close(hard_behavior_cloning_loss(self.model, output, self.obs, action), -lp)

    def test_sampling_and_greedy_waits_contain_condition_and_maximum(self):
        legal = dict(packets=[], plant_mask=[], shovel_mask=0, wait=True)
        for deterministic in (False, True):
            action, _, _ = select_action(self.model, self.output, legal, deterministic=deterministic)
            self.assertEqual(set(action), {'type', 'ticks', 'until'})
            self.assertIn(action['until'], CONDITIONS)
            self.assertIn(action['ticks'], WAIT_TICKS)

    def test_invalid_event_actions_are_rejected_in_sampling_and_batched_replay(self):
        actions = [dict(type='wait', ticks=150), dict(type='wait', ticks=150, until='private_wave'),
                   dict(type='wait', ticks=True, until='timeout'),
                   dict(type='wait', ticks=149, until='timeout'),
                   dict(type='wait', ticks=150, until='timeout', plan='hidden'),
                   dict(type='shovel', col=3, row=2, until='timeout')]
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    select_action(self.model, self.output, self.obs, action=action)
                with self.assertRaises(ValueError):
                    replay_log_probs(self.model, [self.output],
                                     [dict(action=action, legal=legal_summary(self.obs['legal_actions']))])

    def test_fixed_and_legacy_models_reject_event_actions(self):
        for mode in (None, 'fixed'):
            model = GameplayModelV1({**SMALL, **({'wait_mode': mode} if mode else {})})
            with self.assertRaisesRegex(ValueError, 'until'):
                select_action(model, model.step(self.obs), self.obs,
                              action=dict(type='wait', ticks=60, until='wave_changed'))

    def test_condition_head_gradient_is_conditional_on_wait_type(self):
        for action in (dict(type='plant', **self.obs['legal_actions']['plants'][0]),
                       dict(type='shovel', row=2, col=3),
                       dict(type='wait', ticks=60, until='packet_became_ready')):
            self.model.zero_grad(set_to_none=True)
            output = self.model.step(self.obs)
            _, lp, _ = select_action(self.model, output, self.obs, action=action)
            (-lp).backward()
            gradient = self.model.wait_condition.weight.grad
            if action['type'] == 'wait':
                self.assertGreater(float(gradient.abs().sum()), 0.)
            else:
                self.assertIsNone(gradient)

    def test_fixed_control_does_not_score_or_train_condition_head(self):
        model = GameplayModelV1({**SMALL, 'wait_mode': 'fixed'})
        output = model.step(self.obs)
        _, lp, entropy = select_action(model, output, self.obs, action=dict(type='wait', ticks=60))
        expected_lp = (torch.distributions.Categorical(logits=output['type_logits']).log_prob(torch.tensor(2))
                       + torch.distributions.Categorical(logits=output['wait_logits']).log_prob(torch.tensor(0)))
        torch.testing.assert_close(lp, expected_lp)
        (-lp+entropy).backward()
        self.assertIsNone(model.wait_condition.weight.grad)

    def test_fixed_control_collects_and_trains_with_explicit_empty_result_fields(self):
        model = GameplayModelV1({**SMALL, 'wait_mode': 'fixed'}).eval()
        episode = collect_task_episode(model, SyntheticEnv(), TASK, 1, 0, 4)
        for step in episode['transitions']:
            self.assertIsNone(step['wait_result'])
            self.assertIsNone(step['previous_wait_result'])
            self.assertNotIn('until', step['action'])
        before = model.wait_condition.weight.detach().clone()
        losses = update(model, fixture(model), torch.optim.AdamW(model.parameters(), lr=1e-4))
        self.assertLess(losses['first_minibatch_log_prob_max_error'], 5e-5)
        torch.testing.assert_close(before, model.wait_condition.weight, rtol=0, atol=0)

    def test_previous_condition_reason_and_exact_actual_time_reach_the_recurrence(self):
        action = dict(type='wait', ticks=150, until='sun_increased')
        result = wait_result(action, 37)
        first = self.model.step(self.obs, previous_action=action, delta_ticks=37, previous_wait_result=result)
        cases = [(dict(type='wait', ticks=150, until='wave_changed'), 37, None),
                 (action, 37, wait_result(action, 37, reason='defense_lost')),
                 (action, 38, wait_result(action, 38))]
        for other, duration, metadata in cases:
            metadata = metadata or wait_result(other, duration)
            second = self.model.step(self.obs, previous_action=other, delta_ticks=duration,
                                     previous_wait_result=metadata)
            self.assertGreater(float((first['belief']-second['belief']).detach().abs().max()), 0.)

    def test_zero_duration_and_simultaneous_reasons_are_not_dropped(self):
        action = dict(type='wait', ticks=300, until='left_zone_occupied')
        result = wait_result(action, 0, reason='terminal')
        result['triggered'] = ['terminal', 'condition']
        self.model.step(self.obs, previous_action=action, delta_ticks=0, previous_wait_result=result)
        summary = summarize_wait_records([dict(action=action, wait_result=result, actual_ticks=0)])
        self.assertEqual(summary['primary_reason_counts'], dict(terminal=1))
        self.assertEqual(summary['all_trigger_counts'], dict(terminal=1, condition=1))
        self.assertEqual(summary['zero_actual_tick_waits'], 1)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.model.step(self.obs, previous_action=action, delta_ticks=0)
        result['triggered'].reverse()
        with self.assertRaisesRegex(ValueError, 'duration/reason'):
            self.model.step(self.obs, previous_action=action, delta_ticks=0, previous_wait_result=result)

    def test_wait_context_rejects_tick_mismatch_and_extra_metadata(self):
        action = dict(type='wait', ticks=60, until='wave_changed')
        with self.assertRaisesRegex(ValueError, 'duration/reason'):
            self.model.step(self.obs, previous_action=action, delta_ticks=37,
                            previous_wait_result=wait_result(action, 38))
        with self.assertRaisesRegex(ValueError, 'initial'):
            self.model.step(self.obs, delta_ticks=1)
        with self.assertRaisesRegex(ValueError, 'only event waits'):
            self.model.step(self.obs, previous_action=dict(type='shovel', row=2, col=3),
                            previous_wait_result=wait_result(action, 37))

    def test_single_step_and_padded_sequence_likelihood_value_and_hidden_match(self):
        episodes = fixture(self.model)
        outputs, hiddens = self.model.forward_sequences([e['transitions'] for e in episodes], [None, None])
        transitions = [t for e in episodes for t in e['transitions']]
        lp, ent = replay_log_probs(self.model, outputs, transitions)
        offset = 0
        for episode_index, episode in enumerate(episodes):
            hidden = None
            for tr in episode['transitions']:
                tensors, metadata = unpack_tokens(tr['tokens'], torch.device('cpu'))
                out = self.model.step_tokens(tensors, metadata, tr['wave'], hidden, tr['previous_action'],
                    tr['elapsed_since_previous_observation'], tr['events'], tr['previous_wait_result'])
                _, single_lp, single_ent = select_action(self.model, out, tr['legal'], action=tr['action'])
                torch.testing.assert_close(lp[offset], single_lp, rtol=0, atol=5e-5)
                torch.testing.assert_close(ent[offset], single_ent, rtol=0, atol=5e-5)
                torch.testing.assert_close(outputs[offset]['wait_condition_logits'],
                                           out['wait_condition_logits'], rtol=0, atol=5e-5)
                torch.testing.assert_close(self.model.privileged_value_from_extra(outputs[offset], tr['critic_extra']),
                                           self.model.privileged_value_from_extra(out, tr['critic_extra']),
                                           rtol=0, atol=5e-5)
                hidden, offset = out['hidden'], offset+1
            torch.testing.assert_close(hiddens[:, episode_index], hidden[:, 0], rtol=0, atol=5e-5)

    def test_real_cpu_optimizer_replays_all_chunks_before_parameter_changes(self):
        episodes = fixture(self.model)
        before = self.model.wait_condition.weight.detach().clone()
        losses = update(self.model, episodes, torch.optim.AdamW(self.model.parameters(), lr=1e-4))
        self.assertLess(losses['first_minibatch_log_prob_max_error'], 5e-5)
        self.assertGreater(float((before-self.model.wait_condition.weight).abs().sum()), 0.)

    def test_sequence_replay_checks_final_metadata_even_without_a_following_decision(self):
        episodes = fixture(self.model)
        sequence = copy.deepcopy(episodes[0]['transitions'])
        sequence[-1]['wait_result']['actual_ticks'] += 1
        with self.assertRaisesRegex(ValueError, 'duration/reason'):
            self.model.forward_sequences([sequence], [None])
        sequence = copy.deepcopy(episodes[0]['transitions'])
        sequence[-1].pop('previous_wait_result')
        with self.assertRaisesRegex(ValueError, 'omitted wait metadata'):
            self.model.forward_sequences([sequence], [None])

    def test_synthetic_full_model_optimizer_and_rng_resume_next_update_exactly(self):
        episodes = fixture(self.model)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        update(self.model, episodes, optimizer)
        assignments = random.Random(29)
        saved = copy.deepcopy(dict(model=self.model.state_dict(), optimizer=optimizer.state_dict(),
                                   rng=capture_rng(assignments)))
        expected_draws = (random.random(), np.random.rand(), torch.rand(3), assignments.random())
        # Resample probabilities under the updated policy before the next PPO update.
        second = fixture(self.model)
        expected_loss = update(self.model, second, optimizer)
        restored = GameplayModelV1(self.model.config).eval()
        restored.load_state_dict(saved['model'])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        restored_optimizer.load_state_dict(saved['optimizer'])
        restored_assignments = random.Random()
        restore_rng(saved['rng'], restored_assignments)
        actual_draws = (random.random(), np.random.rand(), torch.rand(3), restored_assignments.random())
        self.assertEqual(expected_draws[:2], actual_draws[:2])
        torch.testing.assert_close(expected_draws[2], actual_draws[2], rtol=0, atol=0)
        self.assertEqual(expected_draws[3], actual_draws[3])
        actual_loss = update(restored, fixture(restored), restored_optimizer)
        self.assertEqual(expected_loss, actual_loss)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        for expected, actual in zip(optimizer.state.values(), restored_optimizer.state.values()):
            for name in expected:
                torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)

    def test_raw_shard_preserves_both_wait_metadata_and_digest_tamper_detection(self):
        episode = fixture(self.model)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.npz'
            atomic_numpy(path, episode, compressed=True)
            restored = read_numpy(path)
        self.assertEqual(episode_digest(episode), episode_digest(restored))
        self.assertEqual(episode_hash(episode), episode_hash(restored))
        for field in ('wait_result', 'previous_wait_result'):
            changed = copy.deepcopy(restored)
            tr = next(t for t in changed['transitions'] if t[field] is not None)
            tr[field]['triggered'].append('terminal')
            self.assertNotEqual(episode_digest(episode), episode_digest(changed))
            self.assertNotEqual(episode_hash(episode), episode_hash(changed))

    def test_legacy_episode_digests_stay_identical_to_the_engine_commit(self):
        from test_episode_digest import _episode
        self.assertEqual(episode_digest(_episode()), '0915db43c1a665da68b8fdf2feb2e3b7')
        self.assertEqual(episode_hash(_episode()),
                         '6198c041181d0b6574657185e2b205a2bfb7506b8be0663072f663902f904194')

    def test_environment_factory_and_collector_reject_fixed_env_for_event_model(self):
        with policy_env(self.model.config, '/tmp/not-native-resources') as env:
            self.assertIsInstance(env, EventWaitEnv)
        with policy_env({**SMALL, 'wait_mode': 'fixed'}, '/tmp/not-native-resources') as env:
            self.assertIs(type(env), PvZEnv)
        with self.assertRaisesRegex(ValueError, 'requires EventWaitEnv'):
            collect_task_episode(self.model, SyntheticEnv(), TASK, 1, 0, 4)

    def test_collector_uses_actual_time_for_gamma_shaping_and_next_policy_context(self):
        episode = collect_task_episode(self.model, SyntheticEventEnv(), TASK, 1, 0, 4)
        transitions = episode['transitions']
        self.assertEqual([t['action_duration_ticks'] for t in transitions], [0, 37, 3])
        for index, tr in enumerate(transitions):
            self.assertEqual(tr['wait_result']['actual_ticks'], tr['action_duration_ticks'])
            self.assertEqual(tr['discount'], .99**(tr['action_duration_ticks']/300))
            if index:
                self.assertEqual(tr['previous_wait_result'], transitions[index-1]['wait_result'])
        self.assertAlmostEqual(transitions[-1]['reward'], 1.-transitions[-1]['potential'])

    def test_collector_checks_final_wait_metadata_and_truncation_bootstrap(self):
        with self.assertRaisesRegex(ValueError, 'duration/reason'):
            collect_task_episode(self.model, SyntheticEventEnv(bad_result=True), TASK, 1, 0, 4)
        episode = collect_task_episode(self.model, SyntheticEventEnv(terminal_after=10), TASK, 1, 0, 2,
                                       allow_truncation=True)
        self.assertTrue(episode['truncated'])
        self.assertTrue(np.isfinite(episode['bootstrap_value']))

    def test_evaluator_preserves_raw_interruptions_and_checks_final_metadata(self):
        record = profile.run_episode(SyntheticEventEnv(), TASK, 1, 'checkpoint', self.model, max_actions=4)
        self.assertEqual(record['wait_mode'], 'events')
        self.assertEqual([r['actual_ticks'] for r in record['wait_records']], [0, 37, 3])
        summary = profile.summarize_episodes([record])
        self.assertEqual(summary['wait_summary']['actual_ticks'], 40)
        self.assertEqual(summary['wait_summary']['wait_actions'], 3)
        with self.assertRaisesRegex(ValueError, 'duration/reason'):
            profile.run_episode(SyntheticEventEnv(bad_result=True), TASK, 1, 'checkpoint', self.model,
                                max_actions=4)

    def test_new_and_legacy_checkpoint_loading_requires_correct_version_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.pt'
            for mode in (None, 'fixed', 'events'):
                model = GameplayModelV1({**SMALL, **({'wait_mode': mode} if mode else {})})
                checkpoint = dict(state_dict=model.state_dict(), config=model.config, research_version=1,
                    model_architecture_version=model_architecture_version(model.config),
                    value_semantics='research_explicit_return_v1',
                    provenance=dict(protocol_version=ENV_PROTOCOL_VERSION, observation_version=OBSERVATION_VERSION,
                                    task_version=TASK_VERSION))
                torch.save(checkpoint, path)
                restored, _ = profile.load_checkpoint(path)
                self.assertEqual(restored.config, model.config)
                checkpoint['model_architecture_version'] = 8 if mode is None else 7
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, 'semantics'):
                    profile.load_checkpoint(path)

    def test_formal_configuration_accepts_explicit_modes_without_changing_old_matrix(self):
        root = Path(__file__).resolve().parent.parent
        original = json.loads((root/'experiments/t5/reward_comparison_v2/reward_r0_seed0_v2.json').read_text())
        for mode in ('fixed', 'events'):
            config = copy.deepcopy(original)
            config['model'] = {**SMALL, 'wait_mode': mode}
            config['runtime'].update(deterministic_algorithms=False, cublas_workspace_config=None)
            config['prerequisites'] = []  # Parser fixture only; no experiment is launched.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'parser.json'
                path.write_text(json.dumps(config))
                loaded, _, _ = load_config(path)
                self.assertEqual(loaded['model']['wait_mode'], mode)
        self.assertNotIn('wait_mode', original['model'])


if __name__ == '__main__':
    unittest.main()
