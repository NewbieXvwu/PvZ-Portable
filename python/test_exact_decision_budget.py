"""Boundary reservations, unchanged legacy rollout, bootstrap and course contracts."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from pvz_research import decision_quotas, _assign, load_config, ROOT
from pvz_curriculum import initial_state, observe
from train_pvz_ppo import add_advantages
import train_pvz_ppo_task_family as family


class ExactDecisionBudgetTests(unittest.TestCase):
    def test_explicit_integer_boundaries_and_legacy_config_compatibility(self):
        config = json.loads((ROOT/'experiments/t5/learning_reference256_interrupt_v1/continuous.json').read_text())
        config['prerequisites'] = []  # Parser unit fixture grants no real-run authorization.
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'config.json'
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)[0],config)
            config['budget']['boundary_mode']='exact_decisions_v1'
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)[0],config)
            for mode,budget,nodes in (('unknown',5000,[5000]),('exact_decisions_v1',5000.0,[5000]),
                    ('exact_decisions_v1',5000,[5000.0]),('exact_decisions_v1',5000,[5001])):
                changed=copy.deepcopy(config)
                changed['budget'].update(boundary_mode=mode,decisions=budget)
                changed['evaluation']['decision_nodes']=nodes
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_reservations_cannot_overshoot_and_refill_actual_normal_terminals(self):
        rng = random.Random(771)
        for budget in (1, 2, 19, 20, 21, 501, 5000, 5001):
            count, jobs, all_lengths = 0, 0, []
            while count < budget:
                size = min(20, budget-count)
                ids = list(range(jobs, jobs+size))
                quotas = decision_quotas(ids, budget-count, 100)
                self.assertLessEqual(sum(quotas.values()), budget-count)
                self.assertEqual(list(quotas), ids)
                lengths = [rng.randint(1, quotas[i]) for i in ids]
                self.assertTrue(all(1 <= quotas[i] <= 100 for i in ids))
                count += sum(lengths)
                jobs += size
                all_lengths.extend(lengths)
                self.assertLessEqual(count, budget)
            self.assertEqual(count, budget)
            self.assertEqual(sum(all_lengths), budget)

    def test_integer_balanced_reservations_and_max_actions(self):
        self.assertEqual(decision_quotas([11, 8, 2], 11, 4000), {11:4, 8:4, 2:3})
        self.assertEqual(decision_quotas([11, 8, 2], 99999, 4000), {11:4000, 8:4000, 2:4000})
        for ids, remainder, cap in (([],1,1),([1,1],3,1),([1,2],1,1),([1],True,1),([1],1,0)):
            with self.assertRaises(ValueError):
                decision_quotas(ids,remainder,cap)

    def test_assignments_rng_is_independent_of_quota_bookkeeping(self):
        tasks = [dict(task_id='a',seeds=[1,2]),dict(task_id='b',seeds=[3,4])]
        a, b = random.Random(37), random.Random(37)
        expected = _assign(tasks,[0,1],a,{'a':[],'b':[]},'balanced')
        actual = _assign(tasks,[0,1],b,{'a':[],'b':[]},'balanced')
        for job,quota in decision_quotas([0,1],7,4000).items():
            actual[job]['decision_quota'] = quota
        self.assertEqual(a.getstate(),b.getstate())
        self.assertEqual({i:{k:v for k,v in d.items() if k != 'decision_quota'}
                          for i,d in actual.items()},expected)

    def rollout(self, assignment, result):
        changes = dict(WORKER_MODEL=object(),WORKER_ENV=object(),WORKER_ASSIGNMENTS={9:assignment},
                       WORKER_MAX_ACTIONS=4000,WORKER_REWARD_CONFIG={},WORKER_ALLOW_TRUNCATION=True)
        with patch.multiple(family,**changes), patch.object(family,'collect_task_episode',return_value=copy.deepcopy(result)) as collect:
            saved = family._rollout_worker(9)
            return saved,collect.call_args

    def test_actual_worker_passes_reserved_limit_without_falsifying_terminal(self):
        assignment=dict(action_seed=21,task={},task_seed=44,decision_quota=3)
        episode=dict(terminated=False,truncated=True,won=False,bootstrap_value=.75)
        saved,args = self.rollout(assignment,episode)
        self.assertEqual(args.args[5],3)
        self.assertTrue(args.kwargs['allow_truncation'])
        self.assertEqual(saved['bootstrap_value'],.75)
        self.assertFalse(saved['terminated'])
        self.assertEqual(saved['truncation_reason'],'decision_budget_boundary')
        normal,_ = self.rollout(assignment,dict(terminated=True,truncated=False,won=True))
        self.assertIsNone(normal['truncation_reason'])
        self.assertTrue(normal['won'])

    def test_legacy_assignment_returns_unchanged_episode(self):
        episode=dict(terminated=False,truncated=True,won=False,bootstrap_value=.75)
        saved,args = self.rollout(dict(action_seed=21,task={},task_seed=44),episode)
        self.assertEqual(saved,episode)
        self.assertEqual(args.args[5],4000)

    def test_worker_rejects_noninteger_or_oversized_reservations(self):
        for quota in (True,0,-1,4001,1.5):
            with self.assertRaisesRegex(ValueError,'invalid rollout decision reservation'):
                self.rollout(dict(action_seed=21,task={},task_seed=44,decision_quota=quota),{})

    def test_boundary_truncation_bootstraps_actual_time_and_is_not_course_failure(self):
        tasks = [dict(task_id='a',terrain='day',wave_cap=3)]
        course=initial_state(tasks)
        episode=dict(task_id='a',terminated=False,truncated=True,won=False,bootstrap_value=.75,
            transitions=[dict(value=.2,reward=0.,action_duration_ticks=60)])
        add_advantages([episode],.95,.99)
        self.assertAlmostEqual(episode['transitions'][0]['return'],(.99**.2)*.75)
        observe(course,[episode],dict(window_episodes=8,minimum_window_episodes=2,
                                    uniform_fraction=.25,coverage='terrain_wave_cap'))
        self.assertEqual(course['history']['a'],[])
        self.assertEqual(course['completed']['a'],0)
        self.assertEqual(course['ignored_truncations']['a'],1)


if __name__ == '__main__':
    unittest.main()
