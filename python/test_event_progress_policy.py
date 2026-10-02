"""The progress mask uses public raw predicates and the actual PPO replay path."""
from __future__ import annotations

import copy
import unittest

import torch

from pvz_agent_model import (GameplayModelV1, hard_behavior_cloning_loss,
    model_architecture_version, policy_legal_summary, replay_log_probs, select_action)
from test_event_wait_policy import fixture, update
from test_observation_context import SMALL, source
from pvz_wait_events import CONDITIONS


class ProgressWaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(0)
        self.config={**SMALL,'wait_mode':'events','wait_mask':'progress_v1'}
        self.model=GameplayModelV1(self.config).eval()
        self.obs=source()

    def test_only_visible_already_occupied_condition_is_removed(self):
        for on_board,x,allowed in [(True,160.,False),(True,160.01,True),
                                   (False,100.,True),(True,159.99,False)]:
            with self.subTest(on_board=on_board,x=x):
                obs=copy.deepcopy(self.obs)
                obs['zombies'][0].update(on_board=on_board,x=x)
                mask=policy_legal_summary(obs,self.config)['wait_condition_mask']
                self.assertEqual(mask[CONDITIONS.index('left_zone_occupied')],allowed)
                self.assertTrue(all(value for name,value in zip(CONDITIONS,mask)
                                    if name!='left_zone_occupied'))

    def test_greedy_and_samples_do_not_select_the_instant_condition(self):
        self.obs['zombies'][0]['x']=100.
        out=self.model.step(self.obs)
        out['type_logits']=torch.tensor([-1000.,-1000.,1000.])
        out['wait_condition_logits']=torch.tensor([0.,0.,0.,0.,1000.,0.])
        for deterministic in (True,False):
            for _ in range(20):
                action,lp,entropy=select_action(self.model,out,self.obs,deterministic=deterministic)
                self.assertNotEqual(action['until'],'left_zone_occupied')
                self.assertTrue(torch.isfinite(lp).item())
                self.assertTrue(torch.isfinite(entropy).item())

    def test_saved_legality_is_required_even_for_nonwait_transitions(self):
        out=self.model.step(self.obs)
        for call in (lambda:select_action(self.model,out,dict(packets=[],plant_mask=[],shovel_mask=0,wait=True)),
                     lambda:replay_log_probs(self.model,[out],[dict(action=dict(type='wait',ticks=60,until='timeout'),legal={})])):
            with self.assertRaisesRegex(ValueError,'saved public'):
                call()

    def test_illegal_recorded_wait_is_rejected_by_scalar_and_batch_replay(self):
        self.obs['zombies'][0]['x']=100.
        out=self.model.step(self.obs)
        legal=policy_legal_summary(self.obs,self.config)
        action=dict(type='wait',ticks=60,until='left_zone_occupied')
        for call in (lambda:select_action(self.model,out,legal,action=action),
                     lambda:replay_log_probs(self.model,[out],[dict(action=action,legal=legal)]),
                     lambda:hard_behavior_cloning_loss(self.model,out,self.obs,action)):
            with self.assertRaisesRegex(ValueError,'already satisfied'):
                call()

    def test_scalar_batch_and_cloning_use_the_same_masked_distribution(self):
        self.obs['zombies'][0]['x']=100.
        out=self.model.step(self.obs)
        legal=policy_legal_summary(self.obs,self.config)
        action=dict(type='wait',ticks=150,until='sun_increased')
        _,lp,entropy=select_action(self.model,out,self.obs,action=action)
        batch_lp,batch_entropy=replay_log_probs(self.model,[out],[dict(action=action,legal=legal)])
        torch.testing.assert_close(batch_lp[0],lp)
        torch.testing.assert_close(batch_entropy[0],entropy)
        torch.testing.assert_close(hard_behavior_cloning_loss(self.model,out,self.obs,action),-lp)

    def test_new_policy_version_and_legacy_configuration(self):
        self.assertEqual(model_architecture_version(self.config),13)
        self.assertEqual(model_architecture_version({**SMALL,'wait_mode':'events'}),12)
        for cfg in ({**self.config,'wait_mode':'fixed'}, {**self.config,'wait_mask':False}):
            with self.assertRaisesRegex(ValueError,'wait_mask'):
                GameplayModelV1(cfg)
        legacy=GameplayModelV1({**SMALL,'wait_mode':'events'}).eval()
        legacy.load_state_dict(self.model.state_dict())
        self.obs['zombies'][0]['x']=100.
        out=legacy.step(self.obs)
        out['type_logits']=torch.tensor([-1000.,-1000.,1000.])
        out['wait_condition_logits']=torch.tensor([0.,0.,0.,0.,1000.,0.])
        action,_,_=select_action(legacy,out,self.obs,deterministic=True)
        self.assertEqual(action['until'],'left_zone_occupied')

    def test_real_ppo_path_replays_saved_masks_without_resetting_history(self):
        # Existing mixed-action fixture has no left-zone occupants. Its saved
        # masks are reconstructed from that same public observation.
        legacy=GameplayModelV1({**SMALL,'wait_mode':'events'}).eval()
        legacy.load_state_dict(self.model.state_dict())
        episodes=fixture(legacy)
        legal=policy_legal_summary(self.obs,self.config)
        for episode in episodes:
            for transition in episode['transitions']:
                transition['legal']['wait_condition_mask']=legal['wait_condition_mask']
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=1e-4)
        result=update(self.model,episodes,optimizer)
        self.assertLess(result['first_minibatch_log_prob_max_error'],1e-6)
        self.assertGreater(result['optimizer_steps'],0)


if __name__=='__main__':
    unittest.main()
