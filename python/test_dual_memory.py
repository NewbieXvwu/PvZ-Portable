"""Goal-gradient reach, event timing, ragged replay and real PPO window handling."""
from __future__ import annotations

import copy
import unittest

import torch

from pvz_agent_model import (GameplayModelV1, model_architecture_version,
    observation_tokens, pack_tokens, policy_legal_summary, replay_log_probs, select_action)
from pvz_dual_memory import gradient_start
from test_event_wait_policy import wait_result
from test_observation_context import SMALL, source
from train_pvz_ppo import add_advantages, train_update


def config(cell='gru', **settings):
    return {**SMALL, 'gru_layers':1,'wait_mode':'events','wait_mask':'progress_v1',
            'slow_memory':{'cell':cell,'layers':1,'goal_width':8,'max_ticks':3000,
                           'max_decisions':64,**settings}}


def fixture(model, lengths=(9,6), wave_interval=4):
    episodes=[]
    for job,length in enumerate(lengths):
        hidden,previous,result,delta=None,None,None,0
        transitions=[]
        for index in range(length):
            obs=source();obs.update(tick=index*150,wave=1+index//wave_interval)
            tensors,metadata=observation_tokens(obs,7)
            context=model.planner_context(obs)
            legal=policy_legal_summary(obs,model.config)
            action=dict(type='wait',ticks=150,until='timeout')
            with torch.no_grad():
                output=model.step_tokens(tensors,metadata,obs['wave'],hidden,previous,delta,{},result,
                                         planner_context=context)
                _,lp,_=select_action(model,output,legal,action=action)
                value=model.privileged_value_from_extra(output,[0.]*16)
            wait=wait_result(action,150)
            transitions.append(dict(decision_index=index,tokens=pack_tokens(tensors,metadata),
                wave=obs['wave'],planner_context=context,slow_update=output['slow_update'],
                slow_stage_start=output['slow_stage_start'],legal=legal,action=action,
                previous_action=previous,previous_wait_result=result,wait_result=wait,
                elapsed_since_previous_observation=delta,action_duration_ticks=150,events={},
                log_prob=lp.item(),value=value.item(),critic_extra=[0.]*16,
                reward=1. if index==length-1 else 0.))
            hidden,previous,result,delta=output['hidden'],action,wait,150
        episodes.append(dict(seed=job,task_seed=100+job,task_id='synthetic_dual',result=1,
                             transitions=transitions,terminated=True,truncated=False,won=True))
    add_advantages(episodes,.95,.99)
    return episodes


class DualMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(0)

    def test_versions_and_invalid_slow_settings(self):
        self.assertEqual(model_architecture_version(config()),14)
        self.assertEqual(model_architecture_version({**SMALL,'wait_mode':'events','wait_mask':'progress_v1'}),13)
        for changes in ({'goal_width':17},{'cell':'other'},{'max_decisions':True},{'max_ticks':0}):
            cfg=config();cfg['slow_memory'].update(changes)
            with self.assertRaises(ValueError):GameplayModelV1(cfg)

    def test_gru_and_lstm_ragged_forward_match_actual_step_history(self):
        for cell in ('gru','lstm'):
            model=GameplayModelV1(config(cell)).eval();episodes=fixture(model)
            sequences=[e['transitions'] for e in episodes]
            with torch.no_grad():
                outputs,hidden=model.forward_sequences(sequences,[None]*2)
                lp,_=replay_log_probs(model,outputs,[t for seq in sequences for t in seq])
            expected=torch.tensor([t['log_prob'] for seq in sequences for t in seq])
            self.assertLess((lp-expected).abs().max().item(),1e-6)
            self.assertEqual(hidden.shape,(model.slow_memory.packed_layers,2,16))
            self.assertEqual([o['slow_update'] for o in outputs],
                             [t['slow_update'] for seq in sequences for t in seq])

    def test_goal_and_slow_hidden_are_held_between_real_planning_events(self):
        model=GameplayModelV1(config()).eval();obs=source();obs.update(tick=0,wave=1)
        first=model.step(obs)
        obs['tick']=150
        action=dict(type='wait',ticks=150,until='timeout')
        second=model.step(obs,first['hidden'],previous_action=action,delta_ticks=150,
                          previous_wait_result=wait_result(action,150))
        self.assertTrue(first['slow_update']);self.assertFalse(second['slow_update'])
        torch.testing.assert_close(first['hidden'][1:-1],second['hidden'][1:-1],rtol=0,atol=0,
            msg='Only public control counters may change while slow state and goal are held')

    def test_wave_urgent_elapsed_and_decision_limits_replan(self):
        for kind in ('wave','urgent','ticks','decisions'):
            cfg=config();cfg['slow_memory']['max_decisions']=2
            model=GameplayModelV1(cfg).eval();obs=source();obs.update(tick=0,wave=1)
            first=model.step(obs)
            obs['tick']=1
            if kind=='wave':obs['wave']=2
            if kind=='ticks':obs['tick']=3000
            kwargs={'events':{'plants_eaten':1}} if kind=='urgent' else {}
            result=model.step(obs,first['hidden'],**kwargs)
            if kind=='decisions':result=model.step(obs,result['hidden'])
            self.assertTrue(result['slow_update'],kind)

    def test_actor_loss_reaches_producing_event_and_detached_goal_does_not(self):
        model=GameplayModelV1(config()).eval()
        sequence=fixture(model,lengths=(6,),wave_interval=99)[0]['transitions']
        self.assertEqual(gradient_start(sequence,3),0)
        outputs,_=model.forward_sequences([sequence[:4]],[None])
        lp,_=replay_log_probs(model,[outputs[-1]],[sequence[3]])
        (-lp.mean()).backward()
        norm=sum(p.grad.abs().sum().item() for p in model.slow_memory.parameters() if p.grad is not None)
        self.assertGreater(norm,0.)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():_,hidden=model.forward_sequences([sequence[:3]],[None])
        outputs,_=model.forward_sequences([sequence[3:4]],[hidden[:,0]])
        lp,_=replay_log_probs(model,outputs,[sequence[3]]);(-lp.mean()).backward()
        self.assertTrue(all(p.grad is None or p.grad.count_nonzero().item()==0
                            for p in model.slow_memory.parameters()))

    def test_held_goal_changes_fast_policy_without_changing_schedule(self):
        model=GameplayModelV1(config()).eval()
        sequence=fixture(model,lengths=(6,),wave_interval=99)[0]['transitions']
        with torch.no_grad():
            _,hidden=model.forward_sequences([sequence[:3]],[None])
            normal,_=model.forward_sequences([sequence[3:4]],[hidden[:,0]])
            altered=hidden[:,0].clone();altered[-2,:8]=0.
            fixed,_=model.forward_sequences([sequence[3:4]],[altered])
        self.assertGreater((normal[0]['type_logits']-fixed[0]['type_logits']).abs().max().item(),1e-7)

    def test_real_ppo_keeps_origin_in_gradient_but_counts_only_core_losses(self):
        model=GameplayModelV1(config()).eval();episodes=fixture(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        result=train_update(model,episodes,optimizer,torch.device('cpu'),1,3,.2,.5,.01,2,'dense')
        self.assertLess(result['first_minibatch_log_prob_max_error'],1e-6)
        self.assertGreater(result['slow_gradient_coverage']['max_leading_decisions'],0)
        self.assertGreater(result['slow_gradient_coverage']['max_waves'],1)
        coverage=result['slow_gradient_coverage']
        self.assertEqual(sum(row['core_decisions'] for row in coverage['examples']),15)

    def test_saved_schedule_corruption_is_rejected(self):
        model=GameplayModelV1(config()).eval();episode=fixture(model,lengths=(3,))[0]
        sequence=copy.deepcopy(episode['transitions']);sequence[1]['slow_update']=True
        with self.assertRaisesRegex(ValueError,'schedule'):
            model.forward_sequences([sequence],[None])


if __name__=='__main__':unittest.main()
