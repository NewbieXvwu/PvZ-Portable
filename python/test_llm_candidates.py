"""task52 legality, cache, information boundary and ordinary search integration."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'python')]
from llm_candidates import GeminiCache, Triggers, filter_actions, prompt_for, parse_payload


def observation():
    return dict(tick=30, wave=1, sun=100, plants=[], zombies=[], defenses=[], packets=[],
                grid=[[1]*9 for _ in range(5)], level=7, wave_count=20,
                legal_actions={'plants':[dict(packet=0,row=0,col=0),dict(packet=0,row=1,col=0)]})


class LlmCandidatesTest(unittest.TestCase):
    def test_gateway_fenced_json_is_parsed_without_accepting_freeform_prose(self):
        self.assertEqual(parse_payload('```json\n{"actions":[]}\n```'),dict(actions=[]))
        self.assertIsNone(parse_payload('some prose {"actions":[]}'))
        self.assertIsNone(parse_payload('```json\n{"actions":[]}'))

    def test_engine_legality_strict_types_duplicates_and_limit(self):
        legal = observation()['legal_actions']['plants']
        a = dict(type='plant',packet=0,row=0,col=0)
        for invalid in [dict(a, row=9), dict(a,packet=True), dict(a,row='0')]:
            accepted,c = filter_actions(dict(actions=[a,invalid,a,dict(a,row=1)]),legal)
            self.assertEqual(accepted,[a]);self.assertEqual(c['returned'],4)
            self.assertEqual(c['excess'],1);self.assertEqual(c['duplicates'],1)
            self.assertEqual(c['illegal']+c['malformed'],1)

    def test_malformed_response_does_not_become_an_action(self):
        for bad in [None,[],{},dict(actions='plant')]:
            actions,c = filter_actions(bad,[])
            self.assertEqual(actions,[]);self.assertEqual(c['malformed'],1)

    def test_prompt_renders_scene_but_omits_hidden_audit_and_future(self):
        obs = observation()
        obs.update(privileged_state='HIDDEN_CANARY', fog_loss='SECRET_LOSS', future='FUTURE_CANARY')
        text = prompt_for(obs)
        self.assertIn('第 7 关',text);self.assertIn('引擎合法种植动作',text)
        for hidden in ['HIDDEN_CANARY','SECRET_LOSS','FUTURE_CANARY']:
            self.assertNotIn(hidden,text)

    def test_cache_survives_new_client_and_position_changes_invalidate(self):
        response = dict(candidates=[dict(content=dict(parts=[dict(text=json.dumps(dict(actions=[])))]),
                                         finishReason='STOP')],usageMetadata=dict(promptTokenCount=12))
        transport = Mock(return_value=response)
        with tempfile.TemporaryDirectory() as tmp:
            first = GeminiCache(tmp,transport=transport).propose(observation())
            second = GeminiCache(tmp,transport=transport).propose(observation())
            self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit'])
            self.assertEqual(transport.call_count,1)
            obs=observation();obs['sun']=125
            third=GeminiCache(tmp,transport=transport).propose(obs)
            self.assertFalse(third['cache_hit']);self.assertEqual(transport.call_count,2)
            self.assertNotEqual(first['position'],third['position'])

    def test_illegal_model_response_is_cached_too(self):
        response = dict(candidates=[dict(content=dict(parts=[dict(text='not JSON')]))])
        transport=Mock(return_value=response)
        with tempfile.TemporaryDirectory() as tmp:
            a=GeminiCache(tmp,transport=transport).propose(observation())
            b=GeminiCache(tmp,transport=transport).propose(observation())
            self.assertEqual(a['actions'],[]);self.assertEqual(b['counts']['malformed'],1)
            self.assertEqual(transport.call_count,1)

    def test_cache_lock_identity_is_checked(self):
        response=dict(candidates=[dict(content=dict(parts=[dict(text='{"actions":[]}')]))])
        with tempfile.TemporaryDirectory() as tmp:
            c=GeminiCache(tmp,transport=Mock(return_value=response));a=c.propose(observation())
            path=Path(tmp)/(a['position']+'.json');data=json.loads(path.read_text())
            data['identity']['version']=-1;path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):c.propose(observation())

    def test_trigger_strict_quantile_novel_visibility_and_budget(self):
        t=Triggers(threshold=.2,interval=1200,max_calls=2,adjacent_rows=1)
        self.assertTrue(t.low([.1,.12,.19]));self.assertFalse(t.low([.1,.2,.1]))
        obs=dict(zombies=[dict(type=7,row=3)])
        t.observe(obs);self.assertEqual(t.novel(obs,1),[]);self.assertEqual(t.novel(obs,2),[7])
        t.called(100,[7]);t.observe(obs);self.assertEqual(t.novel(obs,3),[])
        self.assertFalse(t.budget_allows(1299));self.assertTrue(t.budget_allows(1300))
        t.called(1300);self.assertFalse(t.budget_allows(100000))

    def search(self, proposer=None, verify=0):
        import search_play as t
        env=Mock();env.snapshot_fast.return_value=37
        a=dict(type='plant',packet=0,row=0,col=0)
        b=dict(type='plant',packet=0,row=1,col=0)
        args=SimpleNamespace(min_gap=300,per_card=1,student_candidates=0,halving=True,
                             samples=3,verify_samples=verify,horizon=2500)
        def continuations(env,obs,root,jobs,args,stats):
            return [dict(terminal=False,value=.95 if action==b else .7 if action==a else .6)
                    for action,_seed in jobs]
        # Independent verification rejects b, despite its winning preliminary leaves.
        def final_leaf(env,obs,action,*args):
            return dict(terminal=False,value=.5 if action==b else .7)
        stats=dict(searched=0,overrides=0,override_kinds=Counter())
        state={} if proposer is None else dict(llm_proposer=proposer)
        with patch.dict(t._W,state,clear=True),patch.object(t,'_static_scores',return_value=[]),\
             patch.object(t,'_continuations',side_effect=continuations),\
             patch.object(t,'_rollout_policy',side_effect=final_leaf):
            chosen=t._search(env,observation(),a,args,1700,stats)
        env.restore_fast.assert_called_once_with(37);env.release_snapshot.assert_called_once_with(37)
        return chosen,stats

    def test_no_installed_hook_preserves_candidate_source(self):
        chosen,stats=self.search()
        self.assertEqual(chosen['row'],0);self.assertEqual(len(stats['last_targets']),2)

    def test_llm_addition_enters_halving_and_can_win(self):
        b=dict(type='plant',packet=0,row=1,col=0)
        chosen,stats=self.search(lambda *args:[b,b])
        self.assertEqual(chosen,b);self.assertEqual(len(stats['last_targets']),3)
        self.assertEqual(stats['overrides'],1)

    def test_verification_can_reject_preliminary_llm_winner(self):
        b=dict(type='plant',packet=0,row=1,col=0)
        chosen,stats=self.search(lambda *args:[b],verify=9)
        self.assertEqual(chosen['row'],0);self.assertEqual(stats['overrides'],0)
        self.assertFalse(stats['last_verification']['after_override'])


if __name__=='__main__':
    unittest.main()
