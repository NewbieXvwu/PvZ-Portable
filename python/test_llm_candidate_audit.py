"""Tampering must fail an independent check of recorded task52 calls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from audit_llm_candidates import check_event
from llm_candidates import atomic_json, canonical, digest


class CandidateEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cache=Path(self.tmp.name)
        self.a=dict(type='plant',packet=0,row=0,col=1)
        prompt='frame\n公开实体：'+canonical(dict(zombies=[dict(type=7,row=0)]))
        prompt+='\n引擎合法种植动作：'+canonical([dict(packet=0,row=0,col=1)])
        identity=dict(body=dict(contents=[dict(parts=[dict(text=prompt)])]))
        payload=dict(actions=[self.a])
        response=dict(candidates=[dict(content=dict(parts=[dict(text=canonical(payload))]))],
                      usageMetadata=dict(promptTokenCount=9))
        self.saved=dict(identity=identity,response=response,attempts=[dict(ok=True)])
        self.position=digest(identity)
        atomic_json(self.cache/(self.position+'.json'),self.saved)
        chosen=dict(type='plant',packet=0,row=0,col=0)
        self.trace=[dict(tick=0,wave=1,searched=True,chosen=chosen,policy=chosen)]
        self.event=dict(position=self.position,response_sha256=digest(response),payload=payload,
                        actions=[self.a],counts=dict(returned=1,malformed=0,illegal=0,excess=0,duplicates=0,valid=1),
                        usage=response['usageMetadata'],decision_seed=170000,tick=0,wave=1,chosen=chosen,
                        novel_actions=[self.a],suggestion_selected=False,novel_selected=False,
                        trigger='B_low',evidence=dict(policy_leaf_values=[.1,.11,.12]),
                        cache_hit=True,attempts=[])
        self.settings=dict(adjacent_rows=1,terminal_futures=2)

    def check(self,event=None):
        return check_event(event or self.event,self.trace,self.cache,self.settings,.2)

    def test_original_cached_call_checks(self):
        self.check()

    def test_illegal_record_cannot_override_engine_table(self):
        event=deepcopy(self.event);event['actions']=[dict(self.a,col=9)]
        with self.assertRaisesRegex(AssertionError,'legality'):self.check(event)

    def test_cache_poisoning_is_detected(self):
        saved=deepcopy(self.saved)
        saved['response']['usageMetadata']['promptTokenCount']=999
        atomic_json(self.cache/(self.position+'.json'),saved)
        with self.assertRaisesRegex(AssertionError,'changed cached response'):self.check()

    def test_false_selection_claim_is_detected(self):
        event=deepcopy(self.event);event['suggestion_selected']=True
        with self.assertRaises(AssertionError):self.check(event)

    def test_threshold_or_cache_cost_claim_cannot_be_changed(self):
        event=deepcopy(self.event);event['evidence']['policy_leaf_values']=[.1,.2,.12]
        with self.assertRaises(AssertionError):self.check(event)
        event=deepcopy(self.event);event['attempts']=[dict(ok=True)]
        with self.assertRaisesRegex(AssertionError,'cache hit charged'):self.check(event)

    def test_novel_row_must_be_supported_by_public_prompt(self):
        event=deepcopy(self.event);event['trigger']='B_novel'
        event['evidence']=dict(new_types=[7],reference_row=0)
        self.check(event)
        event['evidence']['new_types']=[8]
        with self.assertRaises(AssertionError):self.check(event)


if __name__=='__main__':unittest.main()
