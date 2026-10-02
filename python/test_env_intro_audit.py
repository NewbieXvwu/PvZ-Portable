"""Strict intro audit comparisons and unchanged failed seed inventory."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import research_env_intro_audit as audit

class IntroAuditContract(unittest.TestCase):
    def test_rejects_changed_rng_and_entity_identity(self):
        state={'hidden':{'rng':'a'},'defenses':[{'id':4,'state':1,'x':-21}],'zombies':[{'id':7,'on_board':True}]}
        for path,value in [('rng','b'),('defense_id',5),('x',-160),('zombie_id',8)]:
            changed=copy.deepcopy(state)
            if path=='rng':changed['hidden']['rng']=value
            elif path=='zombie_id':changed['zombies'][0]['id']=value
            else:changed['defenses'][0]['id' if path=='defense_id' else 'x']=value
            traces=[]
            with self.assertRaisesRegex(AssertionError,'intro fidelity differs'):
                audit.require_equal(state,changed,path,lambda *row:traces.append(row))
            self.assertEqual(traces[0][1]['actual'],changed)
    def test_keeps_task_and_seed_order_and_both_controls(self):
        tasks=[{'task_id':str(i),'seeds':[i*100,i*100+1,i*100+2]} for i in range(30)]
        jobs=audit.jobs_for(tasks)
        self.assertEqual(len(jobs),120)
        self.assertEqual([j['job_id'] for j in jobs],list(range(120)))
        self.assertEqual([j['seed'] for j in jobs[:4]],[0,0,1,1])
        self.assertEqual([j['regime'] for j in jobs[:4]],['wait_only','one_first_legal_plant']*2)
        self.assertIs(jobs[0]['task'],tasks[0])
    def test_truncation_or_event_difference_is_not_ignored(self):
        for a,b in [(dict(terminal=False),dict(terminal=True)),(dict(mower_triggered=0),dict(mower_triggered=1))]:
            with self.assertRaises(AssertionError):audit.require_equal(a,b,'event',lambda *row:None)
if __name__=='__main__':unittest.main()
