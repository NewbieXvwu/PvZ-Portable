import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pvz_candidate_coverage import CoverageNet, action_key, features, load, propose
from train_candidate_coverage import marginal


class CoverageTest(unittest.TestCase):
    def test_teacher_off_skips_proposer_and_on_preserves_old_prefix(self):
        from collections import Counter
        from types import SimpleNamespace
        from unittest.mock import Mock
        import search_play as teacher
        actions=[dict(type='plant',packet=0,row=0,col=c) for c in range(4)]
        args=SimpleNamespace(min_gap=0,shovel_candidates=0,per_card=1,student_candidates=2,halving=True)
        outputs=[]
        for enabled in (False,True):
            env=Mock();env.snapshot.return_value=7
            stats={'searched':0,'overrides':0,'override_kinds':Counter()}
            def capture(*pos):
                outputs.append(list(pos[3]))
                return actions[0]
            ranked=[(1.-i*.1,a) for i,a in enumerate(actions)]
            worker={'fast':object(),'student_frame':{},'context':np.zeros(1)}
            if enabled:worker.update(coverage=object(),coverage_budget=2)
            with patch.dict(teacher._W,worker,clear=True), \
                 patch.object(teacher,'_static_scores',side_effect=lambda env,root,a,income:[(1.,x) for x in a]), \
                 patch.object(teacher,'_halving_choice',side_effect=capture), \
                 patch('pvz_rollout_policy.ranked_plants',return_value=ranked) as rank, \
                 patch('pvz_candidate_coverage.features',return_value=(np.zeros(1),np.zeros((5,2)))), \
                 patch('pvz_candidate_coverage.propose',return_value=[actions[3]]) as proposer, \
                 patch('render_episode._frame',return_value={}):
                teacher._search(env,{'tick':0,'legal_actions':{'plants':actions}},actions[0],args,1,stats)
                self.assertEqual(proposer.call_count,int(enabled))
                self.assertEqual(rank.call_count,1)
                env.restore.assert_called_with(7)
                env.release_snapshot.assert_called_once_with(7)
        self.assertEqual(outputs[0],[actions[0],teacher.NOOP,actions[1],actions[2]])
        self.assertEqual(outputs[1],outputs[0]+[actions[3]])

    def test_context_target_rewards_complementary_futures(self):
        scores=np.array([[.5,.5],[.9,.5],[.9,.5],[.5,.9]])
        np.testing.assert_allclose(marginal(scores,[0]),[0,.2,.2,.2])
        np.testing.assert_allclose(marginal(scores,[0,1]),[0,0,0,.2])

    def test_features_use_measured_scores_cost_and_generic_coordinates(self):
        actions=[dict(type='plant',packet=0,row=1,col=2),
                 dict(type='plant',packet=0,row=2,col=3),dict(type='wait')]
        with patch('pvz_value_net.featurize',return_value=(torch.zeros(2,6,10),torch.zeros(3))):
            state,x=features({'packets':[[0,1,True,0,1,125]]},[1,2],actions,
                             [(.7,actions[0]),(.6,actions[1]),(.5,actions[2])],
                             [(.8,{k:v for k,v in actions[1].items() if k!='type'}),(.1,actions[0])])
        self.assertEqual(x.shape,(3,79))
        self.assertEqual(len(state),125)
        self.assertEqual(x[0,12],1)
        np.testing.assert_allclose(x[0,-8:],[125,.1,.7,1,0,.7,0,0],atol=1e-6)
        np.testing.assert_allclose(x[2,-8:],[0,.1,.5,0,0,0,0,1],atol=1e-6)

    def test_greedy_reconditions_deduplicates_and_obeys_budget(self):
        class Conditional:
            def __init__(self):self.sets=[]
            def __call__(self,state,x,selected):
                self.sets.append(list(selected))
                return torch.tensor([0.,.8,.7,.9] if len(selected)==1 else [0.,.1,.9,.8])
        actions=[dict(type='wait')]+[dict(type='plant',packet=0,row=0,col=c) for c in range(3)]
        model=Conditional()
        added=propose(model,np.zeros(1,dtype='float32'),np.zeros((4,2),dtype='float32'),actions,[actions[0]],2)
        self.assertEqual(added,[actions[3],actions[2]])
        self.assertEqual(model.sets,[[0],[0,3]])
        self.assertEqual(propose(model,np.zeros(1),np.zeros((4,2)),actions,actions,8),[])
        with self.assertRaises(ValueError):propose(model,None,None,actions,[],-1)

    def test_checkpoint_roundtrip_is_deterministic_and_schema_checked(self):
        torch.manual_seed(1)
        model=CoverageNet(2,3).eval()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.pt'
            saved={'schema_version':1,'state_dim':2,'action_dim':3,'state_dict':model.state_dict()}
            torch.save(saved,p)
            restored=load(p)
            with torch.inference_mode():
                x=torch.arange(9,dtype=torch.float32).reshape(3,3)
                self.assertTrue(torch.equal(model(torch.ones(2),x,[0]),restored(torch.ones(2),x,[0])))
            torch.save(saved|{'schema_version':9},p)
            with self.assertRaises(ValueError):load(p)


if __name__=='__main__':unittest.main()
