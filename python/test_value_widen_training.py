"""Seed isolation, branch migration and warm-start gates for cross-zone training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_value_widen import Branches, initialize, held
from pvz_value_net import SCALARS,CONTEXT,ENCODER_V1,ValueNet


class ValueTrainingTest(unittest.TestCase):
    def test_branch_seed_isolation_and_indexing(self):
        seeds=[s for s in range(100) if held(s)][:2]+[s for s in range(100) if not held(s)][:2]
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for j in range(2):
                p=Path(directory)/f'{j}.pt';paths.append(p)
                torch.save(dict(boards=torch.full((4,92,6,10),float(j)),
                    scalars=torch.arange(4*(SCALARS+CONTEXT)).reshape(4,-1).float(),
                    won=torch.tensor([0.,1.,0.,1.]),seed=torch.tensor(seeds),
                    config=dict(task_id='day_l7_z13_train')),p)
            d=Branches(paths,{seeds[0]})
            self.assertNotIn(seeds[0],d.seeds)
            self.assertTrue(all(held(seeds[i%4]) for i in d.val))
            self.assertTrue(all(not held(seeds[i%4]) for i in d.fit))
            b,s,y=d.batch(np.array([5,2,7]))
            self.assertEqual(b[:,0,0,0].tolist(),[1.,0.,1.])
            self.assertFalse(s[:,SCALARS:SCALARS+6].any())
            old=torch.load(paths[0],weights_only=False)['scalars']
            torch.testing.assert_close(s[:,SCALARS+6:],old[[1,2,3],SCALARS:])
            self.assertEqual(y.tolist(),[1.,0.,1.])

    def test_warm_start_preserves_legacy_predictions(self):
        torch.manual_seed(11);old=ValueNet(width=8,context=True).eval()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(state_dict=old.state_dict(),width=8,context=True,encoder=ENCODER_V1),path)
            new=initialize(path).eval();board=torch.rand(5,92,6,10);s=torch.rand(5,SCALARS+CONTEXT)
            expanded=torch.cat([s[:,:SCALARS],torch.zeros(5,6),s[:,SCALARS:]],1)
            with torch.inference_mode():torch.testing.assert_close(new(board,expanded),old(board,s),rtol=1e-6,atol=1e-6)
if __name__=='__main__':unittest.main()
