"""Task43 data alignment and bounded evaluation gates; no games or GPU needed."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_value_widen as widen
from train_value_widen import Branches, initialize, held
from pvz_value_net import ValueNet, ENCODER_V1, ENCODER_V2, SCALARS, CONTEXT


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


class ValueWidenTrainingTest(unittest.TestCase):
    def test_v4_warm_start_preserves_every_weight(self):
        model=ValueNet(width=4,context=True,encoder_version=2)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            torch.save(dict(state_dict=model.state_dict(),width=4,context=True,encoder=ENCODER_V2),path)
            actual=widen.initialize(path)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)

    def test_shuffled_cross_shard_rows_keep_scene_labels_and_seed_split(self):
        fit=next(s for s in range(100) if not widen.held(s))
        held=next(s for s in range(100) if widen.held(s))
        excluded=1000
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for i,seed in enumerate((fit,held,excluded)):
                ids=torch.arange(3)+i*3
                scalars=torch.zeros(3,154);scalars[:,114]=1
                path=Path(folder)/f'{i}.pt';paths.append(path)
                torch.save(dict(boards=ids[:,None,None,None].expand(3,92,6,10).half(),
                    scalars_v2=scalars,won=(ids%2).float(),seed=torch.full((3,),seed),
                    position=ids,candidate=ids,packet=ids,kind=torch.zeros(3,dtype=torch.long),
                    config={'task_id':'pool_l21_train_p9k'}),path)
            data=widen.Decisions(paths,{excluded},'pool_')
            np.testing.assert_array_equal(data.fit,[0,1,2])
            np.testing.assert_array_equal(data.val,[3,4,5])
            indices=np.array([4,0,5,2],dtype=np.int64)
            board,scalars,won=data.batch(indices)
            torch.testing.assert_close(board[:,0,0,0],torch.tensor(indices,dtype=torch.float32))
            torch.testing.assert_close(scalars[:,114],torch.ones(4))
            torch.testing.assert_close(won,torch.tensor(indices%2,dtype=torch.float32))
            self.assertEqual(data.rows(indices)['position'].tolist(),indices.tolist())
            self.assertEqual(data.seeds,{fit,held})

    def test_validation_scores_all_rows_in_order_with_bounded_batches(self):
        class Data:
            def batch(self,indices):
                return torch.tensor(indices,dtype=torch.float32)[:,None,None,None],torch.zeros(len(indices),1),None
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.sizes=[]
            def forward(self,board,scalars):self.sizes.append(len(board));return board.flatten()
        model=Model();indices=np.array([7,0,5,3,8])
        actual=widen.score_rows(model,Data(),indices,2,'cpu')
        torch.testing.assert_close(actual,torch.sigmoid(torch.tensor(indices,dtype=torch.float32)))
        self.assertEqual(model.sizes,[2,2,1])

    def test_training_memory_gate_rejects_below_ten_gib(self):
        with patch.object(Path,'read_text',return_value=f'MemAvailable: {10*1024**2-1} kB\n'):
            with self.assertRaisesRegex(RuntimeError,'10 GiB'):
                widen.require_training_memory()


if __name__=='__main__':unittest.main()
