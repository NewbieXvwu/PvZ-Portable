"""Identity-checked row interventions must never silently attach to different examples."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
from pvz_training_rows import RowWeights,update_identity
from dagger_data_design import kept_rows


class RowManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=np.array([1,2,4],dtype=np.int64);self.val=np.array([False,True,False])

    def tearDown(self):self.temp.cleanup()

    def manifest(self,values):
        digest=hashlib.sha256();update_identity(digest,'seed_7/a/frames.npz',self.rows)
        values=np.array(values,np.float32);np.save(self.root/'weights.npy',values)
        spec={'name':'test','rows':3,'multipliers':'weights.npy',
              'multipliers_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
              'row_identity_sha256':digest.hexdigest(),
              'split_sha256':hashlib.sha256(np.packbits(self.val).tobytes()).hexdigest()}
        p=self.root/'plan.json';p.write_text(json.dumps(spec));return p

    def test_weight_identity_split_and_validation_anchor_are_checked(self):
        p=self.manifest([2,1,0]);w=RowWeights(p,self.root)
        w.observe(self.root/'seed_7/a/frames.npz',self.rows);w.finish(3,self.val)
        import torch
        base=torch.tensor([3.,4.,5.]);w.apply(base)
        self.assertEqual(base.tolist(),[6.,4.,0.])
        wrong=RowWeights(p,self.root);wrong.observe(self.root/'seed_7/a/frames.npz',self.rows[::-1])
        with self.assertRaisesRegex(ValueError,'identities'):wrong.finish(3,self.val)
        changed=RowWeights(p,self.root);changed.observe(self.root/'seed_7/a/frames.npz',self.rows)
        with self.assertRaisesRegex(ValueError,'validation split'):changed.finish(3,~self.val)
        bad=RowWeights(self.manifest([1,2,1]),self.root);bad.observe(self.root/'seed_7/a/frames.npz',self.rows)
        with self.assertRaisesRegex(ValueError,'unchanged'):bad.finish(3,self.val)

    def test_invalid_weights_and_checksum_fail_before_loading_boards(self):
        for values in ([0,0,0],[-1,1,1],[np.nan,1,1]):
            with self.assertRaisesRegex(ValueError,'finite'):RowWeights(self.manifest(values),self.root)
        p=self.manifest([1,1,1]);np.save(self.root/'weights.npy',np.array([1,1,2],np.float32))
        with self.assertRaisesRegex(ValueError,'checksum'):RowWeights(p,self.root)

    def test_wait_subsampling_consumes_the_original_global_rng_stream(self):
        hints=np.array([False,True,False,False,True,False]);rows=np.array([1,2,3,5])
        a=random.Random(0);b=random.Random(0)
        expected=np.flatnonzero(np.array([bool(h) or b.random()<.1 for h in hints])[rows])
        self.assertEqual(kept_rows(hints,rows,a).tolist(),expected.tolist())
        self.assertEqual(a.random(),b.random())


class ImmutableCacheTest(unittest.TestCase):
    def test_mutable_weights_are_fresh_and_observation_replay_matches_each_run(self):
        import torch
        from train_rollout_policy_dilution import ImmutableDataCache
        calls=[]
        def load(data,labels,**kwargs):
            calls.append((data,labels))
            kwargs['row_observer'](Path('data/seed_1/x/frames.npz'),np.array([1,3]))
            return (object(),object(),object(),object(),object(),torch.tensor([2.,3.]),[1,1],object(),None)
        cached=ImmutableDataCache(load);observed=[]
        first=cached('data',{'b','a'},row_observer=lambda p,r:observed.append((p,r.tolist())))
        first[5].zero_()
        second=cached('data',{'a','b'},row_observer=lambda p,r:observed.append((p,r.tolist())))
        self.assertEqual(len(calls),1)
        self.assertIs(first[0],second[0])
        self.assertEqual(second[5].tolist(),[2.,3.])
        self.assertEqual(observed[0],observed[1])
        cached('data',{'c'},row_observer=None)
        self.assertEqual(len(calls),2)


class ControlledSelectionTest(unittest.TestCase):
    def test_normalization_and_matching_keep_effective_mass_and_eligibility(self):
        from dagger_data_interventions import normalize_selection,matched_random
        base=np.array([10,1,4],np.float32)
        weights,keep,factor=normalize_selection(base,np.array([False,True,True]))
        self.assertEqual(weights.tolist(),[0,3,3])
        self.assertEqual(float((weights*base).sum()),float(base.sum()))
        compared=np.array([False,True,True,True,False,True])
        soft=np.array([False,False,True,True,False,False])
        chosen=np.array([True,False,True,False,False,True])
        random=matched_random(chosen,compared,soft,np.random.default_rng(1))
        for c in (False,True):
            for s in (False,True):
                mask=(compared==c)&(soft==s)
                self.assertEqual(random[mask].sum(),chosen[mask].sum())
        w,k,f=normalize_selection(base,np.zeros(3,bool))
        self.assertTrue(k.all());self.assertEqual(w.tolist(),[1,1,1])

    def test_old_and_validation_rows_unchanged_and_all_strata_mass_matched(self):
        from dagger_data_interventions import build
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cohort=root/'cohort';cohort.mkdir();out=root/'weights';n=80
            episode=np.repeat(np.arange(4),20);packet=np.tile(np.array([-1,0],np.int8),40)
            soft=(packet>=0)&(np.arange(n)%4==1);val=np.arange(n)%13==0
            weight=np.linspace(1,5,n,dtype=np.float32)
            arrays={'episode':episode,'packet':packet,'cell_soft':soft,'is_val':val,
                    'weight':weight,'spread':np.linspace(0,.2,n,dtype=np.float32),
                    'compared':np.arange(n)%3!=0,'override':np.arange(n)%17==0}
            for k,v in arrays.items():np.save(cohort/(k+'.npy'),v)
            info={'rows':n,'row_identity_sha256':'fixture','split_sha256':'fixture',
                  'episodes':[{'label':label,'multiplier':z} for label,z in [('old',1.3),('old',1.7),('new',1.7),('new',1.7)]]}
            (cohort/'cohort.json').write_text(json.dumps(info))
            report=build(cohort,['old'],out)
            I=np.load(out/'I.npy');R=np.load(out/'R.npy')
            self.assertTrue(np.equal(I[episode<2],1).all());self.assertTrue(np.equal(R[episode<2],1).all())
            for name in ('M','I','R'):
                values=np.load(out/(name+'.npy'));self.assertTrue(np.equal(values[val],1).all())
                for p in (-1,0):
                    for s in (False,True):
                        ix=(episode>=2)&~val&(packet==p)&(soft==s)
                        if name in ('I','R'):
                            self.assertAlmostEqual(float((weight[ix]*values[ix]).sum()),float(weight[ix].sum()),places=4)
            self.assertEqual(int((I[~val]>0).sum()),int((R[~val]>0).sum()))
            for p in (-1,0):
                for s in (False,True):
                    for c in (False,True):
                        ix=(episode>=2)&~val&(packet==p)&(soft==s)&(arrays['compared']==c)
                        self.assertTrue(np.allclose(np.sort(weight[ix]*I[ix]),np.sort(weight[ix]*R[ix]),rtol=2e-6))
            old=root/'old';old.mkdir()
            for k in ('episode','packet','cell_soft','is_val','weight'):np.save(old/(k+'.npy'),arrays[k][:40])
            (old/'cohort.json').write_text(json.dumps({'episodes':info['episodes'][:2]}))
            restored=build(cohort,['old'],root/'restored',old)
            T=np.load(root/'restored/T.npy')
            self.assertTrue(np.equal(T[val],1).all())
            self.assertIn('T',restored['specs'])
            for item in restored['D30_mixture_restoration']:
                prior=np.asarray(item['D30_mass']);actual=np.asarray(item['restored_mass'])
                if prior.sum():self.assertTrue(np.allclose(prior/prior.sum(),actual/actual.sum(),rtol=2e-6))


if __name__=='__main__':unittest.main()
