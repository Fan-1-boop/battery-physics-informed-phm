"""Offline unit tests; no author XJTU files required. Full benchmark needs upstream CSV files."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/rw_selection/benchmark.py'
_spec=importlib.util.spec_from_file_location('rw_selection_v9',SCRIPT)
m=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=m
_spec.loader.exec_module(m)
v8=m.v8


def cells(n=8):
    rr=[]
    for i in range(n):
        cycles=np.arange(1,49)
        features=np.stack([(i+0.1)*np.ones(len(cycles)),
                           0.1*i+cycles*.01,np.ones(len(cycles))*(5+i),
                           0.2*cycles+1+i,cycles*.03+i,cycles*.05+i],axis=1)
        cap=1.9-0.002*cycles-0.02*i
        rr.append(v8.Cell(f'3C_battery-{i+1}','3C',features,cap))
    return rr


class RWSelectionTests(unittest.TestCase):
    def setUp(self):
        self.source=cells()
        self.scaler=v8.fit_scaler(self.source)
        self.model=v8.ridge(self.source,self.scaler,1.0)

    def test_model_quantile_label_free(self):
        pool=cells(8)
        a=m.acquire(pool,self.scaler,self.model,'pred_quartiles',fold=0)
        for c in pool:
            c.capacity[:] = 123456.0
        b=m.acquire(pool,self.scaler,self.model,'pred_quartiles',fold=0)
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),2)

    def test_medoids_exact_1d(self):
        points=np.array([[0.],[1.],[2.],[10.],[11.],[12.]])
        a=m.medoid_pair(points)
        self.assertEqual(len(set(a)),2)
        chosen=points[a,0]
        self.assertTrue(chosen.min()<4 and chosen.max()>9)

    def test_medoids_permutation(self):
        z=np.array([[0.,0.],[1.,1.],[2.,2.],[9.,9.],[10.,10.]])
        ids=m.medoid_pair(z)
        self.assertEqual(len(ids),2)
        self.assertTrue(all(0<=i<len(z) for i in ids))

    def test_capped_feature_selection(self):
        pool=cells(8)
        for p in m.POLICIES[:6]:
            ids=m.acquire(pool,self.scaler,self.model,p,fold=1)
            self.assertEqual(len(set(ids)),2)
            self.assertTrue(all(0<=i<len(pool) for i in ids))

    def test_leave_one_battery_ensemble(self):
        ens=m.source_ensemble(self.source,self.scaler)
        self.assertEqual(ens.shape,(8,7))
        ids=m.acquire(cells(),self.scaler,self.model,'uncertainty_top2',fold=0,ensemble=ens)
        self.assertEqual(len(set(ids)),2)
        self.assertEqual(len(m.acquire(cells(),self.scaler,self.model,
                'uncertainty_weighted_medoids',fold=0,ensemble=ens)),2)

    def test_no_test_label_leakage(self):
        target=cells(8)
        res1=m.run_domain(target,'2C',self.model,self.scaler,policies=('pred_quartiles',),random_repeats=1)
        for c in target:
            c.capacity+=1000  # changes test labels, but not any selection rule
        res2=m.run_domain(target,'2C',self.model,self.scaler,policies=('pred_quartiles',),random_repeats=1)
        seq=lambda r:[tuple(v['id'] for v in z['selected']) for z in r['records']]
        self.assertEqual(seq(res1),seq(res2))
        for rec in res1['records']:
            held=set(res1['fold_indices'][rec['fold']])
            self.assertFalse(held.intersection(z['id'] for z in rec['selected']))

    def test_v8_random_seeds(self):
        n=8
        f=0
        expected=v8.js_shuffle(range(n),20261108+f*101+0*7)[:2]
        got=m.acquire(cells(n),self.scaler,self.model,'random',fold=f,repeat=0)
        self.assertEqual(expected,got)

    def test_empty_or_small(self):
        with self.assertRaises(ValueError):m.acquire([],self.scaler,self.model,'pred_quartiles',fold=0)
        with self.assertRaises(ValueError):m.acquire(cells(1),self.scaler,self.model,'pred_quartiles',fold=0)
        with self.assertRaises(ValueError):m.acquire(cells(3),self.scaler,self.model,'pred_quartiles',fold=0,k=4)

if __name__=='__main__':unittest.main()