import importlib.util
import unittest
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'experiments/rw_multitarget/benchmark.py'
spec=importlib.util.spec_from_file_location('rw_v8',SCRIPT)
m=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m
spec.loader.exec_module(m)

def mockcell(i,group='2C'):
    x=np.array([[4+.01*i+.00001*j, .38+i*.001+.0001*j,220+j, .22, .05,900+j*2] for j in range(24)],float)
    y=1.5+.02*i+.0001*np.arange(24)
    return m.Cell(f'{group}_battery-{i}',group,x,y)

class V8Checks(unittest.TestCase):
    def setUp(self):
        self.src=[mockcell(i,'RW') for i in range(1,9)]
        self.target=[mockcell(i) for i in range(1,9)]
    def test_fivefold_full_unique_holdouts(self):
        fold=m.make_folds(8,'2C')
        self.assertEqual(sorted(x for a in fold for x in a),list(range(8)))
        self.assertEqual(len(fold),5)
    def test_scaler_from_source_only(self):
        stats=m.fit_scaler(self.src)
        altered=[mockcell(i) for i in range(1,9)]
        for c in altered:c.features+=1e5
        self.assertTrue(np.allclose(stats[0],m.fit_scaler(self.src)[0]))
        self.assertFalse(np.allclose(stats[0],m.fit_scaler(altered)[0]))
    def test_cell_level_label_exclusion(self):
        scaler=m.fit_scaler(self.src);model=m.ridge(self.src,scaler,1)
        r=m.run_domain(self.target,'2C',model,scaler,k=2,label_frac=.25,random_repeats=3)
        self.assertEqual(len(r['records']),5*(1+1+3))
        for x in r['records']:
            labels={l['id'] for l in x['labeled']}
            self.assertFalse(labels & set(x['heldout']))
            self.assertTrue(all(l['n_labeled_cycles']==6 for l in x['labeled']))
    def test_no_target_label_changes_acquisition(self):
        scaler=m.fit_scaler(self.src)
        v=m.representative(self.target,scaler,2)
        for c in self.target:c.capacity[:]=1e6
        self.assertEqual(v,m.representative(self.target,scaler,2))
    def test_source_mae_repeat_count(self):
        scaler=m.fit_scaler(self.src);model=m.ridge(self.src,scaler,1)
        r=m.run_domain(self.target,'2C',model,scaler,k=2,label_frac=.25)
        self.assertEqual(r['summary']['source']['n_cell_evaluations'],8)
        self.assertEqual(r['summary']['random']['n_cell_evaluations'],40)
        self.assertEqual(r['summary']['representative']['n_cell_evaluations'],8)
    def test_rng_known_indices(self):
        self.assertEqual(m.js_shuffle(range(8),20261010),[2,7,3,6,5,0,4,1])
    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            m.run_domain(self.target,'2C',np.zeros(7),m.fit_scaler(self.src),label_frac=0)

if __name__=='__main__':unittest.main()