import importlib.util
import sys
from pathlib import Path
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1]/'experiments/rw_multitarget/fixed_budget_v10.py'
spec = importlib.util.spec_from_file_location('phm_fixed_budget_v10',PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def synthetic(group='2C', count=8):
    cells=[]
    for i in range(count):
        t=np.arange(1,51)
        x=np.stack([np.ones(len(t))*(4+i*.001), .1*t+i*.01, t*.04+i,
                    np.ones(len(t))*(.2+i*.01), .01*t+i*.002, t*.6+i],axis=1)
        cap=1.9-i*.01-.001*t
        cells.append(m.v8.Cell(f'{group}_battery-{i+1}',group,x,cap))
    return cells


class V10Tests(unittest.TestCase):
    def setUp(self):
        self.cells = synthetic()
        self.scaler = m.v8.fit_scaler(self.cells)
        self.source = m.v8.ridge(self.cells,self.scaler,1)

    def test_selection_ignores_capacity_labels(self):
        before=m.acquire(self.cells,self.scaler,self.source,'pred_quartiles',fold=2,repeat=0)
        for cell in self.cells: cell.capacity[:]=999.0
        after=m.acquire(self.cells,self.scaler,self.source,'pred_quartiles',fold=2,repeat=0)
        self.assertEqual(before,after)

    def test_budget_exact(self):
        chosen=m.label_prefix(self.cells,[1,3],10)
        self.assertEqual([c.n_cycles for c in chosen],[10,10])
        self.assertEqual(sum(c.n_cycles for c in chosen),20)

    def test_random_v8_unchanged(self):
        self.assertEqual(m.acquire(self.cells,self.scaler,self.source,'random',1,3),
                         m.v8.js_shuffle(range(len(self.cells)),20261108+101+3*7)[:2])

    def test_disjoint_folds(self):
        out=m.evaluate_domain(self.cells,'2C',self.scaler,self.source,budgets=(10,20),random_repeats=2)
        for r in out['records']:
            held=set(r['heldout']);selected={x['id'] for x in r['selected']}
            self.assertFalse(held&selected)
            if r['policy']!='source':
                self.assertEqual(sum(x['n_labeled_cycles'] for x in r['selected']),2*r['budget'])
        self.assertEqual(set().union(*map(set,out['fold_indices'])),set(c.name for c in self.cells))

    def test_fivefold_repeat_accounting(self):
        out=m.evaluate_domain(self.cells,'2C',self.scaler,self.source,budgets=(10,),random_repeats=5)
        self.assertEqual(sum(x['policy']=='random' for x in out['records']),25)
        self.assertEqual(sum(x['policy']=='pred_quartiles' for x in out['records']),5)

if __name__=='__main__': unittest.main()