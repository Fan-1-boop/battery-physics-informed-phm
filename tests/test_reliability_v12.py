"""Offline regression/safety tests of v12 LOCO residual shrinkage."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/"experiments/rw_multitarget/reliability_v12.py"
spec=importlib.util.spec_from_file_location("phm_v12_tests",SCRIPT)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def make_pair(residual1=.1,residual2=.1,n=20):
    cells=[]
    for j,residual in enumerate([residual1,residual2]):
        f=np.array([[4.1+.001*j,.4,210.0,.25,.06,920.0] for _ in range(n)],float)
        y=np.full(n,1.8+residual)
        cells.append(m.v8.Cell(f"3C_battery-{j+1}","3C",f,y))
    return cells


class LOCOUnitTests(unittest.TestCase):
    def setUp(self):
        self.cells=make_pair()
        self.scaler=m.v8.fit_scaler(self.cells)
        self.source=np.array([1.8,0,0,0,0,0,0.],float)

    def test_constant_domain_shift_prefers_full_correction(self):
        lam,details=m.loco_strength(self.cells,self.scaler,self.source)
        self.assertAlmostEqual(lam,1.)
        self.assertEqual(len(details["loco_cells"]),2)

    def test_opposite_direction_shift_disables_correction(self):
        different=make_pair(.15,-.15)
        lam,_=m.loco_strength(different,self.scaler,self.source)
        self.assertAlmostEqual(lam,0.)

    def test_identical_source_prediction_gives_zero(self):
        same=make_pair(0.,0.)
        lam,_=m.loco_strength(same,self.scaler,self.source)
        self.assertAlmostEqual(lam,0.)

    def test_outside_test_capacity_not_used_for_lambda(self):
        chosen=make_pair(.06,.1)
        lam1,_=m.loco_strength(chosen,self.scaler,self.source)
        test=m.v8.Cell("3C_battery-9","3C",
                       np.copy(chosen[0].features),np.full(20,5.0))
        _=m.calc_cell_scores([test],self.scaler,self.source,chosen,lam1)
        test.capacity[:]=10000.0
        lam2,_=m.loco_strength(chosen,self.scaler,self.source)
        self.assertEqual(lam1,lam2)

    def test_reject_duplicate_labeled_cell(self):
        with self.assertRaises(ValueError):
            m.loco_strength([self.cells[0],self.cells[0]],self.scaler,self.source)

    def test_source_model_not_mutated(self):
        before=self.source.copy()
        lam,_=m.loco_strength(self.cells,self.scaler,self.source)
        m.calc_cell_scores(self.cells,self.scaler,self.source,self.cells,lam)
        np.testing.assert_array_equal(before,self.source)

    def test_score_bounds_and_source_baseline_at_zero(self):
        bad=make_pair(.15,-.15)
        lam,_=m.loco_strength(bad,self.scaler,self.source)
        self.assertGreaterEqual(lam,0.)
        self.assertLessEqual(lam,1.)
        scores=m.calc_cell_scores(bad,self.scaler,self.source,bad,lam)
        direct=m.v8.score(bad,self.scaler,self.source)
        for a,b in zip(scores["shrink"],direct):
            self.assertAlmostEqual(a["mae_mAh"],b["mae_mAh"])


if __name__=="__main__":
    unittest.main()
