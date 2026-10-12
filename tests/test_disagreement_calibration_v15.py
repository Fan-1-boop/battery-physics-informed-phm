"""v15 synthetic reliability tests: no target-test capacity leakage."""
from __future__ import annotations
import importlib.util,sys,unittest
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parents[1]/"experiments/rw_multitarget/disagreement_calibration_v15.py"
sp=importlib.util.spec_from_file_location("phm_v15_test",p)
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)

class V15Tests(unittest.TestCase):
    def setUp(self):
        self.scaler=(np.zeros(6),np.ones(6))
        self.source=np.array([1.8,.05,.05,.05,.05,.05,.05])
        self.cells=[
            self.cell("3C_battery-1",0.,.1,10),
            self.cell("3C_battery-2",.1,.1,10)]
    def cell(self,name,x,residual,n=10):
        features=np.full((n,6),x)
        capacity=m.v8.design(features,self.scaler)@self.source+residual
        return m.v8.Cell(name,"3C",features,capacity)

    def test_agreement_no_attenuation(self):
        full=np.array([.12,.08])
        lam,d=m.disagreement_shrinkage(full,full,full)
        np.testing.assert_allclose(lam,[1,1],atol=1e-12)
        np.testing.assert_allclose(d,0,atol=1e-12)

    def test_disagreement_continuous(self):
        full=np.array([.10,.10,.10])
        one=np.array([.10,.20,.30])
        two=np.array([.10,0.,-.10])
        lam,_=m.disagreement_shrinkage(full,one,two)
        np.testing.assert_allclose(lam,[1.,.5,.2],atol=1e-12)

    def test_zero_joint_correction_stays_zero(self):
        lam,_=m.disagreement_shrinkage(np.zeros(3),
                np.array([.1,.2,0]),np.array([-.1,-.2,0]))
        np.testing.assert_allclose(lam,0)

    def test_bounded_weight_even_for_negative_corrections(self):
        rng=np.random.default_rng(2)
        for _ in range(100):
            a=rng.normal(size=10);b=rng.normal(size=10);c=rng.normal(size=10)
            lam,_=m.disagreement_shrinkage(a,b,c)
            self.assertTrue(np.all((lam>=0)&(lam<=1)))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            m.disagreement_shrinkage(np.zeros(4),np.zeros(2),np.zeros(4))

    def test_single_model_uses_only_one_battery(self):
        before=self.source.copy()
        z,alpha=m.single_cell_kernel(self.cells[0],self.scaler,self.source)
        self.assertEqual(z.shape,(10,6))
        self.assertEqual(alpha.shape,(10,))
        np.testing.assert_array_equal(before,self.source)

    def test_zero_residual_is_source_preserving(self):
        cells=[self.cell("3C_battery-1",0,0),self.cell("3C_battery-2",.1,0)]
        full=m.v14.train_local(cells,self.scaler,self.source,"RBF-X")
        a=m.single_cell_kernel(cells[0],self.scaler,self.source)
        b=m.single_cell_kernel(cells[1],self.scaler,self.source)
        old,new,diag=m.score_v15([self.cell("3C_battery-9",.05,0)],
                                   self.scaler,self.source,full,a,b)
        self.assertLess(new[0]["mae_mAh"],1e-7)
        self.assertLess(old[0]["mae_mAh"],1e-7)
        self.assertAlmostEqual(diag[0]["mean_lambda"],0.)

    def test_test_truth_not_needed_in_calibration(self):
        full=m.v14.train_local(self.cells,self.scaler,self.source,"RBF-X")
        a=m.single_cell_kernel(self.cells[0],self.scaler,self.source)
        b=m.single_cell_kernel(self.cells[1],self.scaler,self.source)
        held=self.cell("3C_battery-3",.04,.05,20)
        original=m.score_v15([held],self.scaler,self.source,full,a,b)[2][0]
        held.capacity[:]=1000.
        changed=m.score_v15([held],self.scaler,self.source,full,a,b)[2][0]
        self.assertEqual(original,changed)

    def test_historical_fold_mutation_detected(self):
        r={"fold":0,"repeat":0,"policy":"random","budget":30,
           "selected":["a","b"],"n_labeled_cycles":[30,30],
           "heldout":["c"],
           "ridge":[{"id":"c","mae_mAh":1}],
           "offset":[{"id":"c","mae_mAh":1}],
           "RBF-X":[{"id":"c","mae_mAh":1}]}
        a={**r,"heldout":["d"]}
        with self.assertRaises(AssertionError):
            m.validate_v14_reference(r,a)

    def test_equal_cell_longer_history_not_overweighted(self):
        a=self.cell("3C_battery-1",0.,.1,10)
        b=self.cell("3C_battery-2",0.,-.1,30)
        # Per-cell risk and no intercept penalty: exact cancellation for same input
        for c in [a,b]:
            z,alpha=m.single_cell_kernel(c,self.scaler,self.source)
            correction=float((m.v14.kernel(z[:1],z)@alpha)[0])
            self.assertAlmostEqual(correction, .1 if c is a else -.1,delta=.03)


if __name__=="__main__":unittest.main()
