"""Synthetic offline tests for v14 source-preserving local kernel residual model."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/"experiments/rw_multitarget/source_anchored_kernel_v14.py"
sp=importlib.util.spec_from_file_location("phm_v14_tests",SCRIPT)
v14=importlib.util.module_from_spec(sp)
sys.modules[sp.name]=v14
sp.loader.exec_module(v14)


class KernelV14Tests(unittest.TestCase):
    def setUp(self):
        self.scaler=(np.zeros(6),np.ones(6))
        self.source=np.array([1.8,.1,.1,.1,.1,.1,.1])
        self.a=self.cell("2C_battery-1",0.,.12,10)
        self.b=self.cell("2C_battery-2",.1,.1,10)

    def cell(self,name,feature,residual,n=10):
        x=np.full((n,6),feature,dtype=float)
        y=v14.v8.design(x,self.scaler)@self.source+residual
        return v14.v8.Cell(name,"2C",x,y)

    def test_nonzero_local_residual(self):
        m=v14.train_local([self.a,self.b],self.scaler,self.source,"RBF-XQ")
        prediction,corr,sim=v14.predict(self.a,self.scaler,self.source,m,"RBF-XQ")
        self.assertGreater(np.mean(corr),.05)
        self.assertLess(np.mean(corr),.15)
        self.assertGreaterEqual(sim.min(),0.)
        self.assertTrue(np.isfinite(prediction).all())

    def test_preserves_source_when_outside_support(self):
        # All standardized six features clip at +6 vs training features near 0.
        far=self.cell("2C_battery-3",1000.,0.,5)
        for mode in ("RBF-X","RBF-XQ"):
            model=v14.train_local([self.a,self.b],self.scaler,self.source,mode)
            predicted,corr,sim=v14.predict(far,self.scaler,self.source,model,mode)
            sourcepred=v14.v8.design(far.features,self.scaler)@self.source
            self.assertLess(np.max(np.abs(corr)),1e-5)
            np.testing.assert_allclose(predicted,sourcepred,atol=1e-5,rtol=0)

    def test_zero_residual_preserves_source(self):
        a=self.cell("2C_battery-1",0.,0.,10)
        b=self.cell("2C_battery-2",.1,0.,10)
        model=v14.train_local([a,b],self.scaler,self.source,"RBF-XQ")
        pred,corr,_=v14.predict(self.a,self.scaler,self.source,model,"RBF-XQ")
        np.testing.assert_allclose(corr,0,atol=1e-9)
        np.testing.assert_allclose(pred,v14.v8.design(self.a.features,self.scaler)@self.source,atol=1e-9)

    def test_train_reject_duplicate_target_cell(self):
        with self.assertRaises(ValueError):
            v14.train_local([self.a,self.a],self.scaler,self.source,"RBF-X")

    def test_train_requires_positive_rho(self):
        with self.assertRaises(ValueError):
            v14.train_local([self.a,self.b],self.scaler,self.source,"RBF-XQ",rho=0.)

    def test_source_model_not_changed(self):
        before=self.source.copy()
        _=v14.train_local([self.a,self.b],self.scaler,self.source,"RBF-XQ")
        np.testing.assert_array_equal(before,self.source)

    def test_xq_metric_changes_kernel(self):
        x=np.vstack([np.zeros(6),np.array([1.,-1.,0,0,0,0])])
        z1=v14.space(x,self.scaler,self.source,"RBF-X")
        z2=v14.space(x,self.scaler,self.source,"RBF-XQ")
        self.assertEqual(z1.shape,(2,6))
        self.assertEqual(z2.shape,(2,7))
        # Source predictions equal for this pair, only denominator changes.
        self.assertGreater(v14.kernel(z2,z2)[0,1],v14.kernel(z1,z1)[0,1])
        # For a source-prediction direction, the XQ kernel is strictly different.
        y=np.vstack([np.zeros(6),np.ones(6)])
        self.assertNotAlmostEqual(
            v14.kernel(v14.space(y,self.scaler,self.source,"RBF-XQ"),
                       v14.space(y,self.scaler,self.source,"RBF-XQ"))[0,1],
            v14.kernel(v14.space(y,self.scaler,self.source,"RBF-X"),
                       v14.space(y,self.scaler,self.source,"RBF-X"))[0,1])

    def test_equal_cell_weights_independent_of_cell_history(self):
        a=self.cell("2C_battery-1",0.,.1,10)
        b=self.cell("2C_battery-2",0.,-.1,30)
        for mode in ("RBF-X","RBF-XQ"):
            model=v14.train_local([a,b],self.scaler,self.source,mode)
            _,corr,_=v14.predict(a,self.scaler,self.source,model,mode)
            self.assertAlmostEqual(float(np.mean(corr)),0.,places=7)

    def test_target_test_capacity_does_not_enter_prediction(self):
        model=v14.train_local([self.a,self.b],self.scaler,self.source,"RBF-XQ")
        test=self.cell("2C_battery-4",.02,0,30)
        first=v14.predict(test,self.scaler,self.source,model,"RBF-XQ")[0]
        test.capacity[:]=99
        second=v14.predict(test,self.scaler,self.source,model,"RBF-XQ")[0]
        np.testing.assert_array_equal(first,second)

    def test_bad_archive_selection_detected(self):
        record={"heldout":["a"],"selected":["b","c"],"n_labeled_cycles":[10,10],
                "ridge":[{"id":"a","mae_mAh":20.}],
                "offset":[{"id":"a","mae_mAh":30.}]}
        ref={**record,"selected":["c","d"]}
        with self.assertRaises(AssertionError):
            v14.validate_record(record,ref)


if __name__=="__main__":unittest.main()
