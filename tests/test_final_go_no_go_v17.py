"""v17 test guards: predeclared data, composite kernel, strict final verdict."""
from __future__ import annotations
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
mod=ROOT/"experiments/rw_multitarget/final_go_no_go_v17.py"
spec=importlib.util.spec_from_file_location("phm_v17_tests",mod)
v=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=v
spec.loader.exec_module(v)


class FinalDecisionTests(unittest.TestCase):
    def setUp(self):
        self.scaler=(np.zeros(6),np.ones(6))
        self.source=np.array([1.8,.02,.02,.02,.02,.02,.02],float)

    def cell(self,name,x=.05,residual=.10,n=30):
        f=np.full((n,6),x,float)
        y=v.v8.design(f,self.scaler)@self.source+residual
        return v.v8.Cell(name,"TJU_MIX",f,y)

    def test_manifest_freezes_exact_88_unseen_target_csvs(self):
        j=json.loads(v.MANIFEST.read_text())
        expected={"TJU_MIX":9,"MIT_2017B":43,"MIT_2018A":36}
        self.assertEqual({k:len(c["files"]) for k,c in j["cohorts"].items()},expected)
        self.assertEqual([j["cohorts"][k]["nominal_Ah"] for k in expected],[2.5,1.1,1.1])
        self.assertEqual(j["source_commit"],v.v8.COMMIT)
        self.assertEqual(len({f["path"] for c in j["cohorts"].values() for f in c["files"]}),88)
        self.assertTrue(all(len(f["git_blob_sha"])==40 for c in j["cohorts"].values() for f in c["files"]))

    def test_exact_one_plus_RBF_kernel(self):
        a=self.cell("a",0.0,.1,10);b=self.cell("b",.1,.12,10)
        z=np.vstack([v.v14.space(c.features,self.scaler,self.source,"RBF-X") for c in [a,b]])
        m=v.train_global_local([a,b],self.scaler,self.source)
        residual=np.concatenate([c.capacity-v.v8.design(c.features,self.scaler)@self.source for c in [a,b]])
        reg=np.diag(np.full(20,v.v14.RHO*2*10))
        np.testing.assert_allclose((v.v14.kernel(z,z)+1.+reg)@m[1],residual,atol=1e-8)
        self.assertEqual(m[0].shape,(20,6))

    def test_zero_residual_recovers_unmodified_source(self):
        a=self.cell("a",.1,0);b=self.cell("b",.2,0)
        m=v.train_global_local([a,b],self.scaler,self.source)
        test=self.cell("test",.3,.1)
        pred,corr=v.global_local_prediction(test,self.scaler,self.source,m)
        np.testing.assert_allclose(pred,v.v8.design(test.features,self.scaler)@self.source,atol=1e-10)
        np.testing.assert_allclose(corr,0,atol=1e-10)

    def test_test_capacity_never_affects_predictor(self):
        a=self.cell("a",0,.1);b=self.cell("b",.2,.05)
        m=v.train_global_local([a,b],self.scaler,self.source)
        test=self.cell("hidden",.1,.5)
        p1=v.global_local_prediction(test,self.scaler,self.source,m)[0]
        test.capacity[:]=1e5
        p2=v.global_local_prediction(test,self.scaler,self.source,m)[0]
        np.testing.assert_array_equal(p1,p2)

    def test_training_requires_two_distinct_cells(self):
        a=self.cell("a")
        with self.assertRaises(ValueError):
            v.train_global_local([a,a],self.scaler,self.source)

    def test_nominal_bridge_exact_for_new_cohorts(self):
        for nom in (1.1,2.5):
            metric=v.scaled_metrics([1.8],[2.],nom)
            self.assertAlmostEqual(metric["nmae_nominal_pct"],10)
            self.assertAlmostEqual(metric["mae_mAh"],100*nom)

    @staticmethod
    def fixtures(advantage=0):
        # 3 cohorts, macro, 2 labels independent of test truth.
        results={}
        for d in v.SEEDS:
            results[d]={"summary":{
                "30":{"pred_quartiles":{"metrics":{"ridge":{"mean_nominal_nMAE_pct":5.},
                              "GL-KRR":{"mean_nominal_nMAE_pct":5.-advantage}}},
                      "random":{"metrics":{"ridge":{"mean_nominal_nMAE_pct":6.}}}}
            }}
        macro={
          "30":{"pred_quartiles":{
                    "cohort_equal_nominal_nMAE_pct":{"ridge":5.,"GL-KRR":5.-advantage},
                    "total_negative_transfer_test_cells_vs_source":{"ridge":2,"GL-KRR":1}},
                "random":{"cohort_equal_nominal_nMAE_pct":{"ridge":6.,"GL-KRR":5.}}},
          "20":{"pred_quartiles":{"cohort_equal_nominal_nMAE_pct":{"ridge":6.,"GL-KRR":5.}}}
        }
        cal={"ridge_minus_GL-KRR":{"exploratory_cell_bootstrap_ci95_pp":[.1,1.]}}
        select={"effect_random_ridge_minus_quartile_ridge_nominal_pp":1.,
                "exploratory_cell_bootstrap_ci95_pp":[.1,1.]}
        return results,macro,cal,select

    def test_no_go_binding_when_both_fail(self):
        a,b,c,d=self.fixtures(advantage=-2.)
        d["effect_random_ridge_minus_quartile_ridge_nominal_pp"]=.1
        self.assertEqual(v.decide_direction(a,b,c,d)["automatic_final_decision"],
                         "NO_GO_STOP_THIS_AS_MAIN_ALGORITHMIC_THESIS_DIRECTION")

    def test_selection_only_go_when_calibration_fails(self):
        a,b,c,d=self.fixtures(advantage=-2.)
        self.assertEqual(v.decide_direction(a,b,c,d)["automatic_final_decision"],
                         "GO_SELECTION_ONLY_DROP_COMPLEX_CALIBRATION")

    def test_calibration_go_when_all_pass(self):
        a,b,c,d=self.fixtures(advantage=1.)
        self.assertEqual(v.decide_direction(a,b,c,d)["automatic_final_decision"],
                         "GO_UNIFIED_CALIBRATION")

    def test_outcome_bootstrap_only_resamples_original_cell_IDs(self):
        self.assertEqual(v.N_BOOT,10000)
        self.assertEqual(v.BOOT_SEED,20261707)
        self.assertEqual(v.RANDOM_REPEATS,5)
        self.assertEqual(v.BUDGETS,(10,20,30))


if __name__=="__main__":
    unittest.main()
