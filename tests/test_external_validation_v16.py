"""v16 offline guards for pinned external battery cohort confirmatory evaluation."""
from __future__ import annotations
import importlib.util,sys,unittest,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("phm_v16_tests",ROOT/"experiments/rw_multitarget/external_validation_v16.py")
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)

class V16GuardTests(unittest.TestCase):
    def setUp(self):
        self.scaler=(np.zeros(6),np.ones(6))
        self.source=np.array([1.8,.02,.02,.02,.02,.02,.02])
    @staticmethod
    def cell(name,feat=.1,r=.1,n=30):
        x=np.full((n,6),feat)
        source=np.array([1.8,.02,.02,.02,.02,.02,.02])
        return v.v8.Cell(name,"TJU_NCA",x,v.v8.design(x,(np.zeros(6),np.ones(6)))@source+r)

    def test_fixed_manifest_full_external_total(self):
        data=json.loads(v.MANIFEST.read_text())
        self.assertEqual(set(data["cohorts"]),set(v.SEEDS))
        self.assertEqual(sum(len(x["files"]) for x in data["cohorts"].values()),244)
        self.assertEqual({name:len(d["files"]) for name,d in data["cohorts"].items()},
                         {"TJU_NCA":66,"TJU_NCM":55,"HUST":77,"MIT_2017A":46})
        self.assertEqual([data["cohorts"][k]["nominal_Ah"] for k in v.SEEDS],
                         [3.5,3.5,1.1,1.1])
        self.assertEqual(data["source_commit"],v.v8.COMMIT)
        self.assertEqual(len({x["path"] for coh in data["cohorts"].values() for x in coh["files"]}),244)

    def test_nominal_bridge_proportional_capacity(self):
        # 10% of nominal capacity must be identical in percentage units
        for nominal in (1.1,2.,3.5):
            pseudo_true=np.array([2.,2.])
            pseudo_pred=np.array([1.8,1.8])
            m=v.scaled_metrics(pseudo_pred,pseudo_true,nominal)
            self.assertAlmostEqual(m["nmae_nominal_pct"],10.,places=12)
            self.assertAlmostEqual(m["mae_mAh"],nominal*100.,places=12)

    def test_same_absolute_mAh_can_have_different_nominal_nmae(self):
        p=v.scaled_metrics([1.9],[2.],1.1)
        q=v.scaled_metrics([1.9],[2.],3.5)
        self.assertNotEqual(p["mae_mAh"],q["mae_mAh"])
        self.assertEqual(p["nmae_nominal_pct"],q["nmae_nominal_pct"])

    def test_source_only_scoring_does_not_fit_target(self):
        test=self.cell("TJU_NCA__unseen",.1,.1)
        first=v.score_baselines([test],self.scaler,self.source,3.5)
        test.capacity[:]=1e6
        second=v.score_baselines([test],self.scaler,self.source,3.5)
        # the score changes but source prediction itself, via design+source, never changes
        ref=v.v8.design(test.features,self.scaler)@self.source
        self.assertAlmostEqual(ref[0],1.8+.02*.1*6)
        self.assertNotEqual(first["source"][0]["mae_mAh"],second["source"][0]["mae_mAh"])
        self.assertEqual(first["ridge"],[])
        self.assertEqual(second["offset"],[])

    def test_local_prediction_unaffected_by_test_labels(self):
        a=self.cell("TJU_NCA__a",.1,.05)
        b=self.cell("TJU_NCA__b",.2,.15)
        test=self.cell("TJU_NCA__heldout",.3,.1)
        # diagnostics depend only on features/labeled cells, NOT heldout capacity
        first=v.eval_models([test],[a,b],self.source,self.scaler,3.5)[1][0]
        test.capacity[:]=100
        second=v.eval_models([test],[a,b],self.source,self.scaler,3.5)[1][0]
        self.assertEqual(first,second)

    def test_legacy_kernel_hyperparameters_frozen(self):
        self.assertEqual(v.v14.RHO,.05)
        self.assertEqual(v.v15.RHO,.05)
        self.assertEqual(v.v8.RESIDUAL_ALPHA,10.)
        self.assertEqual(v.v8.FEATURES,(0,4,5,8,12,13))
        self.assertEqual(v.SOURCE_NOMINAL_AH,2.)

    def test_feature_header_exact(self):
        self.assertEqual(len(v.FEATURE_NAMES),17)
        self.assertEqual(v.FEATURE_NAMES[-1],"capacity")
        self.assertEqual(v.FEATURE_NAMES[4],"CC Q")
        self.assertEqual(v.FEATURE_NAMES[12],"CV Q")

    def test_no_duplicate_or_independent_random_repeat_in_summary(self):
        cells=["t1","t2"]
        src=[{"id":k,"mae_mAh":10.,"nmae_nominal_pct":.5} for k in cells]
        fake=[]
        for f in range(5):
            for rep in range(5):
                rows=[{"id":k,"mae_mAh":20.,"nmae_nominal_pct":1.} for k in cells]
                fake.append({"fold":f,"repeat":rep,"budget":30,"policy":"random",
                             "offset":rows,"ridge":rows,"RBF-X":rows,"RBF-XQ":rows,"v15":rows})
        # Intentionally fake duplicated test cells across folds: would average per cell, never count 50 as cells
        out=v.summarize(fake,src,30,"random")
        self.assertEqual(out["n_target_test_cells"],2)
        self.assertAlmostEqual(out["metrics"]["ridge"]["mean_capacity_MAE_mAh"],20.)

    def test_source_normalizer_rejects_wrong_upstream(self):
        data=json.loads(v.MANIFEST.read_text())
        self.assertEqual(data["source_commit"],
                         "cc3cc5053caf38f04e0665f7f88cb109144d035e")

    def test_rank_seed_deterministic(self):
        for seed in v.SEEDS.values():
            a=v.v8.js_shuffle(range(55),seed)
            b=v.v8.js_shuffle(range(55),seed)
            self.assertEqual(a,b)
            self.assertEqual(sorted(a),list(range(55)))


if __name__=="__main__":unittest.main()
