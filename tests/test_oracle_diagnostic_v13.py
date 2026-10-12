"""Offline tests for v13 posthoc oracle diagnostics (no external CSV)."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/"experiments/rw_multitarget/oracle_diagnostic_v13.py"
spec=importlib.util.spec_from_file_location("oracle_v13",SCRIPT)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def sample(policy="pred_quartiles"):
    ids=[f"3C_battery-{i}" for i in range(1,9)]
    folds=[ids[:2],ids[2:4],ids[4:6],ids[6:7],ids[7:]]
    ridge={c:5. for c in ids}
    ridge[ids[0]]=0.
    ridge[ids[1]]=30.
    source=[dict(id=cid,mae_mAh=10.) for cid in ids]
    rec=[]
    for f,held in enumerate(folds):
        for rep in range(5 if policy=="random" else 1):
            rec.append(dict(fold=f,repeat=rep,policy=policy,budget=30,
                            heldout=held,selected=["3C_battery-10","3C_battery-11"],
                            n_labeled_cycles=[30,30],lambda=0.8,
                            ridge=[dict(id=cid,mae_mAh=ridge[cid]) for cid in held],
                            shrink=[dict(id=cid,mae_mAh=ridge[cid]) for cid in held]))
    return dict(domain="3C",n_target_cells=8,folds=folds,source_predictions=source,
                records=rec)


class OracleAuditTests(unittest.TestCase):
    def test_fivefold_oracle_monotone_and_weighted(self):
        res=m.diagnostic(sample(),budget=30,policy="pred_quartiles",assert_v12=False)
        scores=res["metrics_mAh"]
        self.assertAlmostEqual(scores["source"],10.)
        self.assertAlmostEqual(scores["ridge"],7.5)
        self.assertAlmostEqual(scores["domain_oracle"],7.5)
        self.assertAlmostEqual(scores["fold_oracle"],6.25)
        self.assertAlmostEqual(scores["cell_oracle"],5.)
        self.assertEqual(scores["n_cells_ridge_negative_vs_source"],1)

    def test_random_repeats_not_extra_cells(self):
        res=m.diagnostic(sample("random"),budget=30,policy="random",assert_v12=False)
        self.assertEqual(res["n_target_cells"],8)
        self.assertEqual(res["n_random_selection_repeats"],5)
        self.assertEqual(len(res["per_cell"]),8)
        self.assertAlmostEqual(res["metrics_mAh"]["cell_oracle"],5.)

    def test_reject_target_id_leak_in_selection(self):
        data=sample()
        data["records"][0]["selected"][0]="3C_battery-1"
        with self.assertRaises(AssertionError):
            m.diagnostic(data,budget=30,policy="pred_quartiles",assert_v12=False)

    def test_reject_duplicate_heldout_ids(self):
        data=sample()
        data["records"][0]["heldout"]=["3C_battery-1","3C_battery-1"]
        with self.assertRaises(AssertionError):
            m.diagnostic(data,budget=30,policy="pred_quartiles",assert_v12=False)

    def test_reject_budget_change(self):
        data=sample()
        data["records"][2]["n_labeled_cycles"]=[25,30]
        with self.assertRaises(AssertionError):
            m.diagnostic(data,budget=30,policy="pred_quartiles",assert_v12=False)

    def test_reject_missing_random_draw(self):
        data=sample("random")
        data["records"].pop()
        with self.assertRaises(AssertionError):
            m.diagnostic(data,budget=30,policy="random",assert_v12=False)

    def test_reject_duplicate_test_cell(self):
        data=sample()
        data["folds"][1][0]=data["folds"][0][0]
        with self.assertRaises(AssertionError):
            m.diagnostic(data,budget=30,policy="pred_quartiles",assert_v12=False)


if __name__=="__main__":
    unittest.main()
