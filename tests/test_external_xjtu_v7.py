import importlib.util
from pathlib import Path
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "experiments/external_xjtu_v7/benchmark.py"
spec = importlib.util.spec_from_file_location("xjtu_v7_benchmark", PATH)
b = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = b
spec.loader.exec_module(b)


class TestXJTUBenchmark(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.cells = []
        for group in ("3C", "RW"):
            for j in range(6):
                x = rng.normal(size=(24, 6))
                y = 1.7 + 0.05*x[:, 1] + 0.02*x[:, 4] + (0.1 if group == "RW" else 0) + j*0.012
                self.cells.append(b.Cell(f"{group}_battery-{j+1}", group, x, y, "synthetic"))

    def test_no_target_test_cell_labels(self):
        records=b.perform(self.cells, repeats=1)
        self.assertTrue(records)
        for r in records:
            self.assertFalse(set(r["labeled_ids"]) & set(r["heldout_test_ids"]))
            self.assertEqual(len(r["heldout_test_ids"]), 2)
            self.assertTrue(set(r["labeled_ids"]).issubset(set(r["acquisition_pool_ids"])))

    def test_same_holdout_across_selection_methods(self):
        records=b.perform(self.cells, repeats=1)
        grouped={}
        for r in records:
            k=(r["target_domain"],tuple(r["heldout_test_ids"]),r["k"],r["method"])
            grouped.setdefault(k,{})[r["policy"]]=r["heldout_test_ids"]
        self.assertTrue(all(v["random"]==v["representative"] for v in grouped.values()))

    def test_source_only_invariant_to_selection_policy(self):
        records=b.perform(self.cells, repeats=2)
        by={}
        for r in records:
            if r["method"]!="source":continue
            key=(r["target_domain"],tuple(r["heldout_test_ids"]),r["k"],r["repeat"])
            by.setdefault(key,[]).append(r["capacity_MAE_mAh"])
        self.assertTrue(all(len(x)==2 and np.isclose(x[0],x[1],atol=1e-9) for x in by.values()))

    def test_battery_balanced_train_and_nonleakage(self):
        cells=self.cells[:6]
        stats=b.scaler_fit(cells)
        base=b.ridge_fit(cells,stats,1.0)
        self.assertTrue(np.isfinite(base).all())
        y=b.cell_mae_mAh(cells[0],stats,base)
        self.assertGreaterEqual(y,0)

    def test_risk_gate_uses_same_target_test_and_never_reads_test_labels(self):
        ungated=b.perform(self.cells,repeats=1,label_fraction=0.25)
        gated=b.perform(self.cells,repeats=1,label_fraction=0.25,risk_gate=True)
        original=[r for r in gated if r["method"] in ("source","offset","residual_ridge")]
        self.assertEqual(len(original),len(ungated))
        for a,c in zip(ungated,original):
            self.assertEqual(a["heldout_test_ids"],c["heldout_test_ids"])
            self.assertEqual(a["labeled_ids"],c["labeled_ids"])
            self.assertAlmostEqual(a["capacity_MAE_mAh"],c["capacity_MAE_mAh"],places=9)
        for r in gated:
            if r["method"].startswith("gate_"):
                self.assertFalse(set(r["labeled_ids"]) & set(r["heldout_test_ids"]))

    def test_partial_labels_do_not_change_test_ids(self):
        a=b.perform(self.cells,repeats=1,label_fraction=1)
        c=b.perform(self.cells,repeats=1,label_fraction=0.25)
        self.assertEqual([r["heldout_test_ids"] for r in a],[r["heldout_test_ids"] for r in c])
        self.assertEqual([r["labeled_ids"] for r in a],[r["labeled_ids"] for r in c])


if __name__=="__main__":
    unittest.main()