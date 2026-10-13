"""Summarize already frozen v16 external results; NO model refit or tuning.

All per-cell records and full target truth errors live in results/v16/.
This file only produces a compact, easily auditable report and posthoc
kernel-similarity diagnostic from existing run.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/"results/v16"
TARGETS=("TJU_NCA","TJU_NCM","HUST","MIT_2017A")
METHODS=("source","offset","ridge","RBF-X","RBF-XQ","v15")


def main():
    data=json.loads((R/"summary_external_cohorts.json").read_text())
    cohorts={}
    for group in TARGETS:
        d=json.loads((R/f"target_{group}.json").read_text())
        q=d["summary"]["30"]["pred_quartiles"]
        samples=[r for r in d["records"] if r["policy"]=="pred_quartiles" and r["budget"]==30]
        diag=[x for r in samples for x in r["diagnostics"]]
        cohorts[group]={
            "n_raw_csv":d["n_target_files"],
            "n_test_cells":d["n_eligible_target_cells"],
            "n_excluded":len(d["excluded"]),
            "excluded":d["excluded"],
            "nominal_Ah":d["target_nominal_Ah"],
            "N30_quartile":q["metrics"],
            "N30_quartile_mean_observed_kernel_similarity":mean(
                x["mean_max_krr_similarity"] for x in diag),
            "N30_quartile_mean_v15_lambda":mean(
                x["mean_v15_lambda"] for x in diag),
            "n_fold_records":len(samples),
            "n_per_cell_results":len(q["per_cell"])
        }
    all_budgets={}
    for n,policies in data["macro"].items():
        all_budgets[n]={}
        for pol,record in policies.items():
            all_budgets[n][pol]={
                "n_eligible_external_cells":record["n_eligible_external_cells"],
                "nominal_nMAE_pct":record["cohort_equal_nominal_nMAE_pct"],
                "negative_transfer_test_cells":record["total_negative_transfer_test_cells_vs_source"]
            }
    out={
        "source_file":"results/v16/summary_external_cohorts.json",
        "scientific_status":data["status"],
        "n_manifest_csv":data["n_frozen_raw_external_files"],
        "n_eligible":data["n_eligible_external_cells"],
        "n_excluded":data["n_excluded_external_cells"],
        "all_budget_macros":all_budgets,
        "primary_N30_pred_quartiles":cohorts,
        "primary_bootstrap":data["bootstrap_primary"],
        "important_interpretation":"An out-of-domain evaluation of an XJTU-developed model; 4 cohorts from 3 external data families; nominal-capacity bridge fixed a priori",
        "posthoc_support_diagnostic_warning":"Mean kernel similarity is exploratory and must not be used to tune the fixed methods on these same confirmed external target test cells"
    }
    outpath=R/"compact_summary.json"
    outpath.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("v16 compact summary:",json.dumps(out,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
