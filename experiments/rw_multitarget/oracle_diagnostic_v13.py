"""v13 explanatory oracle analysis on *existing* v12 archived predictions.

IMPORTANT: oracle accesses OUTER TEST LABEL-BASED MAE, violating deployable
label constraints BY DESIGN. It is never a legitimate trainable predictor.
No model training or hyperparameter selection is performed in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("2C","3C","R2.5","R3","Sim_satellite")
POLICIES = ("pred_quartiles","random")
BUDGETS = (10,20,30)
TOL = 1e-5


def mean(values):
    if not values:
        raise ValueError("Empty metric")
    return float(np.mean(np.asarray(values,float)))


def diagnostic(domain_data, *, budget, policy, assert_v12=True):
    """Compute domain-, fold- and cell-conditioned test-label cheating oracles.

    Fold/cell oracle average across each historical random selection repeat
    for a test cell. This is NOT a model with access to test-time features
    alone: error values require *full heldout target-capacity labels*.
    """
    src_rows = domain_data["source_predictions"]
    source = {}
    for x in src_rows:
        if x["id"] in source:
            raise AssertionError("Duplicated source heldout ID")
        source[x["id"]] = float(x["mae_mAh"])
    cells = sorted(source)
    n = len(cells)
    if n != domain_data["n_target_cells"]:
        raise AssertionError("Wrong independent test-cell count")
    if sorted(cid for fold in domain_data["folds"] for cid in fold) != cells:
        raise AssertionError("Frozen fivefold IDs do not cover each cell exactly once")
    recs = [r for r in domain_data["records"] if
            r["policy"] == policy and r["budget"] == budget]
    repeats = 5 if policy == "random" else 1
    if len(recs) != 5*repeats:
        raise AssertionError("One acquisition per fold/repeat expected")
    per_cell = defaultdict(list)
    fold_records = []
    keys=set()
    for r in recs:
        fi,rep = r["fold"],r["repeat"]
        if (fi,rep) in keys or fi not in range(5) or rep not in range(repeats):
            raise AssertionError("Invalid duplicate fold/repeat")
        keys.add((fi,rep))
        target_ids=domain_data["folds"][fi]
        if set(target_ids) != set(r["heldout"]) or len(target_ids)!=len(r["heldout"]):
            raise AssertionError("Unfrozen or duplicated heldout")
        if len(r["selected"])!=2 or len(set(r["selected"]))!=2:
            raise AssertionError("Selected batteries must be distinct")
        if set(r["selected"]) & set(target_ids):
            raise AssertionError("Test battery is labeled")
        if list(r["n_labeled_cycles"])!=[budget,budget]:
            raise AssertionError("Label budget changed")
        for k in ("ridge","shrink"):
            if sorted(x["id"] for x in r[k]) != sorted(target_ids):
                raise AssertionError("Predictions missing or duplicate IDs")
        rid={x["id"]:float(x["mae_mAh"]) for x in r["ridge"]}
        shr={x["id"]:float(x["mae_mAh"]) for x in r["shrink"]}
        fold_source=mean([source[i] for i in target_ids])
        fold_ridge=mean([rid[i] for i in target_ids])
        fold_choose_source=fold_source <= fold_ridge
        fold_records.append({
            "fold":fi,"repeat":rep,"selected":r["selected"],
            "heldout":r["heldout"],"lambda":float(r["lambda"]),
            "source_mae_mAh":fold_source,
            "ridge_mae_mAh":fold_ridge,
            "ridge_minus_source_mAh":fold_ridge-fold_source,
            "fold_oracle_choice":"source" if fold_choose_source else "ridge",
            "cell_oracle_source_choices":sum(source[i]<=rid[i] for i in target_ids),
        })
        for cid in target_ids:
            s,v,w=source[cid],rid[cid],shr[cid]
            if not all(np.isfinite(x) and x>=0 for x in (s,v,w)):
                raise AssertionError("Invalid nonnegative MAE")
            per_cell[cid].append({
                "fold":fi,"repeat":rep,"source":s,"ridge":v,"loco":w,
                "domain_oracle_placeholder":None,
                "fold_oracle":s if fold_choose_source else v,
                "cell_oracle":min(s,v),
                "ridge_gain_over_source":s-v,
            })
    if any(len(vals)!=repeats for vals in per_cell.values()) or set(per_cell)!=set(cells):
        raise AssertionError("Random repeats improperly counted as new cells")
    source_val=mean(list(source.values()))
    summaries={}
    for k in ("ridge","loco","fold_oracle","cell_oracle"):
        summaries[k]=mean([mean([v[k] for v in per_cell[cid]]) for cid in cells])
    summaries["source"]=source_val
    summaries["domain_oracle"]=min(source_val,summaries["ridge"])
    eps=1e-8
    chain=["cell_oracle","fold_oracle","domain_oracle"]
    if any(summaries[a] > summaries[b]+eps for a,b in zip(chain,chain[1:])):
        raise AssertionError("Oracle granularity hierarchy failed")
    if summaries["domain_oracle"] > min(summaries["source"],summaries["ridge"])+eps:
        raise AssertionError("Domain oracle invalid")

    per_cell_out=[]
    for cid in cells:
        vals=per_cell[cid]
        rmean=mean([v["ridge"] for v in vals])
        per_cell_out.append({
            "id":cid,"source_mae_mAh":source[cid],
            "ridge_mae_mAh":rmean,
            "loco_mae_mAh":mean([v["loco"] for v in vals]),
            "fold_oracle_mae_mAh":mean([v["fold_oracle"] for v in vals]),
            "cell_oracle_mae_mAh":mean([v["cell_oracle"] for v in vals]),
            "mean_ridge_gain_over_source_mAh":source[cid]-rmean,
            "ridge_negative_transfer_after_repeat_average":rmean > source[cid],
            "n_repeats_with_ridge_negative_transfer":sum(
                v["ridge"]>source[cid] for v in vals)
        })
    summaries["n_cells_ridge_negative_vs_source"]=sum(x["ridge_negative_transfer_after_repeat_average"] for x in per_cell_out)
    summaries["n_cells_oracle_prefers_source"]=sum(x["source_mae_mAh"]<=x["ridge_mae_mAh"] for x in per_cell_out)
    summaries["n_fold_repeats_source_preferred"]=sum(x["fold_oracle_choice"]=="source" for x in fold_records)
    summaries["n_fold_repeats"]=len(fold_records)
    summaries["oracle_gain_cell_vs_ridge_mAh"]=summaries["ridge"]-summaries["cell_oracle"]
    summaries["oracle_gain_fold_vs_ridge_mAh"]=summaries["ridge"]-summaries["fold_oracle"]
    summaries["oracle_gain_cell_vs_fold_mAh"]=summaries["fold_oracle"]-summaries["cell_oracle"]
    summaries["oracle_gain_fold_vs_domain_mAh"]=summaries["domain_oracle"]-summaries["fold_oracle"]
    summaries["negative_transfer_cost_mAh"]=mean([max(0., x["ridge_mae_mAh"]-x["source_mae_mAh"]) for x in per_cell_out])
    summaries["positive_transfer_gain_mAh"]=mean([max(0., x["source_mae_mAh"]-x["ridge_mae_mAh"]) for x in per_cell_out])

    if assert_v12:
        ref=domain_data["summary"][str(budget)][policy]
        for key,old in (("source",ref["source_only_mAh"]),
                        ("ridge",ref["methods"]["ridge"]),
                        ("loco",ref["methods"]["shrink"])):
            if abs(summaries[key]-old)>TOL:
                raise AssertionError(f"Archived v12 mismatch {key}: {summaries[key]} vs {old}")
        if summaries["n_cells_ridge_negative_vs_source"]!=ref["n_cells_negative_vs_source"]["ridge"]:
            raise AssertionError("Negative-transfer counts differ from archived v12")
    return {
        "domain":domain_data["domain"],"budget":budget,"policy":policy,
        "n_target_cells":n,"n_random_selection_repeats":repeats,
        "metrics_mAh":summaries,
        "per_cell":per_cell_out,"fold_repeats":fold_records
    }


def run():
    domains={
        d:json.loads((ROOT/"results/v12"/f"rw_to_{d.replace('.','_')}_loco.json").read_text())
        for d in TARGETS
    }
    results={}
    for n in BUDGETS:
        results[str(n)]={}
        for p in POLICIES:
            by={d:diagnostic(domains[d],budget=n,policy=p) for d in TARGETS}
            keys=("source","ridge","loco","domain_oracle","fold_oracle","cell_oracle")
            macro={key:mean([by[d]["metrics_mAh"][key] for d in TARGETS]) for key in keys}
            if not (macro["cell_oracle"]<=macro["fold_oracle"]+1e-8<=macro["domain_oracle"]+1e-8):
                raise AssertionError("Macro-level hierarchy broken")
            results[str(n)][p]={
                "macro_mAh":macro,
                "macro_oracle_gain_cell_vs_ridge_mAh":macro["ridge"]-macro["cell_oracle"],
                "n_negative_transfer_test_cells":sum(
                    by[d]["metrics_mAh"]["n_cells_ridge_negative_vs_source"] for d in TARGETS),
                "n_unique_test_cells":sum(by[d]["n_target_cells"] for d in TARGETS),
                "by_domain":by
            }
    report={
        "date":"2026-10-12",
        "protocol":"docs/V13_ORACLE_DIAGNOSTIC_PROTOCOL_20261012.md",
        "status":"posthoc_oracle_diagnostic_NOT_deployable_or_externally_confirmatory",
        "leakage_warning":"All oracles exploit full-life target TEST capacity MAE; NEVER use as trained model, operational method, valid predictor or baseline",
        "experimental_lineage":"v12 original archived per-cell scores; no new model training or test fold changes",
        "source_cells":8,"target_cells":47,
        "budgets":BUDGETS,"policies":POLICIES,
        "primary":{"budget":30,"policy":"pred_quartiles"},
        "results":results
    }
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"results/v13/oracle_diagnostics.json")
    args=parser.parse_args()
    report=run()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    condensed={
        n:{p:{"macro_mAh":v["macro_mAh"],"n_negative_transfer_test_cells":v["n_negative_transfer_test_cells"]}
           for p,v in b.items()} for n,b in report["results"].items()}
    print(json.dumps(condensed,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
