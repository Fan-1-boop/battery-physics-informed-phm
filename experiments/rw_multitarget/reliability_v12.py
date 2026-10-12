"""v12 two-labeled-cell LOCO shrinkage: XJTU RW8 -> five target conditions.

DEVELOPMENT DATA only: v8-v10 already explored these 47 targets. Original
pinned author CSVs are read externally and NEVER committed. Selection, test
folds and feature preprocessing remain frozen. No target-test information
enters the LOCO shrinkage coefficient.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V10_PATH = ROOT / "experiments/rw_multitarget/fixed_budget_v10.py"
sp = importlib.util.spec_from_file_location("phm_v10_v12_adapter", V10_PATH)
v10 = importlib.util.module_from_spec(sp)
sys.modules[sp.name] = v10
sp.loader.exec_module(v10)
v8 = v10.v8
TARGETS = ("2C", "3C", "R2.5", "R3", "Sim_satellite")
BUDGETS = (10, 20, 30)
POLICIES = ("random", "pred_quartiles")
REPEATS = 5
TOL_MAH = 1e-6
BOOTSTRAP_SEED = 20261212


def loco_strength(labeled, scaler, source):
    """Unsupervised wrt test: choose lambda using ONLY two labeled batteries.

    The two held-in cells are never used to train their own leave-one-cell-out
    correction. Each cell contributes equal weight to the LOCO squared risk.
    """
    if len(labeled) != 2 or labeled[0].name == labeled[1].name:
        raise ValueError("LOCO shrinkage requires two distinct labeled cells")
    a = b = 0.0
    diagnostics = []
    for held in (0, 1):
        c = labeled[held]
        other = labeled[1 - held]
        independent = v8.ridge([other], scaler, v8.RESIDUAL_ALPHA, base=source)
        x = v8.design(c.features, scaler)
        pred_residual = x @ independent
        true_residual = c.capacity - x @ source
        a += float(np.mean(pred_residual * true_residual))
        b += float(np.mean(pred_residual**2))
        diagnostics.append({
            "heldout_labeled_cell": c.name,
            "trained_on": other.name,
            "n_validation_labels": c.n_cycles,
            "cross_cell_residual_rmse_Ah": float(
                np.mean((true_residual - pred_residual)**2)**0.5),
        })
    lam = 0.0 if b <= 1e-20 else float(np.clip(a / b, 0.0, 1.0))
    return lam, {"cross_product": a, "squared_prediction": b,
                 "loco_cells": diagnostics}


def calc_cell_scores(test, scaler, source, labeled, lam):
    correction = v8.ridge(labeled, scaler, v8.RESIDUAL_ALPHA, base=source)
    offset = float(np.mean([
        np.mean(c.capacity - v8.design(c.features, scaler) @ source)
        for c in labeled
    ]))
    return {
        "ridge": v8.score(test, scaler, source, correction=correction),
        "offset": v8.score(test, scaler, source, offset=offset),
        "shrink": v8.score(test, scaler, source, correction=lam*correction),
    }


def mean_score(records, key):
    per_cell = {}
    for rec in records:
        for x in rec[key]:
            per_cell.setdefault(x["id"], []).append(float(x["mae_mAh"]))
    return {cell: float(np.mean(v)) for cell, v in sorted(per_cell.items())}


def validate_archives(domain, cells, folds, source, scaler, records, source_rows):
    v10_json = json.loads((ROOT / "results/v10" /
                     f"rw_to_{domain.replace('.', '_')}_fixed_N.json").read_text())
    v8_json = json.loads((ROOT / "results/v8" /
                     f"rw_to_{domain.replace('.', '_')}_fivefold.json").read_text())
    errors = []
    expected_folds = [[cells[i].name for i in fold] for fold in folds]
    if expected_folds != v10_json["folds"] or expected_folds != v8_json["fold_indices"]:
        errors.append("Archived folds do not match frozen JS v8 order")
    expected_targets = {x["id"]: x for x in v8_json["target_cells"]}
    for c in cells:
        ref = expected_targets[c.name]
        if c.git_sha != ref["sha"] or c.n_cycles != ref["cycles"]:
            errors.append(f"Original CSV blob/length mismatch for {c.name}")
    ref_source = {p["id"]: p["mae_mAh"] for r in v8_json["results"]
                  if r["policy"] == "source" for p in r["per_cell"]}
    for p in source_rows:
        if abs(ref_source[p["id"]] - p["mae_mAh"]) > TOL_MAH:
            errors.append(f"Source-only reference changed for {p['id']}")
    ref_map = {(r["fold"], r["policy"], r.get("rep", r.get("repeat")), r["budget"]):r
                for r in v10_json["records"]}
    max_diff = 0.0
    n_score_checks = 0
    for r in records:
        key = (r["fold"],r["policy"],r["repeat"],r["budget"])
        expected = ref_map.get(key)
        if expected is None:
            errors.append(f"Missing archived key {key}")
            continue
        if r["selected"] != expected["selected"] or r["heldout"] != expected["heldout"]:
            errors.append(f"Historical selected/test IDs differ: {key}")
        if [r["budget"]]*2 != expected["n_labeled_cycles"]:
            errors.append(f"Historical labeled budget mismatch: {key}")
        for method, archived in (("ridge","ridge"),("offset","bias")):
            actual = {x["id"]:x["mae_mAh"] for x in r[method]}
            old = {x["id"]:x["mae_mAh"] for x in expected[archived]}
            if actual.keys() != old.keys():
                errors.append(f"Prediction IDs mismatch: {key}, {method}")
            for cell in actual.keys() & old.keys():
                n_score_checks += 1
                dif = abs(actual[cell]-old[cell])
                max_diff = max(max_diff, dif)
                if dif > TOL_MAH:
                    errors.append(f"Per-cell MAE mismatch {key} {cell}/{method}: {dif:.6g}")
    if errors:
        raise AssertionError(f"{domain} archive cross-check failed: " + "; ".join(errors[:10]))
    return {"n_legacy_mae_checks":n_score_checks,
            "max_legacy_mae_difference_mAh":max_diff,
            "n_failures":0}


def evaluate_domain(cells, group, source, scaler, budgets=BUDGETS, repeats=REPEATS):
    cells = sorted(cells, key=lambda c:int(c.name.rsplit("-",1)[1]))
    folds = v8.make_folds(len(cells), group)
    records, source_rows = [], []
    for fold, test_ids in enumerate(folds):
        held = set(test_ids)
        test = [c for i,c in enumerate(cells) if i in held]
        pool = [c for i,c in enumerate(cells) if i not in held]
        if len(pool) < 2 or set(c.name for c in test)&set(c.name for c in pool):
            raise AssertionError("Invalid acquisition/test split")
        source_rows.extend(v8.score(test, scaler, source))
        for policy in POLICIES:
            for rep in range(repeats if policy == "random" else 1):
                chosen = v10.acquire(pool,scaler,source,policy,fold,rep)
                if len(set(chosen))!=2: raise AssertionError("Duplicate acquisition")
                for budget in budgets:
                    selected = v10.label_prefix(pool,chosen,budget)
                    if any(c.n_cycles != budget for c in selected):
                        raise AssertionError("Short cycle history violates exact budget")
                    if set(c.name for c in selected)&set(c.name for c in test):
                        raise AssertionError("Labeled test battery!")
                    lam, diag = loco_strength(selected,scaler,source)
                    scores = calc_cell_scores(test,scaler,source,selected,lam)
                    records.append({
                        "fold":fold,"policy":policy,"repeat":rep,"budget":budget,
                        "selected":[c.name for c in selected],
                        "heldout":[c.name for c in test],
                        "n_labeled_cycles":[c.n_cycles for c in selected],
                        "lambda":lam,"loco":diag, **scores
                    })

    audit = validate_archives(group, cells, folds, source, scaler, records, source_rows)
    src = {c["id"]:c["mae_mAh"] for c in source_rows}
    summary = {}
    for budget in budgets:
        summary[str(budget)] = {}
        for policy in POLICIES:
            rec = [r for r in records if r["budget"]==budget and r["policy"]==policy]
            lam = [r["lambda"] for r in rec]
            scores = {method:mean_score(rec,method)
                      for method in ("ridge","offset","shrink")}
            baseline_scores = scores["ridge"]
            summary[str(budget)][policy] = {
                "methods":{method:float(np.mean(list(v.values()))) for method,v in scores.items()},
                "source_only_mAh":float(np.mean(list(src.values()))),
                "mean_lambda":float(np.mean(lam)),
                "fraction_lambda_zero":float(np.mean(np.asarray(lam)<1e-12)),
                "fraction_lambda_one":float(np.mean(np.asarray(lam)>1-1e-12)),
                "min_lambda":float(min(lam)),"max_lambda":float(max(lam)),
                "n_fold_acquisitions":len(rec),
                "n_test_cells":len(src),
                "n_cells_negative_vs_source":{
                    method:sum(v[id]>src[id] for id in src) for method,v in scores.items()
                },
                "n_cells_improved_shrink_vs_ridge":sum(
                    scores["shrink"][id]<baseline_scores[id] for id in src),
                "n_cells_worsened_shrink_vs_ridge":sum(
                    scores["shrink"][id]>baseline_scores[id] for id in src),
                "paired_per_cell": [
                    {"id":cid,"source":src[cid],
                     **{k:scores[k][cid] for k in scores}}
                    for cid in sorted(src,key=lambda x:int(x.rsplit("-",1)[1]))
                ]
            }
    return {"domain":group,"n_target_cells":len(cells),
            "source_sha":[{"id":c.name,"git_blob_sha":c.git_sha,"cycles":c.n_cycles}
                          for c in cells],
            "folds":[[cells[i].name for i in fold] for fold in folds],
            "source_predictions":source_rows,"records":records,
            "summary":summary,"archive_parity":audit}


def bootstrap_difference(domains, budget, policy, a="ridge", b="shrink",
                         seed=BOOTSTRAP_SEED, n_boot=10000):
    arrs = []
    for d in TARGETS:
        percell = domains[d]["summary"][str(budget)][policy]["paired_per_cell"]
        arrs.append(np.array([c[a]-c[b] for c in percell]))
    observed=float(np.mean([np.mean(arr) for arr in arrs]))
    rng=np.random.default_rng(seed)
    boot=np.zeros(n_boot)
    for arr in arrs:
        ids=rng.integers(0,len(arr),size=(n_boot,len(arr)))
        boot+=arr[ids].mean(axis=1)/len(arrs)
    return {"effect_old_minus_new_mAh":observed,
            "exploratory_ci95_mAh":np.quantile(boot,[.025,.975]).tolist(),
            "n_resamples":n_boot,
            "note":"exploratory target-cell stratified paired bootstrap; shared source and training folds dependent, prior post-selection on same data"}


def run(cache:Path, outdir:Path):
    frozen=json.loads((ROOT/"results/v8/rw_source8_model.json").read_text())
    if frozen["source_commit"] != v8.COMMIT:
        raise ValueError("Unexpected original commit")
    rw=[v8.get_cell("RW",i,cache) for i in range(1,9)]
    for c,ref in zip(rw,frozen["source_cells"]):
        if c.name+".csv" != ref["name"] or c.git_sha!=ref["sha"] or c.n_cycles!=ref["cycles"]:
            raise ValueError("RW source original provenance changed")
    scaler=v8.fit_scaler(rw)
    source=v8.ridge(rw,scaler,v8.SOURCE_ALPHA)
    if (not np.allclose(source,frozen["coefficients"],atol=1e-7,rtol=0)
        or not np.allclose(scaler[0],frozen["mean"],atol=1e-7,rtol=0)
        or not np.allclose(scaler[1],frozen["sd"],atol=1e-7,rtol=0)):
        raise AssertionError("Independent RW model does not reproduce frozen source")
    results={}
    outdir.mkdir(parents=True,exist_ok=True)
    for d in TARGETS:
        cells=[v8.get_cell(d,i,cache) for i in range(1,v8.COUNTS[d]+1)]
        result=evaluate_domain(cells,d,source,scaler)
        results[d]=result
        (outdir/f"rw_to_{d.replace('.', '_')}_loco.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(d, {n:{p:{k:round(v,4) for k,v in t["methods"].items()}
                            for p,t in s.items()}
                  for n,s in result["summary"].items()},flush=True)
    macro={}
    for n in BUDGETS:
        macro[str(n)]={}
        for p in POLICIES:
            records=[results[d]["summary"][str(n)][p] for d in TARGETS]
            macro[str(n)][p]={
                "five_domain_macro_mAh":{
                    k:float(np.mean([r["methods"][k] for r in records]))
                    for k in ("ridge","offset","shrink")},
                "source_only_macro_mAh":float(np.mean([r["source_only_mAh"] for r in records])),
                "n_test_cells_negative_vs_source":{
                    k:int(sum(r["n_cells_negative_vs_source"][k] for r in records))
                    for k in ("ridge","offset","shrink")},
                "mean_lambda_per_acquisition":float(np.mean([r["mean_lambda"] for r in records])),
                "paired_ridge_minus_shrink_bootstrap":bootstrap_difference(results,n,p)
            }
    output={
        "date":"2026-10-12",
        "status":"executed_development_set_not_external_confirmation",
        "source_commit":v8.COMMIT,
        "v12_protocol":"docs/V12_LOCO_SHRINKAGE_DEVELOPMENT_PROTOCOL.md",
        "source_weights_retrained":source.tolist(),
        "budgets":BUDGETS,"policies":POLICIES,
        "random_repeats":REPEATS,
        "source_cells":8,"target_cells":47,
        "n_source_and_target_cells":55,
        "target_fold_provenance":"v8 frozen 5-fold IDs",
        "lambda_cv":"cell-equal-weighted two-battery LOCO SSE minimizer clipped [0,1]",
        "sum_archive_parity_checks":sum(r["archive_parity"]["n_legacy_mae_checks"] for r in results.values()),
        "max_archive_mae_abs_diff_mAh":max(r["archive_parity"]["max_legacy_mae_difference_mAh"] for r in results.values()),
        "macro":macro,
        "by_domain":{d:{k:v for k,v in results[d].items() if k in
                         ("n_target_cells","archive_parity","summary")} for d in TARGETS},
        "caveats":[
            "Developer-only reused 47 XJTU target cells; no preregistration on unseen data",
            "N10 selection uses first 12 unlabeled feature cycles; cannot retroactively measure early capacity unless recorded",
            "This is capacity MAE in mAh, not SOH percentage-point error"
        ]
    }
    (outdir/"summary_five_domains.json").write_text(
        json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V12 RUN COMPLETE:", json.dumps({
        "source_and_target_cells":55,
        "archive_score_comparisons":output["sum_archive_parity_checks"],
        "max_abs_archived_mae_delta_mAh":output["max_archive_mae_abs_diff_mAh"],
        "macro":macro
    },ensure_ascii=False),flush=True)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,default=ROOT/"data/external/xjtu_processed")
    ap.add_argument("--outdir",type=Path,default=ROOT/"results/v12")
    args=ap.parse_args()
    run(args.cache,args.outdir)


if __name__=="__main__":
    main()
