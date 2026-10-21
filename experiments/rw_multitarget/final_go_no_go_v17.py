"""v17 final GO/NO-GO on 88 untouched TJU-3 / MIT additional cohorts.

Exactly the previously implemented original Ridge / offset, v14 RBF-X/XQ,
v15 intercell-disagreement kernels. No new algorithm, hyperparameter search,
external outcome tuning, or target test capacity use for fit/selector/scaling.
All source & target CSVs are Git-blob pinned. Capacities share a 2Ah-source
nominal-relative bridge; reports physical target mAh and nMAE (% nominal).
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PINN="wang-fujin/PINN4SOH"
SOURCE_NOMINAL_AH=2.
MANIFEST=ROOT/"docs/v17_final_unseen_manifest.json"
SEEDS={"TJU_MIX":20261701,"MIT_2017B":20261702,"MIT_2018A":20261703}
BUDGETS=(10,20,30)
POLICIES=("random","pred_quartiles")
METHODS=("source","offset","ridge","RBF-X","GL-KRR")
RANDOM_REPEATS=5
BOOT_SEED=20261707
N_BOOT=10000
FEATURE_NAMES=(
    "voltage mean","voltage std","voltage kurtosis","voltage skewness",
    "CC Q","CC charge time","voltage slope","voltage entropy",
    "current mean","current std","current kurtosis","current skewness",
    "CV Q","CV charge time","current slope","current entropy","capacity"
)

s=importlib.util.spec_from_file_location("v15_external_v16",ROOT/"experiments/rw_multitarget/disagreement_calibration_v15.py")
v15=importlib.util.module_from_spec(s)
sys.modules[s.name]=v15
s.loader.exec_module(v15)
v14=v15.v14
v10=v15.v10
v8=v15.v8


def git_sha(raw:bytes):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\x00"+raw).hexdigest()


def download_one(cohort,entry,nominal,cache,upstream):
    name=entry["name"]
    expected_path=entry["path"]
    if Path(expected_path).name!=name or any(x in name for x in ("..","/","\\")):
        raise ValueError(f"Malformed manifest path: {expected_path}")
    f=cache/cohort/name
    if f.exists():
        raw=f.read_bytes()
        if git_sha(raw)!=entry["git_blob_sha"]:
            raise AssertionError(f"Corrupted cached upstream blob: {cohort}/{name}")
    else:
        segments=[urllib.parse.quote(p,safe="") for p in expected_path.split("/")]
        uri=f"https://raw.githubusercontent.com/{PINN}/{upstream}/"+"/".join(segments)
        last_error=None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(uri,timeout=60) as r:
                    raw=r.read()
                if len(raw)>4_000_000:raise ValueError("Unexpectedly large raw CSV")
                break
            except Exception as e:
                last_error=e
                if attempt<2:time.sleep((attempt+1)*.3)
        else:
            raise RuntimeError(f"Unable to download {cohort}/{name}: {last_error}")
        if git_sha(raw)!=entry["git_blob_sha"] or len(raw)!=entry["size_bytes"]:
            raise AssertionError(f"Original official blob/hash/size mismatch: {cohort}/{name}")
        f.parent.mkdir(parents=True,exist_ok=True)
        f.write_bytes(raw)
    if len(raw)!=entry["size_bytes"] or git_sha(raw)!=entry["git_blob_sha"]:
        raise AssertionError(f"Data provenance failed: {name}")
    first=next(csv.reader(raw.decode("utf-8-sig").splitlines()))
    if tuple(first)!=FEATURE_NAMES:
        raise AssertionError(f"Not 17-column charging statistics compatible: {name}: {first}")
    # Parse using exactly the source XJTU filter and the original six features.
    cell=v8.load_csv(raw,name)
    if cell.git_sha!=entry["git_blob_sha"]:
        raise AssertionError("Parser produced different raw blob SHA")
    nameid=cohort+"__"+Path(name).stem
    pseudo_capacity=cell.capacity*(SOURCE_NOMINAL_AH/nominal)
    return v8.Cell(nameid,cohort,cell.features,pseudo_capacity,cell.git_sha),{
        "id":nameid,"file":expected_path,"nominal_Ah":nominal,
        "original_blob_sha":entry["git_blob_sha"],
        "n_valid_cycles":cell.n_cycles,
        "status":"eligible" if cell.n_cycles>=30 else "excluded_fewer_than_30_valid_cycles"}


def load_manifest(cache):
    data=json.loads(MANIFEST.read_text())
    if data["source_commit"]!=v8.COMMIT:
        raise AssertionError("Author fixed commit changed")
    if set(data["cohorts"])!=set(SEEDS):
        raise AssertionError("Unexpected cohort manifest")
    work=[]
    for cohort,manifest in data["cohorts"].items():
        for e in manifest["files"]:
            work.append((cohort,e,float(manifest["nominal_Ah"])))
        if len(manifest["files"])!=manifest["n_files_expected"]:
            raise AssertionError("File count mismatch")
    if len(work)!=88:
        raise AssertionError("v17 must validate exactly 88 raw official external CSVs")
    cells={d:[] for d in SEEDS};audit={d:[] for d in SEEDS}
    with ThreadPoolExecutor(max_workers=12) as exe:
        tasks=[exe.submit(download_one,d,e,q,cache,data["source_commit"]) for d,e,q in work]
        for done in as_completed(tasks):
            cell,entry=done.result()
            audit[cell.group].append(entry)
            if entry["status"]=="eligible":
                cells[cell.group].append(cell)
    for d in SEEDS:
        cells[d].sort(key=lambda c:c.name)
        audit[d].sort(key=lambda x:x["id"])
        if len(cells[d])<7:
            raise AssertionError(f"Cohort {d} has insufficient eligible cells")
        if len({c.name for c in cells[d]})!=len(cells[d]):
            raise AssertionError(f"Duplicate cell ID in {d}")
    return data,cells,audit


def scaled_metrics(pseudo_pred,pseudo_true,nominal):
    diff=np.asarray(pseudo_pred)-np.asarray(pseudo_true)
    if diff.size==0 or not np.isfinite(diff).all():
        raise ValueError("Invalid predictions")
    mae_scaled_2ah=float(np.mean(np.abs(diff)))
    return {
        "mae_mAh":mae_scaled_2ah*(nominal/SOURCE_NOMINAL_AH)*1000.,
        "nmae_nominal_pct":mae_scaled_2ah/SOURCE_NOMINAL_AH*100.
    }


def score_baselines(test,scaler,source,nominal,labeled=None):
    delta=None;offset=0.
    if labeled is not None:
        delta=v8.ridge(labeled,scaler,v8.RESIDUAL_ALPHA,base=source)
        offset=float(np.mean([np.mean(
            c.capacity-v8.design(c.features,scaler)@source) for c in labeled]))
    out={"source":[],"ridge":[],"offset":[]}
    for c in test:
        x=v8.design(c.features,scaler)
        pred=x@source
        out["source"].append({"id":c.name,**scaled_metrics(pred,c.capacity,nominal)})
        if delta is not None:
            out["ridge"].append({"id":c.name,**scaled_metrics(pred+x@delta,c.capacity,nominal)})
            out["offset"].append({"id":c.name,**scaled_metrics(pred+offset,c.capacity,nominal)})
    return out


def train_global_local(labeled,scaler,source):
    """Exactly one new fixed composite KRR K=1+RBF-X; no target test labels.

    The constant and RBF components share v14 rho=.05 and two-cell-equal
    weighting; there are NO tunable new kernel weights or special cohorts.
    """
    if len(labeled)!=2 or labeled[0].name==labeled[1].name:
        raise ValueError("Exactly two distinct labeled target cells are required")
    z=np.vstack([v14.space(c.features,scaler,source,"RBF-X") for c in labeled])
    residual=np.concatenate([
        c.capacity-v8.design(c.features,scaler)@source for c in labeled
    ])
    diag=np.concatenate([np.full(c.n_cycles,v14.RHO*2*c.n_cycles) for c in labeled])
    kernel_joint=v14.kernel(z,z)+1.
    alpha=np.linalg.solve(kernel_joint+np.diag(diag),residual)
    return z,alpha


def global_local_prediction(c,scaler,source,model):
    z=v14.space(c.features,scaler,source,"RBF-X")
    k=v14.kernel(z,model[0])
    correction=(1.+k)@model[1]
    return v8.design(c.features,scaler)@source+correction, correction


def eval_models(test,labeled,source,scaler,nominal):
    old=v14.train_local(labeled,scaler,source,"RBF-X")
    new=train_global_local(labeled,scaler,source)
    rows={"RBF-X":[],"GL-KRR":[]}
    diagnostics=[]
    for c in test:
        pred_x,corr_x,sim_x=v14.predict(c,scaler,source,old,"RBF-X")
        pred_new,corr_new=global_local_prediction(c,scaler,source,new)
        for method,pred in (("RBF-X",pred_x),("GL-KRR",pred_new)):
            rows[method].append({"id":c.name,**scaled_metrics(pred,c.capacity,nominal)})
        diagnostics.append({
            "id":c.name,
            "mean_max_kernel_similarity":float(sim_x.mean()),
            "mean_RBF_correction_scaled_mAh":float(np.abs(corr_x).mean()*1000),
            "mean_GL_correction_scaled_mAh":float(np.abs(corr_new).mean()*1000),
            "GL_global_term_scaled_mAh":float(np.sum(new[1])*1000),
        })
    return rows,diagnostics

def summarize(records,source_rows,n,policy):
    selected=[r for r in records if r["budget"]==n and r["policy"]==policy]
    if len(selected)!=(25 if policy=="random" else 5):
        raise AssertionError("Random repeat/fold count mismatch")
    source={x["id"]:x for x in source_rows}
    s={}
    for method in METHODS:
        if method=="source":
            by={id:[v] for id,v in source.items()}
        else:
            by=defaultdict(list)
            for r in selected:
                for x in r[method]:by[x["id"]].append(x)
        if by.keys()!=source.keys():
            raise AssertionError("Each heldout cell must be represented exactly")
        counts=1 if method=="source" or policy=="pred_quartiles" else 5
        if any(len(v)!=counts for v in by.values()):
            raise AssertionError("Pseudo replication / incomplete random selections")
        s[method]={
            cid:{k:float(np.mean([z[k] for z in vv])) for k in ("mae_mAh","nmae_nominal_pct")}
            for cid,vv in by.items()}
    out={"n_labeled_capacity_observations":2*n,
         "n_target_test_cells":len(source),
         "n_acquisitions":len(selected),
         "metrics":{method:{
             "mean_capacity_MAE_mAh":float(np.mean([row["mae_mAh"] for row in d.values()])),
             "mean_nominal_nMAE_pct":float(np.mean([row["nmae_nominal_pct"] for row in d.values()])),
             "n_negative_transfer_cells_vs_source":sum(
                 row["mae_mAh"]>s["source"][cid]["mae_mAh"] for cid,row in d.items())}
             for method,d in s.items()},
         "per_cell":[{
             "id":cid,
             **{method:s[method][cid] for method in METHODS}}
             for cid in sorted(source)]}
    return out


def run_cohort(domain,cells,nominal,scaler,source,audit):
    folds=v8.js_shuffle(range(len(cells)),SEEDS[domain])
    folds=[folds[i::5] for i in range(5)]
    if sorted(i for fold in folds for i in fold)!=list(range(len(cells))):
        raise AssertionError("Fivefold did not hold out every cell exactly once")
    records=[];source_rows=[]
    for fi,heldids in enumerate(folds):
        heldset=set(heldids)
        test=[c for j,c in enumerate(cells) if j in heldset]
        pool=[c for j,c in enumerate(cells) if j not in heldset]
        score=score_baselines(test,scaler,source,nominal)
        source_rows.extend(score["source"])
        for p in POLICIES:
            for rep in range(RANDOM_REPEATS if p=="random" else 1):
                acquired=v10.acquire(pool,scaler,source,p,fi,rep)
                if len(set(acquired))!=2:raise AssertionError("Two unique target cells required")
                for n in BUDGETS:
                    labeled=v10.label_prefix(pool,acquired,n)
                    if len(labeled)!=2 or any(c.n_cycles!=n for c in labeled):
                        raise AssertionError("Exact 2N early capacity labels not available")
                    if {c.name for c in labeled}&{c.name for c in test}:
                        raise AssertionError("Test capacity label leakage")
                    bas=score_baselines(test,scaler,source,nominal,labeled)
                    new,diag=eval_models(test,labeled,source,scaler,nominal)
                    records.append({
                        "fold":fi,"repeat":rep,"policy":p,"budget":n,
                        "selected":[c.name for c in labeled],
                        "heldout":[c.name for c in test],
                        "n_labeled_capacity_observations":2*n,
                        "offset":bas["offset"],"ridge":bas["ridge"],
                        **new,"diagnostics":diag
                    })
    summary={}
    for n in BUDGETS:
        summary[str(n)]={p:summarize(records,source_rows,n,p) for p in POLICIES}
    return {
        "cohort":domain,"source_Rw_nominal_Ah":SOURCE_NOMINAL_AH,
        "target_nominal_Ah":nominal,
        "n_target_files":len(audit),
        "n_eligible_target_cells":len(cells),
        "excluded": [x for x in audit if x["status"]!="eligible"],
        "file_provenance":audit,
        "folds":[[cells[i].name for i in fold] for fold in folds],
        "source_rows":source_rows,"summary":summary,"records":records
    }


def paired_bootstrap(cohort_results,n=30,policy="pred_quartiles",seed=BOOT_SEED):
    rng=np.random.default_rng(seed)
    pairs=(("ridge","GL-KRR"),("RBF-X","GL-KRR"))
    out={}
    for before,after in pairs:
        arrays=[]
        for cohort in SEEDS:
            rows=cohort_results[cohort]["summary"][str(n)][policy]["per_cell"]
            arrays.append(np.array([row[before]["nmae_nominal_pct"]-
                                    row[after]["nmae_nominal_pct"] for row in rows]))
        obs=float(np.mean([float(np.mean(a)) for a in arrays]))
        boots=np.zeros(N_BOOT)
        for a in arrays:
            idx=rng.integers(0,len(a),size=(N_BOOT,len(a)))
            boots+=a[idx].mean(axis=1)/len(arrays)
        out[before+"_minus_"+after]={
            "cohort_equal_effect_nominal_nMAE_pp":obs,
            "exploratory_cell_bootstrap_ci95_pp":np.quantile(boots,[.025,.975]).tolist(),
            "n_boot":N_BOOT,"seed":seed,
            "limitations":"Previously fixed XJTU methods; 4 external cohorts, same source and overlapping target train/test folds; cell bootstrap descriptive only"
        }
    return out


def selection_bootstrap(cohort_results, budget=30,seed=BOOT_SEED):
    """Fixed source-prediction quartile acquisition vs historical 5x random;
    both use exactly Ridge and the same true heldout cell IDs."""
    rng=np.random.default_rng(seed)
    arrays=[]
    for d in SEEDS:
        random_rows=cohort_results[d]["summary"][str(budget)]["random"]["per_cell"]
        quart_rows=cohort_results[d]["summary"][str(budget)]["pred_quartiles"]["per_cell"]
        ra={row["id"]:row["ridge"]["nmae_nominal_pct"] for row in random_rows}
        qa={row["id"]:row["ridge"]["nmae_nominal_pct"] for row in quart_rows}
        if ra.keys()!=qa.keys():raise AssertionError("Unmatched selection target cells")
        arrays.append(np.asarray([ra[c]-qa[c] for c in sorted(ra)]))
    observed=float(np.mean([np.mean(v) for v in arrays]))
    boots=np.zeros(N_BOOT)
    for a in arrays:
        ind=rng.integers(0,len(a),size=(N_BOOT,len(a)))
        boots+=a[ind].mean(axis=1)/len(arrays)
    return {
       "effect_random_ridge_minus_quartile_ridge_nominal_pp":observed,
       "exploratory_cell_bootstrap_ci95_pp":np.quantile(boots,[.025,.975]).tolist(),
       "seed":seed,"n_boot":N_BOOT,
       "limitations":"Random has five dependent acquisitions averaged within each heldout cell; target training pools and source model overlap; this is descriptive not a valid independent preregistered test"
    }


def decide_direction(results,macro,calibration_boot,selection_boot):
    """Mechanically apply the thresholds written BEFORE accessing v17 targets."""
    primary=macro["30"]["pred_quartiles"]
    ma=primary["cohort_equal_nominal_nMAE_pct"]
    diff=float(ma["ridge"]-ma["GL-KRR"])
    ci=calibration_boot["ridge_minus_GL-KRR"]["exploratory_cell_bootstrap_ci95_pp"]
    cmp_cohort=[]
    for d in SEEDS:
        m=results[d]["summary"]["30"]["pred_quartiles"]["metrics"]
        cmp_cohort.append({
            "cohort":d,
            "ridge_pct":m["ridge"]["mean_nominal_nMAE_pct"],
            "global_local_pct":m["GL-KRR"]["mean_nominal_nMAE_pct"],
            "ridge_minus_GL_pp":m["ridge"]["mean_nominal_nMAE_pct"]-m["GL-KRR"]["mean_nominal_nMAE_pct"]
        })
    n_cohort_improved=sum(x["ridge_minus_GL_pp"]>=0 for x in cmp_cohort)
    n_cohort_harmful=sum(x["ridge_minus_GL_pp"]< -1 for x in cmp_cohort)
    negatives=primary["total_negative_transfer_test_cells_vs_source"]
    secondary=[]
    for n,p in ((20,"pred_quartiles"),(30,"random")):
        mm=macro[str(n)][p]["cohort_equal_nominal_nMAE_pct"]
        secondary.append(mm["GL-KRR"]-mm["ridge"]>0.5)
    cal={
        "gain_at_least_0_5_pp":diff>=.5,
        "paired_descriptive_ci_lower_above_zero":ci[0]>0,
        "at_least_two_of_three_cohorts_nmae_non_worse":n_cohort_improved>=2,
        "no_cohort_more_than_1_0_pp_worse":n_cohort_harmful==0,
        "negative_transfer_count_no_higher_than_ridge":negatives["GL-KRR"]<=negatives["ridge"],
        "not_both_N20_quartile_and_N30_random_excessively_worse":not all(secondary)
    }
    selectq=selection_boot["effect_random_ridge_minus_quartile_ridge_nominal_pp"]
    selectci=selection_boot["exploratory_cell_bootstrap_ci95_pp"]
    select_improved=0
    select_cohort={}
    for d in SEEDS:
        m=results[d]["summary"]["30"]
        rv=m["random"]["metrics"]["ridge"]["mean_nominal_nMAE_pct"]
        qv=m["pred_quartiles"]["metrics"]["ridge"]["mean_nominal_nMAE_pct"]
        select_cohort[d]={"random_ridge_pct":rv,"quartiles_ridge_pct":qv,"gain_pp":rv-qv}
        select_improved+=rv>=qv
    sel={
       "gain_at_least_0_2_pp":selectq>=.2,
       "at_least_two_of_three_cohorts_improved":select_improved>=2,
       "paired_descriptive_ci_lower_above_zero":selectci[0]>0
    }
    calibration_go=all(cal.values())
    selection_go=all(sel.values())
    verdict=("GO_UNIFIED_CALIBRATION" if calibration_go else
             "GO_SELECTION_ONLY_DROP_COMPLEX_CALIBRATION" if selection_go else
             "NO_GO_STOP_THIS_AS_MAIN_ALGORITHMIC_THESIS_DIRECTION")
    return {
       "binding_protocol":"docs/V17_FINAL_GO_NO_GO_PROTOCOL_20261021.md",
       "automatic_final_decision":verdict,
       "candidate_unified_calibration_GO":calibration_go,
       "representative_acquisition_only_GO":selection_go,
       "per_criterion_calibration":cal,
       "per_criterion_selection":sel,
       "observed_ridge_minus_new_macro_pp":diff,
       "observed_random_minus_quartile_ridge_macro_pp":selectq,
       "per_cohort_candidate":cmp_cohort,
       "per_cohort_selection":select_cohort,
       "interpretation_if_NO_GO":"Stop repeated gate/kernel method search and do not portray unified kernel or predicted quartile labeling as a verified novel master-thesis method; archival cross-domain comparison retains value."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=ROOT/"data/external/v17_processed")
    ap.add_argument("--outdir",type=Path,default=ROOT/"results/v17")
    args=ap.parse_args()
    frozen=json.loads((ROOT/"results/v8/rw_source8_model.json").read_text())
    if frozen["source_commit"]!=v8.COMMIT or frozen["feature_indices"]!=list(v8.FEATURES):
        raise AssertionError("Frozen source commit/indices changed")
    rw=[v8.get_cell("RW",i,args.cache/"rw_source") for i in range(1,9)]
    for c,ref in zip(rw,frozen["source_cells"]):
        if c.name+".csv"!=ref["name"] or c.git_sha!=ref["sha"] or c.n_cycles!=ref["cycles"]:
            raise AssertionError("Source raw CSV failed original provenance")
    scaler=v8.fit_scaler(rw)
    source=v8.ridge(rw,scaler,v8.SOURCE_ALPHA)
    for a,b in ((scaler[0],frozen["mean"]),(scaler[1],frozen["sd"]),(source,frozen["coefficients"])):
        if not np.allclose(a,b,atol=1e-7,rtol=0):
            raise AssertionError("RW source-only standardized model changed")
    manifest,bycohort,rawaudit=load_manifest(args.cache)
    args.outdir.mkdir(parents=True,exist_ok=True)
    results={}
    for d in SEEDS:
        qnom=manifest["cohorts"][d]["nominal_Ah"]
        outcome=run_cohort(d,bycohort[d],qnom,scaler,source,rawaudit[d])
        results[d]=outcome
        (args.outdir/f"target_{d}.json").write_text(
            json.dumps(outcome,ensure_ascii=False,indent=2),encoding="utf-8")
        print("COHORT",d,"eligible",len(bycohort[d]),
              "excluded",len(outcome["excluded"]),
              "N30 quartiles",{
                  method:round(val["mean_nominal_nMAE_pct"],3)
                  for method,val in outcome["summary"]["30"]["pred_quartiles"]["metrics"].items()
              },flush=True)
    macro={}
    for n in BUDGETS:
        macro[str(n)]={}
        for pol in POLICIES:
            parts=[results[d]["summary"][str(n)][pol] for d in SEEDS]
            macro[str(n)][pol]={
                "cohort_equal_nominal_nMAE_pct":{
                    method:float(np.mean([s["metrics"][method]["mean_nominal_nMAE_pct"] for s in parts]))
                    for method in METHODS},
                "total_negative_transfer_test_cells_vs_source":{
                    method:int(sum(s["metrics"][method]["n_negative_transfer_cells_vs_source"] for s in parts))
                    for method in METHODS if method!="source"},
                "n_eligible_external_cells":sum(s["n_target_test_cells"] for s in parts),
                "n_target_capacity_truths_per_fold":2*n
            }
    final={
        "date":"2026-10-21",
        "protocol":"docs/V17_FINAL_GO_NO_GO_PROTOCOL_20261021.md",
        "manifest":"docs/v17_final_unseen_manifest.json",
        "status":"v17_predeclared_go_no_go_on_new_external_battery_cohorts",
        "upstream_commit":v8.COMMIT,
        "source_nominal_Ah":SOURCE_NOMINAL_AH,
        "bridge":"source-scaled target pseudo-capacity = Qtarget_Ah *2.0/Qnominal, evaluate actual physical mAh and nominal nMAE %",
        "source_model_coefficients":source.tolist(),
        "source_model_scaler_mean":scaler[0].tolist(),
        "source_model_scaler_sd":scaler[1].tolist(),
        "n_frozen_raw_external_files":sum(len(v["files"]) for v in manifest["cohorts"].values()),
        "n_eligible_external_cells":sum(len(v) for v in bycohort.values()),
        "n_excluded_external_cells":sum(len(v["excluded"]) for v in results.values()),
        "cohorts":{d:{
            "n_target_files":v["n_target_files"],
            "nominal_Ah":v["target_nominal_Ah"],
            "n_eligible":v["n_eligible_target_cells"],
            "excluded":v["excluded"],
            "summary":v["summary"]} for d,v in results.items()},
        "macro":macro,
        "bootstrap_primary":paired_bootstrap(results),
        "bootstrap_acquisition":selection_bootstrap(results),
        "method_names":METHODS,
        "caveats":[
            "All target labels except two early labeled training cells strictly excluded from model fitting; full target test truths are for final scoring only",
            "Methods selected during preceding XJTU experiments, external cohorts not used for fitting/tuning hyperparameters",
            "All external CSVs are originally processed by PINN4SOH, with different nominal sizes and likely chemical/charging domains",
            "Cohort-equal nominal nMAE percent is primary macro; raw mAh comparability differs by nominal capacity",
            "N10 selected using 12 candidate charging cycles requires early label trace preservation"
        ]
    }
    final["binding_go_no_go_decision"]=decide_direction(results,macro,final["bootstrap_primary"],final["bootstrap_acquisition"])
    (args.outdir/"summary_external_cohorts.json").write_text(
        json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V17 FINAL DECISION",json.dumps({
        "n_files":final["n_frozen_raw_external_files"],
        "n_valid":final["n_eligible_external_cells"],
        "n_excluded":final["n_excluded_external_cells"],
        "main_metrics":final["macro"]["30"]["pred_quartiles"],
        "bootstrap":final["bootstrap_primary"],
        "verdict":final["binding_go_no_go_decision"]
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
