"""v14 unified source-anchored local kernel residual capacity calibration.

Exactly two predeclared kernels: six input features (RBF-X) and source-capacity
augmented features (RBF-XQ). New data-free, source-preserving correction:
 f_RW(x) + K(x,XL) (K(XL,XL)+rho*diag(1/w))^{-1}(YL-f_RW(XL))
Test capacities are used only for posthoc MAE, NEVER in model fitting.
v8/v10/v12 original per-cell results and selections are replayed and audited.
This is repeat analysis of previously explored XJTU development cells.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
V10PATH=ROOT/"experiments/rw_multitarget/fixed_budget_v10.py"
sp=importlib.util.spec_from_file_location("phm_v10_v14",V10PATH)
v10=importlib.util.module_from_spec(sp)
sys.modules[sp.name]=v10
sp.loader.exec_module(v10)
v8=v10.v8
DOMAINS=("2C","3C","R2.5","R3","Sim_satellite")
BUDGETS=(10,20,30)
POLICIES=("random","pred_quartiles")
RHO=0.05
REPEATS=5
TOL=1e-6


def space(x,scaler,source,mode):
    """Only frozen RW scaler & source coefficients; test labels unavailable."""
    z=v8.design(x,scaler)[:,1:]
    if mode=="RBF-X":
        return z/np.sqrt(6.)
    if mode=="RBF-XQ":
        projected=z@source[1:]/max(float(np.linalg.norm(source[1:])),1e-8)
        return np.column_stack((z,projected))/np.sqrt(7.)
    raise ValueError(mode)


def kernel(a,b):
    # RBF with fixed lengthscale=1 after source-only normalization.
    d2=np.maximum(np.sum(a*a,axis=1)[:,None]
                  +np.sum(b*b,axis=1)[None,:]-2*a@b.T,0.)
    return np.exp(-.5*d2)


def train_local(labeled,scaler,source,mode,rho=RHO):
    """Battery-balanced kernel Ridge residual model; no target-test inputs."""
    if len(labeled)!=2 or labeled[0].name==labeled[1].name:
        raise ValueError("Exactly two distinct labeled cells are required")
    if rho<=0: raise ValueError("Positive regularization is required")
    z=[]; residual=[]; diag=[]; support=[]
    for c in labeled:
        if c.n_cycles<1: raise ValueError("Empty training cell")
        x=v8.design(c.features,scaler)
        z.append(space(c.features,scaler,source,mode))
        residual.extend((c.capacity-x@source).tolist())
        # Each battery contributes 1/2 of the training risk.
        diag.extend([rho*2*c.n_cycles]*c.n_cycles)
        support.append({"id":c.name,"n_capacity_labels":c.n_cycles})
    points=np.vstack(z)
    k=kernel(points,points)
    alpha=np.linalg.solve(k+np.diag(diag),np.asarray(residual))
    return points,alpha,support


def predict(c,scaler,source,model,mode):
    x=v8.design(c.features,scaler)
    z=space(c.features,scaler,source,mode)
    points,alpha,_=model
    sim=kernel(z,points)
    correction=sim@alpha
    return x@source+correction, correction, np.max(sim,axis=1)


def score_local(test,scaler,source,model,mode):
    rows=[]; diagnostics=[]
    for c in test:
        pred,correction,sim=predict(c,scaler,source,model,mode)
        rows.append({"id":c.name,"mae_mAh":float(np.mean(np.abs(pred-c.capacity))*1000)})
        diagnostics.append({
            "id":c.name,
            "mean_abs_correction_mAh":float(np.abs(correction).mean()*1000),
            "max_abs_correction_mAh":float(np.abs(correction).max()*1000),
            "mean_max_kernel_similarity":float(sim.mean()),
            "min_max_kernel_similarity":float(sim.min()),
        })
    return rows,diagnostics


def validate_record(record,ref):
    if record["heldout"]!=ref["heldout"] or record["selected"]!=ref["selected"]:
        raise AssertionError("Original selection/heldout records differ")
    if record["n_labeled_cycles"]!=ref["n_labeled_cycles"]:
        raise AssertionError("N10/20/30 capacity budget differs")
    checks=0;maxdiff=0.
    for method in ("ridge","offset"):
        a={x["id"]:x["mae_mAh"] for x in record[method]}
        old={x["id"]:x["mae_mAh"] for x in ref[method]}
        if a.keys()!=old.keys(): raise AssertionError("Missing heldout cells")
        for id in a:
            err=abs(a[id]-old[id]);checks+=1;maxdiff=max(maxdiff,err)
            if err>TOL:raise AssertionError(f"v12 replay regression: {id} {method} {err}")
    return checks,maxdiff


def summarize(records,source_rows,budget,policy):
    src={x["id"]:x["mae_mAh"] for x in source_rows}
    out={"source":float(np.mean(list(src.values()))),"n_unique_test_cells":len(src)}
    selected=[r for r in records if r["budget"]==budget and r["policy"]==policy]
    expected=25 if policy=="random" else 5
    if len(selected)!=expected:
        raise AssertionError("Unexpected folds*repeats")
    by={}
    for method in ("ridge","offset","RBF-X","RBF-XQ"):
        cell={}
        for r in selected:
            for x in r[method]:
                cell.setdefault(x["id"],[]).append(x["mae_mAh"])
        if cell.keys()!=src.keys():raise AssertionError("Incomplete target test cells")
        counts=5 if policy=="random" else 1
        if any(len(scores)!=counts for scores in cell.values()):
            raise AssertionError("Repeat imbalance")
        by[method]={id:float(np.mean(v)) for id,v in cell.items()}
        out[method]=float(np.mean(list(by[method].values())))
        out[f"n_negative_transfer_{method}"]=int(
            sum(value>src[id] for id,value in by[method].items()))
    out["n_acquisitions"]=len(selected)
    out["n_labels_per_acquisition"]=budget*2
    out["per_cell"]=[
        {"id":id,"source_mAh":float(src[id]),
         **{f"{method}_mAh":by[method][id] for method in by}}
        for id in sorted(src,key=lambda s:int(s.rsplit("-",1)[1]))
    ]
    return out


def domain_run(cells,domain,scaler,source,runs_archived):
    cells=sorted(cells,key=lambda c:int(c.name.rsplit("-",1)[1]))
    folds=v8.make_folds(len(cells),domain)
    archive_folds=runs_archived["folds"]
    if [[cells[i].name for i in fold] for fold in folds]!=archive_folds:
        raise AssertionError("v12 heldout fold list changed")
    archived_sha={x["id"]:(x["git_blob_sha"],x["cycles"])
                  for x in runs_archived["source_sha"]}
    for c in cells:
        if (c.git_sha,c.n_cycles)!=archived_sha[c.name]:
            raise AssertionError("Original target CSV provenance mismatch: "+c.name)
    archived={(r["fold"],r["repeat"],r["policy"],r["budget"]):r
              for r in runs_archived["records"]}
    records=[];source_rows=[];max_diff=0.;checks=0
    for f,held_ids in enumerate(folds):
        held=set(held_ids)
        test=[c for i,c in enumerate(cells) if i in held]
        pool=[c for i,c in enumerate(cells) if i not in held]
        source_rows.extend(v8.score(test,scaler,source))
        for policy in POLICIES:
            for rep in range(REPEATS if policy=="random" else 1):
                chosen=v10.acquire(pool,scaler,source,policy,f,rep)
                for n in BUDGETS:
                    labeled=v10.label_prefix(pool,chosen,n)
                    if len(labeled)!=2 or any(c.n_cycles!=n for c in labeled):
                        raise AssertionError("Label budget changed")
                    names=[c.name for c in labeled]
                    if set(names)&{c.name for c in test}:
                        raise AssertionError("Heldout capacity leaked into labeled pool")
                    correction=v8.ridge(labeled,scaler,v8.RESIDUAL_ALPHA,base=source)
                    offset=float(np.mean([
                        np.mean(c.capacity-v8.design(c.features,scaler)@source)
                        for c in labeled
                    ]))
                    row={
                        "fold":f,"repeat":rep,"policy":policy,"budget":n,
                        "selected":names,"n_labeled_cycles":[n,n],
                        "heldout":[c.name for c in test],
                        "ridge":v8.score(test,scaler,source,correction=correction),
                        "offset":v8.score(test,scaler,source,offset=offset)
                    }
                    old=archived[(f,rep,policy,n)]
                    add_checks,diff=validate_record(row,old)
                    checks+=add_checks;max_diff=max(max_diff,diff)
                    for mode in ("RBF-X","RBF-XQ"):
                        model=train_local(labeled,scaler,source,mode)
                        score,diag=score_local(test,scaler,source,model,mode)
                        row[mode]=score
                        row[f"{mode}_diagnostics"]=diag
                    records.append(row)
    src_orig={x["id"]:x["mae_mAh"] for x in runs_archived["source_predictions"]}
    if set(src_orig)!={x["id"] for x in source_rows}:
        raise AssertionError("Source test cell identities changed")
    for r in source_rows:
        err=abs(r["mae_mAh"]-src_orig[r["id"]])
        checks+=1;max_diff=max(max_diff,err)
        if err>TOL: raise AssertionError("Source-only prediction changed")
    summaries={}
    for n in BUDGETS:
        summaries[str(n)]={}
        for policy in POLICIES:
            s=summarize(records,source_rows,n,policy)
            old=runs_archived["summary"][str(n)][policy]
            for key,refkey in (("source","source_only_mAh"),
                               ("ridge",None),("offset",None)):
                oldval=(old[refkey] if refkey else
                        old["methods"]["ridge" if key=="ridge" else "offset"])
                if abs(s[key]-oldval)>TOL:
                    raise AssertionError(f"Historical macro mismatch {domain}:{n}:{policy}:{key}")
            summaries[str(n)][policy]=s
    return {
        "domain":domain,"n_target_cells":len(cells),"folds":archive_folds,
        "source_sha":runs_archived["source_sha"],"source_predictions":source_rows,
        "summary":summaries,"records":records,
        "audit":{"n_archived_legacy_per_cell_replay_comparisons":checks,
                 "max_mae_abs_diff_mAh":max_diff,"n_failures":0}
    }


def run(cache,outdir):
    frozen=json.loads((ROOT/"results/v8/rw_source8_model.json").read_text())
    if frozen["source_commit"]!=v8.COMMIT:
        raise ValueError("Wrong pinned upstream author commit")
    rw=[v8.get_cell("RW",i,cache) for i in range(1,9)]
    for c,ref in zip(rw,frozen["source_cells"]):
        if c.name+".csv"!=ref["name"] or c.git_sha!=ref["sha"] or c.n_cycles!=ref["cycles"]:
            raise AssertionError("RW source raw data changed")
    scaler=v8.fit_scaler(rw);source=v8.ridge(rw,scaler,v8.SOURCE_ALPHA)
    for a,b in ((source,frozen["coefficients"]),(scaler[0],frozen["mean"]),(scaler[1],frozen["sd"])):
        if not np.allclose(a,b,rtol=0,atol=1e-7):
            raise AssertionError("Source preprocessing/model changed")
    outdir.mkdir(parents=True,exist_ok=True)
    results={}
    for d in DOMAINS:
        saved=json.loads((ROOT/"results/v12"/f"rw_to_{d.replace('.','_')}_loco.json").read_text())
        target=[v8.get_cell(d,i,cache) for i in range(1,v8.COUNTS[d]+1)]
        res=domain_run(target,d,scaler,source,saved)
        results[d]=res
        (outdir/f"rw_to_{d.replace('.','_')}_kernel.json").write_text(
            json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
        print("DOMAIN",d, {n:{p:{k:round(v,3) for k,v in s.items() if k in
                                      ("source","ridge","offset","RBF-X","RBF-XQ")}
                           for p,s in block.items()}
                           for n,block in res["summary"].items()},flush=True)
    macro={}
    for n in BUDGETS:
        macro[str(n)]={}
        for p in POLICIES:
            subset=[results[d]["summary"][str(n)][p] for d in DOMAINS]
            macro[str(n)][p]={
                "five_domain_equal_weight_mae_mAh":{
                    method:float(np.mean([s[method] for s in subset]))
                    for method in ("source","ridge","offset","RBF-X","RBF-XQ")},
                "negative_transfer_cell_counts":{
                    method:int(sum(s[f"n_negative_transfer_{method}"] for s in subset))
                    for method in ("ridge","offset","RBF-X","RBF-XQ")},
                "n_unique_target_test_cells":47,
                "n_labeled_capacity_truths_per_acquisition":2*n,
            }
    out={
        "status":"development_set_test_NOT_independent_confirmation",
        "upstream_commit":v8.COMMIT,"date":"2026-10-12",
        "protocol":"docs/V14_SOURCE_ANCHORED_LOCAL_KERNEL_PROTOCOL_20261012.md",
        "source_model":"results/v8/rw_source8_model.json",
        "features":list(v8.FEATURES),"budgets":list(BUDGETS),"policies":list(POLICIES),
        "rho":RHO,"kernel_modes":["RBF-X","RBF-XQ"],
        "kernel_rule":"RBF(distance*distance/2) with source-only standardized z; XQ includes normalized source predicted output",
        "no_external_validation":True,
        "n_archived_replayed_cell_scores":sum(r["audit"]["n_archived_legacy_per_cell_replay_comparisons"] for r in results.values()),
        "max_replay_abs_diff_mAh":max(r["audit"]["max_mae_abs_diff_mAh"] for r in results.values()),
        "macro":macro,
        "by_domain":{d:{"n_target_cells":results[d]["n_target_cells"],"summary":results[d]["summary"],
                        "audit":results[d]["audit"]}
                     for d in DOMAINS}
    }
    (outdir/"summary_five_domains.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print("COMPLETE",json.dumps({
        "n_source_and_target_cells":55,
        "archived_replay_checks":out["n_archived_replayed_cell_scores"],
        "max_replay_diff_mAh":out["max_replay_abs_diff_mAh"],
        "macro":macro
    },ensure_ascii=False),flush=True)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,default=ROOT/"data/external/xjtu_processed")
    ap.add_argument("--outdir",type=Path,default=ROOT/"results/v14")
    args=ap.parse_args()
    run(args.cache,args.outdir)


if __name__=="__main__":
    main()
