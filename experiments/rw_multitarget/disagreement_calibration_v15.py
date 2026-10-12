"""v15: one continuous source-preserving correction with inter-cell disagreement.

Two single-labeled-cell RBF-X residual estimators produce a per-observation
empirical disagreement. The existing v14 joint RBF-X residual is smoothly
scaled by its signal-to-disagreement ratio; no switch, no test capacities in
any model or weight. Same 55-cell raw-data provenance and frozen v8-v14 splits.
Development dataset ONLY, not independent confirmation.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SP=importlib.util.spec_from_file_location(
    "phm_v14_under_v15", ROOT/"experiments/rw_multitarget/source_anchored_kernel_v14.py")
v14=importlib.util.module_from_spec(SP)
sys.modules[SP.name]=v14
SP.loader.exec_module(v14)
v8,v10=v14.v8,v14.v10
DOMAINS=v14.DOMAINS
BUDGETS=v14.BUDGETS
POLICIES=v14.POLICIES
REPEATS=v14.REPEATS
RHO=v14.RHO
TOL_MAH=1e-6
EPS_DEN=1e-20


def single_cell_kernel(c,scaler,source):
    """RBF-X trained on ONE cell with per-cycle weight=1/n_cycles."""
    if c.n_cycles < 1: raise ValueError("No labels")
    z=v14.space(c.features,scaler,source,"RBF-X")
    r=c.capacity-v8.design(c.features,scaler)@source
    k=v14.kernel(z,z)
    alpha=np.linalg.solve(k+np.eye(c.n_cycles)*RHO*c.n_cycles,r)
    return z,alpha


def disagreement_shrinkage(full_residual,one_residual,two_residual):
    """Pointwise empirical signal/disagreement shrinkage; no tuning."""
    a=np.asarray(full_residual,dtype=float)
    b=np.asarray(one_residual,dtype=float)
    c=np.asarray(two_residual,dtype=float)
    if a.shape!=b.shape or a.shape!=c.shape:
        raise ValueError("Shape mismatch")
    disagreement=(b-c)*.5
    den=a*a+disagreement*disagreement
    lam=np.divide(a*a,den,out=np.zeros_like(a),where=den>EPS_DEN)
    if not np.isfinite(lam).all():
        raise AssertionError("Nonfinite calibration weight")
    return lam,disagreement


def score_v15(test,scaler,source,full,single_1,single_2):
    all_scores=[];all_diagnostics=[];old_v14_scores=[]
    for c in test:
        x=v8.design(c.features,scaler)
        z=v14.space(c.features,scaler,source,"RBF-X")
        raw=v14.kernel(z,full[0])@full[1]
        g1=v14.kernel(z,single_1[0])@single_1[1]
        g2=v14.kernel(z,single_2[0])@single_2[1]
        lam,dis=disagreement_shrinkage(raw,g1,g2)
        old=x@source+raw
        new=x@source+lam*raw
        old_v14_scores.append({"id":c.name,"mae_mAh":float(np.mean(np.abs(old-c.capacity))*1000)})
        all_scores.append({"id":c.name,"mae_mAh":float(np.mean(np.abs(new-c.capacity))*1000)})
        all_diagnostics.append({
            "id":c.name,
            "mean_lambda":float(lam.mean()),
            "fraction_lambda_le_half":float(np.mean(lam<=.5)),
            "mean_empirical_disagreement_mAh":float(np.mean(np.abs(dis))*1000),
            "mean_abs_joint_correction_mAh":float(np.mean(np.abs(raw))*1000),
            "mean_abs_final_correction_mAh":float(np.mean(np.abs(lam*raw))*1000),
            "min_lambda":float(lam.min()),"max_lambda":float(lam.max())
        })
    return old_v14_scores,all_scores,all_diagnostics


def validate_v14_reference(record,archived):
    for key in ("fold","repeat","policy","budget","selected",
                "n_labeled_cycles","heldout"):
        if record[key]!=archived[key]:
            raise AssertionError(f"Frozen v14 split/acquisition mismatch: {key}")
    comparisons=0;maxdiff=0.
    for name in ("ridge","offset","RBF-X"):
        a={x["id"]:x["mae_mAh"] for x in record[name]}
        b={x["id"]:x["mae_mAh"] for x in archived[name]}
        if a.keys()!=b.keys():raise AssertionError(f"Unmatched IDs {name}")
        for id in a:
            delta=abs(a[id]-b[id]);comparisons+=1;maxdiff=max(delta,maxdiff)
            if delta>TOL_MAH:
                raise AssertionError(f"Changed historical {name} {id}: {delta}")
    return comparisons,maxdiff


def summarize(records,source_rows,budget,policy):
    src={x["id"]:x["mae_mAh"] for x in source_rows}
    data=[r for r in records if r["budget"]==budget and r["policy"]==policy]
    if len(data)!=(25 if policy=="random" else 5):
        raise AssertionError("Unexpected acquisitions")
    permethod={};out={"source_mAh":float(np.mean(list(src.values()))),
                       "n_unique_test_cells":len(src),"n_acquisitions":len(data),
                       "capacity_truth_labels_per_acquisition":2*budget}
    for m in ("ridge","offset","RBF-X","v15"):
        cells=defaultdict(list)
        for r in data:
            for row in r[m]:cells[row["id"]].append(row["mae_mAh"])
        if cells.keys()!=src.keys():raise AssertionError("Heldout target cell set changed")
        repeat_count=(5 if policy=="random" else 1)
        if any(len(v)!=repeat_count for v in cells.values()):
            raise AssertionError("Random repetitions are unbalanced")
        permethod[m]={name:float(np.mean(vals)) for name,vals in cells.items()}
        out[f"{m}_mAh"]=float(np.mean(list(permethod[m].values())))
        out[f"n_negative_vs_source_{m}"]=sum(permethod[m][id]>src[id] for id in src)
    diag=[v for r in data for v in r["v15_diagnostics"]]
    out["mean_lambda_across_test_evaluations"]=float(np.mean([x["mean_lambda"] for x in diag]))
    out["mean_abs_disagreement_mAh"]=float(np.mean([x["mean_empirical_disagreement_mAh"] for x in diag]))
    out["n_test_cell_better_than_v14"]=sum(permethod["v15"][id]<permethod["RBF-X"][id] for id in src)
    out["n_test_cell_worse_than_v14"]=sum(permethod["v15"][id]>permethod["RBF-X"][id] for id in src)
    out["per_cell"]=[
        {"id":id,"source_mAh":src[id],
         **{f"{m}_mAh":permethod[m][id] for m in permethod}}
         for id in sorted(src,key=lambda s:int(s.rsplit("-",1)[1]))]
    return out


def run(cache,outdir):
    frozen=json.loads((ROOT/"results/v8/rw_source8_model.json").read_text())
    rw=[v8.get_cell("RW",i,cache) for i in range(1,9)]
    if frozen["source_commit"]!=v8.COMMIT:raise AssertionError("Source commit changed")
    for c,ref in zip(rw,frozen["source_cells"]):
        if c.name+".csv"!=ref["name"] or c.git_sha!=ref["sha"] or c.n_cycles!=ref["cycles"]:
            raise AssertionError("Original author RW source cell changed")
    scaler=v8.fit_scaler(rw);source=v8.ridge(rw,scaler,v8.SOURCE_ALPHA)
    for v,reference in ((source,frozen["coefficients"]),
                        (scaler[0],frozen["mean"]),(scaler[1],frozen["sd"])):
        if not np.allclose(v,reference,atol=1e-7,rtol=0):
            raise AssertionError("Frozen source coefficients changed")
    outdir.mkdir(exist_ok=True,parents=True)
    results={};nchecks=0;maxdiff=0.
    for domain in DOMAINS:
        cells=[v8.get_cell(domain,i,cache) for i in range(1,v8.COUNTS[domain]+1)]
        cells=sorted(cells,key=lambda c:int(c.name.rsplit("-",1)[1]))
        old=json.loads((ROOT/"results/v14"/f"rw_to_{domain.replace('.','_')}_kernel.json").read_text())
        folds=v8.make_folds(len(cells),domain)
        if [[cells[i].name for i in f] for f in folds]!=old["folds"]:
            raise AssertionError("v14 fold partition changed")
        old_sha={x["id"]:(x["git_blob_sha"],x["cycles"]) for x in old["source_sha"]}
        for c in cells:
            if (c.git_sha,c.n_cycles)!=old_sha[c.name]:
                raise AssertionError("Raw target cell blob or rows differ: "+c.name)
        archived={(r["fold"],r["repeat"],r["policy"],r["budget"]):r for r in old["records"]}
        records=[];source_rows=[]
        for fold,held in enumerate(folds):
            hs=set(held)
            test=[c for i,c in enumerate(cells) if i in hs]
            pool=[c for i,c in enumerate(cells) if i not in hs]
            source_rows.extend(v8.score(test,scaler,source))
            for policy in POLICIES:
                for rep in range(REPEATS if policy=="random" else 1):
                    chosen=v10.acquire(pool,scaler,source,policy,fold,rep)
                    for budget in BUDGETS:
                        labeled=v10.label_prefix(pool,chosen,budget)
                        if len(labeled)!=2 or any(c.n_cycles!=budget for c in labeled):
                            raise AssertionError("Changed early capacity budget")
                        if set(c.name for c in labeled)&set(c.name for c in test):
                            raise AssertionError("Heldout test battery leaked")
                        correction=v8.ridge(labeled,scaler,v8.RESIDUAL_ALPHA,base=source)
                        offset=float(np.mean([
                            np.mean(c.capacity-v8.design(c.features,scaler)@source)
                            for c in labeled]))
                        full=v14.train_local(labeled,scaler,source,"RBF-X")
                        m1=single_cell_kernel(labeled[0],scaler,source)
                        m2=single_cell_kernel(labeled[1],scaler,source)
                        rbf,ours,diags=score_v15(test,scaler,source,full,m1,m2)
                        rec={"fold":fold,"repeat":rep,"policy":policy,"budget":budget,
                             "selected":[c.name for c in labeled],
                             "n_labeled_cycles":[c.n_cycles for c in labeled],
                             "heldout":[c.name for c in test],
                             "ridge":v8.score(test,scaler,source,correction=correction),
                             "offset":v8.score(test,scaler,source,offset=offset),
                             "RBF-X":rbf,"v15":ours,"v15_diagnostics":diags}
                        cnum,diff=validate_v14_reference(rec,archived[(fold,rep,policy,budget)])
                        nchecks+=cnum;maxdiff=max(maxdiff,diff)
                        records.append(rec)
        expected_src={x["id"]:x["mae_mAh"] for x in old["source_predictions"]}
        for x in source_rows:
            if x["id"] not in expected_src or abs(x["mae_mAh"]-expected_src[x["id"]])>TOL_MAH:
                raise AssertionError("Source replay mismatch")
            nchecks+=1
        summary={}
        for n in BUDGETS:
            summary[str(n)]={}
            for pol in POLICIES:
                val=summarize(records,source_rows,n,pol)
                prev=old["summary"][str(n)][pol]
                for label,oldkey in (("source_mAh","source"),("ridge_mAh","ridge"),
                                     ("offset_mAh","offset"),("RBF-X_mAh","RBF-X")):
                    if abs(val[label]-prev[oldkey])>TOL_MAH:
                        raise AssertionError("Old score average mismatch: "+label)
                summary[str(n)][pol]=val
        result={"domain":domain,"n_target_cells":len(cells),"folds":old["folds"],
                "source_sha":old["source_sha"],"source_predictions":source_rows,
                "summary":summary,"records":records}
        results[domain]=result
        (outdir/f"rw_to_{domain.replace('.','_')}_disagreement.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(domain,{n:{p:{m:round(s[m+"_mAh"],3) for m in ("ridge","RBF-X","v15")}
                           for p,s in d.items()} for n,d in summary.items()},flush=True)
    macro={}
    for n in BUDGETS:
        macro[str(n)]={}
        for policy in POLICIES:
            summaries=[results[d]["summary"][str(n)][policy] for d in DOMAINS]
            macro[str(n)][policy]={
                "five_domain_macro_mAh":{method:float(np.mean([s[method+"_mAh"] for s in summaries]))
                     for method in ("ridge","offset","RBF-X","v15")},
                "source_only_macro_mAh":float(np.mean([s["source_mAh"] for s in summaries])),
                "negative_transfer_cell_counts":{method:sum(s["n_negative_vs_source_"+method] for s in summaries)
                     for method in ("ridge","offset","RBF-X","v15")},
                "mean_lambda_across_domains":float(np.mean([s["mean_lambda_across_test_evaluations"] for s in summaries])),
                "n_unique_target_test_cells":47,
                "n_capacity_truths_per_acquisition":2*n}
    summary={"date":"2026-10-12",
             "status":"development_only_not_independent_validation",
             "protocol":"docs/V15_INTERCELL_DISAGREEMENT_CALIBRATION_PROTOCOL_20261012.md",
             "upstream_commit":v8.COMMIT,"source_cells":8,"target_cells":47,
             "rho":RHO,"correction_rule":"g_all^2/(g_all^2+((g1-g2)/2)^2) times g_all",
             "budgets":list(BUDGETS),"policies":list(POLICIES),
             "n_v14_archived_mae_replay_checks":nchecks,
             "max_v14_archived_mae_abs_diff_mAh":maxdiff,
             "macro":macro,
             "by_domain":{d:{"n_target_cells":r["n_target_cells"],"summary":r["summary"]}
                          for d,r in results.items()},
             "caveats":["Intercell disagreement is not a calibrated uncertainty interval",
                        "Same previously explored 47 target batteries, not independent confirmation",
                        "Random five selections per heldout cell are not five independent target cells",
                        "N10 label prefix selection requires first 12 unlabeled target cycles"]}
    (outdir/"summary_five_domains.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V15 COMPLETE",json.dumps({"checks":nchecks,"max_diff":maxdiff,"macro":macro},
                                    ensure_ascii=False),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=ROOT/"data/external/xjtu_processed")
    ap.add_argument("--outdir",type=Path,default=ROOT/"results/v15")
    args=ap.parse_args()
    run(args.cache,args.outdir)


if __name__=="__main__":main()
