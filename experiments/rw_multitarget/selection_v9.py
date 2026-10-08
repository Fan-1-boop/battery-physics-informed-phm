"""v9 RW-source target-battery acquisition comparisons.

Run with a local cache containing the original PINN4SOH XJTU processed CSVs.
Provenance: github.com/wang-fujin/PINN4SOH, fixed commit
cc3cc5053caf38f04e0665f7f88cb109144d035e.

Selection DOES NOT use capacity labels.  Evaluation labels are isolated by
the frozen v8 battery-level five folds.  Uses 2 acquisition batteries, with
capacity labels only from the first 25 percent of their lives.

Usage:
  python experiments/rw_multitarget/selection_v9.py \
    --cache data/external/xjtu_processed --out outputs/v9_selection.json
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

TARGETS={"2C":"rw_to_2C_fivefold.json",
         "3C":"rw_to_3C_fivefold.json",
         "R2.5":"rw_to_R2_5_fivefold.json",
         "R3":"rw_to_R3_fivefold.json",
         "Sim_satellite":"rw_to_Sim_satellite_fivefold.json"}
FEATURES=[0,4,5,8,12,13]

def read_cell(path):
    arr=np.loadtxt(path,delimiter=",",skiprows=1)
    if arr.ndim==1: arr=arr[None,:]
    arr=arr[np.isfinite(arr).all(axis=1)&(arr[:,16]>0)]
    if arr.shape[1]!=17 or not len(arr):raise ValueError(str(path))
    return {"x":arr[:,FEATURES],"y":arr[:,16]}

def design(x,src):
    mean=np.array(src["mean"]); sd=np.array(src["sd"])
    return np.column_stack((np.ones(len(x)),np.clip((x-mean)/sd,-6,6)))

def predict(x,src):
    return design(x,src)@np.array(src["coefficients"])

def descriptors(pool,src):
    z=[]; p=[]
    for c in pool:
        x=c["x"][:12]; z.append(design(x,src)[:,1:].mean(axis=0))
        p.append(float(predict(x,src).mean()))
    return np.asarray(z),np.asarray(p)

def choose(pool,src,policy):
    z,p=descriptors(pool,src)
    if len(pool)<2: raise ValueError("At least two acquisition cells required")
    if policy=="prediction_extremes":
        return [int(p.argmin()),int(p.argmax())]
    if policy=="prediction_median_extreme":
        med=float(np.sort(p)[len(p)//2])
        first=int(np.argmin(abs(p-med)))
        scores=abs(p-p[first]);scores[first]=-1
        return [first,int(scores.argmax())]
    if policy=="novelty_extremes":
        norms=np.linalg.norm(z,axis=1)
        return [int(norms.argmin()),int(norms.argmax())]
    raise ValueError(policy)

def fit_delta(labeled,src,alpha=10.):
    gram=np.zeros((7,7)); rhs=np.zeros(7)
    for c in labeled:
        x=design(c["x"],src)
        y=c["y"]-predict(c["x"],src)
        w=1/len(y)
        gram+=w*x.T@x
        rhs+=w*x.T@y
    gram[1:,1:]+=alpha*np.eye(6)
    return np.linalg.solve(gram,rhs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,default=Path("data/external/xjtu_processed"))
    ap.add_argument("--v8-dir",type=Path,default=Path("results/v8"))
    ap.add_argument("--out",type=Path,default=Path("outputs/v9_selection.json"))
    args=ap.parse_args()
    src=json.loads((args.v8_dir/"rw_source8_model.json").read_text())
    out={}
    for domain,file in TARGETS.items():
        reference=json.loads((args.v8_dir/file).read_text())
        cells={}
        for m in reference["target_cells"]:
            stem=m["id"]; data=read_cell(args.cache/(stem+".csv"))
            cells[stem]=data
        records=[]
        for fold,ids in enumerate(reference["fold_indices"]):
            test=set(ids); pool=[(k,v) for k,v in cells.items() if k not in test]
            for method in ("prediction_extremes","prediction_median_extreme","novelty_extremes"):
                selected=choose([v for _,v in pool],src,method)
                if len(set(selected))!=2:raise AssertionError("Duplicate selection")
                lab=[]
                for i in selected:
                    c=pool[i][1]; n=max(1,math.ceil(.25*len(c["y"])))
                    lab.append({"x":c["x"][:n],"y":c["y"][:n]})
                delta=fit_delta(lab,src)
                for name in ids:
                    c=cells[name]; yhat=predict(c["x"],src)+design(c["x"],src)@delta
                    records.append({"fold":fold,"method":method,"cell":name,
                          "mae_mAh":float(abs(yhat-c["y"]).mean()*1000),
                          "labeled":[pool[i][0] for i in selected]})
        summary={m:float(np.mean([r["mae_mAh"] for r in records if r["method"]==m]))
                 for m in ("prediction_extremes","prediction_median_extreme","novelty_extremes")}
        out[domain]={"summary":summary,"records":records}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v["summary"] for k,v in out.items()},indent=2))

if __name__=="__main__":
    main()
