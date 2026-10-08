"""Simple label-only SOH calibration baselines on exactly the few-shot splits in run.py.

These calibrate ONLY the scalar SOH, not the electrochemical state or the full voltage curve.
Do not interpret improved SOH as evidence of preserved physical interpretability.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import load_model,split_data


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--cache',type=Path,required=True)
    pa.add_argument('--assets',type=Path,required=True)
    pa.add_argument('--out',type=Path,required=True)
    pa.add_argument('--repeats',type=int,default=3)
    pa.add_argument('--shots',default='4,14')
    args=pa.parse_args()
    torch.set_num_threads(1)
    d=np.load(args.cache,allow_pickle=False)
    x=torch.tensor(d['X'],dtype=torch.float32)
    y=torch.tensor(d['Y5'],dtype=torch.float32)
    m=load_model(args.assets)
    with torch.no_grad():p=m(x)[0][:,-1,1].numpy()*100
    ytrue=y[:,-1,1].numpy()*100
    rows=[]
    for fold in range(5):
        for k in map(int,args.shots.split(',')):
            for rep in range(args.repeats):
                tr,te,lab,unlab=split_data(len(x),fold,k,rep)
                # Baseline A: add labeled mean residual to every target test prediction.
                mean_bias=(ytrue[lab]-p[lab]).mean()
                # Baseline B: ridge-regularized affine correction; x anchored at label mean.
                z_lab=np.column_stack([p[lab]-p[lab].mean(),np.ones(len(lab))])
                resid_lab=ytrue[lab]-p[lab]
                reg=np.diag([100.0,1.0])   # pre-specified small-sample shrinkage of slope
                beta=np.linalg.solve(z_lab.T@z_lab+reg,z_lab.T@resid_lab)
                z_test=np.column_stack([p[te]-p[lab].mean(),np.ones(len(te))])
                for mode,yp in [('source_zero',p[te]),('bias_only',p[te]+mean_bias),('ridge_affine',p[te]+z_test@beta)]:
                    err=(yp-ytrue[te])
                    rows.append(dict(fold=fold,repeat=rep,k=k,mode=mode,n_test=len(te),
                        soh_mae_pp=float(np.abs(err).mean()),soh_rmse_pp=float(np.mean(err**2)**0.5),
                        label_ids=';'.join(map(str,lab)),test_ids=';'.join(map(str,te))))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(args.out,index=False)
    print(df.groupby(['k','mode']).agg(mean_mae=('soh_mae_pp','mean'),std=('soh_mae_pp','std')).to_string())

if __name__=='__main__':main()