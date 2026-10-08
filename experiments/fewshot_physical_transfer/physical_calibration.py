"""Ridge correction using model-implied physical latents; NO neural updates.

Exploratory feature-level test for whether electrochemical latent states
carry additive transfer information beyond scalar source SOH calibration.
All normalization fits ONLY the outer training pool, no test features.
Ridge penalty fixed BEFORE tests, target-test never used for selection.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from run import load_model,split_data


def regression(features,train_idx,label_idx,test_idx,src_soh,ytrue,alpha):
    # Standardize features only on the outer non-test training set (unlabeled allowed)
    ave=features[train_idx].mean(axis=0)
    scale=features[train_idx].std(axis=0).clip(min=1e-4)
    X=(features-ave)/scale
    M=np.column_stack([X[label_idx],np.ones(len(label_idx))])
    R=ytrue[label_idx]-src_soh[label_idx]
    # Ridge on standardized predictors, intercept is unpenalized
    reg=np.eye(M.shape[1])*alpha
    reg[-1,-1]=1e-6
    b=np.linalg.solve(M.T@M+reg,M.T@R)
    test=np.column_stack([X[test_idx],np.ones(len(test_idx))])
    return src_soh[test_idx]+test@b


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--alpha',type=float,default=10)
    ap.add_argument('--repeats',type=int,default=3)
    args=ap.parse_args()
    torch.set_num_threads(1)
    data=np.load(args.cache,allow_pickle=False)
    model=load_model(args.assets)
    x=torch.tensor(data['X'],dtype=torch.float32)
    ytrue=data['Y5'][:,-1,1]*100
    with torch.no_grad():
        p,d,lat,sp,sn=model(x)
    src=p[:,-1,1].numpy()*100
    z=np.concatenate([v.numpy() for v in lat],axis=1)
    incons=np.mean(np.abs(p[:,1:,0].numpy()-d[:,1:].numpy()),axis=1)*4200
    features={
      'soh_only_ridge10':src[:,None],
      'soh_physical4_ridge10':np.column_stack([src,z]),
      'soh_physical4_discrep_ridge10':np.column_stack([src,z,incons]),
    }
    rows=[]
    for fold in range(5):
      for k in [4,14]:
       for rep in range(args.repeats):
        tr,te,lab,_=split_data(len(x),fold,k,rep)
        for name,feat in features.items():
         prediction=regression(feat,tr,lab,te,src,ytrue,args.alpha)
         rows.append(dict(fold=fold,repeat=rep,k=k,mode=name,soh_mae_pp=float(np.mean(np.abs(prediction-ytrue[te]))),
                         soh_rmse_pp=float(np.mean((prediction-ytrue[te])**2)**.5),
                         n_test=len(te),label_ids=';'.join(map(str,lab)),test_ids=';'.join(map(str,te))))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.out,index=False)
    print(frame.groupby(['k','mode']).agg(mean_mae=('soh_mae_pp','mean'),std=('soh_mae_pp','std')).to_string())

if __name__=='__main__':main()