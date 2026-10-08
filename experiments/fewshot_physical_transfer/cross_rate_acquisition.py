"""Predefined six-rate extension of fixed low-label acquisition+calibration.

No rate-specific tuning, hyperparameter or label selection based on test labels.
Only the 1C fragment is available when acquiring labels. Every rate shares
exact same selected target cells but uses that rate's k labeled target curves.
"""
import argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from run import load_model,split_data
from label_acquisition import choose_kmeans
from physical_calibration import regression

RATES=['C/80','C/40','0.05A','C/10','C/7','C/5']

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--cache',type=Path,required=True)
 ap.add_argument('--assets',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--alpha',type=float,default=10.0)
 ap.add_argument('--alpha-selection',choices=['fixed','loo'],default='fixed')
 args=ap.parse_args()
 torch.set_num_threads(1)
 dat=np.load(args.cache,allow_pickle=False)
 X=torch.tensor(dat['X'],dtype=torch.float32)
 model=load_model(args.assets)
 with torch.no_grad():p,d,lat,sp,sn=model(X)
 src=p[:,-1,1].numpy()*100
 z=np.concatenate([l.numpy() for l in lat],axis=1)
 residual=np.mean(np.abs(p[:,1:,0].numpy()-d[:,1:].numpy()),axis=1)*4200
 features=np.column_stack([src,z,residual])
 input_arr=X.numpy()
 coarse_input=np.column_stack([input_arr[:,0,0],input_arr[:,-1,0],input_arr[:,-1,1],input_arr[:,:,0].mean(1),input_arr[:,:,0].std(1),input_arr[:,:,1].mean(1)])
 policies={'random':None,'physics_clustering':features,'source_soh_clustering':src[:,None],'input_clustering':coarse_input}
 rows=[]
 for fold in range(5):
  for k in [4,14]:
   for rep in range(3):
    tr,te,lab_random,_=split_data(len(X),fold,k,rep)
    choices={key:(lab_random if features_i is None else choose_kmeans(tr,features_i,k,rep)) for key,features_i in policies.items()}
    for rindex,ratename in enumerate(RATES):
     truth=dat[f'Y{rindex}'][:,-1,1]*100
     for policy,chosen in choices.items():
      alpha=args.alpha
      if args.alpha_selection=='loo':
       # Inner model selection sees ONLY k annotated training cells;
       # test labels remain strictly untouched.
       scores=[]
       for a in (0.1,1.,10.,100.):
        e=[]
        for idx in range(len(chosen)):
         held=chosen[idx:idx+1]
         ins=np.concatenate([chosen[:idx],chosen[idx+1:]])
         pred=regression(features,tr,ins,held,src,truth,alpha=a)
         e.append(abs(float(pred[0]-truth[held[0]])))
        scores.append(np.mean(e))
       alpha=(0.1,1.,10.,100.)[int(np.argmin(scores))]
      yhat=regression(features,tr,chosen,te,src,truth,alpha=alpha)
      errs=yhat-truth[te]
      rows.append(dict(fold=fold,repeat=rep,k=k,rate=ratename,rate_idx=rindex,policy=policy,alpha_used=alpha,
                      soh_mae_pp=float(np.abs(errs).mean()),soh_rmse_pp=float(np.mean(errs**2)**.5),
                      train_ids=';'.join(map(str,tr)),label_ids=';'.join(map(str,chosen)),test_ids=';'.join(map(str,te))))
 args.out.parent.mkdir(parents=True,exist_ok=True)
 df=pd.DataFrame(rows);df.to_csv(args.out,index=False)
 print(df.groupby(['k','policy']).soh_mae_pp.agg(['mean','std']).to_string())
 print(df.groupby(['rate','k','policy']).soh_mae_pp.mean().round(3).unstack('policy').to_string())

if __name__=='__main__':main()