"""Post-hoc (non-selective) test of physical consistency as SOH error proxy.

A physical agreement residual is not the same as independently verified
physical truth; use this diagnostic to assess whether it is a sound
confidence score before using it for validation-free stopping.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr,pearsonr
from run import load_model

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--cache',type=Path,required=True)
 ap.add_argument('--assets',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args()
 torch.set_num_threads(1)
 d=np.load(args.cache,allow_pickle=False)
 X=torch.tensor(d['X'],dtype=torch.float32)
 Y=np.asarray(d['Y5'],dtype=np.float32)
 m=load_model(args.assets)
 with torch.no_grad():
  pred,dv,lat,sp,sn=m(X)
 p=pred.numpy();v=dv.numpy()
 frame=pd.DataFrame(dict(cell_index=np.arange(len(X)),
  abs_soh_error_pp=np.abs(p[:,-1,1]-Y[:,-1,1])*100,
  discrepancy_mV=np.mean(np.abs(p[:,1:,0]-v[:,1:]),axis=1)*4200,
  derived_v_error_mV=np.mean(np.abs(v[:,1:]-Y[:,1:,0]),axis=1)*4200,
  cp_violation=[float(x) for x in ((lat[0].numpy().ravel()>.1)&(lat[0].numpy().ravel()<1.1))==False],
  sp_violation=np.mean((sp.numpy()<0)|(sp.numpy()>1),axis=1)))
 rho,pvalue=spearmanr(frame.abs_soh_error_pp,frame.discrepancy_mV)
 r,ppear=pearsonr(frame.abs_soh_error_pp,frame.discrepancy_mV)
 frame.to_csv(args.out.with_suffix('.csv'),index=False)
 summ=dict(n=int(len(frame)),spearman_rho=float(rho),spearman_p=float(pvalue),pearson_r=float(r),pearson_p=float(ppear),
           discrepancy_mean_mV=float(frame.discrepancy_mV.mean()),soh_MAE_pp=float(frame.abs_soh_error_pp.mean()))
 args.out.with_suffix('.json').write_text(json.dumps(summ,indent=2))
 print(json.dumps(summ,indent=2))

if __name__=='__main__':main()