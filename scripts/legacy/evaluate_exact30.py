import os,json
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
BASE='/mnt/data/che2025_official'; CODE=BASE+'/code'; OUT=BASE+'/exact30'; os.makedirs(OUT,exist_ok=True)
RATES=['C/80','C/40','0.05 A','C/10','C/7','C/5']
D=np.load(BASE+'/cache_dataset2.npz',allow_pickle=True); X=D['X']; files=D['files']
an=pd.read_csv(CODE+'/anode_SiO_Gr_discharge_Cover5_smoothed_dvdq_JS.csv'); ca=pd.read_csv(CODE+'/cathode_NCA_discharge_Cover5_smoothed_dvdq_JS.csv')
ns=torch.tensor(an['SOC_linspace'].values,dtype=torch.float32);nv=torch.tensor(an['Voltage'].values,dtype=torch.float32);ps=torch.tensor(ca['SOC_linspace'].values,dtype=torch.float32);pv=torch.tensor(ca['Voltage'].values,dtype=torch.float32)
def ti(x,xp,fp):
 i=torch.searchsorted(xp,x).clamp(1,len(xp)-1);x0=xp[i-1];x1=xp[i];f0=fp[i-1];f1=fp[i];return f0+(f1-f0)*(x-x0)/(x1-x0)
class E(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(600,128);self.fc2=nn.Linear(128,128);self.fc3=nn.Linear(128,64);self.cp_out=nn.Linear(64,1);self.cn_out=nn.Linear(64,1);self.x0_out=nn.Linear(64,1);self.y0_out=nn.Linear(64,1)
 def forward(self,x):
  x=x.view(x.size(0),-1);x=torch.relu(self.fc1(x));x=torch.relu(self.fc2(x));x=torch.relu(self.fc3(x));return tuple(torch.relu(h(x)) for h in (self.cp_out,self.cn_out,self.x0_out,self.y0_out))
class Dec(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(4,128);self.fc2=nn.Linear(128,256);self.fcv=nn.Linear(256,200);self.fcq=nn.Linear(256,200)
 def forward(self,Cp,Cn,x0,y0):
  x=torch.cat((Cp,Cn,x0,y0),-1);x=torch.relu(self.fc1(x));x=torch.relu(self.fc2(x));return torch.stack((self.fcv(x),self.fcq(x)),-1)
class M(nn.Module):
 def __init__(self):super().__init__();self.encoder=E();self.decoder=Dec()
 def forward(self,x):
  Cp,Cn,x0,y0=self.encoder(x);p=self.decoder(Cp,Cn,x0,y0);sp=y0-p[:,:,1]/Cp;sn=x0-p[:,:,1]/Cn;d=(ti(sp,ps,pv)-ti(sn,ns,nv))/4.2;return p,d,Cp,Cn,x0,y0,sp,sn
def met(p,t):
 p=np.asarray(p).reshape(-1);t=np.asarray(t).reshape(-1);return {'rmse':float(mean_squared_error(t,p)**.5),'mae':float(mean_absolute_error(t,p)),'r2':float(r2_score(t,p))}
kf=list(KFold(n_splits=5,shuffle=True,random_state=0).split(X))
rate_rows=[]; allp=[]; alld=[]; ally=[]; all_states=[]
for r,rate in enumerate(RATES):
 Y=D[f'Y{r}']; ps_,ds_,ys_,sts=[] ,[],[],[]; foldrows=[]
 for f,(tr,te) in enumerate(kf):
  st=torch.load(f'{BASE}/resume_runs/r{r}_f{f}_state.pth',map_location='cpu',weights_only=False)
  m=M();m.load_state_dict(st['beststate']);m.eval()
  with torch.no_grad():p,d,Cp,Cn,x0,y0,sp,sn=m(torch.tensor(X[te],dtype=torch.float32))
  p=p.numpy();d=d.numpy();y=Y[te]; states=np.concatenate([Cp.numpy(),Cn.numpy(),x0.numpy(),y0.numpy()],axis=1)
  np.savez_compressed(f'{OUT}/r{r}_f{f}_pred.npz',pred=p,derived=d,true=y,states=states,test_indices=te,test_files=files[te])
  rr={'rate':rate,'rate_idx':r,'fold':f,'n':len(te),'epoch_reached':int(st['epoch']),'best_epoch':int(st['best_epoch']+1),'best_val_loss':float(st['best']),'stop_reason':'lr_floor' if st['opt']['param_groups'][0]['lr']<5e-7 else ('patience' if st['noimp']>=2000 else 'max_epoch'),
      'soh_pct':met(p[:,-1,1]*100,y[:,-1,1]*100),'q_mAh':met(p[:,:,1]*4840,y[:,:,1]*4840),'voltage_mV':met(p[:,1:,0]*4200,y[:,1:,0]*4200),'derived_voltage_mV':met(d[:,1:]*4200,y[:,1:,0]*4200)}
  with open(f'{OUT}/r{r}_f{f}.json','w') as fh:json.dump(rr,fh,indent=2)
  foldrows.append(rr);ps_.append(p);ds_.append(d);ys_.append(y);sts.append(states)
 p=np.concatenate(ps_);d=np.concatenate(ds_);y=np.concatenate(ys_);states=np.concatenate(sts)
 allp.append(p);alld.append(d);ally.append(y);all_states.append(states)
 mr={'rate':rate,'n':len(p),'SOH_MAE_pct':met(p[:,-1,1]*100,y[:,-1,1]*100)['mae'],'SOH_RMSE_pct':met(p[:,-1,1]*100,y[:,-1,1]*100)['rmse'],'SOH_R2':met(p[:,-1,1]*100,y[:,-1,1]*100)['r2'],'Q_MAE_mAh':met(p[:,:,1]*4840,y[:,:,1]*4840)['mae'],'Q_RMSE_mAh':met(p[:,:,1]*4840,y[:,:,1]*4840)['rmse'],'Q_R2':met(p[:,:,1]*4840,y[:,:,1]*4840)['r2'],'V_MAE_mV':met(p[:,1:,0]*4200,y[:,1:,0]*4200)['mae'],'V_RMSE_mV':met(p[:,1:,0]*4200,y[:,1:,0]*4200)['rmse'],'V_R2':met(p[:,1:,0]*4200,y[:,1:,0]*4200)['r2'],'DV_MAE_mV':met(d[:,1:]*4200,y[:,1:,0]*4200)['mae'],'DV_RMSE_mV':met(d[:,1:]*4200,y[:,1:,0]*4200)['rmse'],'DV_R2':met(d[:,1:]*4200,y[:,1:,0]*4200)['r2'],'mean_best_epoch':float(np.mean([x['best_epoch'] for x in foldrows])),'min_best_epoch':int(min(x['best_epoch'] for x in foldrows)),'max_best_epoch':int(max(x['best_epoch'] for x in foldrows))}
 rate_rows.append(mr)
p=np.concatenate(allp); d=np.concatenate(alld); y=np.concatenate(ally); states=np.concatenate(all_states)
overall={'n':len(p),'SOH_pct':met(p[:,-1,1]*100,y[:,-1,1]*100),'Q_mAh':met(p[:,:,1]*4840,y[:,:,1]*4840),'voltage_mV':met(p[:,1:,0]*4200,y[:,1:,0]*4200),'derived_voltage_mV':met(d[:,1:]*4200,y[:,1:,0]*4200),'paper_dataset2_aggregate':{'SOH_pct':{'rmse':2.54,'mae':1.86,'r2':0.890},'Q_mAh':{'rmse':68.2,'mae':43.1,'r2':0.997},'voltage_mV':{'rmse':27.4,'mae':22.4,'r2':0.996},'derived_voltage_mV':{'rmse':52.4,'mae':43.1,'r2':0.984}},'protocol':'author-equivalent 6 rates x 5 folds; source checkpoint init; seed=123; max 20k; patience 2000; LR floor 5e-7'}
pd.DataFrame(rate_rows).to_csv(OUT+'/ratewise_metrics.csv',index=False)
with open(OUT+'/aggregate_metrics.json','w') as fh:json.dump({'overall':overall,'ratewise':rate_rows},fh,indent=2)
np.savez_compressed(OUT+'/all_predictions.npz',pred=p,derived=d,true=y,states=states)
# plots
fig=plt.figure(figsize=(7,4)); xx=np.arange(6); plt.plot(xx,[r['SOH_MAE_pct'] for r in rate_rows],marker='o',label='Reproduction'); plt.axhline(1.86,ls='--',label='Paper aggregate'); plt.xticks(xx,RATES);plt.ylabel('SOH MAE [percentage points]');plt.xlabel('Target RPT rate');plt.legend();plt.tight_layout();fig.savefig(OUT+'/soh_mae_by_rate.png',dpi=180);plt.close(fig)
fig=plt.figure(figsize=(7,4));plt.plot(xx,[r['V_MAE_mV'] for r in rate_rows],marker='o',label='Predicted OCV');plt.plot(xx,[r['DV_MAE_mV'] for r in rate_rows],marker='o',label='Derived OCV');plt.axhline(22.4,ls='--',label='Paper pred-V aggregate');plt.axhline(43.1,ls=':',label='Paper derived-V aggregate');plt.xticks(xx,RATES);plt.ylabel('Voltage MAE [mV]');plt.xlabel('Target RPT rate');plt.legend();plt.tight_layout();fig.savefig(OUT+'/voltage_mae_by_rate.png',dpi=180);plt.close(fig)
print(pd.DataFrame(rate_rows).to_string(index=False));print('\nOVERALL\n',json.dumps(overall,indent=2))