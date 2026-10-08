import json, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
CODE='/mnt/data/che2025_official/code'; CACHE='/mnt/data/che2025_official/cache_dataset2.npz'; OUT='/mnt/data/che2025_official/results'
import os; os.makedirs(OUT,exist_ok=True)
D=np.load(CACHE,allow_pickle=True);X=D['X'];Y=D['Y5'];files=D['files']
an=pd.read_csv(CODE+'/anode_SiO_Gr_discharge_Cover5_smoothed_dvdq_JS.csv');ca=pd.read_csv(CODE+'/cathode_NCA_discharge_Cover5_smoothed_dvdq_JS.csv')
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
  Cp,Cn,x0,y0=self.encoder(x);p=self.decoder(Cp,Cn,x0,y0);sp=y0-p[:,:,1]/Cp;sn=x0-p[:,:,1]/Cn;d=(ti(sp,ps,pv)-ti(sn,ns,nv))/4.2;return p,d,Cp,Cn,x0,y0

def metrics(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
 return {'rmse':float(mean_squared_error(b,a)**.5),'mae':float(mean_absolute_error(b,a)),'r2':float(r2_score(b,a))}
te=list(KFold(n_splits=5,shuffle=True,random_state=0).split(X))[-1][1]
res={}
preds={}
for label,fn in [('pretrained','best_model.pth'),('finetuned','best_model_fine_tune.pth')]:
 m=M();m.load_state_dict(torch.load(os.path.join(CODE,fn),map_location='cpu',weights_only=True));m.eval()
 with torch.no_grad():p,d,Cp,Cn,x0,y0=m(torch.tensor(X[te]))
 p=p.numpy();d=d.numpy(); y=Y[te]
 rr={
  'soh_pct':metrics(p[:,-1,1]*100,y[:,-1,1]*100),
  'q_mAh':metrics(p[:,:,1]*4840,y[:,:,1]*4840),
  'voltage_mV':metrics(p[:,1:,0]*4200,y[:,1:,0]*4200),
  'derived_voltage_mV':metrics(d[:,1:]*4200,y[:,1:,0]*4200),
 }
 res[label]=rr; preds[label]=(p,d)
res['test_n']=int(len(te));res['test_indices']=te.tolist();res['test_files']=[str(x) for x in files[te]];res['target_rate']='C/5';res['paper_dataset2_aggregate']={'soh_pct':{'rmse':2.54,'mae':1.86,'r2':0.890},'q_mAh':{'rmse':68.2,'mae':43.1,'r2':0.997},'voltage_mV':{'rmse':27.4,'mae':22.4,'r2':0.996},'derived_voltage_mV':{'rmse':52.4,'mae':43.1,'r2':0.984}}
with open(os.path.join(OUT,'dataset2_c5_finalfold_results.json'),'w') as f:json.dump(res,f,indent=2)
p,d=preds['finetuned']; y=Y[te]
np.savez_compressed(os.path.join(OUT,'dataset2_c5_finalfold_predictions.npz'),pred=p,derived=d,true=y,test_indices=te,test_files=files[te])
# capacity scatter
fig=plt.figure(figsize=(5,4));plt.scatter(y[:,-1,1]*4.84,p[:,-1,1]*4.84,s=22);lo=min(y[:,-1,1].min(),p[:,-1,1].min())*4.84;hi=max(y[:,-1,1].max(),p[:,-1,1].max())*4.84;plt.plot([lo,hi],[lo,hi]);plt.xlabel('Measured C/5 capacity [Ah]');plt.ylabel('Predicted capacity [Ah]');plt.tight_layout();fig.savefig(os.path.join(OUT,'c5_finalfold_capacity_scatter.png'),dpi=180);plt.close(fig)
# representative median-error OCV
errs=np.abs(p[:,-1,1]-y[:,-1,1]);idx=int(np.argsort(errs)[len(errs)//2]);
fig=plt.figure(figsize=(5,4));plt.plot(y[idx,:,1]*4.84,y[idx,:,0]*4.2,label='Measured');plt.plot(p[idx,:,1]*4.84,p[idx,:,0]*4.2,label='Predicted');plt.plot(p[idx,:,1]*4.84,d[idx,:]*4.2,label='Derived');plt.xlabel('Q [Ah]');plt.ylabel('Voltage [V]');plt.legend();plt.tight_layout();fig.savefig(os.path.join(OUT,'c5_finalfold_ocv_example.png'),dpi=180);plt.close(fig)
print(json.dumps(res,indent=2))