import os,sys,time,json,random
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
RATE=int(sys.argv[1]);FOLD=int(sys.argv[2]);CHUNK=int(sys.argv[3]) if len(sys.argv)>3 else 2000
BASE='/mnt/data/che2025_official';CODE=BASE+'/code';OUT=BASE+'/resume_runs';os.makedirs(OUT,exist_ok=True)
RUN=f'r{RATE}_f{FOLD}';STATE=f'{OUT}/{RUN}_state.pth';RATES=['C/80','C/40','0.05 A','C/10','C/7','C/5']
D=np.load(BASE+'/cache_dataset2.npz',allow_pickle=True);X=D['X'];Y=D[f'Y{RATE}'];files=D['files']
an=pd.read_csv(CODE+'/anode_SiO_Gr_discharge_Cover5_smoothed_dvdq_JS.csv');ca=pd.read_csv(CODE+'/cathode_NCA_discharge_Cover5_smoothed_dvdq_JS.csv')
ns=torch.tensor(an['SOC_linspace'].values,dtype=torch.float32);nv=torch.tensor(an['Voltage'].values,dtype=torch.float32);ps=torch.tensor(ca['SOC_linspace'].values,dtype=torch.float32);pv=torch.tensor(ca['Voltage'].values,dtype=torch.float32)
def ti(x,xp,fp):
 i=torch.searchsorted(xp,x).clamp(1,len(xp)-1);x0=xp[i-1];x1=xp[i];f0=fp[i-1];f1=fp[i];return f0+(f1-f0)*(x-x0)/(x1-x0)
class E(nn.Module):
 def __init__(self):
  super().__init__();self.fc1=nn.Linear(600,128);self.fc2=nn.Linear(128,128);self.fc3=nn.Linear(128,64);self.cp_out=nn.Linear(64,1);self.cn_out=nn.Linear(64,1);self.x0_out=nn.Linear(64,1);self.y0_out=nn.Linear(64,1)
  with torch.no_grad():
   for h in (self.cp_out,self.cn_out,self.x0_out,self.y0_out):h.weight.fill_(0);h.bias.fill_(.9)
 def forward(self,x):
  x=x.view(x.size(0),-1);x=torch.relu(self.fc1(x));x=torch.relu(self.fc2(x));x=torch.relu(self.fc3(x));return tuple(torch.relu(h(x)) for h in (self.cp_out,self.cn_out,self.x0_out,self.y0_out))
class Dec(nn.Module):
 def __init__(self):super().__init__();self.fc1=nn.Linear(4,128);self.fc2=nn.Linear(128,256);self.fcv=nn.Linear(256,200);self.fcq=nn.Linear(256,200)
 def forward(self,Cp,Cn,x0,y0):
  x=torch.cat((Cp,Cn,x0,y0),-1);x=torch.relu(self.fc1(x));x=torch.relu(self.fc2(x));return torch.stack((self.fcv(x),self.fcq(x)),-1)
class M(nn.Module):
 def __init__(self):super().__init__();self.encoder=E();self.decoder=Dec()
 def forward(self,x):
  Cp,Cn,x0,y0=self.encoder(x);p=self.decoder(Cp,Cn,x0,y0);sp=y0-p[:,:,1]/Cp;sn=x0-p[:,:,1]/Cn;d=(ti(sp,ps,pv)-ti(sn,ns,nv))/4.2;return p,Cp,Cn,x0,y0,d,sp,sn
def lossfun(m,x,y):
 p,Cp,Cn,x0,y0,d,sp,sn=m(x);reg=nn.MSELoss()(p[:,:,0],y[:,:,0])+nn.MSELoss()(p[:,:,1],y[:,:,1]);phy=nn.MSELoss()(d,p[:,:,0]);rng=(torch.mean(torch.relu(d-1))+torch.mean(torch.relu(2.7/4.2-d))+torch.mean(torch.relu(p[:,:,0]-1))+torch.mean(torch.relu(2.7/4.2-p[:,:,0]))+torch.mean(torch.relu(-p[:,:,1]))+torch.mean(torch.relu(p[:,0,1])));con=(torch.mean(torch.relu(Cp-1.1)+torch.relu(.1-Cp))+torch.mean(torch.relu(Cn-1.1)+torch.relu(.1-Cn))+rng+torch.mean(torch.relu(sp-1)+torch.relu(-sp))+torch.mean(torch.relu(sn-1)+torch.relu(-sn)));return reg+phy+.1*con
def setseed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def met(p,t):p=np.asarray(p).reshape(-1);t=np.asarray(t).reshape(-1);return {'rmse':float(mean_squared_error(t,p)**.5),'mae':float(mean_absolute_error(t,p)),'r2':float(r2_score(t,p))}
tridx,teidx=list(KFold(n_splits=5,shuffle=True,random_state=0).split(X))[FOLD]
setseed(123);Xt=torch.tensor(X[tridx],dtype=torch.float32);Yt=torch.tensor(Y[tridx],dtype=torch.float32);Xte=torch.tensor(X[teidx],dtype=torch.float32);Yte=torch.tensor(Y[teidx],dtype=torch.float32);perm=torch.randperm(len(Xt));Xt,Yt=Xt[perm],Yt[perm];n=int(.8*len(Xt));Xtr,Xv=Xt[:n],Xt[n:];Ytr,Yv=Yt[:n],Yt[n:]
setseed(123);m=M();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.95,patience=100)
if os.path.exists(STATE):
 st=torch.load(STATE,map_location='cpu',weights_only=False);m.load_state_dict(st['current']);opt.load_state_dict(st['opt']);sched.load_state_dict(st['sched']);best=st['best'];beststate=st['beststate'];best_epoch=st['best_epoch'];noimp=st['noimp'];start=st['epoch']
else:
 m.load_state_dict(torch.load(CODE+'/best_model.pth',map_location='cpu',weights_only=True));best=float('inf');beststate={k:v.detach().clone() for k,v in m.state_dict().items()};best_epoch=-1;noimp=0;start=0
end=min(start+CHUNK,20000);t0=time.time();stopped=False
for ep in range(start,end):
 m.train();loss=lossfun(m,Xtr,Ytr);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);m.eval();
 with torch.no_grad():vl=lossfun(m,Xv,Yv)
 opt.step();sched.step(vl);v=float(vl.item())
 if v<best:best=v;beststate={k:vv.detach().clone() for k,vv in m.state_dict().items()};best_epoch=ep;noimp=0
 else:noimp+=1
 if noimp>=2000 or opt.param_groups[0]['lr']<5e-7:stopped=True;end=ep+1;break
state={'epoch':end,'current':m.state_dict(),'opt':opt.state_dict(),'sched':sched.state_dict(),'best':best,'beststate':beststate,'best_epoch':best_epoch,'noimp':noimp,'stopped':stopped}
torch.save(state,STATE)
# evaluate current best at chunk boundary
m.load_state_dict(beststate);m.eval();
with torch.no_grad():p,Cp,Cn,x0,y0,d,sp,sn=m(Xte)
p=p.numpy();d=d.numpy();y=Yte.numpy();res={'rate':RATES[RATE],'rate_idx':RATE,'fold':FOLD,'epoch_reached':end,'best_epoch':best_epoch+1,'best_val_loss':best,'no_improvement':noimp,'lr':float(opt.param_groups[0]['lr']),'stopped':stopped,'elapsed_chunk_sec':time.time()-t0,'soh_pct':met(p[:,-1,1]*100,y[:,-1,1]*100),'q_mAh':met(p[:,:,1]*4840,y[:,:,1]*4840),'voltage_mV':met(p[:,1:,0]*4200,y[:,1:,0]*4200),'derived_voltage_mV':met(d[:,1:]*4200,y[:,1:,0]*4200)}
with open(f'{OUT}/{RUN}_e{end}.json','w') as f:json.dump(res,f,indent=2)
print(json.dumps(res),flush=True)