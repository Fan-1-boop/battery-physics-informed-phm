"""Structural smoke test for the Che et al. (Joule 2025) core idea.

IMPORTANT: this uses synthetic, DVA-consistent data. It verifies the implementation
of partial-fragment -> physical latent states -> full pseudo-OCV reconstruction,
not the paper's numerical claims.
"""
from __future__ import annotations
import json, math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED=20251007
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# State order follows paper language: Cn, Cp, x0, y0.
BOUNDS=torch.tensor([[3.4,5.4],[3.4,5.4],[0.70,0.98],[0.02,0.30]],dtype=torch.float32)
NGRID=64
NFRAG=32


def ocp_p_t(y):
    # synthetic cathode OCP, V
    return 4.36 - 1.05*y + 0.055*torch.tanh((0.52-y)*10.0) + 0.025*torch.tanh((0.82-y)*25.0)

def ocp_n_t(x):
    # synthetic graphite/SiOx-like anode OCP, V
    return 0.07 + 0.72*(1.0-x) + 0.035*torch.tanh((0.40-x)*18.0) + 0.018*torch.tanh((0.72-x)*28.0)

def dva_curve_t(theta, s):
    """DVA-like full-cell curve at normalized discharged fraction s in [0,1]."""
    Cn,Cp,x0,y0=[theta[:,i:i+1] for i in range(4)]
    qn=Cn*(x0-0.08)
    qp=Cp*(0.92-y0)
    qmax=torch.minimum(qn,qp).clamp_min(2.5)
    q=s[None,:]*qmax
    x=x0-q/Cn
    y=y0+q/Cp
    v=ocp_p_t(y)-ocp_n_t(x)
    return qmax.squeeze(1), v

def ocp_p_np(y):
    return 4.36 - 1.05*y + 0.055*np.tanh((0.52-y)*10.0) + 0.025*np.tanh((0.82-y)*25.0)

def ocp_n_np(x):
    return 0.07 + 0.72*(1.0-x) + 0.035*np.tanh((0.40-x)*18.0) + 0.018*np.tanh((0.72-x)*28.0)

def dva_np(th, s):
    Cn,Cp,x0,y0=th
    qmax=max(2.5,min(Cn*(x0-0.08),Cp*(0.92-y0)))
    q=s*qmax
    x=x0-q/Cn; y=y0+q/Cp
    return qmax, ocp_p_np(y)-ocp_n_np(x)

@dataclass
class Synth:
    X: np.ndarray; theta: np.ndarray; qmax: np.ndarray; V: np.ndarray; group: np.ndarray


def make_data(n=1800):
    rows=[]; thetas=[]; qmaxs=[]; Vs=[]; groups=[]
    s=np.linspace(0,1,NGRID)
    # Five protocol groups with different rate/SOC-window distortions. Group 4 is held out.
    for i in range(n):
        g=i%5
        efc=np.random.uniform(0,1300)
        z=efc/1300
        # correlated but not identical degradation modes
        Cn=5.15-1.05*z+np.random.normal(0,.045)
        Cp=5.05-0.55*z+np.random.normal(0,.035)
        x0=.94-.105*z+np.random.normal(0,.006)
        y0=.085+.075*z+np.random.normal(0,.005)
        th=np.array([Cn,Cp,x0,y0],dtype=np.float32)
        qm,v=dva_np(th,s)
        # Realistic model discrepancy: small ageing- and protocol-dependent distortion.
        residual=(0.004+0.003*g)*np.sin(np.pi*s)*(0.3+z) + 0.002*np.sin(5*np.pi*s+0.4*g)
        v_target=v+residual
        # Operational charging curve: reverse equilibrium curve plus rate/protocol overpotential.
        rate=[0.2,0.5,1.0,1.5,2.0][g]
        qcharge=qm*s
        vcharge=v_target[::-1] + (0.008+0.010*rate) + 0.003*z
        # Random fragment: start/end independently varying; at least 18% of full span.
        a=np.random.uniform(0.0,0.68); width=np.random.uniform(0.18,min(0.58,1-a)); b=a+width
        qf=np.linspace(a*qm,b*qm,NFRAG)
        vf=np.interp(qf,qcharge,vcharge)
        # Input exactly mirrors the paper's information categories: voltage, capacity, EFC.
        x=np.concatenate([vf, qf, [efc/1300.0]]).astype(np.float32)
        rows.append(x); thetas.append(th); qmaxs.append(qm); Vs.append(v_target.astype(np.float32)); groups.append(g)
    return Synth(np.stack(rows),np.stack(thetas),np.asarray(qmaxs,np.float32),np.stack(Vs),np.asarray(groups))

class Core(nn.Module):
    def __init__(self, physics=True):
        super().__init__(); self.physics=physics
        nin=2*NFRAG+1
        self.enc=nn.Sequential(nn.Linear(nin,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4))
        self.dec=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,128),nn.SiLU(),nn.Linear(128,1+NGRID))
        self.register_buffer('lo',BOUNDS[:,0]); self.register_buffer('hi',BOUNDS[:,1])
        self.register_buffer('s',torch.linspace(0,1,NGRID))
    def forward(self,x):
        raw=self.enc(x)
        theta=self.lo+(self.hi-self.lo)*torch.sigmoid(raw)
        y=self.dec(theta)
        # bounded q prediction and voltage around physically plausible scale
        qpred=2.5+3.0*torch.sigmoid(y[:,0])
        vpred=2.6+2.0*torch.sigmoid(y[:,1:])
        qd,vd=dva_curve_t(theta,self.s)
        return theta,qpred,vpred,qd,vd

def train_once(physics=True, epochs=180):
    data=make_data()
    test=data.group==4; train=~test
    # validation = 20% of training samples, as paper states (sample-level inner validation)
    idx=np.where(train)[0]; rng=np.random.default_rng(SEED); rng.shuffle(idx)
    nv=int(.2*len(idx)); va=idx[:nv]; tr=idx[nv:]; te=np.where(test)[0]
    # normalize inputs using train only
    mu=data.X[tr].mean(0); sd=data.X[tr].std(0)+1e-6
    X=(data.X-mu)/sd
    def ds(ix):
        return TensorDataset(torch.from_numpy(X[ix]),torch.from_numpy(data.qmax[ix]),torch.from_numpy(data.V[ix]),torch.from_numpy(data.theta[ix]))
    dl=DataLoader(ds(tr),batch_size=64,shuffle=True)
    model=Core(physics=physics).to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    best=None; bad=0
    for ep in range(epochs):
        model.train()
        for xb,qb,vb,tb in dl:
            xb,qb,vb=xb.to(DEVICE),qb.to(DEVICE),vb.to(DEVICE)
            th,qp,vp,qd,vd=model(xb)
            # normalized regression terms; physical loss in voltage space
            lq=((qp-qb)/4.8).pow(2).mean(); lv=((vp-vb)/1.5).pow(2).mean()
            lreg=lq+lv
            lphy=((vp-vd)/1.5).pow(2).mean()
            loss=lreg+(1.0*lphy if physics else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        # early stopping
        model.eval(); vals=[]
        with torch.no_grad():
            for xb,qb,vb,tb in DataLoader(ds(va),batch_size=256):
                xb,qb,vb=xb.to(DEVICE),qb.to(DEVICE),vb.to(DEVICE)
                th,qp,vp,qd,vd=model(xb)
                lreg=((qp-qb)/4.8).pow(2).mean()+((vp-vb)/1.5).pow(2).mean()
                lphy=((vp-vd)/1.5).pow(2).mean()
                vals.append(float(lreg+(lphy if physics else 0)))
        val=np.mean(vals)
        if best is None or val<best[0]-1e-6:
            best=(val,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},ep); bad=0
        else:
            bad+=1
        if bad>=25: break
    model.load_state_dict(best[1]); model.eval()
    out=[]
    with torch.no_grad():
        for batch in DataLoader(ds(te),batch_size=256):
            xb,qb,vb,tb=batch; pred=model(xb.to(DEVICE))
            out.append((*[x.cpu().numpy() for x in pred],qb.numpy(),vb.numpy(),tb.numpy()))
    th=np.concatenate([o[0] for o in out]); qp=np.concatenate([o[1] for o in out]); vp=np.concatenate([o[2] for o in out]); qd=np.concatenate([o[3] for o in out]); vd=np.concatenate([o[4] for o in out])
    qt=np.concatenate([o[5] for o in out]); vt=np.concatenate([o[6] for o in out]); tt=np.concatenate([o[7] for o in out])
    # pseudo-SOH normalized to 4.84 Ah, mirroring paper's nominal capacity.
    soh_t=100*qt/4.84; soh_p=100*qp/4.84
    metrics={
        'physics':physics,'device':str(DEVICE),'best_epoch':int(best[2]),'n_train':len(tr),'n_val':len(va),'n_test':len(te),
        'Q_MAE_mAh':float(np.mean(np.abs(qp-qt))*1000),
        'Q_RMSE_mAh':float(np.sqrt(np.mean((qp-qt)**2))*1000),
        'SOH_MAE_pctpoint':float(np.mean(np.abs(soh_p-soh_t))),
        'Vpred_MAE_mV':float(np.mean(np.abs(vp-vt))*1000),
        'Vderived_MAE_mV':float(np.mean(np.abs(vd-vt))*1000),
        'pred_vs_derived_MAE_mV':float(np.mean(np.abs(vp-vd))*1000),
        'theta_MAE':{k:float(np.mean(np.abs(th[:,i]-tt[:,i]))) for i,k in enumerate(['Cn_Ah','Cp_Ah','x0','y0'])},
        'Q_R2':float(r2_score(qt,qp)),
    }
    return metrics

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    m1=train_once(True); m0=train_once(False)
    res={'disclaimer':'Synthetic structural smoke test only; not a numerical reproduction of Che et al. 2025.','with_physics':m1,'without_physics':m0}
    (root/'results'/'synthetic_smoke_metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2))