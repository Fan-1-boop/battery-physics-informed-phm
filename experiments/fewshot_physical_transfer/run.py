"""Target-label-efficient C/5 transfer from Che et al. (2025) pretrained DVA model.

Independent research implementation, NOT an official paper result.
No target test labels/inputs enter adaptation, selection, or early stopping.
Dataset 1 -> Dataset 2 has overlapping physical cells (not cross-cell generalization).
"""
from __future__ import annotations
import argparse, copy, csv, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold

RATE_IDX = 5
Q_NOM_AH = 4.84
V_NOM_V = 4.2


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(600, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        for name in ('cp_out', 'cn_out', 'x0_out', 'y0_out'):
            setattr(self, name, nn.Linear(64, 1))
    def forward(self, x):
        a = x.reshape(x.shape[0], -1)
        a = torch.relu(self.fc1(a))
        a = torch.relu(self.fc2(a))
        a = torch.relu(self.fc3(a))
        return [torch.relu(getattr(self, k)(a)) for k in ('cp_out', 'cn_out', 'x0_out', 'y0_out')]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,256)
        self.fcv = nn.Linear(256,200)
        self.fcq = nn.Linear(256,200)
    def forward(self, latents):
        x = torch.cat(latents, dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.stack([self.fcv(x), self.fcq(x)], dim=-1)


def interpolate(x, xp, fp):
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, xp.numel()-1)
    x0, x1 = xp[idx-1], xp[idx]
    return fp[idx-1]+(fp[idx]-fp[idx-1])*(x-x0)/(x1-x0)


class PhysicalModel(nn.Module):
    def __init__(self, anode, cathode):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        for prefix, table in (('n', anode), ('p', cathode)):
            soc = np.asarray(table['SOC_linspace'], dtype=np.float32)
            volt = np.asarray(table['Voltage'], dtype=np.float32)
            if np.any(np.diff(soc) <= 0):
                raise ValueError(f'{prefix} SOC must strictly increase')
            self.register_buffer(f'{prefix}s', torch.tensor(soc), persistent=False)
            self.register_buffer(f'{prefix}v', torch.tensor(volt), persistent=False)
    def forward(self, x):
        lat = self.encoder(x)
        pred = self.decoder(lat)
        cp, cn, x0, y0 = lat
        sp = y0 - pred[:,:,1]/cp.clamp_min(1e-6)
        sn = x0 - pred[:,:,1]/cn.clamp_min(1e-6)
        derived = (interpolate(sp, self.ps, self.pv) - interpolate(sn, self.ns, self.nv))/V_NOM_V
        return pred, derived, lat, sp, sn


def bound_loss(pred, derived, lat, sp, sn):
    cp, cn, _, _ = lat
    rel = lambda x, low, high: (torch.relu(low-x) + torch.relu(x-high)).mean()
    return (rel(cp,.1,1.1)+rel(cn,.1,1.1)+rel(sp,0,1)+rel(sn,0,1)
            +rel(pred[:,:,0],2.7/V_NOM_V,1)+rel(derived,2.7/V_NOM_V,1)
            +torch.relu(-pred[:,:,1]).mean()+torch.relu(pred[:,0,1]).mean())


def losses(m, x, target=None):
    pred, dv, lat, sp, sn = m(x)
    phy = ((pred[:,:,0]-dv)**2).mean()
    bound = bound_loss(pred,dv,lat,sp,sn)
    if target is None:
        return phy, bound
    reg = ((pred[:,:,0]-target[:,:,0])**2).mean()+((pred[:,:,1]-target[:,:,1])**2).mean()
    return reg, phy, bound


def evaluate(m, x, y):
    with torch.no_grad():
        pred, derived, lat, sp, sn = m(x)
    p=pred.numpy(); d=derived.numpy(); t=y.numpy()
    cp,cn,_,_= [v.numpy() for v in lat]
    s_p=sp.numpy(); s_n=sn.numpy()
    mae=lambda a,b:float(np.mean(np.abs(a-b)))
    violation=lambda a,lo,hi:float(100*np.mean((a < lo)|(a > hi)))
    return dict(soh_mae_pp=mae(p[:,-1,1]*100,t[:,-1,1]*100),
         soh_rmse_pp=float(np.mean(((p[:,-1,1]-t[:,-1,1])*100)**2)**.5),
         q_mae_mAh=mae(p[:,:,1]*4840,t[:,:,1]*4840),
         vpred_mae_mV=mae(p[:,1:,0]*4200,t[:,1:,0]*4200),
         vderived_mae_mV=mae(d[:,1:]*4200,t[:,1:,0]*4200),
         vconsistency_mae_mV=mae(p[:,1:,0]*4200,d[:,1:]*4200),
         cp_violation_pct=violation(cp,.1,1.1),cn_violation_pct=violation(cn,.1,1.1),
         sp_violation_pct=violation(s_p,0,1),sn_violation_pct=violation(s_n,0,1))


def split_data(n, fold, k, repeat, outer_seed=0):
    if not 0<=fold<5: raise ValueError('fold must be 0..4')
    tr, te = list(KFold(n_splits=5, shuffle=True,random_state=outer_seed).split(np.arange(n)))[fold]
    rng=np.random.default_rng(1000+repeat*173+fold*29)
    lab=np.sort(rng.choice(tr, size=min(k,len(tr)), replace=False))
    unlab=np.setdiff1d(tr,lab)
    assert not (set(lab) & set(unlab))
    assert not (set(tr) & set(te))
    assert not (set(te) & (set(lab)|set(unlab)))
    return tr, te, lab, unlab


def run_once(m0, x, y, fold, k, repeat, mode, epochs, lr, lamb_prox, unlabeled_weight, label_seed=None):
    tr, te, lab, unlab = split_data(x.shape[0],fold,k,repeat)
    seed_all(123+fold*17+repeat*1009)
    model=copy.deepcopy(m0)
    if mode == 'zero':
        return dict(fold=fold,repeat=repeat,k=0,mode=mode,epochs=0,n_train=len(tr),n_test=len(te),
                    label_ids='',test_ids=';'.join(map(str,te)),train_ids=';'.join(map(str,tr)),
                    **evaluate(model,x[te],y[te]))
    assert mode in ('full','encoder','encoder_prox','encoder_prox_unlab')
    for p in model.decoder.parameters(): p.requires_grad=(mode=='full')
    params=[p for p in model.parameters() if p.requires_grad]
    orig=[p.detach().clone() for p in params]
    optimizer=torch.optim.AdamW(params,lr=lr)
    xt,yt=x[lab],y[lab]
    xu=x[unlab]
    t0=time.time()
    for ep in range(epochs):
        model.train()
        reg,phy,bound=losses(model,xt,yt)
        total=reg+phy+0.1*bound
        if mode in ('encoder_prox','encoder_prox_unlab'):
            prox=sum(torch.mean((p-v)**2) for p,v in zip(params,orig))
            total=total+lamb_prox*prox
        if mode=='encoder_prox_unlab':
            pu,bu=losses(model,xu)
            total=total+unlabeled_weight*(pu+0.1*bu)
        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(params,1)
        optimizer.step()
    model.eval()
    res=evaluate(model,x[te],y[te])
    return dict(fold=fold,repeat=repeat,k=k,mode=mode,epochs=epochs,n_train=len(tr),n_test=len(te),
                label_ids=';'.join(map(str,lab)),test_ids=';'.join(map(str,te)),
                train_ids=';'.join(map(str,tr)), elapsed_sec=round(time.time()-t0,2), **res)


def load_model(assets_dir):
    a=pd.read_csv(assets_dir/'anode_SiO_Gr_discharge_Cover5_smoothed_dvdq_JS.csv')
    c=pd.read_csv(assets_dir/'cathode_NCA_discharge_Cover5_smoothed_dvdq_JS.csv')
    model=PhysicalModel(a,c)
    src=torch.load(assets_dir/'best_model.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(src, strict=True)
    model.eval()
    return model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--folds',default='0,1')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--shots',default='4,14')
    parser.add_argument('--modes',default='full,encoder,encoder_prox,encoder_prox_unlab')
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--prox',type=float,default=0.01)
    parser.add_argument('--unlabeled-weight',type=float,default=0.1)
    args=parser.parse_args()
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    d=np.load(args.cache,allow_pickle=False)
    x=torch.tensor(d['X'],dtype=torch.float32)
    y=torch.tensor(d['Y5'],dtype=torch.float32)
    if x.shape!=(91,200,3) or y.shape!=(91,200,2): raise ValueError(f'unexpected cache shape {x.shape},{y.shape}')
    model=load_model(args.assets)
    args.out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.out/'protocol.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    folds=list(map(int,args.folds.split(',')))
    shots=list(map(int,args.shots.split(',')))
    modes=args.modes.split(',')
    runs=[]
    for fold in folds:
        runs.append(run_once(model,x,y,fold,0,0,'zero',0,args.lr,args.prox,args.unlabeled_weight))
        for k in shots:
            for rep in range(args.repeats):
                for mode in modes:
                    q=run_once(model,x,y,fold,k,rep,mode,args.epochs,args.lr,args.prox,args.unlabeled_weight)
                    runs.append(q)
                    print(json.dumps({a:q[a] for a in ('fold','repeat','k','mode','soh_mae_pp','vconsistency_mae_mV','elapsed_sec')}),flush=True)
                    pd.DataFrame(runs).to_csv(args.out/'per_run.csv',index=False)
    df=pd.DataFrame(runs)
    df.to_csv(args.out/'per_run.csv',index=False)
    ag=df.groupby(['k','mode']).agg(n=('soh_mae_pp','size'),soh_mae_mean=('soh_mae_pp','mean'),
        soh_mae_std=('soh_mae_pp','std'),consistency_mV=('vconsistency_mae_mV','mean'),
        cp_viol_pct=('cp_violation_pct','mean'),sp_viol_pct=('sp_violation_pct','mean')).reset_index()
    ag.to_csv(args.out/'summary.csv',index=False)
    print(ag.to_string(index=False),flush=True)

if __name__=='__main__':main()