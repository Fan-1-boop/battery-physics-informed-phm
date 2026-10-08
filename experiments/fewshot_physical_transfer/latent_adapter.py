"""Four-parameter, physically constrained latent domain shift adapter.

Pre-registered structure: z_target = z_source + 0.25*tanh(delta), where
z=(C_p,C_n,x_0,y_0). Encoder and decoder are frozen. Comparison is
on the exact same held-out splits as run.py, with no target-test access.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from run import load_model,losses,evaluate,seed_all,split_data,interpolate,V_NOM_V

class ShiftedModel(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.base=source
        for p in self.base.parameters(): p.requires_grad_(False)
        self.delta=nn.Parameter(torch.zeros(4))
    def forward(self,x):
        base=self.base
        original=base.encoder(x)
        offsets=0.25*torch.tanh(self.delta)
        cp,cn,x0,y0=[torch.relu(z+offsets[j]) for j,z in enumerate(original)]
        lat=[cp,cn,x0,y0]
        pred=base.decoder(lat)
        sp=y0-pred[:,:,1]/cp.clamp_min(1e-6)
        sn=x0-pred[:,:,1]/cn.clamp_min(1e-6)
        derived=(interpolate(sp,base.ps,base.pv)-interpolate(sn,base.ns,base.nv))/V_NOM_V
        return pred,derived,lat,sp,sn

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--shots',default='4,14')
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--lr',type=float,default=0.01)
    args=p.parse_args()
    torch.set_num_threads(1)
    d=np.load(args.cache,allow_pickle=False)
    X=torch.tensor(d['X'],dtype=torch.float32)
    Y=torch.tensor(d['Y5'],dtype=torch.float32)
    src=load_model(args.assets)
    rows=[]
    for fold in range(5):
        for k in map(int,args.shots.split(',')):
            for rep in range(args.repeats):
                _,te,lab,_=split_data(len(X),fold,k,rep)
                seed_all(123+fold*17+rep*1009)
                m=ShiftedModel(copy.deepcopy(src))
                opt=torch.optim.Adam([m.delta],lr=args.lr)
                for i in range(args.epochs):
                    reg,phy,bd=losses(m,X[lab],Y[lab])
                    # physically constrained adaptation + zero-mean prior on corrections
                    obj=reg+phy+0.1*bd+0.01*torch.square(0.25*torch.tanh(m.delta)).mean()
                    opt.zero_grad();obj.backward();opt.step()
                m.eval()
                metric=evaluate(m,X[te],Y[te])
                rows.append(dict(fold=fold,repeat=rep,k=k,mode='latent_shift_4d',epochs=args.epochs,
                    label_ids=';'.join(map(str,lab)),test_ids=';'.join(map(str,te)),
                    delta_cp=float((.25*torch.tanh(m.delta[0])).item()),
                    delta_cn=float((.25*torch.tanh(m.delta[1])).item()),
                    delta_x0=float((.25*torch.tanh(m.delta[2])).item()),
                    delta_y0=float((.25*torch.tanh(m.delta[3])).item()),**metric))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(args.out,index=False)
    print(df.groupby('k').agg(soh_mae=('soh_mae_pp','mean'),soh_std=('soh_mae_pp','std'),
        consistency=('vconsistency_mae_mV','mean'),cp_viol=('cp_violation_pct','mean')).to_string())

if __name__=='__main__':main()